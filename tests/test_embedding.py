"""Tests for ViTPatchEmbedding: shapes, gradients, parameter registration,
and the linear-projection / strided-convolution equivalence."""

import pytest
import torch
from torch import nn

from vit_tokenizer import ViTPatchEmbedding, patchify, sinusoidal_positional_encoding


def make_module(**overrides):
    config = dict(image_size=32, patch_size=16, in_channels=3, d_model=64)
    config.update(overrides)
    return ViTPatchEmbedding(**config)


class TestShapes:
    def test_batched_output(self):
        module = make_module()
        tokens = module(torch.randn(8, 3, 32, 32))
        assert tokens.shape == (8, 4 + 1, 64)  # 4 patches + [CLS]

    def test_single_image_output(self):
        module = make_module()
        tokens = module(torch.randn(3, 32, 32))
        assert tokens.shape == (5, 64)

    def test_vit_base_dimensions(self):
        """The canonical ViT-Base/16 configuration: 224^2 images, 16^2 patches
        -> 196 patches, 197 tokens."""
        module = ViTPatchEmbedding(image_size=224, patch_size=16, d_model=768)
        assert module.num_patches == 196
        assert module.sequence_length == 197

    def test_rectangular_image(self):
        module = make_module(image_size=(32, 64))
        tokens = module(torch.randn(2, 3, 32, 64))
        assert tokens.shape == (2, 8 + 1, 64)

    def test_wrong_input_size_raises(self):
        module = make_module()
        with pytest.raises(ValueError, match="Expected images"):
            module(torch.randn(1, 3, 64, 64))

    def test_indivisible_config_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            make_module(image_size=33)


class TestParameters:
    def test_learnable_parameters_registered(self):
        """The PoC kept these on a Dataset, invisible to any optimizer. As an
        nn.Module the projection weight/bias and class token must all appear
        in parameters()."""
        module = make_module()
        names = {name for name, _ in module.named_parameters()}
        assert names == {"projection.weight", "projection.bias", "class_token"}

    def test_sinusoidal_table_is_buffer_not_parameter(self):
        module = make_module(positional_encoding="sinusoidal-1d")
        assert "positional_table" in dict(module.named_buffers())
        assert "positional_table" not in dict(module.named_parameters())

    def test_learnable_table_is_parameter(self):
        module = make_module(positional_encoding="learnable")
        assert "positional_table" in dict(module.named_parameters())

    def test_bias_starts_at_zero(self):
        module = make_module()
        assert torch.equal(module.projection.bias, torch.zeros(64))

    def test_gradients_flow_to_all_parameters(self):
        module = make_module()
        tokens = module(torch.randn(2, 3, 32, 32))
        tokens.sum().backward()
        for name, parameter in module.named_parameters():
            assert parameter.grad is not None, f"no gradient reached {name}"
            assert parameter.grad.abs().sum() > 0, f"zero gradient at {name}"

    def test_unknown_encoding_raises(self):
        with pytest.raises(ValueError, match="positional_encoding"):
            make_module(positional_encoding="fourier")


class TestForwardSemantics:
    def test_class_token_prepended(self):
        """With 'none' positional encoding, token 0 must be exactly the class
        token for every image in the batch."""
        module = make_module(positional_encoding="none")
        tokens = module(torch.randn(3, 3, 32, 32))
        for b in range(3):
            assert torch.allclose(tokens[b, 0], module.class_token[0, 0])

    def test_positional_encoding_added(self):
        """tokens(sinusoidal) - tokens(none) must equal the encoding table."""
        torch.manual_seed(0)
        with_pe = make_module(positional_encoding="sinusoidal-1d")
        torch.manual_seed(0)
        without_pe = make_module(positional_encoding="none")
        images = torch.randn(2, 3, 32, 32)
        difference = with_pe(images) - without_pe(images)
        expected = sinusoidal_positional_encoding(5, 64)
        assert torch.allclose(difference[0], expected, atol=1e-5)
        assert torch.allclose(difference[1], expected, atol=1e-5)

    def test_forward_does_not_mutate_buffer(self):
        """The PoC added the encoding in place; repeated passes would then
        accumulate garbage. The table must be identical across calls."""
        module = make_module()
        before = module.positional_table.clone()
        for _ in range(3):
            module(torch.randn(2, 3, 32, 32))
        assert torch.equal(module.positional_table, before)

    def test_deterministic_given_seed(self):
        torch.manual_seed(7)
        first = make_module()
        torch.manual_seed(7)
        second = make_module()
        images = torch.randn(2, 3, 32, 32)
        assert torch.equal(first(images), second(images))

    def test_projection_equals_strided_convolution(self):
        """docs/MATH.md section 3.2: a linear layer on flattened patches is
        exactly a Conv2d with kernel_size = stride = P. Copying the linear
        weight into the conv kernel must reproduce the projected patches."""
        torch.manual_seed(0)
        patch_size, in_channels, d_model = 8, 3, 32
        images = torch.randn(2, in_channels, 24, 24)

        linear = nn.Linear(in_channels * patch_size**2, d_model)
        conv = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        with torch.no_grad():
            conv.weight.copy_(linear.weight.view(d_model, in_channels, patch_size, patch_size))
            conv.bias.copy_(linear.bias)

        via_linear = linear(patchify(images, patch_size))  # (B, N, d)
        via_conv = conv(images).flatten(2).transpose(1, 2)  # (B, d, gh, gw) -> (B, N, d)
        assert torch.allclose(via_linear, via_conv, atol=1e-5)


class TestPositionalVariants:
    @pytest.mark.parametrize(
        "kind", ["sinusoidal-1d", "sinusoidal-2d", "learnable", "none"]
    )
    def test_all_variants_run(self, kind):
        module = make_module(positional_encoding=kind)
        tokens = module(torch.randn(2, 3, 32, 32))
        assert tokens.shape == (2, 5, 64)

    def test_2d_cls_position_is_zero_vector(self):
        module = make_module(positional_encoding="sinusoidal-2d")
        assert torch.equal(module.positional_table[0, 0], torch.zeros(64))

    def test_2d_table_matches_grid_encoding(self):
        from vit_tokenizer import sinusoidal_positional_encoding_2d

        module = make_module(positional_encoding="sinusoidal-2d")
        expected = sinusoidal_positional_encoding_2d(2, 2, 64)
        assert torch.equal(module.positional_table[0, 1:], expected)
