"""Tests for the MiniViT model: shapes, gradient flow, positional variants,
and — most importantly — evidence that the whole stack actually learns."""

import pytest
import torch

from vit_tokenizer import MiniViT, TransformerEncoderBlock


def make_model(**overrides):
    config = dict(
        image_size=32, patch_size=8, d_model=32, depth=2, num_heads=4, num_classes=3
    )
    config.update(overrides)
    return MiniViT(**config)


class TestEncoderBlock:
    def test_shape_preserved(self):
        block = TransformerEncoderBlock(d_model=32, num_heads=4)
        tokens = torch.randn(2, 10, 32)
        assert block(tokens).shape == tokens.shape

    def test_returns_attention(self):
        block = TransformerEncoderBlock(d_model=32, num_heads=4)
        tokens, weights = block(torch.randn(2, 10, 32), return_attention=True)
        assert tokens.shape == (2, 10, 32)
        assert weights.shape == (2, 4, 10, 10)

    def test_residual_path_exists(self):
        """Pre-norm blocks keep an identity path: zeroing every weight must
        reduce the block to (nearly) the identity function."""
        block = TransformerEncoderBlock(d_model=32, num_heads=4)
        with torch.no_grad():
            for parameter in block.parameters():
                parameter.zero_()
        tokens = torch.randn(2, 10, 32)
        assert torch.allclose(block(tokens), tokens, atol=1e-6)


class TestMiniViT:
    def test_logits_shape(self):
        model = make_model()
        assert model(torch.randn(4, 3, 32, 32)).shape == (4, 3)

    def test_single_image(self):
        model = make_model()
        assert model(torch.randn(3, 32, 32)).shape == (3,)

    @pytest.mark.parametrize(
        "kind", ["sinusoidal-1d", "sinusoidal-2d", "learnable", "none", "rotary"]
    )
    def test_all_positional_variants_run(self, kind):
        model = make_model(positional_encoding=kind)
        assert model(torch.randn(2, 3, 32, 32)).shape == (2, 3)

    def test_rotary_disables_additive_table(self):
        model = make_model(positional_encoding="rotary")
        assert torch.equal(
            model.embedding.positional_table,
            torch.zeros_like(model.embedding.positional_table),
        )
        assert all(block.attention.rope_base is not None for block in model.blocks)

    def test_attention_maps_returned(self):
        model = make_model(depth=3)
        _, attention_maps = model(torch.randn(2, 3, 32, 32), return_attention=True)
        assert len(attention_maps) == 3
        sequence_length = model.embedding.sequence_length
        for weights in attention_maps:
            assert weights.shape == (2, 4, sequence_length, sequence_length)

    def test_gradients_reach_every_parameter(self):
        """One backward pass must touch everything — including the tokenizer's
        projection and class token, the original PoC's blind spot."""
        model = make_model()
        logits = model(torch.randn(2, 3, 32, 32))
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
        for name, parameter in model.named_parameters():
            assert parameter.grad is not None, f"no gradient reached {name}"

    def test_vit_base_parameter_count(self):
        """docs/MATH.md section 11.3: the ~12 d^2 L + d(C P^2 + 1) estimate
        should land within a few percent of ViT-Base's published ~86M."""
        model = MiniViT(
            image_size=224, patch_size=16, d_model=768, depth=12,
            num_heads=12, num_classes=1000, positional_encoding="learnable",
        )
        total = sum(p.numel() for p in model.parameters())
        assert 84e6 < total < 89e6

    def test_overfits_tiny_batch(self):
        """The canonical 'does it learn?' test: a model that cannot drive the
        loss to ~zero on 8 fixed samples has a broken gradient path somewhere.
        Exercises the full stack: patchify -> projection -> [CLS] -> positions
        -> attention -> MLP -> head."""
        torch.manual_seed(0)
        model = make_model(d_model=64)
        images = torch.randn(8, 3, 32, 32)
        labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        initial_loss = None
        for _ in range(150):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(images), labels)
            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < 0.05, f"loss only reached {final_loss:.4f}"
        assert final_loss < initial_loss / 10
        assert (model(images).argmax(dim=1) == labels).all()
