"""Tests for patch extraction: shapes, ordering, exact invertibility."""

import pytest
import torch

from vit_tokenizer import patch_grid_shape, patchify, unpatchify


class TestPatchGridShape:
    def test_square_image(self):
        assert patch_grid_shape(224, 16) == (14, 14)

    def test_rectangular_image(self):
        assert patch_grid_shape((64, 96), 32) == (2, 3)

    def test_indivisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            patch_grid_shape(225, 16)

    def test_nonpositive_patch_raises(self):
        with pytest.raises(ValueError, match="positive"):
            patch_grid_shape(224, 0)


class TestPatchify:
    def test_single_image_shape(self):
        image = torch.randn(3, 224, 224)
        patches = patchify(image, 16)
        assert patches.shape == (14 * 14, 3 * 16 * 16)

    def test_batched_shape(self):
        images = torch.randn(8, 3, 64, 64)
        patches = patchify(images, 32)
        assert patches.shape == (8, 4, 3 * 32 * 32)

    def test_batched_matches_single(self):
        images = torch.randn(4, 3, 32, 32)
        batched = patchify(images, 16)
        for i in range(4):
            assert torch.equal(batched[i], patchify(images[i], 16))

    def test_matches_naive_crop_loop(self):
        """patchify must agree with the obvious (slow) double loop, including
        raster ordering of patches and (C, P, P) layout within each patch."""
        torch.manual_seed(0)
        image = torch.randn(3, 8, 12)
        patch_size = 4
        patches = patchify(image, patch_size)

        index = 0
        for top in range(0, 8, patch_size):
            for left in range(0, 12, patch_size):
                block = image[:, top : top + patch_size, left : left + patch_size]
                assert torch.equal(patches[index], block.reshape(-1)), (
                    f"patch {index} (top={top}, left={left}) mismatched"
                )
                index += 1
        assert index == patches.shape[0]

    def test_single_channel(self):
        image = torch.randn(1, 16, 16)
        assert patchify(image, 8).shape == (4, 64)

    def test_wrong_rank_raises(self):
        with pytest.raises(ValueError, match="shape"):
            patchify(torch.randn(224, 224), 16)

    def test_indivisible_raises(self):
        with pytest.raises(ValueError, match="not divisible"):
            patchify(torch.randn(3, 224, 224), 15)


class TestUnpatchify:
    @pytest.mark.parametrize("shape,patch_size", [
        ((3, 224, 224), 16),
        ((3, 64, 96), 32),
        ((1, 28, 28), 7),
        ((5, 3, 32, 32), 8),
    ])
    def test_round_trip_is_exact(self, shape, patch_size):
        """patchify and unpatchify are pure reindexings, so the round trip
        must be bit-for-bit exact, not merely allclose."""
        torch.manual_seed(1)
        images = torch.randn(*shape)
        image_size = shape[-2:]
        recovered = unpatchify(patchify(images, patch_size), patch_size, image_size)
        assert torch.equal(recovered, images)

    def test_wrong_patch_count_raises(self):
        with pytest.raises(ValueError, match="patches"):
            unpatchify(torch.randn(5, 3 * 16 * 16), 16, 224)

    def test_indivisible_patch_dim_raises(self):
        with pytest.raises(ValueError, match="P\\^2"):
            unpatchify(torch.randn(16, 100), 16, 64)
