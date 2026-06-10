"""Patch extraction and reconstruction for Vision Transformers.

A ViT consumes an image not as a pixel grid but as a *sequence* of flattened
patches. For an image ``x`` of shape ``(C, H, W)`` and patch size ``P`` (with
``P | H`` and ``P | W``), the image is cut into

    N = (H / P) * (W / P)

non-overlapping ``P x P`` blocks, and each block is flattened into a vector of
length ``C * P^2``. The result is a matrix of shape ``(N, C * P^2)`` whose rows
are ordered row-major over the patch grid (left-to-right, top-to-bottom),
matching the raster order used by the original ViT paper.

Within a single patch vector, the layout is ``(C, P, P)`` flattened row-major:
all red pixels first, then green, then blue. This is exactly the layout
``nn.Conv2d`` uses for its kernels, which is what makes the linear projection
in :mod:`vit_tokenizer.embedding` provably equivalent to a strided convolution
(see ``docs/MATH.md`` section 3 and the test
``tests/test_embedding.py::test_projection_equals_strided_convolution``).

See ``docs/MATH.md`` section 2 for the full derivation.
"""

from __future__ import annotations

import torch

__all__ = ["patch_grid_shape", "patchify", "unpatchify"]


def patch_grid_shape(image_size: int | tuple[int, int], patch_size: int) -> tuple[int, int]:
    """Return ``(grid_h, grid_w)``, the number of patches along each axis.

    Raises ``ValueError`` if the image is not evenly divisible into patches,
    because silently cropping or padding would change the data without the
    caller knowing. Resize the image first (e.g. with
    :class:`vit_tokenizer.dataset.ResizeAndPad`).
    """
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
    height, width = image_size
    if patch_size <= 0:
        raise ValueError(f"patch_size must be positive, got {patch_size}")
    if height % patch_size != 0 or width % patch_size != 0:
        raise ValueError(
            f"Image size {height}x{width} is not divisible by patch size "
            f"{patch_size}. Resize or pad the image so both dimensions are "
            f"multiples of {patch_size}."
        )
    return height // patch_size, width // patch_size


def patchify(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Split image(s) into a sequence of flattened non-overlapping patches.

    Args:
        images: ``(C, H, W)`` or batched ``(B, C, H, W)`` tensor.
        patch_size: side length ``P`` of the square patches.

    Returns:
        ``(N, C*P^2)`` for a single image or ``(B, N, C*P^2)`` for a batch,
        where ``N = (H/P) * (W/P)``. Patches appear in raster order; each row
        is one patch flattened in ``(C, P, P)`` order.

    ``unfold(dim, size, step)`` extracts sliding windows as a view (no copy):
    one call over the height axis and one over the width axis yields shape
    ``(B, C, H/P, W/P, P, P)``. We then permute the channel axis *inside* the
    patch axes so each row of the output is a complete patch, and call
    ``.contiguous()`` because ``view`` requires the logical layout to match
    physical memory after the permute.
    """
    squeeze_batch = images.dim() == 3
    if squeeze_batch:
        images = images.unsqueeze(0)
    if images.dim() != 4:
        raise ValueError(
            f"Expected images of shape (C, H, W) or (B, C, H, W), got {tuple(images.shape)}"
        )
    batch, channels, height, width = images.shape
    grid_h, grid_w = patch_grid_shape((height, width), patch_size)

    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    # (B, C, grid_h, grid_w, P, P) -> (B, grid_h, grid_w, C, P, P): rows of the
    # flattened output must each hold one whole patch, not one channel slice.
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    patches = patches.view(batch, grid_h * grid_w, channels * patch_size * patch_size)
    return patches.squeeze(0) if squeeze_batch else patches


def unpatchify(
    patches: torch.Tensor,
    patch_size: int,
    image_size: int | tuple[int, int],
) -> torch.Tensor:
    """Inverse of :func:`patchify`: reassemble flattened patches into image(s).

    Args:
        patches: ``(N, C*P^2)`` or ``(B, N, C*P^2)`` tensor as produced by
            :func:`patchify`.
        patch_size: side length ``P`` used when patchifying.
        image_size: original ``H`` (int) or ``(H, W)``.

    Returns:
        ``(C, H, W)`` or ``(B, C, H, W)`` tensor. ``unpatchify(patchify(x))``
        reproduces ``x`` exactly (bit-for-bit; both are pure reindexings).
    """
    squeeze_batch = patches.dim() == 2
    if squeeze_batch:
        patches = patches.unsqueeze(0)
    if patches.dim() != 3:
        raise ValueError(
            f"Expected patches of shape (N, C*P^2) or (B, N, C*P^2), got {tuple(patches.shape)}"
        )
    grid_h, grid_w = patch_grid_shape(image_size, patch_size)
    batch, num_patches, patch_dim = patches.shape
    if num_patches != grid_h * grid_w:
        raise ValueError(
            f"Got {num_patches} patches but image size {image_size} with patch "
            f"size {patch_size} implies {grid_h * grid_w}."
        )
    if patch_dim % (patch_size * patch_size) != 0:
        raise ValueError(
            f"Patch vectors of length {patch_dim} are not divisible by "
            f"P^2={patch_size * patch_size}; wrong patch_size?"
        )
    channels = patch_dim // (patch_size * patch_size)

    images = patches.view(batch, grid_h, grid_w, channels, patch_size, patch_size)
    images = images.permute(0, 3, 1, 4, 2, 5).contiguous()
    images = images.view(batch, channels, grid_h * patch_size, grid_w * patch_size)
    return images.squeeze(0) if squeeze_batch else images
