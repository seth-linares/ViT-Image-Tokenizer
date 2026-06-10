"""vit_tokenizer: a tested, documented image tokenizer for Vision Transformers.

Turns images into the token sequences a ViT consumes:

    images (B, C, H, W) -> patchify -> project -> [CLS] -> + positions
                        -> tokens (B, N+1, d_model)

Quickstart::

    import torch
    from vit_tokenizer import ViTPatchEmbedding

    tokenizer = ViTPatchEmbedding(image_size=224, patch_size=16, d_model=512)
    images = torch.randn(8, 3, 224, 224)
    tokens = tokenizer(images)          # (8, 197, 512)

The math behind every step is derived in ``docs/MATH.md`` and verified
numerically by the test suite in ``tests/``.
"""

from .dataset import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    Compose,
    ImageFolderDataset,
    Normalize,
    ResizeAndPad,
    default_transform,
    to_tensor,
)
from .embedding import ViTPatchEmbedding
from .patching import patch_grid_shape, patchify, unpatchify
from .positional import (
    frequency_bands,
    sinusoidal_positional_encoding,
    sinusoidal_positional_encoding_2d,
)

__version__ = "1.0.0"

__all__ = [
    "IMAGENET_MEAN",
    "IMAGENET_STD",
    "Compose",
    "ImageFolderDataset",
    "Normalize",
    "ResizeAndPad",
    "ViTPatchEmbedding",
    "default_transform",
    "frequency_bands",
    "patch_grid_shape",
    "patchify",
    "sinusoidal_positional_encoding",
    "sinusoidal_positional_encoding_2d",
    "to_tensor",
    "unpatchify",
]
