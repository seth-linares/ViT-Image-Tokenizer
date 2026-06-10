"""The ViT patch-embedding module: images in, transformer-ready tokens out.

Pipeline (shapes for a batch of ``B`` images, ``N`` patches, model width ``d``)::

    images   (B, C, H, W)
      | patchify                       split into flattened patches
    patches  (B, N, C*P^2)
      | linear projection  z = x W^T + b
    tokens   (B, N, d)
      | prepend learnable [CLS] token
    tokens   (B, N+1, d)
      | + positional encoding
    tokens   (B, N+1, d)               ready for the transformer encoder

This is an ``nn.Module`` — not a ``Dataset`` — because the projection matrix,
bias, and class token are *learnable parameters*: they must be registered on a
module that is part of the model so the optimizer updates them and gradients
flow through them on every forward pass. (The original PoC stored them on a
``Dataset``, where ``model.parameters()`` never sees them and ``DataLoader``
workers would each train a private copy that is thrown away.)
"""

from __future__ import annotations

import torch
from torch import nn

from .patching import patch_grid_shape, patchify
from .positional import (
    sinusoidal_positional_encoding,
    sinusoidal_positional_encoding_2d,
)

__all__ = ["ViTPatchEmbedding"]

_ENCODING_CHOICES = ("sinusoidal-1d", "sinusoidal-2d", "learnable", "none")


class ViTPatchEmbedding(nn.Module):
    """Tokenize a batch of images for a Vision Transformer.

    Args:
        image_size: input height/width ``H`` (int for square, or ``(H, W)``).
            Must be divisible by ``patch_size``.
        patch_size: side length ``P`` of the square patches.
        in_channels: image channels ``C`` (3 for RGB).
        d_model: transformer width ``d`` that patches are projected to.
        positional_encoding: one of
            ``"sinusoidal-1d"`` — fixed encoding over raster position (the
            original Transformer scheme; the [CLS] token takes position 0 and
            patches take positions 1..N);
            ``"sinusoidal-2d"`` — fixed factorized row/column encoding, which
            respects the image's 2-D geometry (the [CLS] token gets a zero
            vector since it has no spatial location);
            ``"learnable"`` — a trainable ``(N+1, d)`` table, as used by the
            actual ViT paper (Dosovitskiy et al. 2021);
            ``"none"`` — no positional information (for ablation).
        base: wavelength base of the sinusoidal encodings (10,000 in the
            original paper; larger spreads the wavelengths further apart).

    Sinusoidal tables are registered as *buffers*: they travel with the module
    (``.to(device)``, ``state_dict``) but receive no gradients.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int] = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        d_model: int = 512,
        positional_encoding: str = "sinusoidal-1d",
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        if positional_encoding not in _ENCODING_CHOICES:
            raise ValueError(
                f"positional_encoding must be one of {_ENCODING_CHOICES}, "
                f"got {positional_encoding!r}"
            )
        if isinstance(image_size, int):
            image_size = (image_size, image_size)

        self.image_size = image_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.d_model = d_model
        self.positional_encoding_kind = positional_encoding
        self.grid_shape = patch_grid_shape(image_size, patch_size)
        self.num_patches = self.grid_shape[0] * self.grid_shape[1]
        self.patch_dim = in_channels * patch_size * patch_size

        self.projection = nn.Linear(self.patch_dim, d_model)
        # Xavier keeps Var(output) = Var(input) through the projection so
        # activations neither explode nor vanish (docs/MATH.md section 7);
        # the zero bias adds no signal until training says otherwise.
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.zeros_(self.projection.bias)

        # Shaped (1, 1, d) so it broadcasts/expands over the batch in forward.
        # ViT/BERT initialize the class token from a small truncated normal:
        # near zero so it starts as a blank slate, but not exactly zero.
        self.class_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.class_token, std=0.02)

        sequence_length = self.num_patches + 1
        if positional_encoding == "learnable":
            self.positional_table = nn.Parameter(torch.zeros(1, sequence_length, d_model))
            nn.init.trunc_normal_(self.positional_table, std=0.02)
        else:
            if positional_encoding == "sinusoidal-1d":
                table = sinusoidal_positional_encoding(sequence_length, d_model, base)
            elif positional_encoding == "sinusoidal-2d":
                grid_table = sinusoidal_positional_encoding_2d(*self.grid_shape, d_model, base)
                table = torch.cat([torch.zeros(1, d_model), grid_table], dim=0)
            else:  # "none"
                table = torch.zeros(sequence_length, d_model)
            self.register_buffer("positional_table", table.unsqueeze(0))

    @property
    def sequence_length(self) -> int:
        """Tokens produced per image: ``N`` patches plus the [CLS] token."""
        return self.num_patches + 1

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Map ``(B, C, H, W)`` images to ``(B, N+1, d)`` token sequences.

        A single ``(C, H, W)`` image is also accepted and returns ``(N+1, d)``.
        """
        squeeze_batch = images.dim() == 3
        if squeeze_batch:
            images = images.unsqueeze(0)
        batch, channels, height, width = images.shape
        if channels != self.in_channels or (height, width) != self.image_size:
            raise ValueError(
                f"Expected images of shape (B, {self.in_channels}, "
                f"{self.image_size[0]}, {self.image_size[1]}), got {tuple(images.shape)}"
            )

        patches = patchify(images, self.patch_size)  # (B, N, C*P^2)
        tokens = self.projection(patches)  # (B, N, d)
        class_tokens = self.class_token.expand(batch, -1, -1)  # (B, 1, d)
        tokens = torch.cat([class_tokens, tokens], dim=1)  # (B, N+1, d)
        # Out-of-place add: in-place `+=` would write into the autograd graph
        # (and, for sinusoidal tables, risk corrupting the shared buffer).
        tokens = tokens + self.positional_table
        return tokens.squeeze(0) if squeeze_batch else tokens

    def extra_repr(self) -> str:
        return (
            f"image_size={self.image_size}, patch_size={self.patch_size}, "
            f"in_channels={self.in_channels}, d_model={self.d_model}, "
            f"num_patches={self.num_patches}, "
            f"positional_encoding={self.positional_encoding_kind!r}"
        )
