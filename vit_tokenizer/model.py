"""A minimal but complete Vision Transformer built on the tokenizer.

This is the smallest honest answer to "does the tokenizer actually work?":
tokens from :class:`~vit_tokenizer.embedding.ViTPatchEmbedding` flow through
``depth`` pre-norm transformer encoder blocks, and the final [CLS] state is
read out by a linear classification head.

Each block is the standard pre-norm (PreLN) layout::

    x = x + Attention(LayerNorm(x))
    x = x + MLP(LayerNorm(x))

Pre-norm puts an *identity path* from the loss to every layer — the residual
stream is never rescaled by a LayerNorm sitting on it — which is why it
trains stably without the warmup tricks the original post-norm transformer
needed (docs/MATH.md section 11). The MLP is the usual
``Linear(d, 4d) -> GELU -> Linear(4d, d)`` expansion.

Trainability is part of the test suite:
``tests/test_model.py::test_overfits_tiny_batch`` checks that the full model
(tokenizer included) drives the loss to ~zero on a small batch, and
``examples/train_shapes.py`` trains it to high accuracy on a synthetic shape
classification task on CPU in about a minute.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .embedding import ViTPatchEmbedding

__all__ = ["TransformerEncoderBlock", "MiniViT"]


class TransformerEncoderBlock(nn.Module):
    """One pre-norm encoder block: attention sublayer + MLP sublayer."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        rope_base: float | None = None,
    ) -> None:
        super().__init__()
        hidden = int(d_model * mlp_ratio)
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, rope_base=rope_base)
        self.mlp_norm = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_model),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        attended, weights = self.attention(self.attention_norm(tokens), return_attention=True)
        tokens = tokens + attended
        tokens = tokens + self.mlp(self.mlp_norm(tokens))
        return (tokens, weights) if return_attention else tokens


class MiniViT(nn.Module):
    """Tokenizer + transformer encoder + classification head, end to end.

    Args:
        image_size, patch_size, in_channels, d_model, base: forwarded to
            :class:`ViTPatchEmbedding`.
        depth: number of encoder blocks ``L``.
        num_heads: attention heads per block.
        num_classes: output classes for the linear head.
        mlp_ratio: hidden width of each block's MLP as a multiple of ``d``.
        positional_encoding: the four :class:`ViTPatchEmbedding` schemes plus
            ``"rotary"``, which disables the additive table and instead
            applies RoPE to queries and keys inside every attention layer.

    Parameter count is roughly ``12 d^2`` per block plus the tokenizer's
    ``d(CP^2 + 1)`` (docs/MATH.md section 11.3) — at d_model=768, depth=12
    this reproduces ViT-Base's ~86M.
    """

    def __init__(
        self,
        image_size: int | tuple[int, int] = 64,
        patch_size: int = 8,
        in_channels: int = 3,
        d_model: int = 96,
        depth: int = 4,
        num_heads: int = 4,
        num_classes: int = 10,
        mlp_ratio: float = 4.0,
        positional_encoding: str = "sinusoidal-2d",
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        rotary = positional_encoding == "rotary"
        self.embedding = ViTPatchEmbedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            d_model=d_model,
            positional_encoding="none" if rotary else positional_encoding,
            base=base,
        )
        self.blocks = nn.ModuleList(
            TransformerEncoderBlock(
                d_model, num_heads, mlp_ratio, rope_base=base if rotary else None
            )
            for _ in range(depth)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.positional_encoding_kind = positional_encoding

    def forward(
        self,
        images: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        """Map ``(B, C, H, W)`` images to ``(B, num_classes)`` logits.

        With ``return_attention=True`` also returns one ``(B, H, S, S)``
        attention-weight tensor per block, for visualization
        (:func:`vit_tokenizer.visualize.plot_attention_map`).
        """
        squeeze_batch = images.dim() == 3
        if squeeze_batch:
            images = images.unsqueeze(0)

        tokens = self.embedding(images)
        attention_maps = []
        for block in self.blocks:
            tokens, weights = block(tokens, return_attention=True)
            attention_maps.append(weights)

        # The [CLS] state at position 0 is the model's summary of the image.
        logits = self.head(self.final_norm(tokens[:, 0]))
        if squeeze_batch:
            logits = logits.squeeze(0)
            attention_maps = [weights.squeeze(0) for weights in attention_maps]
        return (logits, attention_maps) if return_attention else logits
