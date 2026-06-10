"""Multi-head self-attention from scratch.

Scaled dot-product attention (Vaswani et al. 2017, section 3.2):

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_h) ) V

Each token's query is compared against every token's key; the resulting
scores are normalized into a probability distribution per row, which then
mixes the value vectors. Multi-head attention runs ``H`` of these in parallel
on ``d_h = d/H``-dimensional projections and concatenates the results.

Two properties are worth internalizing (both proven in docs/MATH.md section 9
and verified in tests/test_attention.py):

* **The 1/sqrt(d_h) scaling is a variance correction.** For unit-variance
  inputs, ``q . k`` has variance ``d_h``; without the scaling the softmax
  saturates and its gradient collapses toward zero.
* **Self-attention is permutation-equivariant.** Shuffle the input tokens and
  the output shuffles identically — attention itself has no notion of
  position. Position must be injected either additively before the first
  block (sinusoidal/learnable tables) or multiplicatively inside attention
  (rotary embeddings, the ``rope_base`` option here).

This implementation is numerically identical to ``torch.nn.MultiheadAttention``
(``tests/test_attention.py::test_matches_torch_multihead_attention`` copies
weights across and asserts elementwise agreement) — it exists to make the
mechanism readable, not to replace the fused kernel.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .positional import rope_rotate

__all__ = ["MultiHeadSelfAttention"]


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention over ``(B, S, d)`` token sequences.

    Args:
        d_model: token width ``d``; must be divisible by ``num_heads``.
        num_heads: number of parallel attention heads ``H``.
        rope_base: if set (e.g. ``10_000.0``), rotary positional embeddings
            with this wavelength base are applied to the queries and keys of
            every head, making the attention *scores* a function of relative
            position (docs/MATH.md section 10). Requires an even head
            dimension. ``None`` (default) leaves attention position-free.
    """

    def __init__(self, d_model: int, num_heads: int, rope_base: float | None = None) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} is not divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.rope_base = rope_base
        if rope_base is not None and self.head_dim % 2 != 0:
            raise ValueError(
                f"Rotary embeddings rotate dimension *pairs*, so the head "
                f"dimension must be even; got d_model/num_heads = {self.head_dim}."
            )

        # Four affine maps: token -> query/key/value, and mixed values -> token.
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        for projection in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)

    def forward(
        self,
        tokens: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Attend over ``(B, S, d)`` tokens; optionally return the weights.

        Returns ``(B, S, d)`` output, plus the ``(B, H, S, S)`` attention
        weights (rows sum to 1) when ``return_attention=True``.
        """
        if tokens.dim() != 3:
            raise ValueError(f"Expected (B, S, d_model) tokens, got {tuple(tokens.shape)}")
        batch, seq_len, _ = tokens.shape

        # (B, S, d) -> (B, H, S, d_h): each head sees its own d_h-dim slice.
        def split_heads(projected: torch.Tensor) -> torch.Tensor:
            return projected.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        queries = split_heads(self.q_proj(tokens))
        keys = split_heads(self.k_proj(tokens))
        values = split_heads(self.v_proj(tokens))

        if self.rope_base is not None:
            positions = torch.arange(seq_len, device=tokens.device)
            queries = rope_rotate(queries, positions, self.rope_base)
            keys = rope_rotate(keys, positions, self.rope_base)

        scores = queries @ keys.transpose(-2, -1) / math.sqrt(self.head_dim)
        attention = torch.softmax(scores, dim=-1)  # (B, H, S, S), rows sum to 1

        mixed = attention @ values  # (B, H, S, d_h)
        mixed = mixed.transpose(1, 2).reshape(batch, seq_len, self.d_model)
        output = self.out_proj(mixed)
        return (output, attention) if return_attention else output

    def extra_repr(self) -> str:
        rope = f", rope_base={self.rope_base:g}" if self.rope_base is not None else ""
        return f"d_model={self.d_model}, num_heads={self.num_heads}{rope}"
