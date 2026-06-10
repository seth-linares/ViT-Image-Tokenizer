"""Sinusoidal positional encodings (Vaswani et al. 2017, section 3.5).

Self-attention is permutation-equivariant: shuffle the input tokens and the
outputs shuffle identically. Without an explicit position signal the model
cannot tell where a patch sits in the image, so we add a deterministic
position-dependent vector to every token.

For position ``pos`` and pair index ``j in {0, ..., d/2 - 1}``::

    omega_j        = base^(-2j / d)          (angular frequency)
    PE[pos, 2j]    = sin(omega_j * pos)
    PE[pos, 2j+1]  = cos(omega_j * pos)

The frequencies form a geometric progression from ``omega_0 = 1`` down to
``omega_{d/2-1} ~ 1/base``, i.e. wavelengths from ``2*pi`` up to roughly
``2*pi*base``. Low dimensions oscillate fast and disambiguate nearby
positions; high dimensions oscillate slowly and encode coarse position.

Key provable properties (all verified numerically in
``tests/test_positional.py``, derivations in ``docs/MATH.md`` section 5):

* every entry lies in ``[-1, 1]`` and ``||PE[pos]||^2 = d/2`` exactly, so the
  encoding never drowns out the patch embedding it is added to;
* ``PE[pos] . PE[pos+k] = sum_j cos(omega_j * k)`` depends only on the offset
  ``k``, which is what lets attention learn relative-position patterns;
* ``PE[pos+k]`` is a fixed *linear* function (a block-diagonal rotation) of
  ``PE[pos]`` for every fixed ``k``.
"""

from __future__ import annotations

import torch

__all__ = [
    "frequency_bands",
    "rope_rotate",
    "sinusoidal_positional_encoding",
    "sinusoidal_positional_encoding_2d",
]


def frequency_bands(d_model: int, base: float = 10_000.0) -> torch.Tensor:
    """Angular frequencies ``omega_j = base^(-2j/d)`` for ``j = 0..ceil(d/2)-1``.

    This is the readable form of the ``div_term`` you will see in most
    transformer codebases, which compute the algebraically identical

        exp(2j * (-ln(base) / d)) = base^(-2j/d)

    via ``torch.exp``. The identity ``a^x = e^(x ln a)`` makes the two equal
    up to float rounding; ``tests/test_positional.py`` asserts this.
    """
    if d_model <= 0:
        raise ValueError(f"d_model must be positive, got {d_model}")
    pair_index = torch.arange(0, d_model, 2, dtype=torch.float32)
    return base ** (-pair_index / d_model)


def sinusoidal_positional_encoding(
    num_positions: int,
    d_model: int,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Build the ``(num_positions, d_model)`` sinusoidal encoding table.

    Even columns hold ``sin``, odd columns hold ``cos`` of the same angles.
    Position 0 is the all-(0, 1, 0, 1, ...) vector since ``sin 0 = 0`` and
    ``cos 0 = 1``. Odd ``d_model`` is supported (the final cos column is
    simply dropped), though even widths are the norm.
    """
    if num_positions <= 0:
        raise ValueError(f"num_positions must be positive, got {num_positions}")
    positions = torch.arange(num_positions, dtype=torch.float32).unsqueeze(1)
    omega = frequency_bands(d_model, base)
    angles = positions * omega  # (num_positions, ceil(d/2)) by broadcasting

    encoding = torch.zeros(num_positions, d_model)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : d_model // 2])
    return encoding


def sinusoidal_positional_encoding_2d(
    grid_h: int,
    grid_w: int,
    d_model: int,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Factorized 2-D encoding: first half encodes the row, second half the column.

    The 1-D encoding indexes patches by their raster position, so patch
    ``(r, 0)`` and patch ``(r-1, W-1)`` — spatially far apart — get adjacent
    encodings. The standard fix (used e.g. by DETR and MAE) is to encode the
    row index and the column index independently with ``d/2`` dimensions each
    and concatenate:

        PE2D[(r, c)] = concat( PE_{d/2}[r],  PE_{d/2}[c] )

    Two patches in the same row then share their entire first half, and two
    patches in the same column share their second half. Rows are returned in
    raster order to align with :func:`vit_tokenizer.patching.patchify`.
    """
    if d_model % 2 != 0:
        raise ValueError(f"2-D encoding needs an even d_model, got {d_model}")
    half = d_model // 2
    row_encoding = sinusoidal_positional_encoding(grid_h, half, base)
    col_encoding = sinusoidal_positional_encoding(grid_w, half, base)
    encoding = torch.cat(
        [
            row_encoding.unsqueeze(1).expand(grid_h, grid_w, half),
            col_encoding.unsqueeze(0).expand(grid_h, grid_w, half),
        ],
        dim=-1,
    )
    return encoding.reshape(grid_h * grid_w, d_model)


def rope_rotate(
    vectors: torch.Tensor,
    positions: torch.Tensor,
    base: float = 10_000.0,
) -> torch.Tensor:
    """Rotary positional embedding (RoPE; Su et al. 2021).

    Where the sinusoidal table *adds* position to the token once at the input,
    RoPE *rotates* the query/key vectors inside every attention layer: the
    pair ``(x[2j], x[2j+1])`` of the token at position ``m`` is rotated by the
    angle ``omega_j * m``, with the same frequencies
    ``omega_j = base^(-2j/d)`` as the additive encoding::

        out[2j]   = x[2j] * cos(omega_j m) - x[2j+1] * sin(omega_j m)
        out[2j+1] = x[2j] * sin(omega_j m) + x[2j+1] * cos(omega_j m)

    This is exactly the block-rotation matrix of docs/MATH.md section 5.6
    applied *multiplicatively*. Because rotations compose
    (``R(m)^T R(n) = R(n - m)``), the attention score between a query at
    position ``m`` and a key at position ``n`` satisfies

        < R(m) q, R(n) k > = < q, R(n - m) k >

    — it depends on the *relative* offset ``n - m`` only, never on absolute
    position. Rotations are also isometries, so token norms are untouched.
    Both facts are verified in ``tests/test_positional.py::TestRope``;
    derivation in docs/MATH.md section 10.

    Args:
        vectors: ``(..., S, d)`` tensor (any leading batch/head axes) with
            even last dimension ``d``.
        positions: ``(S,)`` integer or float positions, one per token.
        base: wavelength base shared with the sinusoidal encoding.

    Returns:
        Tensor of the same shape with each token rotated by its position.
    """
    d_model = vectors.shape[-1]
    if d_model % 2 != 0:
        raise ValueError(f"RoPE rotates dimension pairs; got odd width {d_model}")
    if positions.dim() != 1 or positions.shape[0] != vectors.shape[-2]:
        raise ValueError(
            f"positions must be one-dimensional with one entry per token; got "
            f"{tuple(positions.shape)} for {vectors.shape[-2]} tokens"
        )
    omega = frequency_bands(d_model, base).to(vectors.device)
    angles = positions.to(vectors.dtype).unsqueeze(-1) * omega  # (S, d/2)
    cos, sin = torch.cos(angles), torch.sin(angles)

    even, odd = vectors[..., 0::2], vectors[..., 1::2]
    rotated = torch.empty_like(vectors)
    rotated[..., 0::2] = even * cos - odd * sin
    rotated[..., 1::2] = even * sin + odd * cos
    return rotated
