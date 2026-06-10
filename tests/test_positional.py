"""Tests for sinusoidal positional encodings.

Each mathematical property claimed in docs/MATH.md section 5 is verified
numerically here:

* equivalence of the readable power form and the conventional exp/log form;
* agreement with the paper's formula evaluated entry by entry;
* boundedness and the exact norm ||PE[pos]||^2 = d/2;
* offset-only dependence of dot products (Toeplitz similarity);
* the relative-position property: PE[pos+k] = R_k @ PE[pos] for a fixed
  block-diagonal rotation matrix R_k.
"""

import math

import numpy as np
import pytest
import torch

from vit_tokenizer import (
    frequency_bands,
    sinusoidal_positional_encoding,
    sinusoidal_positional_encoding_2d,
)


class TestFrequencyBands:
    def test_matches_exp_log_form(self):
        """The PoC's original question: base^(-2j/d) == exp(2j * -ln(base)/d).
        This is the identity a^x = e^{x ln a}; both styles appear in the wild."""
        d_model = 512
        conventional = np.exp(
            np.arange(0, d_model, 2, dtype=np.float32) * -(np.log(10_000.0) / d_model)
        )
        readable = frequency_bands(d_model).numpy()
        np.testing.assert_allclose(readable, conventional, rtol=1e-6)

    def test_geometric_progression(self):
        """Consecutive frequencies have the constant ratio base^(-2/d)."""
        d_model, base = 64, 10_000.0
        omega = frequency_bands(d_model, base)
        ratios = omega[1:] / omega[:-1]
        expected = base ** (-2.0 / d_model)
        assert torch.allclose(ratios, torch.full_like(ratios, expected), rtol=1e-5)

    def test_endpoints(self):
        """omega_0 = 1 always; the last frequency approaches 1/base."""
        omega = frequency_bands(128, base=10_000.0)
        assert omega[0] == pytest.approx(1.0)
        assert omega[-1] == pytest.approx(10_000.0 ** (-126 / 128))


class TestSinusoidalEncoding:
    def test_shape(self):
        assert sinusoidal_positional_encoding(50, 512).shape == (50, 512)

    def test_matches_paper_formula_entrywise(self):
        """PE[pos, 2j] = sin(pos / base^{2j/d}), PE[pos, 2j+1] = cos(...),
        checked against a direct scalar-by-scalar evaluation."""
        num_positions, d_model, base = 12, 10, 10_000.0
        encoding = sinusoidal_positional_encoding(num_positions, d_model, base)
        for pos in range(num_positions):
            for j in range(d_model // 2):
                angle = pos / base ** (2 * j / d_model)
                assert encoding[pos, 2 * j].item() == pytest.approx(math.sin(angle), abs=1e-5)
                assert encoding[pos, 2 * j + 1].item() == pytest.approx(math.cos(angle), abs=1e-5)

    def test_position_zero(self):
        """sin(0)=0 and cos(0)=1, so row 0 is (0, 1, 0, 1, ...)."""
        encoding = sinusoidal_positional_encoding(5, 8)
        assert torch.equal(encoding[0], torch.tensor([0.0, 1.0] * 4))

    def test_bounded(self):
        encoding = sinusoidal_positional_encoding(200, 256)
        assert encoding.abs().max() <= 1.0

    def test_exact_norm(self):
        """sin^2 + cos^2 = 1 for each of the d/2 pairs, so ||PE[pos]||^2 = d/2
        for every position (docs/MATH.md section 5.4)."""
        d_model = 128
        encoding = sinusoidal_positional_encoding(100, d_model)
        norms_squared = (encoding ** 2).sum(dim=1)
        assert torch.allclose(norms_squared, torch.full((100,), d_model / 2), atol=1e-3)

    def test_rows_unique(self):
        """No two positions share an encoding (within a sequence far shorter
        than the longest wavelength)."""
        encoding = sinusoidal_positional_encoding(500, 64)
        distances = torch.cdist(encoding, encoding)
        distances.fill_diagonal_(float("inf"))
        assert distances.min() > 1e-3

    def test_dot_product_depends_only_on_offset(self):
        """PE[p] . PE[q] = sum_j cos(omega_j (p-q)): a function of p-q alone,
        making the Gram matrix Toeplitz (docs/MATH.md section 5.5)."""
        encoding = sinusoidal_positional_encoding(60, 64)
        gram = encoding @ encoding.T
        for offset in (1, 5, 17):
            band = torch.diagonal(gram, offset=offset)
            assert torch.allclose(band, band[0].expand_as(band), atol=1e-3)

    def test_dot_product_closed_form(self):
        """The same dot products, against the analytic sum of cosines."""
        d_model, base = 32, 10_000.0
        encoding = sinusoidal_positional_encoding(40, d_model, base)
        omega = frequency_bands(d_model, base)
        p, q = 7, 29
        expected = torch.cos(omega * (p - q)).sum()
        actual = encoding[p] @ encoding[q]
        assert actual.item() == pytest.approx(expected.item(), abs=1e-4)

    def test_relative_position_is_linear(self):
        """For fixed offset k there is one matrix R_k, independent of pos, with
        PE[pos + k] = R_k @ PE[pos]. R_k is block-diagonal with 2x2 rotation
        blocks [[cos(w k), sin(w k)], [-sin(w k), cos(w k)]] per frequency
        (docs/MATH.md section 5.6). This linearity is the design reason for
        the sinusoidal form: relative offsets become learnable linear maps."""
        num_positions, d_model, base, k = 50, 64, 10_000.0, 9
        encoding = sinusoidal_positional_encoding(num_positions, d_model, base)
        omega = frequency_bands(d_model, base)

        rotation = torch.zeros(d_model, d_model)
        for j, w in enumerate(omega):
            c, s = torch.cos(w * k), torch.sin(w * k)
            rotation[2 * j, 2 * j] = c
            rotation[2 * j, 2 * j + 1] = s
            rotation[2 * j + 1, 2 * j] = -s
            rotation[2 * j + 1, 2 * j + 1] = c

        shifted = encoding[: num_positions - k] @ rotation.T
        assert torch.allclose(shifted, encoding[k:], atol=1e-4)

    def test_odd_d_model(self):
        """Odd widths are unusual but must not crash or leave stale zeros in
        the sine columns."""
        encoding = sinusoidal_positional_encoding(10, 7)
        assert encoding.shape == (10, 7)
        assert encoding[1].abs().sum() > 0

    def test_invalid_args_raise(self):
        with pytest.raises(ValueError):
            sinusoidal_positional_encoding(0, 64)
        with pytest.raises(ValueError):
            sinusoidal_positional_encoding(10, 0)


class TestSinusoidal2D:
    def test_shape(self):
        assert sinusoidal_positional_encoding_2d(7, 7, 512).shape == (49, 512)

    def test_factorization(self):
        """Patch (r, c) is the concatenation of the row-r and column-c 1-D
        encodings, each of width d/2, in raster order."""
        grid_h, grid_w, d_model = 3, 4, 16
        encoding = sinusoidal_positional_encoding_2d(grid_h, grid_w, d_model)
        rows = sinusoidal_positional_encoding(grid_h, d_model // 2)
        cols = sinusoidal_positional_encoding(grid_w, d_model // 2)
        for r in range(grid_h):
            for c in range(grid_w):
                token = encoding[r * grid_w + c]
                assert torch.equal(token[: d_model // 2], rows[r])
                assert torch.equal(token[d_model // 2 :], cols[c])

    def test_same_row_shares_first_half(self):
        encoding = sinusoidal_positional_encoding_2d(4, 4, 32)
        first, second = encoding[0], encoding[3]  # row 0, columns 0 and 3
        assert torch.equal(first[:16], second[:16])
        assert not torch.equal(first[16:], second[16:])

    def test_odd_d_model_raises(self):
        with pytest.raises(ValueError, match="even"):
            sinusoidal_positional_encoding_2d(4, 4, 33)


class TestRope:
    """Rotary embeddings (docs/MATH.md section 10): rope_rotate applies the
    section-5.6 rotation matrices multiplicatively to token vectors."""

    def test_shape_preserved(self):
        from vit_tokenizer import rope_rotate

        vectors = torch.randn(2, 4, 10, 32)  # (batch, heads, seq, head_dim)
        positions = torch.arange(10)
        assert rope_rotate(vectors, positions).shape == vectors.shape

    def test_position_zero_is_identity(self):
        from vit_tokenizer import rope_rotate

        vectors = torch.randn(1, 5, 16)
        rotated = rope_rotate(vectors, torch.zeros(5))
        assert torch.allclose(rotated, vectors, atol=1e-6)

    def test_norm_preserved(self):
        """Rotations are isometries: every token keeps its norm exactly."""
        from vit_tokenizer import rope_rotate

        vectors = torch.randn(3, 20, 64)
        rotated = rope_rotate(vectors, torch.arange(20))
        assert torch.allclose(
            rotated.norm(dim=-1), vectors.norm(dim=-1), atol=1e-4
        )

    def test_matches_explicit_rotation_matrix(self):
        """rope_rotate at position m must equal multiplication by the
        block-diagonal matrix of 2x2 rotations by omega_j * m."""
        from vit_tokenizer import frequency_bands, rope_rotate

        torch.manual_seed(0)
        d_model, position = 16, 7
        vector = torch.randn(d_model)
        omega = frequency_bands(d_model)

        rotation = torch.zeros(d_model, d_model)
        for j, w in enumerate(omega):
            c, s = torch.cos(w * position), torch.sin(w * position)
            rotation[2 * j, 2 * j] = c
            rotation[2 * j, 2 * j + 1] = -s
            rotation[2 * j + 1, 2 * j] = s
            rotation[2 * j + 1, 2 * j + 1] = c

        rotated = rope_rotate(vector.view(1, 1, d_model), torch.tensor([position]))
        assert torch.allclose(rotated.flatten(), rotation @ vector, atol=1e-5)

    def test_scores_depend_only_on_relative_position(self):
        """The defining property: <R(m) q, R(n) k> = <q, R(n-m) k>. Place the
        same q and k at shifted positions; the dot product must not change."""
        from vit_tokenizer import rope_rotate

        torch.manual_seed(0)
        d_model, seq_len = 32, 40
        query = torch.randn(d_model)
        key = torch.randn(d_model)
        # Every token identical, so position is the only varying quantity.
        queries = query.expand(seq_len, d_model)
        keys = key.expand(seq_len, d_model)
        positions = torch.arange(seq_len)
        rotated_queries = rope_rotate(queries, positions)
        rotated_keys = rope_rotate(keys, positions)

        offset = 6
        scores = [
            rotated_queries[m] @ rotated_keys[m + offset]
            for m in range(0, seq_len - offset, 7)
        ]
        for score in scores[1:]:
            assert score.item() == pytest.approx(scores[0].item(), abs=1e-3)

    def test_odd_width_raises(self):
        from vit_tokenizer import rope_rotate

        with pytest.raises(ValueError, match="pairs"):
            rope_rotate(torch.randn(1, 4, 15), torch.arange(4))

    def test_mismatched_positions_raise(self):
        from vit_tokenizer import rope_rotate

        with pytest.raises(ValueError, match="one entry per token"):
            rope_rotate(torch.randn(1, 4, 16), torch.arange(5))
