"""Tests for MultiHeadSelfAttention.

The headline checks: numerical equivalence with torch.nn.MultiheadAttention,
the permutation-equivariance property of position-free attention (and its
deliberate *loss* under RoPE), and the variance argument behind the
1/sqrt(d_h) scaling.
"""

import math

import pytest
import torch
from torch import nn

from vit_tokenizer import MultiHeadSelfAttention


class TestShapesAndBasics:
    def test_output_shape(self):
        attention = MultiHeadSelfAttention(d_model=64, num_heads=4)
        tokens = torch.randn(2, 10, 64)
        assert attention(tokens).shape == (2, 10, 64)

    def test_attention_weights_shape_and_rows_sum_to_one(self):
        attention = MultiHeadSelfAttention(d_model=64, num_heads=4)
        _, weights = attention(torch.randn(2, 10, 64), return_attention=True)
        assert weights.shape == (2, 4, 10, 10)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 4, 10), atol=1e-5)
        assert (weights >= 0).all()

    def test_indivisible_heads_raise(self):
        with pytest.raises(ValueError, match="divisible"):
            MultiHeadSelfAttention(d_model=64, num_heads=5)

    def test_rope_odd_head_dim_raises(self):
        with pytest.raises(ValueError, match="even"):
            MultiHeadSelfAttention(d_model=9, num_heads=3, rope_base=10_000.0)

    def test_wrong_rank_raises(self):
        attention = MultiHeadSelfAttention(d_model=64, num_heads=4)
        with pytest.raises(ValueError, match="Expected"):
            attention(torch.randn(10, 64))

    def test_gradients_flow(self):
        attention = MultiHeadSelfAttention(d_model=32, num_heads=2)
        attention(torch.randn(2, 5, 32)).sum().backward()
        for name, parameter in attention.named_parameters():
            assert parameter.grad is not None, f"no gradient reached {name}"


class TestEquivalenceWithTorch:
    def test_matches_torch_multihead_attention(self):
        """Copy our weights into nn.MultiheadAttention; outputs must agree
        elementwise. This pins down every convention (head split order,
        scaling, projection layout) against the reference implementation."""
        torch.manual_seed(0)
        d_model, num_heads = 64, 4
        ours = MultiHeadSelfAttention(d_model, num_heads)
        theirs = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        with torch.no_grad():
            theirs.in_proj_weight.copy_(
                torch.cat([ours.q_proj.weight, ours.k_proj.weight, ours.v_proj.weight])
            )
            theirs.in_proj_bias.copy_(
                torch.cat([ours.q_proj.bias, ours.k_proj.bias, ours.v_proj.bias])
            )
            theirs.out_proj.weight.copy_(ours.out_proj.weight)
            theirs.out_proj.bias.copy_(ours.out_proj.bias)

        tokens = torch.randn(3, 17, d_model)
        our_output, our_weights = ours(tokens, return_attention=True)
        their_output, their_weights = theirs(tokens, tokens, tokens, need_weights=True)
        assert torch.allclose(our_output, their_output, atol=1e-5)
        # nn.MultiheadAttention returns weights averaged over heads by default.
        assert torch.allclose(our_weights.mean(dim=1), their_weights, atol=1e-5)


class TestPermutationEquivariance:
    def test_equivariant_without_positions(self):
        """Attn(Pi x) = Pi Attn(x): attention alone carries no positional
        information (docs/MATH.md section 9.4) — which is exactly why the
        tokenizer must add it."""
        torch.manual_seed(1)
        attention = MultiHeadSelfAttention(d_model=32, num_heads=4)
        tokens = torch.randn(1, 12, 32)
        permutation = torch.randperm(12)
        permuted_first = attention(tokens[:, permutation])
        attended_first = attention(tokens)[:, permutation]
        assert torch.allclose(permuted_first, attended_first, atol=1e-5)

    def test_rope_breaks_equivariance(self):
        """With rotary embeddings the score depends on relative position, so
        shuffling tokens must change the (re-shuffled) output."""
        torch.manual_seed(1)
        attention = MultiHeadSelfAttention(d_model=32, num_heads=4, rope_base=10_000.0)
        tokens = torch.randn(1, 12, 32)
        permutation = torch.roll(torch.arange(12), shifts=3)
        permuted_first = attention(tokens[:, permutation])
        attended_first = attention(tokens)[:, permutation]
        assert not torch.allclose(permuted_first, attended_first, atol=1e-4)


class TestScaling:
    def test_score_variance_is_unit_after_scaling(self):
        """For iid unit-variance q and k, Var(q.k) = d_h, so dividing by
        sqrt(d_h) restores variance ~1 (docs/MATH.md section 9.2). Checked
        statistically over a large sample."""
        torch.manual_seed(0)
        head_dim = 256
        queries = torch.randn(2000, head_dim)
        keys = torch.randn(2000, head_dim)
        scores = (queries * keys).sum(dim=-1) / math.sqrt(head_dim)
        assert scores.var().item() == pytest.approx(1.0, abs=0.1)
        assert scores.mean().item() == pytest.approx(0.0, abs=0.1)

    def test_unscaled_softmax_saturates(self):
        """The failure mode the scaling prevents: at d_h = 256 the unscaled
        scores have std 16 and softmax concentrates almost all mass on one
        entry, killing gradients."""
        torch.manual_seed(0)
        head_dim, seq_len = 256, 32
        queries = torch.randn(seq_len, head_dim)
        keys = torch.randn(seq_len, head_dim)
        saturated = torch.softmax(queries @ keys.T, dim=-1)
        scaled = torch.softmax(queries @ keys.T / math.sqrt(head_dim), dim=-1)
        assert saturated.max(dim=-1).values.mean() > 0.9   # nearly one-hot rows
        assert scaled.max(dim=-1).values.mean() < 0.5      # healthy spread
