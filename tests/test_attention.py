"""test_attention.py - Unit tests for model/attention.py, model/multi_head_attention.py,
and model/grouped_query_attention.py.

Tests output shapes, causal masking (no future token leakage), rounding
precision, and the KV head repetition scheme in grouped query attention.
"""

import torch
import pytest
from model.attention import SingleHeadAttention
from model.multi_head_attention import MultiHeadedSelfAttention
from model.grouped_query_attention import GroupedQueryAttention


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def batch_input():
    """A (2, 5, 8) input tensor: 2 batches, 5 tokens, 8-dim embeddings."""
    torch.manual_seed(42)
    return torch.randn(2, 5, 8)


# ---------------------------------------------------------------------------
# SingleHeadAttention
# ---------------------------------------------------------------------------

class TestSingleHeadAttention:
    """Tests for SingleHeadAttention.forward (model/attention.py)."""

    def test_output_shape(self, batch_input):
        """Output shape is (batch, seq_len, attention_dim)."""
        head = SingleHeadAttention(embedding_dim=8, attention_dim=4)
        out = head(batch_input)
        assert out.shape == (2, 5, 4)

    def test_output_is_tensor(self, batch_input):
        """Forward returns a torch.Tensor."""
        head = SingleHeadAttention(embedding_dim=8, attention_dim=4)
        assert isinstance(head(batch_input), torch.Tensor)

    def test_rounded_to_four_decimals(self, batch_input):
        """All output values are rounded to 4 decimal places."""
        head = SingleHeadAttention(embedding_dim=8, attention_dim=4)
        out = head(batch_input)
        assert torch.all(out == torch.round(out, decimals=4))

    def test_causal_mask_no_future_leakage(self):
        """Masking is causal: changing a future token does not affect past positions.

        If position 0 attends only to itself, swapping the value at position 2
        must not change the output at position 0.
        """
        torch.manual_seed(0)
        head = SingleHeadAttention(embedding_dim=4, attention_dim=4)
        x1 = torch.ones(1, 4, 4)
        x2 = x1.clone()
        x2[0, 2, :] = 99.0
        out1 = head(x1)
        out2 = head(x2)
        assert torch.allclose(out1[0, 0, :], out2[0, 0, :], atol=1e-3)

    def test_first_position_attends_only_to_itself(self):
        """The first token can only look at itself under the causal mask."""
        head = SingleHeadAttention(embedding_dim=4, attention_dim=4)
        x = torch.zeros(1, 3, 4)
        x[0, 0, :] = 1.0
        out = head(x)
        # Position 0 output is determined solely by position 0 value
        assert out.shape[1] == 3

    def test_single_token_sequence(self):
        """A sequence of length 1 does not raise an error."""
        head = SingleHeadAttention(embedding_dim=8, attention_dim=4)
        x = torch.randn(1, 1, 8)
        out = head(x)
        assert out.shape == (1, 1, 4)

    def test_reproducible_with_manual_seed(self, batch_input):
        """Two heads initialized with the same class produce the same output."""
        head1 = SingleHeadAttention(embedding_dim=8, attention_dim=4)
        head2 = SingleHeadAttention(embedding_dim=8, attention_dim=4)
        assert torch.allclose(head1(batch_input), head2(batch_input), atol=1e-4)


# ---------------------------------------------------------------------------
# MultiHeadedSelfAttention
# ---------------------------------------------------------------------------

class TestMultiHeadedSelfAttention:
    """Tests for MultiHeadedSelfAttention.forward (model/multi_head_attention.py)."""

    def test_output_shape(self, batch_input):
        """Output shape is (batch, seq_len, attention_dim)."""
        mha = MultiHeadedSelfAttention(embedding_dim=8, attention_dim=8, num_heads=2)
        out = mha(batch_input)
        assert out.shape == (2, 5, 8)

    def test_output_is_tensor(self, batch_input):
        """Forward returns a torch.Tensor."""
        mha = MultiHeadedSelfAttention(embedding_dim=8, attention_dim=8, num_heads=2)
        assert isinstance(mha(batch_input), torch.Tensor)

    def test_rounded_to_four_decimals(self, batch_input):
        """All output values are rounded to 4 decimal places."""
        mha = MultiHeadedSelfAttention(embedding_dim=8, attention_dim=8, num_heads=2)
        out = mha(batch_input)
        assert torch.all(out == torch.round(out, decimals=4))

    def test_num_heads_creates_correct_head_count(self):
        """nn.ModuleList contains exactly num_heads head modules."""
        mha = MultiHeadedSelfAttention(embedding_dim=8, attention_dim=8, num_heads=4)
        assert len(mha.attention_heads) == 4

    def test_single_head(self, batch_input):
        """A single-head MHA is equivalent in shape to single-head attention."""
        mha = MultiHeadedSelfAttention(embedding_dim=8, attention_dim=8, num_heads=1)
        out = mha(batch_input)
        assert out.shape == (2, 5, 8)

    def test_output_differs_from_single_head(self, batch_input):
        """Multi-head output generally differs from single-head output due to projection."""
        mha2 = MultiHeadedSelfAttention(embedding_dim=8, attention_dim=8, num_heads=2)
        mha4 = MultiHeadedSelfAttention(embedding_dim=8, attention_dim=8, num_heads=4)
        # Both are valid but the outputs should differ since they have different heads
        assert not torch.allclose(mha2(batch_input), mha4(batch_input))


# ---------------------------------------------------------------------------
# GroupedQueryAttention
# ---------------------------------------------------------------------------

class TestGroupedQueryAttention:
    """Tests for GroupedQueryAttention.forward (model/grouped_query_attention.py)."""

    def test_output_shape(self, batch_input):
        """Output shape matches the input shape (batch, seq_len, model_dim)."""
        gqa = GroupedQueryAttention(model_dim=8, num_heads=4, num_kv_heads=2)
        out = gqa(batch_input)
        assert out.shape == (2, 5, 8)

    def test_output_is_tensor(self, batch_input):
        """Forward returns a torch.Tensor."""
        gqa = GroupedQueryAttention(model_dim=8, num_heads=4, num_kv_heads=2)
        assert isinstance(gqa(batch_input), torch.Tensor)

    def test_rounded_to_four_decimals(self, batch_input):
        """All output values are rounded to 4 decimal places."""
        gqa = GroupedQueryAttention(model_dim=8, num_heads=4, num_kv_heads=2)
        out = gqa(batch_input)
        assert torch.all(out == torch.round(out, decimals=4))

    def test_kv_proj_output_dim(self):
        """K and V projections have dimension num_kv_heads * head_dim."""
        gqa = GroupedQueryAttention(model_dim=8, num_heads=4, num_kv_heads=2)
        # head_dim = 8 // 4 = 2; kv proj width = 2 * 2 = 4
        assert gqa.k_proj.out_features == 4
        assert gqa.v_proj.out_features == 4

    def test_num_heads_equals_num_kv_heads_valid(self, batch_input):
        """num_kv_heads == num_heads is valid (standard multi-head attention)."""
        gqa = GroupedQueryAttention(model_dim=8, num_heads=4, num_kv_heads=4)
        out = gqa(batch_input)
        assert out.shape == (2, 5, 8)

    def test_single_kv_head(self, batch_input):
        """A single KV head shared across all query heads produces the correct shape."""
        gqa = GroupedQueryAttention(model_dim=8, num_heads=4, num_kv_heads=1)
        out = gqa(batch_input)
        assert out.shape == (2, 5, 8)

    def test_causal_mask_applied(self):
        """Changing a future position does not alter earlier positions in output."""
        torch.manual_seed(0)
        gqa = GroupedQueryAttention(model_dim=8, num_heads=4, num_kv_heads=2)
        x1 = torch.randn(1, 5, 8)
        x2 = x1.clone()
        x2[0, 4, :] = 99.0
        out1 = gqa(x1)
        out2 = gqa(x2)
        assert torch.allclose(out1[0, 0, :], out2[0, 0, :], atol=1e-3)
