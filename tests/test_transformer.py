"""test_transformer.py - Unit tests for model/transformer.py.

Tests the Pre-LN TransformerBlock: output shape, residual connection
preservation, rounding, and the FFN sublayer internal structure.
"""

import torch
import pytest
from model.transformer import TransformerBlock


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def block():
    """Return a TransformerBlock with model_dim=8, num_heads=2."""
    return TransformerBlock(model_dim=8, num_heads=2)


@pytest.fixture
def batch_input():
    """A (2, 5, 8) input tensor: 2 batches, 5 tokens, 8-dim model."""
    torch.manual_seed(7)
    return torch.randn(2, 5, 8)


# ---------------------------------------------------------------------------
# TransformerBlock
# ---------------------------------------------------------------------------

class TestTransformerBlock:
    """Tests for TransformerBlock.forward (model/transformer.py)."""

    def test_output_shape(self, block, batch_input):
        """Output shape matches the input shape (batch, seq_len, model_dim)."""
        out = block(batch_input)
        assert out.shape == batch_input.shape

    def test_output_is_tensor(self, block, batch_input):
        """Forward returns a torch.Tensor."""
        assert isinstance(block(batch_input), torch.Tensor)

    def test_rounded_to_four_decimals(self, block, batch_input):
        """All output values are rounded to 4 decimal places."""
        out = block(batch_input)
        assert torch.all(out == torch.round(out, decimals=4))

    def test_residual_changes_input(self, block, batch_input):
        """The output is not identical to the input (the block computes something)."""
        out = block(batch_input)
        assert not torch.allclose(out, batch_input, atol=1e-4)

    def test_different_model_dims(self):
        """TransformerBlock works for various (model_dim, num_heads) combinations."""
        for model_dim, num_heads in [(4, 1), (16, 4), (32, 8)]:
            block = TransformerBlock(model_dim=model_dim, num_heads=num_heads)
            x = torch.randn(1, 3, model_dim)
            out = block(x)
            assert out.shape == (1, 3, model_dim)

    def test_sublayers_instantiated(self, block):
        """Block contains attention, feed forward, and two LayerNorm sublayers."""
        assert hasattr(block, "multihead_attention")
        assert hasattr(block, "linear_nn")
        assert hasattr(block, "first_norm")
        assert hasattr(block, "second_norm")

    def test_ffn_up_projects_4x(self, block):
        """The FFN up-projection maps model_dim to 4 * model_dim."""
        assert block.linear_nn.up_projection.out_features == 8 * 4

    def test_ffn_down_projects_back(self, block):
        """The FFN down-projection maps 4 * model_dim back to model_dim."""
        assert block.linear_nn.down_projection.out_features == 8

    def test_single_token_sequence(self, block):
        """A sequence length of 1 does not raise an error."""
        x = torch.randn(1, 1, 8)
        out = block(x)
        assert out.shape == (1, 1, 8)

    def test_batch_size_one(self):
        """A batch size of 1 processes correctly."""
        block = TransformerBlock(model_dim=8, num_heads=2)
        x = torch.randn(1, 4, 8)
        out = block(x)
        assert out.shape == (1, 4, 8)

    def test_output_dtype_is_float(self, block, batch_input):
        """Output tensor dtype is a floating point type."""
        out = block(batch_input)
        assert out.is_floating_point()
