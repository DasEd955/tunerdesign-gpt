"""test_embeddings_and_encoding.py - Unit tests for model/embeddings.py and model/positional_encoding.py.

Tests token embedding lookup (shape, values, rounding) and sinusoidal
positional encoding (shape, formula correctness, rounding, boundary
handling for odd d_model).
"""

import math
import numpy as np
import pytest
from model.embeddings import Solution as EmbeddingsSolution
from model.positional_encoding import Solution as PESolution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def emb():
    """Return an EmbeddingsSolution instance."""
    return EmbeddingsSolution()


@pytest.fixture
def pe():
    """Return a PESolution instance."""
    return PESolution()


@pytest.fixture
def embed_matrix():
    """A small (5, 4) embedding matrix with known values."""
    return np.arange(20, dtype=np.float64).reshape(5, 4)


# ---------------------------------------------------------------------------
# Embedding lookup
# ---------------------------------------------------------------------------

class TestEmbeddingLookup:
    """Tests for EmbeddingsSolution.lookup."""

    def test_output_shape(self, emb, embed_matrix):
        """Output shape is (len(token_ids), embed_dim)."""
        token_ids = np.array([0, 2, 4])
        out = emb.lookup(embed_matrix, token_ids)
        assert out.shape == (3, 4)

    def test_correct_rows_returned(self, emb, embed_matrix):
        """The correct embedding rows are retrieved for each token ID."""
        token_ids = np.array([1, 3])
        out = emb.lookup(embed_matrix, token_ids)
        assert np.array_equal(np.round(embed_matrix[1], 5), out[0])
        assert np.array_equal(np.round(embed_matrix[3], 5), out[1])

    def test_single_token(self, emb, embed_matrix):
        """A single token ID returns a 2D array with one row."""
        out = emb.lookup(embed_matrix, np.array([2]))
        assert out.shape == (1, 4)

    def test_repeated_token_ids(self, emb, embed_matrix):
        """Repeated token IDs return the same row multiple times."""
        out = emb.lookup(embed_matrix, np.array([0, 0, 0]))
        assert np.all(out == out[0])

    def test_rounded_to_five_decimals(self, emb, embed_matrix):
        """Output values are rounded to exactly 5 decimal places."""
        matrix = np.random.default_rng(42).random((10, 8))
        out = emb.lookup(matrix, np.array([0, 5, 9]))
        assert np.all(out == np.round(out, 5))

    def test_first_row(self, emb, embed_matrix):
        """Token ID 0 returns the first row of the embedding matrix."""
        out = emb.lookup(embed_matrix, np.array([0]))
        assert np.array_equal(out[0], np.round(embed_matrix[0], 5))

    def test_last_row(self, emb, embed_matrix):
        """Token ID (vocab_size - 1) returns the last row of the embedding matrix."""
        out = emb.lookup(embed_matrix, np.array([4]))
        assert np.array_equal(out[0], np.round(embed_matrix[4], 5))


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class TestPositionalEncoding:
    """Tests for PESolution.get_positional_encoding."""

    def test_output_shape(self, pe):
        """Output shape is (seq_len, d_model)."""
        out = pe.get_positional_encoding(seq_len=6, d_model=8)
        assert out.shape == (6, 8)

    def test_even_columns_are_sine(self, pe):
        """Even-indexed columns match the sine formula."""
        seq_len, d_model = 4, 8
        out = pe.get_positional_encoding(seq_len, d_model)
        positions = np.arange(seq_len).reshape(-1, 1)
        i = np.arange(0, d_model, 2)
        expected_sin = np.round(np.sin(positions / (10000 ** (i / d_model))), 5)
        assert np.allclose(out[:, 0::2], expected_sin, atol=1e-4)

    def test_odd_columns_are_cosine(self, pe):
        """Odd-indexed columns match the cosine formula."""
        seq_len, d_model = 4, 8
        out = pe.get_positional_encoding(seq_len, d_model)
        positions = np.arange(seq_len).reshape(-1, 1)
        i = np.arange(0, d_model, 2)
        expected_cos = np.round(np.cos(positions / (10000 ** (i / d_model))), 5)
        assert np.allclose(out[:, 1::2], expected_cos, atol=1e-4)

    def test_first_position_sine_is_zero(self, pe):
        """PE[0, 0] = sin(0) = 0."""
        out = pe.get_positional_encoding(seq_len=3, d_model=4)
        assert abs(float(out[0, 0])) < 1e-4

    def test_first_position_cosine_is_one(self, pe):
        """PE[0, 1] = cos(0) = 1."""
        out = pe.get_positional_encoding(seq_len=3, d_model=4)
        assert abs(float(out[0, 1]) - 1.0) < 1e-4

    def test_rounded_to_five_decimals(self, pe):
        """Output values are rounded to exactly 5 decimal places."""
        out = pe.get_positional_encoding(seq_len=5, d_model=6)
        assert np.all(out == np.round(out, 5))

    def test_single_position(self, pe):
        """A single position encoding returns a (1, d_model) matrix."""
        out = pe.get_positional_encoding(seq_len=1, d_model=4)
        assert out.shape == (1, 4)

    def test_odd_d_model(self, pe):
        """An odd d_model does not raise an error and returns the correct shape."""
        out = pe.get_positional_encoding(seq_len=4, d_model=5)
        assert out.shape == (4, 5)

    def test_values_bounded(self, pe):
        """All output values are in [-1, 1] because they are pure sine/cosine."""
        out = pe.get_positional_encoding(seq_len=20, d_model=16)
        assert float(out.max()) <= 1.0 + 1e-5
        assert float(out.min()) >= -1.0 - 1e-5
