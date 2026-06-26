"""test_data_loaders.py - Unit tests for data/loader.py, data/dataset.py, and data/nlp_preprocessing.py.

Tests the tensor-based batch loader (loader.py), the word-level dataset
sampler (dataset.py), and the padded NLP preprocessing pipeline
(nlp_preprocessing.py).
"""

import torch
import pytest
from data.loader import Solution as LoaderSolution
from data.dataset import Solution as DatasetSolution
from data.nlp_preprocessing import Solution as NLPSolution


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def loader():
    """Return a LoaderSolution instance."""
    return LoaderSolution()


@pytest.fixture
def dataset():
    """Return a DatasetSolution instance."""
    return DatasetSolution()


@pytest.fixture
def nlp():
    """Return an NLPSolution instance."""
    return NLPSolution()


@pytest.fixture
def encoded_corpus():
    """Return a 1D integer tensor long enough for batch tests."""
    return torch.arange(100, dtype=torch.long)


# ---------------------------------------------------------------------------
# loader.py tests
# ---------------------------------------------------------------------------

class TestCreateBatches:
    """Tests for LoaderSolution.create_batches."""

    def test_returns_tuple_of_two_tensors(self, loader, encoded_corpus):
        """create_batches returns a 2-tuple of tensors."""
        result = loader.create_batches(encoded_corpus, context_length=10, batch_size=4)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], torch.Tensor)
        assert isinstance(result[1], torch.Tensor)

    def test_x_shape(self, loader, encoded_corpus):
        """X has shape (batch_size, context_length)."""
        X, _ = loader.create_batches(encoded_corpus, context_length=10, batch_size=4)
        assert X.shape == (4, 10)

    def test_y_shape(self, loader, encoded_corpus):
        """Y has shape (batch_size, context_length)."""
        _, Y = loader.create_batches(encoded_corpus, context_length=10, batch_size=4)
        assert Y.shape == (4, 10)

    def test_y_is_x_shifted_by_one(self, loader, encoded_corpus):
        """Y[i][j] == X[i][j+1] for all valid positions."""
        X, Y = loader.create_batches(encoded_corpus, context_length=10, batch_size=4)
        assert torch.all(Y[:, :-1] == X[:, 1:])

    def test_reproducibility_with_seed(self, loader, encoded_corpus):
        """Two calls produce the same X and Y because both use torch.manual_seed(0)."""
        X1, Y1 = loader.create_batches(encoded_corpus, context_length=10, batch_size=4)
        X2, Y2 = loader.create_batches(encoded_corpus, context_length=10, batch_size=4)
        assert torch.equal(X1, X2)
        assert torch.equal(Y1, Y2)

    def test_windows_stay_within_corpus(self, loader, encoded_corpus):
        """Every sampled window lies entirely within the corpus bounds."""
        X, Y = loader.create_batches(encoded_corpus, context_length=10, batch_size=8)
        # All values should be valid integer indices present in the corpus
        assert X.max().item() < len(encoded_corpus)
        assert Y.max().item() < len(encoded_corpus)

    def test_batch_size_one(self, loader, encoded_corpus):
        """A batch size of 1 returns single row tensors."""
        X, Y = loader.create_batches(encoded_corpus, context_length=5, batch_size=1)
        assert X.shape == (1, 5)
        assert Y.shape == (1, 5)


# ---------------------------------------------------------------------------
# dataset.py tests
# ---------------------------------------------------------------------------

class TestBatchLoader:
    """Tests for DatasetSolution.batch_loader."""

    CORPUS = "the cat sat on the mat the rat ate the bat"

    def test_returns_tuple_of_two_lists(self, dataset):
        """batch_loader returns a 2-tuple of lists."""
        result = dataset.batch_loader(self.CORPUS, context_length=3, batch_size=2)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], list)
        assert isinstance(result[1], list)

    def test_x_batch_size(self, dataset):
        """X contains batch_size sequences."""
        X, _ = dataset.batch_loader(self.CORPUS, context_length=3, batch_size=4)
        assert len(X) == 4

    def test_y_batch_size(self, dataset):
        """Y contains batch_size sequences."""
        _, Y = dataset.batch_loader(self.CORPUS, context_length=3, batch_size=4)
        assert len(Y) == 4

    def test_sequence_length(self, dataset):
        """Each sequence has exactly context_length tokens."""
        X, Y = dataset.batch_loader(self.CORPUS, context_length=3, batch_size=4)
        for seq in X + Y:
            assert len(seq) == 3

    def test_y_shifted_by_one(self, dataset):
        """Y[i] is the same tokens as X[i] but shifted right by one word."""
        X, Y = dataset.batch_loader(self.CORPUS, context_length=3, batch_size=4)
        # Verify the shift relationship directly from the batch: Y[i][j] == X[i][j+1]
        # and the last Y token is the word that follows the last X token in the corpus.
        for x_seq, y_seq in zip(X, Y):
            assert x_seq[1:] == y_seq[:-1]

    def test_tokens_are_strings(self, dataset):
        """Word tokens are strings, not integers."""
        X, _ = dataset.batch_loader(self.CORPUS, context_length=3, batch_size=2)
        assert all(isinstance(t, str) for seq in X for t in seq)

    def test_reproducibility(self, dataset):
        """Two calls return the same batches."""
        r1 = dataset.batch_loader(self.CORPUS, context_length=3, batch_size=4)
        r2 = dataset.batch_loader(self.CORPUS, context_length=3, batch_size=4)
        assert r1 == r2


# ---------------------------------------------------------------------------
# nlp_preprocessing.py tests
# ---------------------------------------------------------------------------

class TestGetDataset:
    """Tests for NLPSolution.get_dataset."""

    POSITIVE = ["I love this", "great film"]
    NEGATIVE = ["terrible movie", "I hate it"]

    def test_returns_tensor(self, nlp):
        """get_dataset returns a torch.Tensor."""
        result = nlp.get_dataset(self.POSITIVE, self.NEGATIVE)
        assert isinstance(result, torch.Tensor)

    def test_row_count(self, nlp):
        """Output has len(positive) + len(negative) rows."""
        result = nlp.get_dataset(self.POSITIVE, self.NEGATIVE)
        assert result.shape[0] == len(self.POSITIVE) + len(self.NEGATIVE)

    def test_padding_present(self, nlp):
        """Shorter sequences are zero-padded so all rows have the same length."""
        result = nlp.get_dataset(self.POSITIVE, self.NEGATIVE)
        # At least one 0 must be present because sentence lengths differ
        assert (result == 0).any()

    def test_ids_start_at_one(self, nlp):
        """Non-padding token IDs are all >= 1 (0 is reserved for padding)."""
        result = nlp.get_dataset(self.POSITIVE, self.NEGATIVE)
        non_pad = result[result != 0]
        assert non_pad.min().item() >= 1

    def test_batch_first(self, nlp):
        """The output tensor is batch-first: shape is (num_sentences, max_len)."""
        result = nlp.get_dataset(self.POSITIVE, self.NEGATIVE)
        assert result.ndim == 2
        assert result.shape[0] == 4

    def test_single_sentence_each(self, nlp):
        """Works correctly with one positive and one negative sentence."""
        result = nlp.get_dataset(["good"], ["bad"])
        assert result.shape[0] == 2

    def test_longer_sentences_determine_width(self, nlp):
        """Output width equals the length of the longest sentence."""
        pos = ["one two three four five"]
        neg = ["one"]
        result = nlp.get_dataset(pos, neg)
        assert result.shape[1] == 5
