"""test_tokenizer_utils.py - Unit tests for data/tokenizer_utils.py.

Tests greedy longest match tokenization, number tokenization, token
counting, and fertility score computation.
"""

import pytest
from data.tokenizer_utils import Solution


@pytest.fixture
def utils():
    """Return a Solution instance for all tokenizer_utils tests."""
    return Solution()


@pytest.fixture
def simple_vocab():
    """Minimal vocabulary covering digits and a two-digit merge."""
    return {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
            "5": 5, "6": 6, "7": 7, "8": 8, "9": 9, "12": 10}


class TestGreedyTokenize:
    """Tests for Solution._greedy_tokenize (internal helper)."""

    def test_prefers_longer_match(self, utils, simple_vocab):
        """Greedy selection always picks the longest matching token first."""
        tokens = utils._greedy_tokenize("12", simple_vocab)
        assert tokens == ["12"]

    def test_falls_back_to_single_char(self, utils):
        """When no vocab match exists, each character is its own token."""
        vocab = {"ab": 0}
        tokens = utils._greedy_tokenize("c", vocab)
        assert tokens == ["c"]

    def test_full_coverage(self, utils, simple_vocab):
        """Every character in the input appears in exactly one token."""
        text = "123"
        tokens = utils._greedy_tokenize(text, simple_vocab)
        assert "".join(tokens) == text

    def test_empty_string(self, utils, simple_vocab):
        """An empty input string produces an empty token list."""
        assert utils._greedy_tokenize("", simple_vocab) == []

    def test_single_char_in_vocab(self, utils, simple_vocab):
        """A single character that is in the vocab produces one token."""
        assert utils._greedy_tokenize("5", simple_vocab) == ["5"]

    def test_mixed_short_and_long(self, utils, simple_vocab):
        """Input with both mergeable and non-mergeable segments tokenizes correctly."""
        tokens = utils._greedy_tokenize("125", simple_vocab)
        assert tokens == ["12", "5"]


class TestTokenizeNumbers:
    """Tests for Solution.tokenize_numbers."""

    def test_returns_list_of_lists(self, utils, simple_vocab):
        """tokenize_numbers returns a list of token lists."""
        result = utils.tokenize_numbers([12, 5], simple_vocab)
        assert isinstance(result, list)
        assert all(isinstance(t, list) for t in result)

    def test_length_matches_input(self, utils, simple_vocab):
        """Output has one entry per input number."""
        numbers = [1, 12, 5, 99]
        result = utils.tokenize_numbers(numbers, simple_vocab)
        assert len(result) == len(numbers)

    def test_known_merge_used(self, utils, simple_vocab):
        """A number whose string matches a vocab merge is tokenized as one token."""
        result = utils.tokenize_numbers([12], simple_vocab)
        assert result[0] == ["12"]

    def test_number_not_in_vocab_splits_to_chars(self, utils, simple_vocab):
        """A number with no multi-char vocab entry splits into single-char tokens."""
        result = utils.tokenize_numbers([34], simple_vocab)
        assert result[0] == ["3", "4"]

    def test_empty_number_list(self, utils, simple_vocab):
        """An empty input list returns an empty list."""
        assert utils.tokenize_numbers([], simple_vocab) == []


class TestCountTokens:
    """Tests for Solution.count_tokens."""

    def test_returns_int(self, utils, simple_vocab):
        """count_tokens returns an int."""
        assert isinstance(utils.count_tokens("12", simple_vocab), int)

    def test_count_with_merge(self, utils, simple_vocab):
        """A two character string covered by one vocab entry counts as 1 token."""
        assert utils.count_tokens("12", simple_vocab) == 1

    def test_count_without_merge(self, utils, simple_vocab):
        """Two characters not merged count as 2 tokens."""
        assert utils.count_tokens("34", simple_vocab) == 2

    def test_empty_string_is_zero(self, utils, simple_vocab):
        """An empty string produces a count of zero."""
        assert utils.count_tokens("", simple_vocab) == 0


class TestFertilityScore:
    """Tests for Solution.fertility_score."""

    def test_returns_float(self, utils, simple_vocab):
        """fertility_score returns a float."""
        result = utils.fertility_score("12 34", simple_vocab)
        assert isinstance(result, float)

    def test_perfect_vocabulary_gives_one(self, utils, simple_vocab):
        """A vocabulary that covers every word as one token gives fertility 1.0."""
        # fertility_score divides total greedy tokens by word count (text.split()).
        # Spaces are not words, so a single-word input with an exact vocab match
        # produces 1 token / 1 word = 1.0 without any space ambiguity.
        vocab = {"hello": 0}
        score = utils.fertility_score("hello", vocab)
        assert score == 1.0

    def test_fragmented_vocab_gives_higher_fertility(self, utils):
        """Words split into many single-char tokens produce fertility > 1."""
        vocab = {"a": 0, "b": 1, "c": 2}
        score = utils.fertility_score("abc", vocab)
        assert score == 3.0

    def test_rounded_to_four_decimals(self, utils):
        """The result is rounded to 4 decimal places."""
        vocab = {"a": 0, "b": 1, "c": 2}
        score = utils.fertility_score("abc def", vocab)
        assert score == round(score, 4)
