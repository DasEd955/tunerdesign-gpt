"""test_tokenizer.py - Unit tests for data/tokenizer.py.

Tests the BPE merge learning algorithm: correct merge selection,
lexicographic tiebreaking, non-overlapping left to right merging,
early stopping, and deterministic output.
"""

import pytest
from data.tokenizer import Solution


@pytest.fixture
def tokenizer():
    """Return a Solution instance for all tokenizer tests."""
    return Solution()


class TestGetMerges:
    """Tests for Solution.get_merges."""

    def test_returns_list(self, tokenizer):
        """get_merges returns a list."""
        assert isinstance(tokenizer.get_merges("aabb", 1), list)

    def test_each_merge_is_two_element_list(self, tokenizer):
        """Every merge entry is a two element list of strings."""
        merges = tokenizer.get_merges("aabb", 2)
        for merge in merges:
            assert isinstance(merge, list)
            assert len(merge) == 2
            assert all(isinstance(t, str) for t in merge)

    def test_most_frequent_pair_chosen(self, tokenizer):
        """The first merge selects the most frequent adjacent pair."""
        # "aab" -> pairs: (a,a)x1, (a,b)x1 -> tie -> lexicographic: (a,a) < (a,b)
        merges = tokenizer.get_merges("aab", 1)
        assert merges[0] == ["a", "a"]

    def test_lexicographic_tiebreak(self, tokenizer):
        """Ties in frequency are broken by lexicographic order."""
        # "abba" -> pairs: (a,b)x1, (b,b)x1, (b,a)x1 -> tie -> (a,b) is lex smallest
        merges = tokenizer.get_merges("abba", 1)
        assert merges[0] == ["a", "b"]

    def test_num_merges_respected(self, tokenizer):
        """Number of merges performed does not exceed num_merges."""
        merges = tokenizer.get_merges("abcabcabc", 2)
        assert len(merges) <= 2

    def test_stops_early_when_no_pairs(self, tokenizer):
        """Merging stops early when only one token remains."""
        # After merging "aa" into "aa", then "aa" has no adjacent pair
        merges = tokenizer.get_merges("aa", 10)
        assert len(merges) == 1
        assert merges[0] == ["a", "a"]

    def test_stops_early_single_char_corpus(self, tokenizer):
        """A single character corpus produces no merges."""
        merges = tokenizer.get_merges("a", 5)
        assert merges == []

    def test_non_overlapping_merge(self, tokenizer):
        """Merges are applied left to right without overlapping."""
        # "aaaa" -> first merge: (a,a) -> "aa aa" -> two "aa" tokens
        # second merge: (aa,aa) -> one "aaaa" token
        merges = tokenizer.get_merges("aaaa", 2)
        assert merges[0] == ["a", "a"]
        assert merges[1] == ["aa", "aa"]

    def test_merge_recorded_correctly(self, tokenizer):
        """The merge record matches the actual pair that was merged."""
        merges = tokenizer.get_merges("abab", 1)
        assert merges[0] == ["a", "b"]

    def test_deterministic_output(self, tokenizer):
        """Calling get_merges twice with the same input gives the same result."""
        corpus = "the cat sat on the mat"
        assert tokenizer.get_merges(corpus, 5) == tokenizer.get_merges(corpus, 5)

    def test_zero_merges(self, tokenizer):
        """Requesting zero merges returns an empty list."""
        assert tokenizer.get_merges("hello", 0) == []

    def test_merges_accumulate_subwords(self, tokenizer):
        """Repeated identical pairs produce compound subword tokens across steps."""
        # "ababab": (a,b) is most frequent -> "ab ab ab" -> (ab,ab) -> "abab ab"
        merges = tokenizer.get_merges("ababab", 2)
        assert merges[0] == ["a", "b"]
        assert merges[1] == ["ab", "ab"]
