"""test_vocab.py - Unit tests for data/vocab.py.

Tests the character-level vocabulary builder and its encode/decode
round-trip helpers: build_vocab, encode, and decode.
"""

import pytest
from data.vocab import Solution


@pytest.fixture
def vocab():
    """Return a Solution instance for all vocab tests."""
    return Solution()


class TestBuildVocab:
    """Tests for Solution.build_vocab."""

    def test_unique_chars_mapped(self, vocab):
        """All unique characters in the corpus appear in stoi."""
        text = "hello"
        stoi, _ = vocab.build_vocab(text)
        assert set(stoi.keys()) == set("hello")

    def test_sorted_alphabetically(self, vocab):
        """Characters are assigned IDs in alphabetical order."""
        stoi, _ = vocab.build_vocab("cab")
        assert stoi["a"] < stoi["b"] < stoi["c"]

    def test_itos_is_inverse_of_stoi(self, vocab):
        """itos[stoi[c]] == c for every character in the corpus."""
        stoi, itos = vocab.build_vocab("abcde")
        for char, idx in stoi.items():
            assert itos[idx] == char

    def test_ids_start_at_zero(self, vocab):
        """The minimum assigned ID is 0."""
        stoi, _ = vocab.build_vocab("xyz")
        assert min(stoi.values()) == 0

    def test_ids_are_contiguous(self, vocab):
        """IDs form a contiguous range from 0 to vocab_size - 1."""
        stoi, _ = vocab.build_vocab("hello world")
        ids = sorted(stoi.values())
        assert ids == list(range(len(ids)))

    def test_single_char_corpus(self, vocab):
        """A single character corpus produces a vocab of size 1."""
        stoi, itos = vocab.build_vocab("a")
        assert len(stoi) == 1
        assert stoi["a"] == 0
        assert itos[0] == "a"

    def test_repeated_chars_deduplicated(self, vocab):
        """Repeated characters produce only one vocab entry."""
        stoi, _ = vocab.build_vocab("aaaa")
        assert len(stoi) == 1

    def test_whitespace_included_in_vocab(self, vocab):
        """Whitespace characters are treated as valid vocabulary entries."""
        stoi, _ = vocab.build_vocab("a b")
        assert " " in stoi

    def test_returns_tuple_of_two_dicts(self, vocab):
        """build_vocab returns a 2-tuple of dicts."""
        result = vocab.build_vocab("test")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], dict)
        assert isinstance(result[1], dict)


class TestEncode:
    """Tests for Solution.encode."""

    def test_encode_produces_list_of_ints(self, vocab):
        """encode returns a list of integers."""
        stoi, _ = vocab.build_vocab("abc")
        result = vocab.encode("abc", stoi)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    def test_encode_length_matches_input(self, vocab):
        """Output length equals the length of the input string."""
        text = "hello"
        stoi, _ = vocab.build_vocab(text)
        assert len(vocab.encode(text, stoi)) == len(text)

    def test_encode_correct_ids(self, vocab):
        """Each character is mapped to the correct integer ID."""
        stoi, _ = vocab.build_vocab("abc")
        result = vocab.encode("bac", stoi)
        assert result == [stoi["b"], stoi["a"], stoi["c"]]

    def test_encode_single_char(self, vocab):
        """A single character string encodes to a one element list."""
        stoi, _ = vocab.build_vocab("z")
        assert vocab.encode("z", stoi) == [0]


class TestDecode:
    """Tests for Solution.decode."""

    def test_decode_produces_string(self, vocab):
        """decode returns a str."""
        _, itos = vocab.build_vocab("abc")
        assert isinstance(vocab.decode([0, 1, 2], itos), str)

    def test_decode_correct_chars(self, vocab):
        """Each integer ID maps back to the correct character."""
        stoi, itos = vocab.build_vocab("abc")
        ids = [stoi["c"], stoi["a"], stoi["b"]]
        assert vocab.decode(ids, itos) == "cab"

    def test_roundtrip_encode_decode(self, vocab):
        """encode followed by decode is the identity function."""
        text = "hello world"
        stoi, itos = vocab.build_vocab(text)
        assert vocab.decode(vocab.encode(text, stoi), itos) == text

    def test_empty_ids_gives_empty_string(self, vocab):
        """Decoding an empty list returns an empty string."""
        _, itos = vocab.build_vocab("abc")
        assert vocab.decode([], itos) == ""
