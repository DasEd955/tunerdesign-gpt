"""test_app.py - Unit and integration tests for app.py.

Tests the vocabulary encoding helper, the response generation function, and the
Gradio interface builder in isolation using a lightweight in-memory model so that
no saved checkpoint file is required on disk.
"""

import torch
import pytest
from unittest.mock import MagicMock, patch

from app import (
    encode_prompt,
    generate_response,
    build_respond_fn,
    build_interface,
)
from model.model import GPTConfig, create_model
from data.vocab import Solution as VocabSolution


# ---------------------------------------------------------------------------
# Shared Fixtures
# ---------------------------------------------------------------------------

CORPUS = (
    "All the world's a stage, and all the men and women merely players: "
    "they have their exits and their entrances, and one man in his time plays many parts, "
    "his acts being seven ages."
)


@pytest.fixture
def vocab_maps():
    """Return (stoi, itos) built from CORPUS."""
    vocab = VocabSolution()
    stoi, itos = vocab.build_vocab(CORPUS)
    return stoi, itos


@pytest.fixture
def small_config(vocab_maps):
    """A minimal GPTConfig for fast in-memory tests."""
    stoi, _ = vocab_maps
    return GPTConfig(
        vocab_size=len(stoi),
        context_length=8,
        model_dim=16,
        num_blocks=1,
        num_heads=2,
        batch_size=2,
        epochs=1,
        lr=1e-2,
    )


@pytest.fixture
def small_model(small_config):
    """A freshly initialized (untrained) GPT in eval mode."""
    model = create_model(small_config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# encode_prompt
# ---------------------------------------------------------------------------

class TestEncodePrompt:
    """Tests for the encode_prompt helper in app.py."""

    def test_returns_2d_tensor(self, vocab_maps):
        """encode_prompt returns a 2-D tensor of shape (1, T)."""
        stoi, _ = vocab_maps
        result = encode_prompt("All", stoi)
        assert result.ndim == 2
        assert result.shape[0] == 1

    def test_length_matches_known_chars(self, vocab_maps):
        """Encoded length equals the number of characters present in the vocabulary."""
        stoi, _ = vocab_maps
        prompt = "All"
        expected_len = sum(1 for ch in prompt if ch in stoi)
        result = encode_prompt(prompt, stoi)
        assert result.shape[1] == expected_len

    def test_unknown_chars_are_dropped(self, vocab_maps):
        """Characters not in stoi are silently excluded from the output tensor."""
        stoi, _ = vocab_maps
        prompt = "All\x00\x01"
        result = encode_prompt(prompt, stoi)
        expected_len = sum(1 for ch in prompt if ch in stoi)
        assert result.shape[1] == expected_len

    def test_fully_unknown_prompt_returns_fallback(self, vocab_maps):
        """A prompt with no vocabulary characters returns a (1, 1) tensor of token 0."""
        stoi, _ = vocab_maps
        result = encode_prompt("\x00\x01\x02", stoi)
        assert result.shape == (1, 1)
        assert result[0, 0].item() == 0

    def test_output_dtype_is_long(self, vocab_maps):
        """The returned tensor has dtype torch.long (int64)."""
        stoi, _ = vocab_maps
        result = encode_prompt("stage", stoi)
        assert result.dtype == torch.long

    def test_empty_prompt_returns_fallback(self, vocab_maps):
        """An empty prompt returns the fallback (1, 1) tensor of token 0."""
        stoi, _ = vocab_maps
        result = encode_prompt("", stoi)
        assert result.shape == (1, 1)


# ---------------------------------------------------------------------------
# generate_response
# ---------------------------------------------------------------------------

class TestGenerateResponse:
    """Tests for generate_response in app.py."""

    def test_returns_string(self, small_model, small_config, vocab_maps):
        """generate_response returns a str."""
        stoi, itos = vocab_maps
        result = generate_response(small_model, small_config, stoi, itos, "All", new_chars=5)
        assert isinstance(result, str)

    def test_output_length_matches_new_chars(self, small_model, small_config, vocab_maps):
        """The returned string has exactly new_chars characters."""
        stoi, itos = vocab_maps
        result = generate_response(small_model, small_config, stoi, itos, "stage", new_chars=10)
        assert len(result) == 10

    def test_output_chars_in_vocab(self, small_model, small_config, vocab_maps):
        """Every character in the output belongs to the training vocabulary."""
        stoi, itos = vocab_maps
        result = generate_response(small_model, small_config, stoi, itos, "All", new_chars=15)
        valid = set(itos.values())
        assert all(ch in valid for ch in result)

    def test_deterministic_across_calls(self, small_model, small_config, vocab_maps):
        """Calling generate_response twice with the same inputs yields identical output."""
        stoi, itos = vocab_maps
        kwargs = dict(new_chars=10)
        r1 = generate_response(small_model, small_config, stoi, itos, "All the world", **kwargs)
        r2 = generate_response(small_model, small_config, stoi, itos, "All the world", **kwargs)
        assert r1 == r2

    def test_model_set_to_eval_mode(self, small_model, small_config, vocab_maps):
        """generate_response forces the model into eval mode before generation."""
        stoi, itos = vocab_maps
        small_model.train()
        generate_response(small_model, small_config, stoi, itos, "All", new_chars=3)
        assert not small_model.training


# ---------------------------------------------------------------------------
# build_respond_fn
# ---------------------------------------------------------------------------

class TestBuildRespondFn:
    """Tests for the respond closure returned by build_respond_fn."""

    def test_returns_string(self, small_model, small_config, vocab_maps):
        """The respond function returns a str."""
        stoi, itos = vocab_maps
        respond = build_respond_fn(small_model, small_config, stoi, itos)
        result = respond("All the world", [])
        assert isinstance(result, str)

    def test_empty_history_accepted(self, small_model, small_config, vocab_maps):
        """respond accepts an empty history list without error."""
        stoi, itos = vocab_maps
        respond = build_respond_fn(small_model, small_config, stoi, itos)
        result = respond("stage", [])
        assert isinstance(result, str)

    def test_non_empty_history_accepted(self, small_model, small_config, vocab_maps):
        """respond accepts a populated history list without error."""
        stoi, itos = vocab_maps
        respond = build_respond_fn(small_model, small_config, stoi, itos)
        history = [
            {"role": "user", "content": "All"},
            {"role": "assistant", "content": "test"},
        ]
        result = respond("the world", history)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# build_interface
# ---------------------------------------------------------------------------

class TestBuildInterface:
    """Tests for build_interface in app.py."""

    def test_returns_chat_interface(self, small_model, small_config, vocab_maps):
        """build_interface returns a gr.ChatInterface instance."""
        import gradio as gr
        stoi, itos = vocab_maps
        interface = build_interface(small_model, small_config, stoi, itos)
        assert isinstance(interface, gr.ChatInterface)

    def test_title_is_set(self, small_model, small_config, vocab_maps):
        """The interface title is 'Tuner Design GPT'."""
        stoi, itos = vocab_maps
        interface = build_interface(small_model, small_config, stoi, itos)
        assert interface.title == "Tuner Design GPT"

    def test_description_contains_disclaimer(self, small_model, small_config, vocab_maps):
        """The interface description contains the educational disclaimer text."""
        stoi, itos = vocab_maps
        interface = build_interface(small_model, small_config, stoi, itos)
        assert "inference only" in interface.description
        assert "educational" in interface.description

