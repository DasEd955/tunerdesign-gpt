"""test_gpt.py - Unit tests for model/gpt.py and model/model.py.

Tests the full GPT forward pass (shape, rounding, logit range), the
GPTConfig dataclass defaults, and the create_model / save_model / load_model
round-trip (checkpoint integrity and weight equality).
"""

import os
import torch
import pytest
from model.gpt import GPT
from model.model import GPTConfig, create_model, save_model, load_model


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_config():
    """A minimal GPTConfig that runs quickly in tests."""
    return GPTConfig(
        vocab_size=20,
        context_length=8,
        model_dim=16,
        num_blocks=2,
        num_heads=2,
        batch_size=2,
        epochs=2,
        lr=1e-3,
    )


@pytest.fixture
def small_model(small_config):
    """A freshly initialized GPT from small_config."""
    return create_model(small_config)


@pytest.fixture
def token_input(small_config):
    """A (2, 8) integer token tensor within the vocabulary range."""
    return torch.randint(0, small_config.vocab_size, (2, small_config.context_length))


# ---------------------------------------------------------------------------
# GPTConfig
# ---------------------------------------------------------------------------

class TestGPTConfig:
    """Tests for the GPTConfig dataclass."""

    def test_required_field_vocab_size(self):
        """GPTConfig requires vocab_size; all others have defaults."""
        config = GPTConfig(vocab_size=50)
        assert config.vocab_size == 50

    def test_default_context_length(self):
        """Default context_length is 32."""
        assert GPTConfig(vocab_size=10).context_length == 32

    def test_default_model_dim(self):
        """Default model_dim is 128."""
        assert GPTConfig(vocab_size=10).model_dim == 128

    def test_default_num_blocks(self):
        """Default num_blocks is 3."""
        assert GPTConfig(vocab_size=10).num_blocks == 3

    def test_default_num_heads(self):
        """Default num_heads is 4."""
        assert GPTConfig(vocab_size=10).num_heads == 4

    def test_custom_overrides(self):
        """Custom values override all defaults."""
        config = GPTConfig(vocab_size=100, context_length=16, model_dim=64)
        assert config.vocab_size == 100
        assert config.context_length == 16
        assert config.model_dim == 64


# ---------------------------------------------------------------------------
# GPT forward pass
# ---------------------------------------------------------------------------

class TestGPTForward:
    """Tests for GPT.forward (model/gpt.py)."""

    def test_output_shape(self, small_model, token_input, small_config):
        """Output shape is (batch, seq_len, vocab_size)."""
        out = small_model(token_input)
        assert out.shape == (2, small_config.context_length, small_config.vocab_size)

    def test_output_is_tensor(self, small_model, token_input):
        """Forward returns a torch.Tensor."""
        assert isinstance(small_model(token_input), torch.Tensor)

    def test_rounded_to_four_decimals(self, small_model, token_input):
        """All logit values are rounded to 4 decimal places."""
        out = small_model(token_input)
        assert torch.all(out == torch.round(out, decimals=4))

    def test_output_is_float(self, small_model, token_input):
        """Output dtype is a floating point type (raw logits, not probabilities)."""
        out = small_model(token_input)
        assert out.is_floating_point()

    def test_no_softmax_applied(self, small_model, token_input):
        """Logits are not constrained to [0, 1] (no softmax in forward)."""
        out = small_model(token_input)
        assert not (torch.all(out >= 0) and torch.all(out <= 1))

    def test_single_token_context(self, small_config):
        """A context of length 1 does not raise an error."""
        model = create_model(small_config)
        x = torch.randint(0, small_config.vocab_size, (1, 1))
        out = model(x)
        assert out.shape == (1, 1, small_config.vocab_size)

    def test_batch_size_one(self, small_config):
        """A batch size of 1 processes correctly."""
        model = create_model(small_config)
        x = torch.randint(0, small_config.vocab_size, (1, small_config.context_length))
        out = model(x)
        assert out.shape == (1, small_config.context_length, small_config.vocab_size)

    def test_different_batches_produce_different_outputs(self, small_model, small_config):
        """Two distinct input sequences generally produce distinct logit tensors."""
        torch.manual_seed(1)
        x1 = torch.randint(0, small_config.vocab_size, (1, small_config.context_length))
        x2 = torch.randint(0, small_config.vocab_size, (1, small_config.context_length))
        if not torch.equal(x1, x2):
            assert not torch.allclose(small_model(x1), small_model(x2))


# ---------------------------------------------------------------------------
# create_model
# ---------------------------------------------------------------------------

class TestCreateModel:
    """Tests for create_model (model/model.py)."""

    def test_returns_gpt_instance(self, small_config):
        """create_model returns a GPT instance."""
        assert isinstance(create_model(small_config), GPT)

    def test_embedding_table_size(self, small_model, small_config):
        """Word embedding table has vocab_size rows and model_dim columns."""
        assert small_model.word_embeddings.weight.shape == (
            small_config.vocab_size, small_config.model_dim
        )

    def test_position_embedding_size(self, small_model, small_config):
        """Position embedding table has context_length rows and model_dim columns."""
        assert small_model.position_embeddings.weight.shape == (
            small_config.context_length, small_config.model_dim
        )

    def test_num_transformer_blocks(self, small_model, small_config):
        """The sequential block stack contains num_blocks TransformerBlock modules."""
        assert len(small_model.transformer_blocks) == small_config.num_blocks


# ---------------------------------------------------------------------------
# save_model / load_model
# ---------------------------------------------------------------------------

class TestCheckpointRoundtrip:
    """Tests for save_model and load_model (model/model.py)."""

    def test_save_creates_file(self, small_model, small_config, tmp_path):
        """save_model writes a file at the specified path."""
        path = str(tmp_path / "test_gpt.pt")
        save_model(small_model, small_config, path)
        assert os.path.exists(path)

    def test_load_returns_gpt_and_config(self, small_model, small_config, tmp_path):
        """load_model returns a (GPT, GPTConfig) 2-tuple."""
        path = str(tmp_path / "test_gpt.pt")
        save_model(small_model, small_config, path)
        loaded_model, loaded_config = load_model(path)
        assert isinstance(loaded_model, GPT)
        assert isinstance(loaded_config, GPTConfig)

    def test_loaded_config_matches_saved(self, small_model, small_config, tmp_path):
        """The config restored from the checkpoint equals the original config."""
        path = str(tmp_path / "test_gpt.pt")
        save_model(small_model, small_config, path)
        _, loaded_config = load_model(path)
        assert loaded_config == small_config

    def test_loaded_weights_match_saved(self, small_model, small_config, tmp_path, token_input):
        """Model weights and outputs are identical before and after the round trip."""
        path = str(tmp_path / "test_gpt.pt")
        # Both models must be in eval mode to disable dropout before comparing outputs.
        small_model.eval()
        original_out = small_model(token_input)
        save_model(small_model, small_config, path)
        loaded_model, _ = load_model(path)
        loaded_model.eval()
        loaded_out = loaded_model(token_input)
        assert torch.allclose(original_out, loaded_out, atol=1e-4)

    def test_save_creates_parent_dirs(self, small_model, small_config, tmp_path):
        """save_model creates missing parent directories automatically."""
        path = str(tmp_path / "nested" / "dir" / "gpt.pt")
        save_model(small_model, small_config, path)
        assert os.path.exists(path)
