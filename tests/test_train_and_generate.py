"""test_train_and_generate.py - Unit tests for train.py and generate.py.

Tests the AdamW training loop (return type, rounding, loss behavior) and
the autoregressive text generator (output length, character coverage,
determinism).
"""

import torch
import pytest
from train import Solution as TrainSolution
from generate import Solution as GenerateSolution
from model.model import GPTConfig, create_model
from data.vocab import Solution as VocabSolution


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

CORPUS = (
    "the cat sat on the mat the rat ate the bat "
    "the cat sat on the mat the rat ate the bat"
)


@pytest.fixture
def small_config():
    """A fast-training GPTConfig for use in training and generation tests."""
    vocab = VocabSolution()
    stoi, _ = vocab.build_vocab(CORPUS)
    return GPTConfig(
        vocab_size=len(stoi),
        context_length=8,
        model_dim=16,
        num_blocks=1,
        num_heads=2,
        batch_size=2,
        epochs=5,
        lr=1e-2,
    )


@pytest.fixture
def small_model(small_config):
    """A freshly initialized GPT for training tests."""
    return create_model(small_config)


@pytest.fixture
def encoded_data():
    """Encoded integer tensor of CORPUS."""
    vocab = VocabSolution()
    stoi, _ = vocab.build_vocab(CORPUS)
    return torch.tensor(vocab.encode(CORPUS, stoi), dtype=torch.long)


@pytest.fixture
def itos():
    """Integer-to-character mapping for CORPUS."""
    vocab = VocabSolution()
    _, itos = vocab.build_vocab(CORPUS)
    return itos


# ---------------------------------------------------------------------------
# train.py
# ---------------------------------------------------------------------------

class TestTrain:
    """Tests for TrainSolution.train (train.py)."""

    @pytest.fixture
    def trainer(self):
        """Return a TrainSolution instance."""
        return TrainSolution()

    def test_returns_float(self, trainer, small_model, encoded_data, small_config):
        """train() returns a float."""
        result = trainer.train(
            small_model, encoded_data,
            small_config.epochs, small_config.context_length,
            small_config.batch_size, small_config.lr
        )
        assert isinstance(result, float)

    def test_rounded_to_four_decimals(self, trainer, small_model, encoded_data, small_config):
        """Returned loss is rounded to exactly 4 decimal places."""
        loss = trainer.train(
            small_model, encoded_data,
            small_config.epochs, small_config.context_length,
            small_config.batch_size, small_config.lr
        )
        assert loss == round(loss, 4)

    def test_loss_is_positive(self, trainer, small_model, encoded_data, small_config):
        """Cross-entropy loss is always a positive value."""
        loss = trainer.train(
            small_model, encoded_data,
            small_config.epochs, small_config.context_length,
            small_config.batch_size, small_config.lr
        )
        assert loss > 0.0

    def test_loss_decreases_over_more_epochs(self, trainer, small_config, encoded_data):
        """Training for many more epochs yields a lower final loss than a handful of steps."""
        model_few = create_model(small_config)
        model_many = create_model(small_config)

        # Use a large epoch gap (1 vs 300) so convergence is reliable even on a
        # tiny corpus and model where a small gap can produce flaky results due to
        # per-epoch seed variation.
        loss_few = trainer.train(
            model_few, encoded_data, 1,
            small_config.context_length, small_config.batch_size, small_config.lr
        )
        loss_many = trainer.train(
            model_many, encoded_data, 300,
            small_config.context_length, small_config.batch_size, small_config.lr
        )
        assert loss_many < loss_few

    def test_model_weights_change_after_training(self, trainer, small_model, encoded_data, small_config):
        """At least one model parameter changes after one training step."""
        before = small_model.word_embeddings.weight.clone()
        trainer.train(
            small_model, encoded_data, 1,
            small_config.context_length, small_config.batch_size, small_config.lr
        )
        after = small_model.word_embeddings.weight
        assert not torch.equal(before, after)

    def test_single_epoch(self, trainer, small_model, encoded_data, small_config):
        """Training for a single epoch does not raise an error."""
        loss = trainer.train(
            small_model, encoded_data, 1,
            small_config.context_length, small_config.batch_size, small_config.lr
        )
        assert isinstance(loss, float)


# ---------------------------------------------------------------------------
# generate.py
# ---------------------------------------------------------------------------

class TestGenerate:
    """Tests for GenerateSolution.generate (generate.py)."""

    @pytest.fixture
    def generator(self):
        """Return a GenerateSolution instance."""
        return GenerateSolution()

    @pytest.fixture
    def trained_model(self, small_model, encoded_data, small_config):
        """A lightly trained model suitable for generation tests."""
        trainer = TrainSolution()
        trainer.train(
            small_model, encoded_data, 10,
            small_config.context_length, small_config.batch_size, small_config.lr
        )
        small_model.eval()
        return small_model

    @pytest.fixture
    def seed_context(self, encoded_data, small_config):
        """A (1, context_length) seed tensor drawn from the corpus."""
        return encoded_data[:small_config.context_length].unsqueeze(0)

    def test_returns_string(self, generator, trained_model, seed_context, small_config, itos):
        """generate() returns a str."""
        result = generator.generate(
            trained_model, new_chars=5,
            context=seed_context,
            context_length=small_config.context_length,
            int_to_char=itos,
        )
        assert isinstance(result, str)

    def test_output_length(self, generator, trained_model, seed_context, small_config, itos):
        """The generated string has exactly new_chars characters."""
        result = generator.generate(
            trained_model, new_chars=10,
            context=seed_context,
            context_length=small_config.context_length,
            int_to_char=itos,
        )
        assert len(result) == 10

    def test_characters_in_vocab(self, generator, trained_model, seed_context, small_config, itos):
        """Every generated character belongs to the training vocabulary."""
        result = generator.generate(
            trained_model, new_chars=20,
            context=seed_context,
            context_length=small_config.context_length,
            int_to_char=itos,
        )
        valid_chars = set(itos.values())
        assert all(c in valid_chars for c in result)

    def test_deterministic_with_seeded_generator(self, generator, trained_model, seed_context, small_config, itos):
        """Two calls with the same model and context produce the same output."""
        kwargs = dict(
            new_chars=15,
            context=seed_context,
            context_length=small_config.context_length,
            int_to_char=itos,
        )
        result1 = generator.generate(trained_model, **kwargs)
        result2 = generator.generate(trained_model, **kwargs)
        assert result1 == result2

    def test_zero_new_chars_returns_empty_string(self, generator, trained_model, seed_context, small_config, itos):
        """Requesting 0 new characters returns an empty string."""
        result = generator.generate(
            trained_model, new_chars=0,
            context=seed_context,
            context_length=small_config.context_length,
            int_to_char=itos,
        )
        assert result == ""

    def test_context_cropped_when_too_long(self, generator, trained_model, small_config, itos, encoded_data):
        """Contexts longer than context_length are cropped without error."""
        long_context = encoded_data[:small_config.context_length + 10].unsqueeze(0)
        result = generator.generate(
            trained_model, new_chars=5,
            context=long_context,
            context_length=small_config.context_length,
            int_to_char=itos,
        )
        assert len(result) == 5
