"""test_integration.py - Integration tests for the end-to-end GPT pipeline.

Each test exercises multiple stages together to verify that the modules
compose correctly: data pipeline to training, training to generation,
checkpoint save/load to generation, and the full run() entry point.
"""

import os
import torch
import pytest
from data.vocab import Solution as VocabSolution
from data.loader import Solution as LoaderSolution
from model.model import GPTConfig, create_model, save_model, load_model, run
from train import Solution as TrainSolution
from generate import Solution as GenerateSolution


# ---------------------------------------------------------------------------
# Shared corpus and helpers
# ---------------------------------------------------------------------------

CORPUS = (
    "all the world is a stage and all the men and women are merely players "
    "they have their exits and their entrances and one man in his time plays many parts"
)


def _build_encoded(text: str):
    """Return (stoi, itos, encoded_tensor) for the given text."""
    vocab = VocabSolution()
    stoi, itos = vocab.build_vocab(text)
    data = torch.tensor(vocab.encode(text, stoi), dtype=torch.long)
    return stoi, itos, data


def _small_config(vocab_size: int) -> GPTConfig:
    """Return a minimal GPTConfig suitable for fast integration tests."""
    return GPTConfig(
        vocab_size=vocab_size,
        context_length=8,
        model_dim=16,
        num_blocks=1,
        num_heads=2,
        batch_size=4,
        epochs=5,
        lr=1e-2,
    )


# ---------------------------------------------------------------------------
# Data pipeline to training
# ---------------------------------------------------------------------------

class TestDataPipelineToTraining:
    """Verify that the data pipeline feeds the training loop correctly."""

    def test_encoded_corpus_used_in_training(self):
        """Encoding text via vocab.py and passing the tensor to train.py runs without error."""
        stoi, _, data = _build_encoded(CORPUS)
        config = _small_config(len(stoi))
        model = create_model(config)
        trainer = TrainSolution()
        loss = trainer.train(
            model, data, config.epochs,
            config.context_length, config.batch_size, config.lr
        )
        assert isinstance(loss, float)
        assert loss > 0.0

    def test_loader_output_drives_gpt_forward(self):
        """Batches from loader.py have the correct shape to feed the GPT forward pass."""
        stoi, _, data = _build_encoded(CORPUS)
        config = _small_config(len(stoi))
        model = create_model(config)
        loader = LoaderSolution()
        X, Y = loader.create_batches(data, context_length=config.context_length, batch_size=config.batch_size)
        logits = model(X)
        assert logits.shape == (config.batch_size, config.context_length, config.vocab_size)

    def test_logit_shape_matches_vocab_size(self):
        """GPT logit width equals the vocabulary size derived from the corpus."""
        stoi, _, data = _build_encoded(CORPUS)
        config = _small_config(len(stoi))
        model = create_model(config)
        x = data[:config.context_length].unsqueeze(0)
        logits = model(x)
        assert logits.shape[-1] == len(stoi)


# ---------------------------------------------------------------------------
# Training to generation
# ---------------------------------------------------------------------------

class TestTrainingToGeneration:
    """Verify that a trained model produces valid generated text."""

    def test_trained_model_generates_valid_chars(self):
        """Characters produced by generate.py after training all belong to the corpus vocab."""
        stoi, itos, data = _build_encoded(CORPUS)
        config = _small_config(len(stoi))
        model = create_model(config)

        trainer = TrainSolution()
        trainer.train(
            model, data, config.epochs,
            config.context_length, config.batch_size, config.lr
        )
        model.eval()

        seed = data[:config.context_length].unsqueeze(0)
        generator = GenerateSolution()
        text = generator.generate(model, new_chars=20, context=seed,
                                  context_length=config.context_length, int_to_char=itos)
        valid = set(itos.values())
        assert all(c in valid for c in text)

    def test_generated_length_is_exact(self):
        """generate.py produces exactly the requested number of characters."""
        stoi, itos, data = _build_encoded(CORPUS)
        config = _small_config(len(stoi))
        model = create_model(config)

        trainer = TrainSolution()
        trainer.train(
            model, data, config.epochs,
            config.context_length, config.batch_size, config.lr
        )
        model.eval()

        seed = data[:config.context_length].unsqueeze(0)
        generator = GenerateSolution()
        text = generator.generate(model, new_chars=30, context=seed,
                                  context_length=config.context_length, int_to_char=itos)
        assert len(text) == 30


# ---------------------------------------------------------------------------
# Checkpoint save/load to generation
# ---------------------------------------------------------------------------

class TestCheckpointToGeneration:
    """Verify that a model saved and reloaded continues to generate the same output."""

    def test_loaded_model_generates_same_output(self, tmp_path):
        """Saving and loading a model preserves generation output exactly."""
        stoi, itos, data = _build_encoded(CORPUS)
        config = _small_config(len(stoi))
        model = create_model(config)

        trainer = TrainSolution()
        trainer.train(
            model, data, config.epochs,
            config.context_length, config.batch_size, config.lr
        )

        path = str(tmp_path / "checkpoint.pt")
        save_model(model, config, path)

        loaded_model, loaded_config = load_model(path)
        loaded_model.eval()
        model.eval()

        seed = data[:config.context_length].unsqueeze(0)
        generator = GenerateSolution()
        kwargs = dict(new_chars=15, context=seed,
                      context_length=config.context_length, int_to_char=itos)
        text_original = generator.generate(model, **kwargs)
        text_loaded = generator.generate(loaded_model, **kwargs)
        assert text_original == text_loaded

    def test_loaded_config_enables_correct_forward_shape(self, tmp_path):
        """A model reconstructed from checkpoint produces the expected logit shape."""
        stoi, itos, data = _build_encoded(CORPUS)
        config = _small_config(len(stoi))
        model = create_model(config)
        path = str(tmp_path / "checkpoint.pt")
        save_model(model, config, path)
        loaded_model, loaded_config = load_model(path)
        x = data[:loaded_config.context_length].unsqueeze(0)
        logits = loaded_model(x)
        assert logits.shape == (1, loaded_config.context_length, loaded_config.vocab_size)


# ---------------------------------------------------------------------------
# Full run() entry point
# ---------------------------------------------------------------------------

class TestRunEntryPoint:
    """Verify the run() function executes the full pipeline without error."""

    def test_run_completes_without_error(self, tmp_path, capsys):
        """run() on a short corpus writes a checkpoint and prints generated text."""
        path = str(tmp_path / "run_model.pt")
        run(
            training_text=CORPUS,
            save_path=path,
            context_length=8,
            model_dim=16,
            num_blocks=1,
            num_heads=2,
            batch_size=4,
            epochs=5,
            lr=1e-2,
            new_chars=10,
        )
        assert os.path.exists(path)

    def test_run_prints_generated_text(self, tmp_path, capsys):
        """run() prints a non-empty generated text block."""
        path = str(tmp_path / "run_model2.pt")
        run(
            training_text=CORPUS,
            save_path=path,
            context_length=8,
            model_dim=16,
            num_blocks=1,
            num_heads=2,
            batch_size=4,
            epochs=5,
            lr=1e-2,
            new_chars=8,
        )
        captured = capsys.readouterr()
        assert "Generated text" in captured.out

    def test_run_checkpoint_is_loadable(self, tmp_path):
        """The checkpoint written by run() can be loaded with load_model."""
        path = str(tmp_path / "run_checkpoint.pt")
        run(
            training_text=CORPUS,
            save_path=path,
            context_length=8,
            model_dim=16,
            num_blocks=1,
            num_heads=2,
            batch_size=4,
            epochs=3,
            lr=1e-2,
            new_chars=5,
        )
        loaded_model, loaded_config = load_model(path)
        assert isinstance(loaded_config, GPTConfig)
        assert loaded_config.context_length == 8


# ---------------------------------------------------------------------------
# Cross-module reproducibility
# ---------------------------------------------------------------------------

class TestReproducibility:
    """Verify that fixed seeds produce identical outputs across the full pipeline."""

    def test_two_training_runs_produce_same_loss(self):
        """Training the same corpus twice from identical seeds gives the same final loss."""
        stoi, _, data = _build_encoded(CORPUS)
        config = _small_config(len(stoi))
        trainer = TrainSolution()

        model1 = create_model(config)
        model2 = create_model(config)

        loss1 = trainer.train(model1, data, config.epochs,
                              config.context_length, config.batch_size, config.lr)
        loss2 = trainer.train(model2, data, config.epochs,
                              config.context_length, config.batch_size, config.lr)
        assert loss1 == loss2

    def test_encode_decode_roundtrip_through_pipeline(self):
        """Encoding a corpus and immediately decoding returns the original text."""
        vocab = VocabSolution()
        stoi, itos = vocab.build_vocab(CORPUS)
        ids = vocab.encode(CORPUS, stoi)
        recovered = vocab.decode(ids, itos)
        assert recovered == CORPUS
