"""model.py - Top-level GPT model factory, persistence, and end-to-end training pipeline.

Provides GPTConfig, model construction, checkpoint save/load, and the run() entry point
that wires together vocabulary building, training, saving, and autoregressive text
generation in a single call.
"""

import os
import torch
from dataclasses import dataclass

from .gpt import GPT
from data.vocab import Solution as VocabSolution
from train import Solution as TrainSolution
from generate import Solution as GenerateSolution


@dataclass
class GPTConfig:
    """Hyperparameter bundle for constructing and training a GPT model.

    Attributes:
        vocab_size: Number of unique tokens in the character-level vocabulary.
        context_length: Maximum sequence length the model attends over.
        model_dim: Embedding and hidden dimension used throughout all layers.
        num_blocks: Number of stacked transformer blocks.
        num_heads: Number of attention heads per block (must evenly divide model_dim).
        batch_size: Number of training examples drawn per gradient step.
        epochs: Total number of gradient-update iterations during training.
        lr: Learning rate passed to the AdamW optimizer.
    """

    vocab_size: int
    context_length: int = 32
    model_dim: int = 128
    num_blocks: int = 3
    num_heads: int = 4
    batch_size: int = 8
    epochs: int = 200
    lr: float = 1e-3


def create_model(config: GPTConfig) -> GPT:
    """Instantiate a GPT model from a GPTConfig.

    Args:
        config: Hyperparameter bundle describing the model architecture.

    Returns:
        GPT: A freshly initialized (untrained) GPT instance.
    """
    return GPT(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        model_dim=config.model_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
    )


def save_model(model: GPT, config: GPTConfig, path: str = "saved_model/gpt.pt") -> None:
    """Serialize model weights and config to a PyTorch checkpoint file.

    Creates any missing parent directories before writing. The checkpoint
    bundles both the state dict and GPTConfig so load_model() can reconstruct
    the architecture without additional arguments.

    Args:
        model: Trained GPT instance whose state dict will be saved.
        config: The GPTConfig used to build the model.
        path: Filesystem path for the output ``.pt`` file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, path)


def load_model(path: str = "saved_model/gpt.pt") -> tuple[GPT, GPTConfig]:
    """Restore a GPT model and its config from a checkpoint file.

    Reads the checkpoint written by save_model(), reconstructs the architecture
    via create_model(), and loads the saved weights.

    Args:
        path: Path to the ``.pt`` checkpoint file.

    Returns:
        tuple[GPT, GPTConfig]: The restored model (in training mode) and its config.
    """
    checkpoint = torch.load(path, weights_only=False)
    config: GPTConfig = checkpoint["config"]
    model = create_model(config)
    model.load_state_dict(checkpoint["model_state"])
    return model, config


def run(
    training_text: str,
    save_path: str = "saved_model/gpt.pt",
    context_length: int = 32,
    model_dim: int = 128,
    num_blocks: int = 3,
    num_heads: int = 4,
    batch_size: int = 8,
    epochs: int = 200,
    lr: float = 1e-3,
    new_chars: int = 200,
) -> None:
    """Build a vocabulary, train a GPT, save the checkpoint, and generate sample text.

    Full end-to-end pipeline: encodes training_text into a character-level integer
    tensor, builds a GPTConfig, trains with AdamW, saves the checkpoint, then runs
    autoregressive generation from the first context_length tokens and prints the result.

    Args:
        training_text: Raw string used as the entire training corpus.
        save_path: Destination path for the saved ``.pt`` checkpoint.
        context_length: Number of tokens per training sequence and generation window.
        model_dim: Embedding and hidden dimension for all layers.
        num_blocks: Number of stacked transformer blocks.
        num_heads: Number of attention heads per block.
        batch_size: Number of examples per gradient step.
        epochs: Total training iterations.
        lr: AdamW learning rate.
        new_chars: Number of new characters to autoregressively generate after training.
    """
    vocab = VocabSolution()
    stoi, itos = vocab.build_vocab(training_text)
    data = torch.tensor(vocab.encode(training_text, stoi), dtype=torch.long)

    config = GPTConfig(
        vocab_size=len(stoi),
        context_length=context_length,
        model_dim=model_dim,
        num_blocks=num_blocks,
        num_heads=num_heads,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
    )

    print(f"Vocab size: {config.vocab_size} | Corpus: {len(training_text):,} chars")
    print(f"Config: context={context_length}, dim={model_dim}, blocks={num_blocks}, heads={num_heads}")

    model = create_model(config)

    print(f"\nTraining for {epochs} epochs...")
    trainer = TrainSolution()
    final_loss = trainer.train(model, data, epochs, context_length, batch_size, lr)
    print(f"Final loss: {final_loss}")

    save_model(model, config, save_path)
    print(f"Model saved to {save_path}")

    model.eval()
    start = data[:context_length].unsqueeze(0)
    generator = GenerateSolution()
    text = generator.generate(
        model=model,
        new_chars=new_chars,
        context=start,
        context_length=context_length,
        int_to_char=itos,
    )
    print(f"\nGenerated text:\n{'-' * 60}\n{text}\n{'-' * 60}")


if __name__ == "__main__":
    training_text = (
        "All the world's a stage, and all the men and women merely players: "
        "they have their exits and their entrances, and one man in his time plays many parts, "
        "his acts being seven ages."
    ) * 5
    run(training_text)
