import os
import torch
from dataclasses import dataclass

from .gpt import GPT
from data.vocab import Solution as VocabSolution
from train import Solution as TrainSolution
from generate import Solution as GenerateSolution


@dataclass
class GPTConfig:
    vocab_size: int
    context_length: int = 32
    model_dim: int = 128
    num_blocks: int = 3
    num_heads: int = 4
    batch_size: int = 8
    epochs: int = 200
    lr: float = 1e-3


def create_model(config: GPTConfig) -> GPT:
    return GPT(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        model_dim=config.model_dim,
        num_blocks=config.num_blocks,
        num_heads=config.num_heads,
    )


def save_model(model: GPT, config: GPTConfig, path: str = "saved_model/gpt.pt") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": config}, path)


def load_model(path: str = "saved_model/gpt.pt") -> tuple[GPT, GPTConfig]:
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
