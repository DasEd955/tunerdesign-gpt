#!/usr/bin/env python3
"""
Demonstration: GPT learns to generate Shakespeare-like text.
"""
import torch
from model.gpt import GPT
from data.vocab import Solution as VocabSolution
from train import Solution as TrainSolution
from generate import Solution as GenerateSolution

# Simulated training text (repeating pattern to make it learnable)
training_text = """All the world's a stage, and all the men and women merely players:
they have their exits and their entrances, and one man in his time plays many parts,
his acts being seven ages. At first, the infant, mewling and puking in the nurse's arms.
Then the whining school-boy, with his satchel and shining morning face, creeping like snail
unwilling to school. And then the lover, sighing like furnace, with a woeful ballad made
to his mistress' eyebrow. Then a soldier, full of strange oaths and bearded like the pard,
jealous in honour, sudden and quick in quarrel, seeking the bubble reputation even in
the cannon's mouth. And then the justice, in fair round belly with good capon lined, with
eyes severe and beard of formal cut, full of wise saws and modern instances. The sixth age
shifts into the lean and slippered pantaloon, with spectacles on nose and pouch on side;
his youthful hose, well saved, a world too wide for his shrunk shank, and his big manly
voice, turning again toward childish treble, pipes and whistles in his sound. Last scene of
all, that ends this strange eventful history, is second childishness and mere oblivion,
sans teeth, sans eyes, sans taste, sans everything."""

# Repeat for more training data
training_text = training_text * 3

# Build vocabulary
vocab_solution = VocabSolution()
stoi, itos = vocab_solution.build_vocab(training_text)
vocab_size = len(stoi)

# Encode training data
encoded = vocab_solution.encode(training_text, stoi)
data = torch.tensor(encoded, dtype=torch.long)

print("=" * 70)
print("GPT GENERATION DEMO")
print("=" * 70)
print(f"\nTraining on {len(training_text):,} characters")
print(f"Vocab size: {vocab_size} unique characters")
print(f"Sample unique characters: {sorted(stoi.keys())[:15]}")

# Create and train model
context_length = 32
model_dim = 128
num_blocks = 3
num_heads = 4
batch_size = 8
epochs = 200
lr = 0.001

print(f"\nModel config:")
print(f"  Context length: {context_length}")
print(f"  Model dim: {model_dim}")
print(f"  Transformer blocks: {num_blocks}")
print(f"  Attention heads: {num_heads}")
print(f"  Training epochs: {epochs}")

model = GPT(vocab_size, context_length, model_dim, num_blocks, num_heads)

# Train with progress
print(f"\nTraining...")
train_solution = TrainSolution()
optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

losses = []
for epoch in range(epochs):
    torch.manual_seed(epoch)
    iX = torch.randint(len(data) - context_length, (batch_size,))
    X = torch.stack([data[i:i + context_length] for i in iX])
    Y = torch.stack([data[i + 1:i + 1 + context_length] for i in iX])

    logits = model(X)
    B, T, C = logits.shape
    loss = torch.nn.functional.cross_entropy(logits.view(B * T, C), Y.view(B * T))

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    if (epoch + 1) % 50 == 0:
        print(f"  Epoch {epoch+1:3d}: loss = {loss.item():.4f}")

print(f"  Final loss: {losses[-1]:.4f} (started at {losses[0]:.4f})")
print(f"  Learning: {'YES' if (losses[0] - losses[-1] > 0.1) else 'minimal'}")

# Generate text
model.eval()
print(f"\nGenerated text (with different starting contexts):")
print("-" * 70)

generate_solution = GenerateSolution()

for start_idx in range(min(10, len(data) - context_length)):
    prompt_ids = data[start_idx : start_idx + 8].unsqueeze(0)
    context_ids = data[start_idx : start_idx + context_length].unsqueeze(0)

    generated = generate_solution.generate(
        model=model,
        new_chars=80,
        context=context_ids,
        context_length=context_length,
        int_to_char=itos
    )

    prompt_text = vocab_solution.decode(prompt_ids[0].tolist(), itos)
    print(f"\nPrompt: ...{prompt_text}...")
    print(f"Generated: {generated}")

print("\n" + "=" * 70)
print("[OK] GPT training and generation complete!")
print("=" * 70)
