import torch
import torch.nn as nn
import torch.nn.functional as F

# The GPT model is provided for you. It returns raw logits (not probabilities).
# You only need to implement the training loop below.

class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        # Train the GPT model using AdamW and cross_entropy loss.
        # For each epoch: seed with torch.manual_seed(epoch).
        # Sample a batch: randomly select starting position in the training data & extract input-target pairs
        # Forward pass: feed input tokens through the model; output shape is (B, T, C)
            # B -> batch size, T -> Context Length, C -> vocab size
        # Compute loss: Reshape logits to (B * T, V) & targets to (B * T), then apply cross-entropy
            # This treats each position as an independent classification problem
        # Backward pass: loss.backward() computes gradients for every parameter
        # Update: optimizer.step() applies AdamW updates
        # Return the final loss rounded to 4 decimals.
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

        for epoch in range(epochs):
            torch.manual_seed(epoch)
            iX = torch.randint(len(data) - context_length, (batch_size,))
            X = torch.stack([data[i:i + context_length] for i in iX])    
            Y = torch.stack([data[i + 1:i + 1 + context_length] for i in iX])
            
            logits = model(X)
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B * T, C), Y.view(B * T))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        return round(loss.item(), 4)