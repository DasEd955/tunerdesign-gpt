"""train.py - AdamW training loop for the character-level GPT model.

The GPT model is provided externally and returns raw logits (not probabilities).
This module implements only the gradient update loop used to fit the model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Solution:
    def train(self, model: nn.Module, data: torch.Tensor, epochs: int, context_length: int, batch_size: int, lr: float) -> float:
        """Train the GPT model using AdamW and cross-entropy loss.

        For each epoch: seeds with torch.manual_seed(epoch), samples a random batch of
        (input, target) pairs by selecting batch_size starting positions from data, runs a
        forward pass to get logits of shape (B, T, C), reshapes to (B*T, V) and (B*T,) for
        cross-entropy, then runs the backward pass and an AdamW parameter update.

        Args:
            model: GPT model that returns logits with shape (batch, context, vocab).
            data: 1D tensor of integer token IDs representing the full training corpus.
            epochs: Number of gradient update iterations to perform.
            context_length: Number of tokens in each input/target sequence (T).
            batch_size: Number of sequences sampled per gradient step (B).
            lr: Learning rate for the AdamW optimizer.

        Returns:
            float: The final training loss rounded to 4 decimal places.
        """
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