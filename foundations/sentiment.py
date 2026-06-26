"""sentiment.py - Bag-of-words sentiment classifier using learned embeddings.

Maps each token to a 16-dimensional embedding, averages over the sequence
(bag-of-words), projects to a scalar, and applies sigmoid for a binary
sentiment probability.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    """Bag-of-words sentiment classifier.

    Architecture: Embedding(vocabulary_size, 16) -> mean over T -> Linear(16, 1) -> Sigmoid
    """

    def __init__(self, vocabulary_size: int):
        """Initialize the embedding, linear, and sigmoid layers.

        Args:
            vocabulary_size: Number of unique tokens in the vocabulary;
                determines the number of rows in the embedding table.
        """
        super().__init__()
        torch.manual_seed(0)
        # Layers: Embedding(vocabulary_size, 16) -> Linear(16, 1) -> Sigmoid
        self.embedding = nn.Embedding(vocabulary_size, 16)
        self.linear = nn.Linear(16, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: TensorType[int]) -> TensorType[float]:
        """Classify sentiment of a batch of padded token-ID sequences.

        The embedding layer outputs a (B, T, 16) tensor which is averaged across
        the token dimension (dim=1) into a (B, 16) tensor before the linear layer.

        Args:
            x: Integer token ID tensor of shape (B, T).

        Returns:
            TensorType[float]: Sigmoid probability tensor of shape (B, 1),
            rounded to 4 decimal places.
        """
        # Hint: The embedding layer outputs a B, T, embed_dim tensor
        # but you should average it into a B, embed_dim tensor before using the Linear layer

        # Return a B, 1 tensor and round to 4 decimal places
        embeddings = self.embedding(x)
        averaged = torch.mean(embeddings, dim=1)
        projected = self.linear(averaged)
        return torch.round(self.sigmoid(projected), decimals=4)


