"""attention.py - Single-head causal self-attention module.

Implements the fundamental scaled dot product attention operation with a causal
(lower triangular) mask so each position can only attend to itself and earlier positions.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class SingleHeadAttention(nn.Module):
    """Single-head scaled dot product causal self-attention.

    Projects the input into key, query, and value spaces, computes masked attention
    scores, and returns the weighted value aggregation.
    """

    def __init__(self, embedding_dim: int, attention_dim: int):
        """Initialize the three projection matrices.

        Instantiation order (key, query, value) is fixed to ensure reproducible
        weights when torch.manual_seed(0) is set.

        Args:
            embedding_dim: Dimensionality of the input token embeddings.
            attention_dim: Dimensionality of the key/query/value projections.
        """
        super().__init__()
        torch.manual_seed(0)
        # Create three linear projections (Key, Query, Value) with bias=False
        # Instantiation order matters for reproducible weights: key, query, value
        self.key = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.query = nn.Linear(embedding_dim, attention_dim, bias=False)
        self.value = nn.Linear(embedding_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        """Compute causal self-attention for the input sequence.

        Projects input through K, Q, V layers, computes scaled dot product scores,
        applies a causal lower triangular mask (filling future positions with -inf),
        normalizes with softmax, and returns the weighted value aggregation rounded
        to 4 decimal places.

        Args:
            embedded: Input tensor of shape (batch, seq_len, embedding_dim).

        Returns:
            TensorType[float]: Output tensor of shape (batch, seq_len, attention_dim),
            rounded to 4 decimal places.
        """
        # 1. Project input through K, Q, V linear layers
        # 2. Compute attention scores: (Q @ K^T) / sqrt(attention_dim)
        # 3. Apply causal mask: use torch.tril(torch.ones(...)) to build lower-triangular matrix,
        #    then masked_fill positions where mask == 0 with float('-inf')
        # 4. Apply softmax(dim=2) to masked scores
        # 5. Return (scores @ V) rounded to 4 decimal places
        key = self.key(embedded)
        query = self.query(embedded)
        value = self.value(embedded)
        
        scores = query @ torch.transpose(key, 1, 2)
        context_length, attention_dim = key.shape[1], key.shape[2]
        scores = scores / (attention_dim ** 0.5)

        lower_triangular = torch.tril(torch.ones(context_length, context_length))
        mask = lower_triangular == 0
        scores = scores.masked_fill(mask, float("-inf"))
        scores = nn.functional.softmax(scores, dim=2)

        return torch.round(scores @ value, decimals=4)

