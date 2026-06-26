"""multi_head_attention.py - Multi-head causal self-attention with an output projection.

Runs num_heads independent SingleHeadAttention heads in parallel, concatenates their
outputs along the feature dimension, and applies a final linear projection (W_O).
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class MultiHeadedSelfAttention(nn.Module):
    """Multi-head causal self-attention layer.

    Creates num_heads SingleHeadAttention instances, each operating on a
    (attention_dim // num_heads) sub-space, then combines them with an output projection.
    """

    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        """Build the attention heads and output projection.

        Each head uses attention_dim // num_heads as its per-head dimension.
        The output projection maps the concatenated head outputs back to attention_dim.

        Args:
            embedding_dim: Dimensionality of the input token embeddings.
            attention_dim: Total attention output dimension (split evenly across heads).
            num_heads: Number of parallel attention heads.
        """
        super().__init__()
        torch.manual_seed(0)
        # Create num_heads SingleHeadAttention instances using nn.ModuleList
        # Each head size = attention_dim // num_heads
        # Use: self.SingleHeadAttention(embedding_dim, head_size)
        # After the heads, add an output projection: nn.Linear(attention_dim, attention_dim, bias=False)
        self.attention_heads = nn.ModuleList()
        for head in range(num_heads):
            self.attention_heads.append(self.SingleHeadAttention(embedding_dim, attention_dim // num_heads))
        self.projection = nn.Linear(attention_dim, attention_dim, bias=False)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        """Run all heads, concatenate, and project through W_O.

        Runs each head on the input, concatenates outputs along dim=2,
        passes the result through the output projection (W_O), and returns
        the result rounded to 4 decimal places.

        Args:
            embedded: Input of shape (batch, seq_len, embedding_dim).

        Returns:
            TensorType[float]: Output of shape (batch, seq_len, attention_dim),
            rounded to 4 decimal places.
        """
        head_outputs = list()
        for head in self.attention_heads:
            head_outputs.append(head(embedded))
        concat = torch.cat(head_outputs, dim=2)
        return torch.round(self.projection(concat), decimals=4)

    class SingleHeadAttention(nn.Module):
        """One attention head: causal scaled dot product attention in a projected subspace."""

        def __init__(self, embedding_dim: int, attention_dim: int):
            """Initialize the K, Q, V projection matrices.

            Args:
                embedding_dim: Dimensionality of the input embeddings.
                attention_dim: Projection dimension for this head.
            """
            super().__init__()
            torch.manual_seed(0)
            self.key_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.query_gen = nn.Linear(embedding_dim, attention_dim, bias=False)
            self.value_gen = nn.Linear(embedding_dim, attention_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            """Compute causal scaled dot product attention for this head.

            Args:
                embedded: Input of shape (batch, seq_len, embedding_dim).

            Returns:
                TensorType[float]: Output of shape (batch, seq_len, attention_dim).
            """
            k = self.key_gen(embedded)
            q = self.query_gen(embedded)
            v = self.value_gen(embedded)

            scores = q @ torch.transpose(k, 1, 2) # @ is the same as torch.matmul()
            context_length, attention_dim = k.shape[1], k.shape[2]
            scores = scores / (attention_dim ** 0.5)

            lower_triangular = torch.tril(torch.ones(context_length, context_length))
            mask = lower_triangular == 0
            scores = scores.masked_fill(mask, float('-inf'))
            scores = nn.functional.softmax(scores, dim = 2)

            return scores @ v
