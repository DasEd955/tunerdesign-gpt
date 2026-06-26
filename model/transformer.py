"""transformer.py - Pre-LN Transformer block with multi-head attention and feed forward sublayers.

Uses Pre-LN architecture: LayerNorm is applied BEFORE each sub-layer, not after.
This differs from the original "Attention is All You Need" diagram but trains more stably.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class TransformerBlock(nn.Module):
    """A single Pre-LN transformer block: multi-head self-attention followed by an MLP.

    Each sublayer is wrapped in a residual connection:
        x = x + attention(layer_norm_1(x))
        x = x + feed_forward(layer_norm_2(x))
    """

    def __init__(self, model_dim: int, num_heads: int):
        """Build the block's sublayers in a fixed order for reproducible weights.

        Instantiation order: MultiHeadedSelfAttention, VanillaNeuralNetwork,
        first LayerNorm, second LayerNorm.

        Args:
            model_dim: Embedding dimension shared across all sublayers.
            num_heads: Number of parallel attention heads (must evenly divide model_dim).
        """
        super().__init__()
        torch.manual_seed(0)
        # Instantiate in this order:
        # 1. self.MultiHeadedSelfAttention(model_dim, num_heads)
        # 2. self.VanillaNeuralNetwork(model_dim)
        # 3. Two nn.LayerNorm(model_dim) instances
        self.multihead_attention = self.MultiHeadedSelfAttention(model_dim, num_heads)
        self.linear_nn = self.VanillaNeuralNetwork(model_dim)
        self.first_norm = nn.LayerNorm(model_dim)
        self.second_norm = nn.LayerNorm(model_dim)

    def forward(self, embedded: TensorType[float]) -> TensorType[float]:
        """Apply two Pre-LN residual sublayers to the input sequence.

        Two residual connections with Pre-LN:
            x = x + attention(layer_norm_1(x))
            x = x + feed_forward(layer_norm_2(x))

        Args:
            embedded: Input tensor of shape (batch, seq_len, model_dim).

        Returns:
            TensorType[float]: Output of the same shape, rounded to 4 decimal places.
        """
        torch.manual_seed(0)
        # Two residual connections with Pre-LN:
        #   x = x + attention(layer_norm_1(x))
        #   x = x + feed_forward(layer_norm_2(x))
        # Return result rounded to 4 decimal places
        embedded = embedded + self.multihead_attention(self.first_norm(embedded))
        embedded = embedded + self.linear_nn(self.second_norm(embedded))
        return torch.round(embedded, decimals=4)

    class MultiHeadedSelfAttention(nn.Module):
        """Multi-head causal self-attention with a learned output projection."""

        class SingleHeadAttention(nn.Module):
            """One attention head: projects to head_size and applies causal scaled dot product attention."""

            def __init__(self, model_dim: int, head_size: int):
                """Initialize the K, Q, V projections for a single head.

                Args:
                    model_dim: Dimensionality of the input embeddings.
                    head_size: Projection dimension for this head (model_dim // num_heads).
                """
                super().__init__()
                torch.manual_seed(0)
                self.key_gen = nn.Linear(model_dim, head_size, bias=False)
                self.query_gen = nn.Linear(model_dim, head_size, bias=False)
                self.value_gen = nn.Linear(model_dim, head_size, bias=False)

            def forward(self, embedded: TensorType[float]) -> TensorType[float]:
                """Compute causal scaled dot product attention for this head.

                Args:
                    embedded: Input of shape (batch, seq_len, model_dim).

                Returns:
                    TensorType[float]: Output of shape (batch, seq_len, head_size).
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

        def __init__(self, model_dim: int, num_heads: int):
            """Build num_heads SingleHeadAttention instances and an output projection.

            Args:
                model_dim: Total model dimension; each head operates on model_dim // num_heads.
                num_heads: Number of parallel attention heads.
            """
            super().__init__()
            torch.manual_seed(0)
            self.att_heads = nn.ModuleList()
            for i in range(num_heads):
                self.att_heads.append(self.SingleHeadAttention(model_dim, model_dim // num_heads))
            self.output_proj = nn.Linear(model_dim, model_dim, bias=False)

        def forward(self, embedded: TensorType[float]) -> TensorType[float]:
            """Run all heads in parallel, concatenate, and apply the output projection.

            Args:
                embedded: Input of shape (batch, seq_len, model_dim).

            Returns:
                TensorType[float]: Output of shape (batch, seq_len, model_dim).
            """
            head_outputs = []
            for head in self.att_heads:
                head_outputs.append(head(embedded))
            concatenated = torch.cat(head_outputs, dim = 2)
            return self.output_proj(concatenated)

    class VanillaNeuralNetwork(nn.Module):
        """Position-wise feed forward sublayer: up-project by 4x, ReLU, down-project, Dropout."""

        def __init__(self, model_dim: int):
            """Initialize the two linear layers and dropout (p=0.2).

            Args:
                model_dim: Input/output dimension; the hidden layer uses model_dim * 4.
            """
            super().__init__()
            torch.manual_seed(0)
            self.up_projection = nn.Linear(model_dim, model_dim * 4)
            self.relu = nn.ReLU()
            self.down_projection = nn.Linear(model_dim * 4, model_dim)
            self.dropout = nn.Dropout(0.2) # using p = 0.2

        def forward(self, x: TensorType[float]) -> TensorType[float]:
            """Apply the up-projection, ReLU, down-projection, and dropout.

            Args:
                x: Input of shape (batch, seq_len, model_dim).

            Returns:
                TensorType[float]: Output of the same shape.
            """
            torch.manual_seed(0)
            return self.dropout(self.down_projection(self.relu(self.up_projection(x))))
