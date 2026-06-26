"""grouped_query_attention.py - Grouped-Query Attention (GQA) as used in modern LLMs.

GQA reduces memory bandwidth during inference by sharing a smaller set of key/value
heads (num_kv_heads) across multiple query heads (num_heads). Each KV head is
repeated (num_heads // num_kv_heads) times so standard scaled dot product attention
can be applied without changing the Q projection.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention: multiple query heads share a reduced set of KV heads.

    Reduces the KV cache size from num_heads to num_kv_heads while keeping
    the same query capacity, improving inference memory efficiency.
    """

    def __init__(self, model_dim: int, num_heads: int, num_kv_heads: int):
        """Initialize the Q, K, V, and output projection matrices.

        Q projects to num_heads * head_dim; K and V project to num_kv_heads * head_dim.

        Args:
            model_dim: Input/output embedding dimension.
            num_heads: Total number of query heads (must evenly divide model_dim).
            num_kv_heads: Number of key/value heads (must evenly divide num_heads).
        """
        super().__init__()
        torch.manual_seed(0)
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = model_dim // num_heads

        self.q_proj = nn.Linear(model_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, num_kv_heads * self.head_dim, bias=False)
        self.output_proj = nn.Linear(num_heads * self.head_dim, model_dim, bias=False)

    def forward(self, x: TensorType[float]) -> TensorType[float]:
        """Compute grouped-query causal self-attention.

        Projects x into Q, K, V, reshapes into heads (Q has num_heads, K/V have
        num_kv_heads), expands K and V by repeating each KV head
        (num_heads // num_kv_heads) times, computes scaled dot product attention
        with a causal mask, concatenates heads, and applies the output projection.

        Args:
            x: Input tensor of shape (batch, seq_len, model_dim).

        Returns:
            TensorType[float]: Output of shape (batch, seq_len, model_dim),
            rounded to 4 decimal places.
        """
        B, T, D = x.shape

        # 1. Project x into Q, K, V using the projection layers
        # 2. Reshape into heads: Q has num_heads, K and V have num_kv_heads
        # 3. Expand K, V by repeating each KV head (num_heads // num_kv_heads) times
        # 4. Compute scaled dot-product attention with causal mask
        # 5. Concatenate heads and apply output projection
        # 6. Return rounded output (decimals=4)
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_kv_heads, self.head_dim).transpose(1, 2)

        repeats = self.num_heads // self.num_kv_heads
        k, v = k.repeat_interleave(repeats, dim=1), v.repeat_interleave(repeats, dim=1)

        scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        mask = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)

        out = (weights @ v).transpose(1, 2).contiguous().view(B, T, -1)
        return torch.round(self.output_proj(out), decimals=4)