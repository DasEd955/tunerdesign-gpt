"""kv_cache.py - Key/Value cache for efficient autoregressive inference.

During generation each new token only needs to attend to itself plus previously
computed keys and values. KVCache accumulates those tensors so CachedAttention
avoids recomputing them on every step.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class KVCache:
    """Accumulates key and value tensors across generation steps.

    On the first call to update(), the cache is initialized with the given tensors.
    Subsequent calls concatenate along the sequence dimension (dim=1) so the
    full history is always available.
    """

    def __init__(self):
        """Initialize an empty cache with no stored keys or values."""
        self.cache_k: Optional[torch.Tensor] = None  # (batch, seq_len, model_dim)
        self.cache_v: Optional[torch.Tensor] = None

    def update(self, new_k: torch.Tensor, new_v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new keys and values to the cache and return the full history.

        On the first call, initializes the cache with the given tensors.
        Subsequent calls concatenate along the sequence dimension (dim=1).

        Args:
            new_k: Key tensor for the current step, shape (batch, new_seq, model_dim).
            new_v: Value tensor for the current step, shape (batch, new_seq, model_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Full accumulated (keys, values) tensors.
        """
        if self.cache_k is None:
            self.cache_k = new_k
            self.cache_v = new_v
        else:
            self.cache_k = torch.cat([self.cache_k, new_k], dim=1)
            self.cache_v = torch.cat([self.cache_v, new_v], dim=1)
        return (self.cache_k, self.cache_v)

    def clear(self):
        """Reset the cache, discarding all accumulated keys and values."""
        self.cache_k = None
        self.cache_v = None


class CachedAttention(nn.Module):
    """Attention module that uses a KVCache to avoid recomputing keys and values.

    On each forward call, projects x into Q, K, V; appends K and V to the cache;
    then attends over the full cached history. This makes each generation step O(1)
    in computation for the KV projection rather than O(seq_len).
    """

    def __init__(self, model_dim: int):
        """Initialize the Q, K, V projection matrices.

        Args:
            model_dim: Input/output embedding dimension; all three projections map
                model_dim to model_dim.
        """
        super().__init__()
        torch.manual_seed(0)
        self.q_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[KVCache] = None) -> Tuple[torch.Tensor, KVCache]:
        """Project x, update the KV cache, and compute attention over the full history.

        Projects x into Q, K, V. Creates a new KVCache if none is provided, then
        updates it with the current K and V. Computes scaled dot product attention
        using Q against the full cached K and V.

        Args:
            x: Input tensor of shape (batch, seq_len, model_dim).
            kv_cache: Existing KVCache to extend, or None to start a fresh cache.

        Returns:
            Tuple[torch.Tensor, KVCache]: Attention output rounded to 4 decimal places
            and the updated KVCache for the next generation step.
        """
        # 1. Project x into Q, K, V using the linear layers
        # 2. If kv_cache is None, create a new KVCache
        # 3. Update the cache with the new K and V
        # 4. Compute scaled dot-product attention using Q and the full cached K, V
        # 5. Return (rounded output, kv_cache)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        if kv_cache is None:
            kv_cache = KVCache()
        full_k, full_v = kv_cache.update(k, v)

        scores = (q @ full_k.transpose(-2, -1)) * (full_k.shape[-1] ** -0.5)
        weights = torch.softmax(scores, dim=-1)
        output = weights @ full_v

        return (torch.round(output, decimals=4), kv_cache) 