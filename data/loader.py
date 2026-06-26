"""loader.py - Tensor-based batch creator for integer encoded text.

Samples random (input, target) tensor pairs from a 1D encoded token sequence.
Y is X shifted right by one position, implementing the standard next-token
prediction objective.
"""

import torch
from torchtyping import TensorType
from typing import Tuple


class Solution:
    def create_batches(self, data: TensorType[int], context_length: int, batch_size: int) -> Tuple[TensorType[int], TensorType[int]]:
        """Sample a batch of input/target token ID tensors from an encoded corpus.

        Picks batch_size random starting positions (seeded with torch.manual_seed(0))
        and extracts context_length-length windows. Y is X shifted right by 1 so that
        Y[i][j] = data[start_i + j + 1], implementing next-token prediction.

        Args:
            data: 1D tensor of integer token IDs representing the full corpus.
            context_length: Number of tokens per training example (T).
            batch_size: Number of examples per batch (B).

        Returns:
            Tuple[TensorType[int], TensorType[int]]: (X, Y) each of shape
            (batch_size, context_length).
        """
        # data: 1D tensor of encoded text (integer token IDs)
        # context_length: number of tokens in each training example
        # batch_size: number of examples per batch
        #
        # Return (X, Y) where:
        # - X has shape (batch_size, context_length)
        # - Y has shape (batch_size, context_length)
        # - Y is X shifted right by 1 (Y[i][j] = data[start_i + j + 1])
        #
        # Use torch.manual_seed(0) before generating random start indices
        # Use torch.randint to pick random starting positions
        torch.manual_seed(0)
        iX = torch.randint(len(data) - context_length, (batch_size,))
        X = torch.stack([data[i:i + context_length] for i in iX])
        Y = torch.stack([data[i + 1:i + 1 + context_length] for i in iX])
        return (X, Y)