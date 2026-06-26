"""embeddings.py - Token embedding lookup via direct matrix indexing.

Demonstrates the fundamental embedding operation: selecting rows from a
(vocab_size, embed_dim) weight matrix using integer token IDs.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:
    def lookup(self, embeddings: NDArray[np.float64], token_ids: NDArray[np.int64]) -> NDArray[np.float64]:
        """Retrieve embedding vectors for a sequence of token IDs.

        Selects rows from the embedding matrix using token_ids as indices.
        Equivalent to a one-hot multiply but implemented as direct array indexing.

        Args:
            embeddings: Weight matrix of shape (vocab_size, embed_dim).
            token_ids: 1D array of integer token IDs to look up.

        Returns:
            NDArray[np.float64]: Selected rows of shape (len(token_ids), embed_dim),
            rounded to 5 decimal places.
        """
        # embeddings: (vocab_size, embed_dim) matrix
        # token_ids: 1D array of integer token IDs
        # Return the embedding vectors for the given token IDs
        # return np.round(your_answer, 5)
        return np.round(embeddings[token_ids], 5)
