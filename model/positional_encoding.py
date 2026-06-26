"""positional_encoding.py - Sinusoidal positional encoding from "Attention Is All You Need".

Computes a (seq_len, d_model) matrix where even columns are sine and odd columns are
cosine of position scaled frequencies. No loops are used; broadcasting handles the
full matrix at once.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_positional_encoding(self, seq_len: int, d_model: int) -> NDArray[np.float64]:
        """Compute sinusoidal positional encodings for a sequence.

        Fills a (seq_len, d_model) matrix using the formulas:
            PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
            PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

        Uses np.arange() and broadcasting to compute all values at once without
        any explicit loops. Even columns (0::2) receive sine values; odd columns
        (1::2) receive cosine values.

        Args:
            seq_len: Number of positions (rows) in the encoding matrix.
            d_model: Model dimension (columns); determines the frequency range.

        Returns:
            NDArray[np.float64]: Positional encoding matrix of shape (seq_len, d_model),
            rounded to 5 decimal places.
        """
        # PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
        #
        # Hint: Use np.arange() to create position and dimension index vectors,
        # then compute all values at once with broadcasting (no loops needed).
        # Assign sine to even columns (PE[:, 0::2]) and cosine to odd columns (PE[:, 1::2]).
        # Round to 5 decimal places.
        
        # Create an empty matrix of shape (seq_len x d_model), initially full of 0s
        PE = np.zeros((seq_len, d_model))
        # Create position vector -> Shape becomes (seq_len x 1)
            # np.arange(seq_len) -> Creates integers from 0 to seq_len - 1; ex. np.arange(5) -> [0 1 2 3 4]
            # .reshape(-1, 1) -> Turns the 1D array into a column vector 
        position = np.arange(seq_len).reshape(-1, 1)
        # Compute frequency scaling terms (creates the denominator values)
            # np.arange(0, d_model, 2) -> Creates even indices only, ex. np.arange(0, 6, 2) -> [0 2 4]
        div_term = 10000 ** (np.arange(0, d_model, 2) / d_model)
        # Fill in all even columns with sin(x)
            # : -> all rows
            # 0::2 -> start at column 0, step by 2
        # Fill in all odd columns with cos(x)
            # : -> all rows
            # 1::2 -> start at column 1, step by 2
            # Slice div_term so dimensions match when d_model is odd
        PE[:, 0::2] = np.sin(position / div_term)
        PE[:, 1::2] = np.cos(position / div_term[:PE[:, 1::2].shape[1]])

        # Return positional encodings rounded to 5 decimal places
        return np.round(PE, 5)