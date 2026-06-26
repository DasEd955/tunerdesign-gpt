"""normalization.py - Layer normalization forward pass.

Implements the standard Layer Normalization operation: zero-centers and unit-scales
a feature vector using its own mean and variance, then applies a learned affine
transform (gamma, beta).
"""

import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], gamma: NDArray[np.float64], beta: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply layer normalization to a 1D feature vector.

        Normalizes x using its mean and variance (eps=1e-5), then scales and shifts
        with the learned parameters gamma and beta:
            x_hat = (x - mean) / sqrt(var + eps)
            out = gamma * x_hat + beta

        Args:
            x: 1D feature vector to normalize.
            gamma: 1D scale parameter of the same length as x.
            beta: 1D shift parameter of the same length as x.

        Returns:
            NDArray[np.float64]: Normalized and affine transformed vector,
            rounded to 5 decimal places.
        """
        # x: 1D feature vector
        # gamma: 1D scale parameter (same length as x)
        # beta: 1D shift parameter (same length as x)
        # eps = 1e-5
        # Normalize: x_hat = (x - mean) / sqrt(var + eps)
        # Scale and shift: out = gamma * x_hat + beta
        # return np.round(your_answer, 5)
        mean = np.mean(x)
        var = np.var(x)
        epsilon = 1e-5
        out = ((x - mean) / (np.sqrt(var + epsilon)) * gamma) + beta
        return np.round(out, 5)
