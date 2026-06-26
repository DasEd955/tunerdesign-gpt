"""rms_normalization.py - RMS Normalization forward pass.

RMSNorm is similar to LayerNorm but omits mean-centering and the beta shift
parameter. It normalizes by the Root Mean Square of the input and scales by gamma,
making it simpler and slightly cheaper than full Layer Normalization.
"""

import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        """Apply RMS Normalization to input vector x.

        Computes the Root Mean Square of x (with eps for numerical stability),
        divides x by it to normalize, then scales by gamma. No mean-centering or
        beta shift is applied (unlike LayerNorm).

        Args:
            x: Input vector to normalize.
            gamma: Per-element scale parameter (same length as x).
            eps: Small constant added inside the square root to prevent division by zero.

        Returns:
            List[float]: Normalized and scaled output, rounded to 4 decimal places.
        """
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        # Normalize x, then scale by gamma
        # Return result rounded to 4 decimal places as a list

        # Compute Root-Mean-Square (RMS)
        # Divide by standard deviation (normalize)
        # Output: Scale by gamma
        rms_x = np.sqrt(np.mean(np.square(x) + eps))
        x_hat = x / rms_x
        output = gamma * x_hat

        # Return the normalized output as a list of floats
        return np.round(output, 4)
