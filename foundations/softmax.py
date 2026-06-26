"""softmax.py - Numerically stable softmax activation.

Computes softmax by subtracting max(z) before exponentiation to prevent overflow,
then normalizes so all outputs sum to 1.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute the softmax of a 1D logit array.

        Subtracts max(z) before computing exp for numerical stability,
        then divides by the sum so outputs are a valid probability distribution.

        Args:
            z: 1D NumPy array of raw logit values.

        Returns:
            NDArray[np.float64]: Probability distribution over classes,
            rounded to 4 decimal places.
        """
        # z is a 1D NumPy array of logits
        # Hint: subtract max(z) for numerical stability before computing exp
        # return np.round(your_answer, 4)
        exp_z = np.exp(z - np.max(z))
        z = exp_z / np.sum(exp_z)
        return np.round(z, 4)
