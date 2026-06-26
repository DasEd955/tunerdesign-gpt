"""activations.py - Element-wise neural network activation functions.

Implements sigmoid and ReLU, the two most fundamental activation functions
used in feedforward and classification networks.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def sigmoid(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the sigmoid activation function element-wise.

        Formula: 1 / (1 + e^(-z))

        Args:
            z: 1D NumPy array of pre-activation values.

        Returns:
            NDArray[np.float64]: Sigmoid-activated output rounded to 5 decimal places.
        """
        # z is a 1D NumPy array
        # Formula: 1 / (1 + e^(-z))
        # return np.round(your_answer, 5)
        z = 1 / (1 + np.exp(-z))
        return np.round(z, 5)

    def relu(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply the ReLU activation function element-wise.

        Formula: max(0, z) applied to each element.

        Args:
            z: 1D NumPy array of pre-activation values.

        Returns:
            NDArray[np.float64]: ReLU-activated output (negative values clamped to 0).
        """
        # z is a 1D NumPy array
        # Formula: max(0, z) element-wise
        return np.maximum(0, z)
            
