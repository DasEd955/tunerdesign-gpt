"""neuron.py - Single artificial neuron forward pass with configurable activation.

Models the fundamental computation unit of a neural network: a weighted sum of
inputs plus a bias, followed by an activation function (sigmoid or ReLU).
"""

import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        """Compute the output of a single neuron.

        Pre-activation: z = dot(x, w) + b
        Then applies the specified activation:
            'sigmoid': 1 / (1 + exp(-z))
            'relu':    max(0, z)
        Any other string returns z unchanged.

        Args:
            x: 1D input array.
            w: 1D weight array (same length as x).
            b: Scalar bias term.
            activation: Activation function name ('sigmoid' or 'relu').

        Returns:
            float: Activated output rounded to 5 decimal places.
        """
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        #
        # Pre-activation: z = dot(x, w) + b
        # Sigmoid: σ(z) = 1 / (1 + exp(-z))
        # ReLU: max(0, z)
        # return round(your_answer, 5)
        z, result = np.dot(x, w) + b, 0.0
        if activation == "sigmoid".lower():
            result = 1.0 / (1.0 + np.exp(-z))
        elif activation == "relu".lower():
            result = np.maximum(0.0, z)
        else:
            result = z
        return np.round(result, 5)
