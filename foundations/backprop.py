"""backprop.py - Single neuron forward and backward pass with sigmoid activation.

Demonstrates the complete gradient computation for a single sigmoid neuron:
forward pass to compute prediction, MSE loss, and analytic backpropagation
to obtain weight and bias gradients.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def sigmoid(self, z: float) -> float:
        """Compute the sigmoid of a scalar value.

        Args:
            z: Scalar pre-activation value.

        Returns:
            float: sigmoid(z) = 1 / (1 + e^(-z)).
        """
        return 1 / (1 + np.exp(-z))

    def backward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, y_true: float) -> Tuple[NDArray[np.float64], float]:
        """Compute gradients for a single sigmoid neuron via backpropagation.

        Forward pass: z = dot(x, w) + b, y_hat = sigmoid(z)
        Loss: L = 0.5 * (y_hat - y_true)^2
        Returns the gradients dL/dw and dL/db derived by chain rule.

        Args:
            x: 1D input array.
            w: 1D weight array (same length as x).
            b: Scalar bias term.
            y_true: True target value.

        Returns:
            Tuple[NDArray[np.float64], float]: (dL_dw, dL_db) both rounded to
            5 decimal places.
        """
        # x: 1D input array
        # w: 1D weight array
        # b: scalar bias
        # y_true: true target value
        #
        # Forward: z = dot(x, w) + b, y_hat = sigmoid(z)
        # Loss: L = 0.5 * (y_hat - y_true)^2
        # Return: (dL_dw rounded to 5 decimals, dL_db rounded to 5 decimals)
        z = np.dot(x, w) + b
        y_hat = self.sigmoid(z)
        loss = 0.5 * np.square(y_hat - y_true)
        dL_dw = (y_hat - y_true) * (y_hat * (1 - y_hat)) * x
        dL_db = (y_hat - y_true) * (y_hat * (1 - y_hat))
        return (np.round(dL_dw, 5), np.round(dL_db, 5))