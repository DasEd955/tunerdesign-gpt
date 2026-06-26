"""mlp.py - Multilayer perceptron forward pass with ReLU hidden activations.

Implements a generic MLP forward pass for an arbitrary number of layers.
ReLU is applied after every hidden layer; the output layer has no activation.
"""

import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        """Run a multilayer perceptron forward pass.

        Applies each weight matrix and bias vector in sequence. ReLU is applied
        after every hidden layer (all layers except the last); the final layer
        produces raw output with no activation.

        Args:
            x: 1D input array.
            weights: List of 2D weight matrices, one per layer.
            biases: List of 1D bias vectors, one per layer (same length as weights).

        Returns:
            NDArray[np.float64]: Output of the final layer, rounded to 5 decimal places.
        """
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        h = x
        for i in range(len(weights)):
            h = np.dot(h, weights[i]) + biases[i]
            if i < len(weights) - 1:
                h = np.maximum(h, 0)
        return np.round(h, 5)
