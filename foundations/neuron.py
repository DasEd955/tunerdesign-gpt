import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
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
