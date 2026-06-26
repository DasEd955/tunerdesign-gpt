"""multi_layer_backprop.py - Two layer MLP forward and backward pass.

Implements manual backpropagation through a two layer ReLU network using NumPy.
Returns both the MSE loss and the weight/bias gradients for both layers,
demonstrating the chain rule applied to a concrete two layer architecture.
"""

import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        """Compute the forward pass, MSE loss, and all gradients for a two layer MLP.

        Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        Loss: MSE = mean((predictions - y_true)^2)

        Runs the forward pass to obtain predictions, then propagates gradients
        backward through both linear layers and the ReLU nonlinearity via chain rule.

        Args:
            x: 1D input vector.
            W1: 2D weight matrix for the first layer.
            b1: 1D bias vector for the first layer.
            W2: 2D weight matrix for the second layer.
            b2: 1D bias vector for the second layer.
            y_true: True target values.

        Returns:
            dict: Keys 'loss' (float), 'dW1' (2D list), 'db1' (1D list),
            'dW2' (2D list), 'db2' (1D list), all rounded to 4 decimal places.
        """
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)

        # Convert inputs to NumPy arrays
        x = np.array(x)
        W1 = np.array(W1)
        b1 = np.array(b1)
        W2 = np.array(W2)
        b2 = np.array(b2)
        y_true = np.array(y_true)

        # Forward Pass -> Computes predictions
        z1 = x @ W1.T + b1
        a1 = np.maximum(0, z1)
        z2 = a1 @ W2.T + b2
        loss = np.mean((z2 - y_true) ** 2)

        # Backward Pass (Backpropagation) -> Computes gradients
        n = len(y_true) if y_true.ndim > 0 else 1
        dz2 = 2 * (z2 - y_true) / n
        dW2 = dz2.reshape(-1, 1) @ a1.reshape(1, -1)
        db2 = dz2

        da1 = dz2.reshape(1, -1) @ W2
        da1 = da1.flatten()
        dz1 = da1 = da1 * (z1 > 0).astype(float)
        dW1 = dz1.reshape(-1, 1) @ x.reshape(1, -1)
        db1 = dz1

        return {
            'loss': round(float(loss), 4),
            'dW1': np.round(dW1, 4).tolist(),
            'db1': np.round(db1, 4).tolist(),
            'dW2': np.round(dW2, 4).tolist(),
            'db2': np.round(db2, 4).tolist(),
        }