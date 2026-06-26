"""loss.py - Binary and categorical cross-entropy loss functions.

Implements the two standard classification losses used in neural networks.
Both clip predictions with a small epsilon to prevent log(0) instability.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def binary_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        """Compute binary cross-entropy loss for binary classification.

        Clips y_pred to [1e-7, 1-1e-7] to avoid log(0).
        Formula: -mean(y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred))

        Args:
            y_true: Ground truth labels (0 or 1).
            y_pred: Predicted probabilities in [0, 1].

        Returns:
            float: Mean binary cross-entropy loss, rounded to 4 decimal places.
        """
        # y_true: true labels (0 or 1)
        # y_pred: predicted probabilities
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return np.round(loss, 4)

    def categorical_cross_entropy(self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]) -> float:
        """Compute categorical cross-entropy loss for multi-class classification.

        Clips y_pred to [1e-7, 1-1e-7] to avoid log(0). Sums log-probabilities
        over classes per sample, then averages over samples.
        Formula: -mean(sum(y_true * log(y_pred), axis=1))

        Args:
            y_true: One-hot encoded true labels of shape (n_samples, n_classes).
            y_pred: Predicted class probabilities of shape (n_samples, n_classes).

        Returns:
            float: Mean categorical cross-entropy loss, rounded to 4 decimal places.
        """
        # y_true: one-hot encoded true labels (shape: n_samples x n_classes)
        # y_pred: predicted probabilities (shape: n_samples x n_classes)
        # Hint: add a small epsilon (1e-7) to y_pred to avoid log(0)
        # return round(your_answer, 4)
        epsilon = 1e-7
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        loss = -np.mean(np.sum(y_true * np.log(y_pred), axis = 1))
        return np.round(loss, 4)
