"""linear_regression.py - Linear regression prediction and mean squared error.

Implements the two core operations of a linear model: computing predictions
via matrix-vector multiplication and evaluating MSE loss against ground truth.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        """Compute linear model predictions for a feature matrix.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            weights: Weight vector of shape (n_features,).

        Returns:
            NDArray[np.float64]: Predicted values of shape (n_samples,),
            rounded to 5 decimal places.
        """
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        return np.round(np.dot(X, weights), 5)

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        """Compute mean squared error between predictions and ground truth.

        Args:
            model_prediction: Predicted values array of shape (n_samples,).
            ground_truth: True target values array of shape (n_samples,).

        Returns:
            float: MSE = sum((pred - truth)^2) / n, rounded to 5 decimal places.
        """
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        return np.round(np.sum(np.square(model_prediction - ground_truth)) / len(ground_truth), 5)
