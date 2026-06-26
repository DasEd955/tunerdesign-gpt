"""linear_regression_training.py - Gradient descent training for linear regression.

Extends linear_regression.py with an analytic gradient computation and a
full training loop that iteratively applies gradient descent updates to fit
a linear model via MSE minimization.
"""

import numpy as np
from numpy.typing import NDArray


class Solution:
    def get_derivative(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64], N: int, X: NDArray[np.float64], desired_weight: int) -> float:
        """Compute the MSE gradient with respect to one weight.

        Uses the analytic derivative: -2 * dot(ground_truth - prediction, X[:, j]) / N.

        Args:
            model_prediction: Current predicted values of shape (N,).
            ground_truth: True target values of shape (N,).
            N: Number of training samples (len(X)).
            X: Feature matrix of shape (N, n_features).
            desired_weight: Column index j of the weight to differentiate with respect to.

        Returns:
            float: Partial derivative of MSE with respect to weights[desired_weight].
        """
        # note that N is just len(X)
        return -2 * np.dot(ground_truth - model_prediction, X[:, desired_weight]) / N

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.squeeze(np.matmul(X, weights))

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Train a linear regression model using gradient descent.

        For each iteration: computes predictions, computes the gradient for every
        weight index via get_derivative(), and updates all weights by subtracting
        learning_rate * gradient.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            Y: Target values of shape (n_samples,).
            num_iterations: Number of gradient descent steps.
            initial_weights: Starting weight vector of shape (n_features,).

        Returns:
            NDArray[np.float64]: Trained weight vector rounded to 5 decimal places.
        """
        # For each iteration:
        #   1. Compute predictions with get_model_prediction(X, weights)
        #   2. For each weight index j, compute gradient with get_derivative()
        #   3. Update: weights[j] -= learning_rate * gradient
        # Return np.round(final_weights, 5)
        weights = initial_weights.copy()
        for iteration in range(num_iterations):
            prediction = self.get_model_prediction(X, weights)
            gradients = np.zeros_like(weights)
            for j in range(len(weights)):
                gradients[j] = self.get_derivative(prediction, Y, len(Y), X, j)
            weights -= self.learning_rate * gradients
        return np.round(weights, 5)