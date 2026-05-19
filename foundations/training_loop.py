import numpy as np
from numpy.typing import NDArray
from typing import Tuple


class Solution:
    def train(self, X: NDArray[np.float64], y: NDArray[np.float64], epochs: int, lr: float) -> Tuple[NDArray[np.float64], float]:
        # X: (n_samples, n_features)
        # y: (n_samples,) targets
        # epochs: number of training iterations
        # lr: learning rate
        #
        # Model: y_hat = X @ w + b
        # Loss: MSE = (1/n) * sum((y_hat - y)^2)
        # Initialize w = zeros, b = 0
        # return (np.round(w, 5), round(b, 5))

        # Initialize:
            # w -> Learned weight vector; initially a 1D array of zeros of size X.shape[1]
            # b -> Bias
            # n -> Size of the dataset (rows) 
        w, b = np.zeros(X.shape[1]), 0.0
        n = X.shape[0]

        # Iterate through all training epochs
            # Model (Forward Pass): y_hat = X @ w + b
            # Use Mean Squared Error Loss: (1/n) * np.sum(np.square(y_hat - y))
            # Gradients:
                # Compute error vector to simplify: y_hat - y
                # dL_dw = (2/n) * X.T @ (error)
                # dL_db = (2/n) * np.sum(error)
            # Update:
                # w -= lr * dL_dw
                # b -= lr * dL_db
        for epoch in range(epochs):
            y_hat = X @ w + b
            mse_loss = (1/n) * np.sum(np.square(y_hat - y))
            error = y_hat - y
            dL_dw = (2/n) * X.T @ (error)
            dL_db = (2/n) * np.sum(error)
            w -= lr * dL_dw
            b -= lr * dL_db

        # Return the output as a tuple (w, b), the learned weight vector & bias
        return (np.round(w, 5), round(b, 5))
