"""pytorch_basics.py - Fundamental PyTorch tensor operations.

Covers the core tensor manipulation and loss primitives needed to build
neural network training loops: reshaping, averaging, concatenation, and MSE loss.
All outputs are rounded to 4 decimal places.
"""

import torch
import torch.nn
from torchtyping import TensorType


class Solution:
    def reshape(self, to_reshape: TensorType[float]) -> TensorType[float]:
        """Reshape a 2D tensor from (M, N) to (M*N/2, 2).

        Args:
            to_reshape: Input tensor of shape (M, N).

        Returns:
            TensorType[float]: Reshaped tensor of shape (M*N//2, 2),
            rounded to 4 decimal places.
        """
        # Reshape (M, N) tensor to (M*N/2, 2)
        # Use torch.reshape(tensor, new_shape)
        m, n = to_reshape.shape
        return torch.round(torch.reshape(to_reshape, (m * n // 2, 2)), decimals = 4)

    def average(self, to_avg: TensorType[float]) -> TensorType[float]:
        """Compute the column-wise mean (average across rows, dim=0).

        Args:
            to_avg: Input 2D tensor.

        Returns:
            TensorType[float]: 1D tensor of column means, rounded to 4 decimal places.
        """
        # Compute column-wise mean (average across rows)
        # Use torch.mean(tensor, dim=0)
        return torch.round(torch.mean(to_avg, dim=0), decimals = 4)

    def concatenate(self, cat_one: TensorType[float], cat_two: TensorType[float]) -> TensorType[float]:
        """Join two tensors side-by-side along dim=1.

        Args:
            cat_one: First tensor.
            cat_two: Second tensor (must have the same number of rows as cat_one).

        Returns:
            TensorType[float]: Concatenated tensor, rounded to 4 decimal places.
        """
        # Join two tensors side-by-side along dim=1
        # Use torch.cat((a, b), dim=1)
        return torch.round(torch.cat((cat_one, cat_two), dim=1), decimals = 4)

    def get_loss(self, prediction: TensorType[float], target: TensorType[float]) -> TensorType[float]:
        """Compute Mean Squared Error between prediction and target tensors.

        Args:
            prediction: Predicted values tensor.
            target: Ground truth values tensor (same shape as prediction).

        Returns:
            TensorType[float]: Scalar MSE loss, rounded to 4 decimal places.
        """
        # Compute Mean Squared Error between prediction and target
        # Use torch.nn.functional.mse_loss(prediction, target)
        return torch.round(torch.nn.functional.mse_loss(prediction, target), decimals = 4)
