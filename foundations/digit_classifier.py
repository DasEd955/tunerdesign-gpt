"""digit_classifier.py - MNIST-style digit classification network.

A two layer MLP for classifying 28x28 grayscale images (flattened to 784 features)
into 10 digit classes. Each of the 10 output units produces an independent sigmoid
probability rather than a softmax, so thresholding any output gives a binary
confidence estimate per digit.
"""

import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):
    """Two layer MLP digit classifier.

    Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid

    Input(784): raw pixels arrive as a flat 784-element vector (28x28).
    Linear(512): hidden layer projects to 512 features.
    ReLU: introduces nonlinearity.
    Dropout(0.2): regularizes to reduce overfitting.
    Linear(10) + Sigmoid: final 10 sigmoid outputs represent confidence that the
        image shows each digit (0-9).
    """

    def __init__(self):
        """Initialize all layers with torch.manual_seed(0) for reproducibility."""
        super().__init__()
        torch.manual_seed(0)
        # Architecture: Linear(784, 512) -> ReLU -> Dropout(0.2) -> Linear(512, 10) -> Sigmoid
            # Input(784) -> The raw pixels arrive as a flat 784-element vector (28x28)
            # Linear(512) -> The hidden layer projects to 512 features
            # ReLU -> Introduces nonlinearity
            # Dropout(0.2) -> Dropout (20%) regularizes
            # Linear(10) & Sigmoid() -> The final 10 sigmoid outputs represent the confidence that the image shows that digit (0-9)
        self.first_linear = nn.Linear(784, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.projection = nn.Linear(512, 10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        """Run the forward pass and return per-digit confidence scores.

        Args:
            images: Input tensor of shape (batch_size, 784).

        Returns:
            TensorType[float]: Sigmoid outputs of shape (batch_size, 10),
            rounded to 4 decimal places.
        """
        torch.manual_seed(0)
        # images shape: (batch_size, 784)
        # Return the model's prediction to 4 decimal places
        x = self.first_linear(images)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.projection(x)
        x = self.sigmoid(x)
        return torch.round(x, decimals=4)
