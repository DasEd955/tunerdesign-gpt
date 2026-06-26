"""weight_init.py - Xavier and Kaiming weight initialization strategies.

Implements the two dominant weight initialization methods for deep networks plus
a diagnostic that measures how activation variance propagates layer by layer,
making it easy to see which init keeps variance stable versus exploding/vanishing.
"""

import torch
import torch.nn as nn
import math
from typing import List


class Solution:

    def xavier_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        """Generate a weight matrix using Xavier (Glorot) normal initialization.

        Computes std = sqrt(2 / (fan_in + fan_out)) and samples weights from
        N(0, std^2). Goal: keep variance of activations stable across layers;
        works best for tanh/sigmoid activations but is often used generally.

        Args:
            fan_in: Number of input features (columns) for this layer.
            fan_out: Number of output features (rows) for this layer.

        Returns:
            List[List[float]]: Weight matrix of shape (fan_out, fan_in),
            rounded to 4 decimal places, as a nested list.
        """
        # Return a (fan_out x fan_in) weight matrix using Xavier/Glorot normal initialization
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        # Goal: Keep variance of activations stable across layers; works best for tanh/sigmoid, but often used generally
        
        # Set seed so randomness is reproducible (same weights every run)
        torch.manual_seed(0)
        # Compute standard deviation based on layer sizes
        # Xavier tries to balance input + output variance
        std = math.sqrt(2.0 / (fan_in + fan_out))
        # Sample weights from normal distribution N(0, std^2)
        # Shape: (fan_out, fan_in) since each neuron has fan_in inputs 
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals = 4).tolist()
        

    def kaiming_init(self, fan_in: int, fan_out: int) -> List[List[float]]:
        """Generate a weight matrix using Kaiming (He) normal initialization.

        Computes std = sqrt(2 / fan_in) and samples weights from N(0, std^2).
        Designed specifically for ReLU networks; compensates for ReLU dropping
        roughly 50% of activations, keeping variance stable through depth.

        Args:
            fan_in: Number of input features (columns) for this layer.
            fan_out: Number of output features (rows) for this layer.

        Returns:
            List[List[float]]: Weight matrix of shape (fan_out, fan_in),
            rounded to 4 decimal places, as a nested list.
        """
        # Return a (fan_out x fan_in) weight matrix using Kaiming/He normal initialization (for ReLU)
        # Use torch.manual_seed(0) for reproducibility
        # Round to 4 decimal places and return as nested list
        # Designed specifically for ReLU networks; compensates for ReLU "dropping" ~50% of activations
        
        # Set seed so randomness is reproducible (same weights every run)
        torch.manual_seed(0)
        # Only depends on fan_in since ReLU reduces variance
        std = math.sqrt(2.0 / fan_in)
        # Same Idea: Gaussian weights scaled by std
        weights = torch.randn(fan_out, fan_in) * std
        return torch.round(weights, decimals = 4).tolist()

    def check_activations(self, num_layers: int, input_dim: int, hidden_dim: int, init_type: str) -> List[float]:
        """Simulate a deep network forward pass and report per-layer activation std.

        Builds num_layers weight matrices using the specified init_type ('xavier',
        'kaiming', or 'random'), passes a random input through each linear + ReLU
        layer, and records the standard deviation of activations at each layer.

        This simulates a deep neural network forward pass and measures how activation
        variance changes layer by layer, revealing whether the initialization keeps
        signal strength stable, vanishing, or exploding.

        Args:
            num_layers: Number of layers to simulate.
            input_dim: Dimensionality of the input to the first layer.
            hidden_dim: Dimensionality of all subsequent layers.
            init_type: One of 'xavier', 'kaiming', or 'random' (unscaled baseline).

        Returns:
            List[float]: Standard deviation of activations after each layer's ReLU,
            rounded to 2 decimal places.
        """
        # Forward random input through num_layers with the given init_type.
        # Use torch.manual_seed(0) once at the start.
        # Return the std of activations after each layer, rounded to 2 decimals.
        # This function simulates a deep neural network forward pass & measures how activation variance changes layer by layer

        # Set seed ONCE so input + weights are reproducible
        torch.manual_seed(0)

        # Build layer sizes:
            # First layer: input_dim -> hidden_dim
            # Remaining layers: hidden_dim -> hidden_dim
        dims = [input_dim] + [hidden_dim] * num_layers
        # Store weight matrices for each layer
        weights = list()

        # Create weight matrix for each layer 
        for layer in range(num_layers):
            # Choose init strategy
            if init_type == "xavier":
                std = math.sqrt(2.0 / (dims[layer] + dims[layer + 1]))
            elif init_type == "kaiming":
                std = math.sqrt(2.0 / dims[layer])
            elif init_type == "random":
                # Baseline: no scaling (bad init case)
                std = 1.0
            # Sameple weight matrix for this layer
            # Shape: (output_size, input_size)
            w = torch.randn(dims[layer + 1], dims[layer]) * std
            weights.append(w) 
        
        # Start with a random input vector (batch size = 1)
        x = torch.randn(1, input_dim)
        stds = list()

        # Forward pass through each layer
        for w in weights:
            # Linear transformation: xW^T
            # This mixes inputs using learned weights
            x = x @ w.T
            # Non-linearity: ReLU
            # Removes negative values, introduces sparsity
            x = torch.relu(x)
            # Measure how spread out activations are at this layer
            stds.append(round(x.std().item(), 2))

        # Return list of activation standard deviations
        return stds