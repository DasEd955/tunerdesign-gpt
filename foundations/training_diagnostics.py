"""training_diagnostics.py - Activation and gradient statistics for neural network health checks.

Provides tools for diagnosing common training pathologies by inspecting what
actually flows through the network during the forward and backward passes.
Detects dead neurons, exploding/vanishing gradients, and activation collapse
without requiring the user to instrument the model manually.
"""

import torch
import torch.nn as nn
from typing import List, Dict


class Solution:

    def compute_activation_stats(self, model: nn.Module, x: torch.Tensor) -> List[Dict[str, float]]:
        """Collect activation statistics after each Linear layer in the model.

        Runs a forward pass layer by layer (torch.no_grad()) and records mean,
        std, and dead_fraction after each nn.Linear. Checks the outputs of each
        Linear layer during the forward pass.

        A neuron is counted as dead if its activation is <= 0 for all samples
        in the batch (checking across dim=0 for 2D activations).

        Args:
            model: Neural network to inspect.
            x: Input batch tensor.

        Returns:
            List[Dict[str, float]]: One dict per nn.Linear layer with keys
            'mean', 'std', and 'dead_fraction', all rounded to 4 decimal places.
        """
        # Forward pass through model layer by layer
        # After each nn.Linear, record: mean, std, dead_fraction
        # Run with torch.no_grad(). Round to 4 decimals.
        # Key Idea: Checks the outputs of each Linear layer during the forward pass

        # Create an empty list to store states for every nn.Linear layer
        stats = list()

        # with torch.no_grad -> Clears previously accumulated gradients (since only inspecting, not training)
            # Iterate layer-by-layer through the model
            # Feed activations through the current layer (x = module(x))
        with torch.no_grad():
            for module in model.children():
                x = module(x)

                # Only collect stats for Linear layers
                    # Compute the mean & standard deviation
                if isinstance(module, nn.Linear):
                    mean = round(x.mean().item(), 4)
                    std = round(x.std().item(), 4)

                    # Dead Neurons:
                        # Checks if activations have shape [batch_size, features] (2D)
                            # (x <= 0).all(dim=0) -> Creates a boolean tensor & check across the batch dimension
                            # .float().mean().item() -> converts [True, False] to [1.0, 0.0], then averages 0.5
                    if x.dim() >= 2:
                        dead_frac = round(((x <= 0).all(dim= 0)).float().mean().item(), 4)
                    else:
                        dead_frac = round((x <= 0).float().mean().item(), 4)
                    
                    # Store the layer statistics
                    stats.append({"mean" : mean, "std" : std, "dead_fraction" : dead_frac})
        
        # Return activation statistics map for all nn.Linear layers
        return stats 

    def compute_gradient_stats(self, model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> List[Dict[str, float]]:
        """Collect weight gradient statistics for each Linear layer after a backward pass.

        Zeroes gradients, runs a forward pass with nn.MSELoss, calls loss.backward(),
        and then records mean, std, and L2 norm of the weight gradient for each
        nn.Linear. Checks whether gradients are too small, too large, or healthy
        during backpropagation.

        Args:
            model: Neural network to inspect.
            x: Input batch tensor.
            y: Target tensor used with nn.MSELoss.

        Returns:
            List[Dict[str, float]]: One dict per nn.Linear layer with keys
            'mean', 'std', and 'norm', all rounded to 4 decimal places.
        """
        # Forward + backward pass with nn.MSELoss
        # For each nn.Linear layer's weight gradient, record: mean, std, norm
        # Call model.zero_grad() first. Round to 4 decimals.
        # Key Idea: Checks whether gradients are too small, too large, or healthy during backpropagation

        # model.zero_grad() -> Disable PyTorch gradient accumulation
        # output = model(x) -> Normal forward pass
        # loss = nn.MSELoss()(output, y) -> Computes Mean Squared Error
        # loss.backward() -> Runs backpropagation
        model.zero_grad()
        output = model(x)
        loss = nn.MSELoss()(output, y)
        loss.backward()
        stats = list()

        # Iterate through layers in the model
            # Only compute grad & statistics if layer is nn.Linear
                # grad = module.weight.grad -> Gets gradients of the weight matrix
                # grad.mean() -> Average gradient value
                # grad.std() -> Measures spread of gradients
                # torch.norm(grad) -> Computes L2 norm (measures overall gradient magnitude)
            # Store the layer statistics 
        for module in model.children():
            if isinstance(module, nn.Linear):
                grad = module.weight.grad
                mean = round(grad.mean().item(), 4)
                std = round(grad.std().item(), 4)
                norm = round(torch.norm(grad).item(), 4)
                stats.append({"mean" : mean, "std" : std, "norm" : norm})

        # Return gradient statistics map for all nn.Linear layers
        return stats

    def diagnose(self, activation_stats: List[Dict[str, float]], gradient_stats: List[Dict[str, float]]) -> str:
        """Classify network training health from activation and gradient statistics.

        Converts stats into a label. Checks in priority order:
            1. 'dead_neurons' if any layer's dead_fraction > 0.5
            2. 'exploding_gradients' if any gradient norm > 1000
            3. 'vanishing_gradients' if the last layer's gradient norm < 1e-5,
               or if any activation std < 0.1
            4. 'exploding_gradients' if any activation std > 10.0
            5. 'healthy' otherwise

        Args:
            activation_stats: Output of compute_activation_stats().
            gradient_stats: Output of compute_gradient_stats().

        Returns:
            str: One of 'dead_neurons', 'exploding_gradients',
            'vanishing_gradients', or 'healthy'.
        """
        # Classify network health based on the stats
        # Return: 'dead_neurons', 'exploding_gradients', 'vanishing_gradients', or 'healthy'
        # Check in priority order (see problem description for thresholds)
        # Key Idea: Converts stats into a label
        
        # Dead Neurons Check:
            # If over half the neurons never activate -> return "dead_neurons"
        for s in activation_stats:
            if s["dead_fraction"] > 0.5:
                return "dead_neurons"

        # Exploding Gradient Check:
            # Huge gradients -> updates become unstable & weights jump wildly
        for s in gradient_stats:
            if s["norm"] > 1000:
                return "exploding_gradients"
        
        # Vanishing Gradient Check:
            # Looks at the final layer
            # If even the last layer gets almost no gradient -> learning is basically dead 
        if gradient_stats and gradient_stats[-1]["norm"] < 1e-5:
            return "vanishing_gradients"

        # Activation Std Checks:
            # if s["std"] < 0.1 -> Activations collapsed; network outputs nearly constant values (equivalent to weak signal flow)
            # if s["std"] > 10.0 -> Exploding activations; each layer magnifies values too much -> often leads to NaNs later
        for s in activation_stats:
            if s["std"] < 0.1:
                return "vanishing_gradients"
            if s["std"] > 10.0:
                return "exploding_gradients"

        return "healthy"
