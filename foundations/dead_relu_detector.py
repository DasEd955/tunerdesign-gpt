"""dead_relu_detector.py - Dead neuron detection and diagnosis for ReLU networks.

A neuron is "dead" if it outputs 0 for every sample in a batch after a ReLU.
This module detects dead fractions per ReLU layer and maps those fractions to a
recommended fix (leaky ReLU, reinitialization, lower learning rate, or healthy).
"""

import torch
import torch.nn as nn
from typing import List


class Solution:

    def detect_dead_neurons(self, model: nn.Module, x: torch.Tensor) -> List[float]:
        """Measure the fraction of dead neurons after each ReLU layer.

        Runs a forward pass through the model layer by layer (torch.no_grad()).
        After each nn.ReLU, computes the fraction of neurons that output 0 for
        ALL samples in the batch.

        Args:
            model: Neural network whose ReLU activations will be inspected.
            x: Input batch tensor to pass through the model.

        Returns:
            List[float]: One dead fraction per ReLU layer, rounded to 4 decimal places.
        """
        # Forward pass through the model.
        # After each ReLU layer, compute the fraction of neurons that are dead.
        # A neuron is dead if it outputs 0 for ALL samples in the batch.
        # Return a list of dead fractions (one per ReLU layer), rounded to 4 decimals.
        dead_fractions = list()
        with torch.no_grad():
            for module in model.children():
                x = module(x)
                if isinstance(module, nn.ReLU):
                    dead = (x == 0).all(dim=0).float().mean().item()
                    dead_fractions.append(round(dead, 4))
        return dead_fractions

    def suggest_fix(self, dead_fractions: List[float]) -> str:
        """Recommend a fix based on the per-layer dead neuron fractions.

        Checks in priority order:
            1. 'use_leaky_relu' if any layer has dead fraction > 0.5
            2. 'reinitialize' if the first layer has dead fraction > 0.3
            3. 'reduce_learning_rate' if dead fraction strictly increases with depth
               AND the last layer's fraction > 0.1
            4. 'healthy' if max dead fraction < 0.1
            5. 'healthy' otherwise

        Args:
            dead_fractions: List of per-ReLU-layer dead neuron fractions.

        Returns:
            str: One of 'use_leaky_relu', 'reinitialize', 'reduce_learning_rate',
            or 'healthy'.
        """
        if len(dead_fractions) == 0:
            return "healthy"
        
        if max(dead_fractions) > 0.5:
            return "use_leaky_relu"
        
        if dead_fractions[0] > 0.3:
            return "reinitialize"
        
        if len(dead_fractions) >= 2:
            increasing = all(
                dead_fractions[i] < dead_fractions[i + 1]
                for i in range(len(dead_fractions) - 1)
            )
            if increasing and dead_fractions[-1] > 0.1:
                return "reduce_learning_rate"

        if max(dead_fractions) < 0.1:
            return "healthy"
        
        return "healthy"
