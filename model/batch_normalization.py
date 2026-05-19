import numpy as np
from typing import Tuple, List


class Solution:
    def batch_norm(self, x: List[List[float]], gamma: List[float], beta: List[float],
                   running_mean: List[float], running_var: List[float],
                   momentum: float, eps: float, training: bool) -> Tuple[List[List[float]], List[float], List[float]]:
        # During training: normalize using batch statistics, then update running stats
        # During inference: normalize using running stats (no batch stats needed)
        # Apply affine transform: y = gamma * x_hat + beta
        # Return (y, running_mean, running_var), all rounded to 4 decimals as lists
        
        # Convert x, gamma, beta, running_mean, running_var into NumPy arrays
        x = np.array(x)
        gamma = np.array(gamma)
        beta = np.array(beta)
        running_mean = np.array(running_mean, dtype=np.float64)
        running_var = np.array(running_var, dtype=np.float64)

        # Training Mode:
            # Compute batch_mean, batch_var, x_hat
            # Update thr unning statistics for later use in inference (running_mean & running_var) 
        if training:
            batch_mean = np.mean(x, axis=0)
            batch_var = np.var(x, axis=0)
            x_hat = (x - batch_mean) / np.sqrt(batch_var + eps)
            running_mean = (1 - momentum) * running_mean + (momentum * batch_mean)
            running_var = (1 - momentum) * running_var + (momentum * batch_var)
        # Inference Mode:
            # Use the running statistics instead of batch statistics
        else:
            x_hat = (x - running_mean) / np.sqrt(running_var + eps)
        
        # Compute y_out
        y_out = gamma * x_hat + beta

        # Output: A tuple of 3 values
            # 1. The normalized output as a 2D list (y_out)
            # 2. The updated running_mean as a 1D list
            # 3. The updated running_var as a 1D list
        return (np.round(y_out, 4), np.round(running_mean, 4), np.round(running_var, 4))