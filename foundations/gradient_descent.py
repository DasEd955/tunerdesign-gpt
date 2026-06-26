"""gradient_descent.py - Gradient descent minimization of f(x) = x^2.

Demonstrates the core gradient descent update rule on a simple scalar
objective to build intuition for how learning rate and iteration count
control convergence toward the minimum.
"""


class Solution:
    def get_minimizer(self, iterations: int, learning_rate: float, init: int) -> float:
        """Minimize f(x) = x^2 using gradient descent from an initial value.

        Applies the update x = x - learning_rate * f'(x) where f'(x) = 2x,
        iterating for the specified number of steps.

        Args:
            iterations: Number of gradient descent steps to take.
            learning_rate: Step size multiplied by the gradient each iteration.
            init: Starting value of x.

        Returns:
            float: Final value of x after all iterations, rounded to 5 decimal places.
        """
        # Objective function: f(x) = x^2
        # Derivative:         f'(x) = 2x
        # Update rule:        x = x - learning_rate * f'(x)
        # Round final answer to 5 decimal places
        x_new = init
        for iteration in range(iterations):
            x_new -= learning_rate * (2 * x_new)
        return round(x_new, 5)
