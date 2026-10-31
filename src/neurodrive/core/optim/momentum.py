"""SGD with momentum: gradient descent that remembers which way it has been going.

Plain SGD re-measures the slope at every step. Momentum behaves like a heavy ball rolling
downhill: steps in a consistent direction add up and speed it up, while gradients that keep
flipping sign (a zig-zag across a narrow valley) partly cancel.

For every parameter θ with gradient g, learning rate η, momentum coefficient β (the fraction of
the old velocity that is kept, between 0 and 1) and velocity v (starting at 0):

    v_new = β · v + g
    θ_new = θ − η · v_new

This is the convention PyTorch uses for SGD with momentum. With β = 0 it is exactly plain SGD. If
the gradient stays the same g, v approaches g / (1 − β), so with β = 0.9 steps grow up to 10-fold.

Worked example: L(w) = (w − 3)², η = 0.1, β = 0.9, w = 0. Step 1: g = −6, v = −6, w = 0.6.
Step 2: g = −4.8, v = 0.9 · (−6) − 4.8 = −10.2, w = 0.6 + 1.02 = 1.62.
"""

import numpy as np

from neurodrive.core.optim.base import Array, check_learning_rate, check_pairs


class Momentum:
    """SGD with momentum. Keeps one velocity array per parameter between calls to ``step``."""

    def __init__(self, learning_rate: float, beta: float = 0.9) -> None:
        self.learning_rate = check_learning_rate(learning_rate)
        beta = float(beta)
        if not 0.0 <= beta < 1.0:
            raise ValueError(f"beta must be in [0, 1), got {beta}")
        self.beta = beta
        self.velocities: list[Array] | None = None

    def reset(self) -> None:
        """Forget the velocities, so the next step behaves like a first step."""
        self.velocities = None

    def step(self, params: list[Array], grads: list[Array]) -> list[Array]:
        """Update the velocities, then return [θ − η·v for every parameter] (inputs untouched)."""
        check_pairs(params, grads)
        if self.velocities is None:
            self.velocities = [np.zeros(np.shape(p)) for p in params]
        elif len(self.velocities) != len(params) or any(
            v.shape != np.shape(p) for v, p in zip(self.velocities, params, strict=True)
        ):
            raise ValueError("parameter shapes changed since the last step; call reset() first")
        self.velocities = [self.beta * v + g for v, g in zip(self.velocities, grads, strict=True)]
        return [p - self.learning_rate * v for p, v in zip(params, self.velocities, strict=True)]
