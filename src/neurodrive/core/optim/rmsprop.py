"""RMSProp: gradient descent with a separate, self-adjusting step size for every parameter.

SGD uses one learning rate for all parameters, which is too large for those with big gradients
and too small for those with tiny ones. RMSProp (Root Mean Square Propagation) divides each
gradient by the typical size of that parameter's recent gradients, so all parameters move at a
similar pace.

For every parameter θ with gradient g, learning rate η, decay ρ (the fraction of the old average
that is kept, between 0 and 1), a tiny ε that prevents division by zero, and s the running
average of g² (starting at 0):

    s_new = ρ · s + (1 − ρ) · g²
    θ_new = θ − η · g / (√s_new + ε)

The running average is an *exponential moving average*: recent gradients count most, older ones
fade by a factor ρ per step. √s is the root mean square of recent gradients, their typical size
with the sign ignored. ε is added after the square root (PyTorch's convention).

Worked example: η = 0.1, ρ = 0.9, first step with gradients g = [10, 0.1]: s = [10, 0.001],
√s = [3.1623, 0.031623], step = 0.1 · [3.1623, 3.1623] = [0.3162, 0.3162]. A gradient 100 times
larger gives the same step. On a first step the size is always η / √(1 − ρ).
"""

import numpy as np

from neurodrive.core.optim.base import Array, check_learning_rate, check_pairs


class RMSProp:
    """RMSProp. Keeps one running average of squared gradients per parameter between steps."""

    def __init__(self, learning_rate: float, rho: float = 0.9, eps: float = 1e-8) -> None:
        self.learning_rate = check_learning_rate(learning_rate)
        rho, eps = float(rho), float(eps)
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {rho}")
        if not np.isfinite(eps) or eps <= 0.0:
            raise ValueError(f"eps must be positive and finite, got {eps}")
        self.rho, self.eps = rho, eps
        self.square_averages: list[Array] | None = None

    def reset(self) -> None:
        """Forget the running averages, so the next step behaves like a first step."""
        self.square_averages = None

    def step(self, params: list[Array], grads: list[Array]) -> list[Array]:
        """Update the averages s, then return [θ − η·g / (√s + ε)] (inputs untouched)."""
        check_pairs(params, grads)
        if self.square_averages is None:
            self.square_averages = [np.zeros(np.shape(p)) for p in params]
        elif len(self.square_averages) != len(params) or any(
            s.shape != np.shape(p) for s, p in zip(self.square_averages, params, strict=True)
        ):
            raise ValueError("parameter shapes changed since the last step; call reset() first")
        rho = self.rho
        self.square_averages = [
            rho * s + (1.0 - rho) * np.square(g)
            for s, g in zip(self.square_averages, grads, strict=True)
        ]
        return [
            p - self.learning_rate * g / (np.sqrt(s) + self.eps)
            for p, g, s in zip(params, grads, self.square_averages, strict=True)
        ]
