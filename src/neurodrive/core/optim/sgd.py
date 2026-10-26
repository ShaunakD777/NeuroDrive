"""Stochastic gradient descent (SGD): the simplest optimizer.

For every parameter θ with gradient g = ∂L/∂θ, and a learning rate η (a small positive number
chosen by the user, the length of each stride downhill):

    θ_new = θ − η · g

A positive gradient means the loss rises when θ rises, so θ is moved down; a negative gradient
moves it up. "Stochastic" refers to where g comes from: a random minibatch rather than the whole
dataset, so g is a noisy estimate of the true downhill direction. The rule itself is the same.

Worked example: L(w) = (w − 3)², so g = 2(w − 3). From w = 0 with η = 0.1: g = −6 and
w_new = 0 − 0.1 · (−6) = 0.6.
"""

from neurodrive.core.optim.base import Array, check_learning_rate, check_pairs


class SGD:
    """Plain gradient descent step with learning rate ``learning_rate`` (η). Keeps no memory."""

    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = check_learning_rate(learning_rate)

    def step(self, params: list[Array], grads: list[Array]) -> list[Array]:
        """Return [θ − η·g for every pair]; the inputs are left untouched."""
        check_pairs(params, grads)
        return [p - self.learning_rate * g for p, g in zip(params, grads, strict=True)]
