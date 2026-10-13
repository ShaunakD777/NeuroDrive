"""A single artificial neuron: the Phase 1 teaching object.

A neuron has one weight ``w_i`` per input and one bias ``b``. Given inputs ``x_i`` it first forms
the weighted sum ``z = w_1*x_1 + w_2*x_2 + ... + b`` and then applies its activation function
``f`` to get the output ``a = f(z)``. Weights say how strongly (and in which direction) each input
pushes; the bias shifts where the neuron "switches on".
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.activations import get_activation


class Neuron:
    """One neuron: weights, a bias and a named activation function.

    ``weights`` is a 1-D sequence of numbers (one per input), ``bias`` a single number and
    ``activation`` one of the names in ``neurodrive.core.activations.ACTIVATIONS``.
    """

    def __init__(self, weights: ArrayLike, bias: float, activation: str = "tanh") -> None:
        w = np.array(weights, dtype=np.float64)
        if w.ndim != 1 or w.size == 0:
            raise ValueError(f"weights must be a non-empty 1-D sequence, got shape {w.shape}")
        self.weights: NDArray[np.float64] = w
        self.bias = float(bias)
        self.activation = activation
        self._f = get_activation(activation)  # fails early on an unknown name

    @property
    def n_inputs(self) -> int:
        """How many inputs the neuron reads."""
        return int(self.weights.size)

    def weighted_sum(self, inputs: ArrayLike) -> NDArray[np.float64]:
        """Compute z = sum_i(w_i * x_i) + b.

        ``inputs`` has shape (n_inputs,) for one example or (N, n_inputs) for N examples at once.
        The result has shape () or (N,) respectively.
        """
        x = np.asarray(inputs, dtype=np.float64)
        if x.ndim not in (1, 2) or x.shape[-1] != self.n_inputs:
            raise ValueError(
                f"expected inputs of shape ({self.n_inputs},) or (N, {self.n_inputs}), "
                f"got {x.shape}"
            )
        return x @ self.weights + self.bias

    def forward_with_sum(
        self, inputs: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return ``(z, a)``: the weighted sum and the activated output a = f(z).

        The Neuron view shows both, the sum before the squash and the output after it.
        """
        z = self.weighted_sum(inputs)
        return z, self._f(z)

    def forward(self, inputs: ArrayLike) -> NDArray[np.float64]:
        """Return the output a = f(z) for ``inputs`` (shape (n_inputs,) or (N, n_inputs))."""
        return self.forward_with_sum(inputs)[1]
