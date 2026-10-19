"""The forward pass: pushing an observation through the network, layer by layer.

Start with ``a_0``, the observation. Each layer ``l = 1..L`` does two steps:

1. the *weighted sums* ``z_l = W_l a_{l-1} + b_l`` (every neuron adds up its weighted inputs and
   its bias), then
2. the *activation* ``a_l = f_l(z_l)`` (every neuron squashes its sum).

The last ``a_L`` is the network's output. The :class:`ForwardTrace` keeps every ``a_l`` and
``z_l``, not just the output, because the Network view draws them, the explanations read them
and backpropagation (Phase 3) needs them (docs/ARCHITECTURE.md §6.9).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.activations import get_activation
from neurodrive.core.nn.network import Network


@dataclass(frozen=True)
class ForwardTrace:
    """Every intermediate value of one forward pass.

    ``activations[l]`` is ``a_l``: entry 0 is the observation, entry L the output, so there are
    L + 1 of them. ``weighted_sums[l - 1]`` is ``z_l``, so there are L of them. For a batch of N
    observations each entry has shape (N, layer width); for a single observation the batch
    dimension is absent and each entry has shape (layer width,).
    """

    activations: list[NDArray[np.float64]]
    weighted_sums: list[NDArray[np.float64]]

    @property
    def output(self) -> NDArray[np.float64]:
        """The network's output ``a_L``."""
        return self.activations[-1]


def forward(network: Network, observations: ArrayLike) -> ForwardTrace:
    """Run ``observations`` through ``network`` and return the full trace.

    ``observations`` has shape (n_inputs,) for one example or (N, n_inputs) for N at once. All N
    rows are computed together as ``Z = A W^T + b`` (the row-vector form of ``z = W a + b``).
    Raises ``ValueError`` if the last dimension is not ``spec.n_inputs``.
    """
    spec = network.spec
    a = np.asarray(observations, dtype=np.float64)
    if a.ndim not in (1, 2) or a.shape[-1] != spec.n_inputs:
        raise ValueError(
            f"expected observations of shape ({spec.n_inputs},) or (N, {spec.n_inputs}), "
            f"got {a.shape}"
        )
    single = a.ndim == 1
    a = a.reshape(1, -1) if single else a
    activations = [a]
    weighted_sums = []
    for layer in range(spec.n_layers):
        z = a @ network.weights[layer].T + network.biases[layer]
        a = get_activation(spec.activation_name(layer))(z)
        weighted_sums.append(z)
        activations.append(a)
    if single:
        activations = [x[0] for x in activations]
        weighted_sums = [x[0] for x in weighted_sums]
    return ForwardTrace(activations, weighted_sums)


def predict(network: Network, observations: ArrayLike) -> NDArray[np.float64]:
    """The network's output only (``forward(...).output``)."""
    return forward(network, observations).output
