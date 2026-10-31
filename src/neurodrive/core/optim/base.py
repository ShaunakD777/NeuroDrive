"""The optimizer contract: how every optimizer is used, and how a network becomes a parameter list.

Backprop (``core.nn.backward``) says, for every parameter θ, how the loss L changes when θ
changes: the gradient g = ∂L/∂θ. An *optimizer* is the rule that turns those gradients into new
parameter values. Every optimizer in NeuroDrive follows the same contract:

* It is built with its settings (for example a learning rate η) and keeps any memory it needs
  between steps (SGD has none; Momentum, RMSProp and Adam do).
* ``step(params, grads)`` takes two equal-length lists of arrays, the parameters and their
  gradients (same shape pairwise), and returns a *new* list of updated arrays. It never modifies
  the arrays it is given.

A network's parameters are presented as the list [W_0, b_0, W_1, b_1, ...], the same order as
``Network.flatten()`` (docs/ARCHITECTURE.md §6.8), so optimizers never need to know about layers.
"""

from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from neurodrive.core.nn.backward import NetworkGradients
from neurodrive.core.nn.network import Network

Array = NDArray[np.float64]


class Optimizer(Protocol):
    """Anything with a ``step`` that returns updated copies of the parameters."""

    def step(self, params: list[Array], grads: list[Array]) -> list[Array]: ...


def check_pairs(params: list[Array], grads: list[Array]) -> None:
    """Raise ``ValueError`` unless the two lists have equal length and pairwise equal shapes."""
    if len(params) != len(grads):
        raise ValueError(f"{len(params)} parameter arrays but {len(grads)} gradient arrays")
    for i, (p, g) in enumerate(zip(params, grads, strict=True)):
        if np.shape(p) != np.shape(g):
            raise ValueError(
                f"parameter {i} has shape {np.shape(p)} but its gradient has shape {np.shape(g)}"
            )


def check_learning_rate(learning_rate: float) -> float:
    """Return η as a float, requiring it to be finite and strictly positive."""
    lr = float(learning_rate)
    if not np.isfinite(lr) or lr <= 0.0:
        raise ValueError(f"learning_rate must be positive and finite, got {learning_rate}")
    return lr


def params_of(network: Network) -> list[Array]:
    """The network's parameters as [W_0, b_0, W_1, b_1, ...] (the arrays themselves, not copies)."""
    out: list[Array] = []
    for w, b in zip(network.weights, network.biases, strict=True):
        out += [w, b]
    return out


def grads_of(grads: NetworkGradients) -> list[Array]:
    """The gradients of a backward pass in the same order as :func:`params_of`."""
    out: list[Array] = []
    for gw, gb in zip(grads.weights, grads.biases, strict=True):
        out += [gw, gb]
    return out


def network_from_params(network: Network, params: list[Array]) -> Network:
    """A new network with ``network``'s spec and the parameter list ``params`` (order as above)."""
    n = network.spec.n_layers
    if len(params) != 2 * n:
        raise ValueError(f"expected {2 * n} parameter arrays for {n} layers, got {len(params)}")
    return Network(network.spec, params[0::2], params[1::2])
