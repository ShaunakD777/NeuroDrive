"""Initialisation: choosing a network's starting numbers.

Learning has to start somewhere. Two things matter about the start:

* **Symmetry.** If all the neurons of a layer start with identical weights they compute identical
  things, and anything that treats them alike (evolution's mutations apart, every gradient step
  in Phase 3) keeps them identical. Random starting weights break the tie.
* **Scale.** With weights too large, the weighted sums are huge and tanh saturates at -1 or +1
  whatever the input; with weights too small, the signal shrinks layer after layer.

Schemes (biases always start at 0). For a layer with ``n_in`` inputs and ``n_out`` neurons:

* ``"zeros"``  : every weight is 0. Useful as a baseline and to demonstrate symmetry.
* ``"uniform"``: every weight is drawn evenly from [-s, s] with the same ``s`` (``scale``) in
  every layer. Its variance is s^2 / 3.
* ``"xavier"`` : every weight is drawn evenly from [-a, a] with a = sqrt(6 / (n_in + n_out)).
  Its variance is a^2 / 3 = 2 / (n_in + n_out), so wide layers get smaller weights.

Every random number comes from the generator passed in (``neurodrive.core.rng``).
"""

import math

import numpy as np

from neurodrive.core.nn.network import Network
from neurodrive.core.nn.population import Population
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import Rng

SCHEMES = ("zeros", "uniform", "xavier")
DEFAULT_SCALE = 0.5


def weight_limit(scheme: str, n_in: int, n_out: int, scale: float = DEFAULT_SCALE) -> float:
    """The bound a of the interval [-a, a] that weights are drawn from (0 for ``"zeros"``).

    ``n_in`` and ``n_out`` are the layer's input and neuron counts; ``scale`` is s for
    ``"uniform"`` and ignored otherwise. Raises ``ValueError`` for an unknown scheme.
    """
    if scheme == "zeros":
        return 0.0
    if scheme == "uniform":
        if scale < 0:
            raise ValueError(f"scale must be non-negative, got {scale}")
        return float(scale)
    if scheme == "xavier":
        return math.sqrt(6.0 / (n_in + n_out))
    raise ValueError(f"unknown init scheme {scheme!r}; valid names: {', '.join(SCHEMES)}")


def init_network(
    spec: NetworkSpec, rng: Rng, scheme: str = "xavier", scale: float = DEFAULT_SCALE
) -> Network:
    """A network with starting weights drawn from ``rng`` by ``scheme``; all biases are 0.

    Layers are drawn in order, each weight matrix in one go, so the same generator state always
    gives the same network. ``"zeros"`` draws nothing.
    """
    weights, biases = [], []
    for layer in range(spec.n_layers):
        n_out, n_in = spec.weight_shape(layer)
        limit = weight_limit(scheme, n_in, n_out, scale)
        if limit == 0.0:
            weights.append(np.zeros((n_out, n_in)))
        else:
            weights.append(rng.uniform(-limit, limit, size=(n_out, n_in)))
        biases.append(np.zeros(n_out))
    return Network(spec, weights, biases)


def init_population(
    spec: NetworkSpec, size: int, rng: Rng, scheme: str = "xavier", scale: float = DEFAULT_SCALE
) -> Population:
    """``size`` independently initialised networks, drawn one after another from ``rng``.

    Member ``i`` equals the ``i``-th call of :func:`init_network` on the same generator.
    """
    if size < 1:
        raise ValueError(f"size must be at least 1, got {size}")
    return Population.from_networks([init_network(spec, rng, scheme, scale) for _ in range(size)])
