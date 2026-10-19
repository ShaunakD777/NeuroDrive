"""A population: N different networks of the same shape, evaluated together.

Driving many cars at once (a generation of an evolution run, or 20 random brains side by side)
means giving car ``i`` its *own* network ``i``. Looping over the networks is slow, so the weights
are *stacked*: layer ``l`` keeps one array ``W_l`` of shape (N, n_out, n_in) and one array
``b_l`` of shape (N, n_out), where index ``i`` is network ``i``. One batched matrix product then
computes all N networks at once. Each car's result is exactly what its network alone would give:

    z_l[i] = W_l[i] a_{l-1}[i] + b_l[i],   a_l[i] = f_l(z_l[i]).
"""

from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.activations import get_activation
from neurodrive.core.nn.forward import ForwardTrace
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec


class Population:
    """N networks sharing one :class:`NetworkSpec`, with stacked weights.

    ``weights[l]`` has shape (N, n_out, n_in) and ``biases[l]`` shape (N, n_out) for layer
    ``l``. Prefer :meth:`from_networks` or :meth:`from_genomes` to build one. The constructor
    copies the arrays and checks every shape against ``spec``.
    """

    def __init__(
        self,
        spec: NetworkSpec,
        weights: Sequence[ArrayLike],
        biases: Sequence[ArrayLike],
    ) -> None:
        if len(weights) != spec.n_layers or len(biases) != spec.n_layers:
            raise ValueError(f"spec has {spec.n_layers} layers; got {len(weights)}/{len(biases)}")
        self.spec = spec
        self.weights: list[NDArray[np.float64]] = [np.array(w, dtype=np.float64) for w in weights]
        self.biases: list[NDArray[np.float64]] = [np.array(b, dtype=np.float64) for b in biases]
        n = self.weights[0].shape[0] if self.weights[0].ndim == 3 else 0
        if n < 1:
            raise ValueError("a population needs at least one network")
        for layer in range(spec.n_layers):
            want_w = (n, *spec.weight_shape(layer))
            want_b = (n, *spec.bias_shape(layer))
            if self.weights[layer].shape != want_w:
                raise ValueError(f"W{layer} has shape {self.weights[layer].shape}, want {want_w}")
            if self.biases[layer].shape != want_b:
                raise ValueError(f"b{layer} has shape {self.biases[layer].shape}, want {want_b}")
        self.size = n

    @classmethod
    def from_networks(cls, networks: Sequence[Network]) -> "Population":
        """Stack networks that all have the same spec; network ``i`` becomes member ``i``."""
        if len(networks) == 0:
            raise ValueError("a population needs at least one network")
        spec = networks[0].spec
        if any(net.spec != spec for net in networks):
            raise ValueError("all networks in a population must share one spec")
        return cls(
            spec,
            [np.stack([net.weights[layer] for net in networks]) for layer in range(spec.n_layers)],
            [np.stack([net.biases[layer] for net in networks]) for layer in range(spec.n_layers)],
        )

    @classmethod
    def from_genomes(cls, spec: NetworkSpec, genomes: ArrayLike) -> "Population":
        """Build from a matrix of shape (N, n_parameters), one flat genome per row.

        Each row is read in the layout of :meth:`Network.flatten`: per layer, W row by row, then b.
        """
        g = np.asarray(genomes, dtype=np.float64)
        if g.ndim != 2 or g.shape[1] != spec.n_parameters:
            raise ValueError(f"genomes must have shape (N, {spec.n_parameters}), got {g.shape}")
        n = g.shape[0]
        weights, biases, pos = [], [], 0
        for layer in range(spec.n_layers):
            n_out, n_in = spec.weight_shape(layer)
            weights.append(g[:, pos : pos + n_out * n_in].reshape(n, n_out, n_in))
            pos += n_out * n_in
            biases.append(g[:, pos : pos + n_out])
            pos += n_out
        return cls(spec, weights, biases)

    def network(self, index: int) -> Network:
        """Member ``index`` as a stand-alone :class:`Network` (a copy)."""
        if not 0 <= index < self.size:
            raise IndexError(f"member {index} out of range for a population of {self.size}")
        return Network(
            self.spec,
            [w[index] for w in self.weights],
            [b[index] for b in self.biases],
        )

    def genomes(self) -> NDArray[np.float64]:
        """All genomes as a matrix of shape (N, n_parameters); row ``i`` is member ``i``."""
        parts = []
        for w, b in zip(self.weights, self.biases, strict=True):
            parts.append(w.reshape(self.size, -1))
            parts.append(b)
        return np.concatenate(parts, axis=1)

    def forward(self, observations: ArrayLike) -> ForwardTrace:
        """Give row ``i`` of ``observations`` to member ``i`` and return the full trace.

        ``observations`` has shape (N, n_inputs). Every trace entry has a leading dimension N:
        ``activations[l]`` is (N, width_l) and ``weighted_sums[l - 1]`` is (N, width_l).
        """
        spec = self.spec
        a = np.asarray(observations, dtype=np.float64)
        want = (self.size, spec.n_inputs)
        if a.shape != want:
            raise ValueError(f"expected observations of shape {want}, got {a.shape}")
        activations = [a]
        weighted_sums = []
        for layer in range(spec.n_layers):
            # (N, n_out, n_in) @ (N, n_in, 1) -> (N, n_out, 1): one matrix-vector product per car
            z = (self.weights[layer] @ a[:, :, None])[:, :, 0] + self.biases[layer]
            a = get_activation(spec.activation_name(layer))(z)
            weighted_sums.append(z)
            activations.append(a)
        return ForwardTrace(activations, weighted_sums)

    def predict(self, observations: ArrayLike) -> NDArray[np.float64]:
        """The outputs only, shape (N, n_outputs)."""
        return self.forward(observations).output
