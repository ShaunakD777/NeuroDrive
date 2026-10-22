"""A network's parameters: one weight matrix and one bias vector per layer.

``Network`` holds the numbers; the shape comes from its :class:`NetworkSpec`. Besides the
matrices it can be seen as one long list of numbers, the *genome* (flat vector), laid out as in
docs/ARCHITECTURE.md §6.8: ``W_0`` row by row, then ``b_0``, then ``W_1``, ``b_1`` and so on.
Learning methods that mutate or nudge "all the numbers" work on that vector; the forward pass
works on the matrices.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.nn.spec import NetworkSpec


class Network:
    """Weights ``W_l`` (shape (n_out, n_in)) and biases ``b_l`` (shape (n_out,)) for every layer.

    The constructor copies the arrays into float64 and checks every shape against ``spec``, so
    a ``Network`` is always consistent with its spec. ``weights`` and ``biases`` are lists with
    one entry per layer, in order.
    """

    def __init__(
        self,
        spec: NetworkSpec,
        weights: list[ArrayLike],
        biases: list[ArrayLike],
    ) -> None:
        if len(weights) != spec.n_layers or len(biases) != spec.n_layers:
            raise ValueError(
                f"spec has {spec.n_layers} layers but got {len(weights)} weight matrices "
                f"and {len(biases)} bias vectors"
            )
        self.spec = spec
        self.weights: list[NDArray[np.float64]] = []
        self.biases: list[NDArray[np.float64]] = []
        for layer, (w, b) in enumerate(zip(weights, biases, strict=True)):
            w = np.array(w, dtype=np.float64)
            b = np.array(b, dtype=np.float64)
            if w.shape != spec.weight_shape(layer):
                raise ValueError(
                    f"W{layer} has shape {w.shape}, expected {spec.weight_shape(layer)}"
                )
            if b.shape != spec.bias_shape(layer):
                raise ValueError(f"b{layer} has shape {b.shape}, expected {spec.bias_shape(layer)}")
            self.weights.append(w)
            self.biases.append(b)

    @classmethod
    def zeros(cls, spec: NetworkSpec) -> "Network":
        """A network whose every weight and bias is 0 (random starts come in task 2.4)."""
        return cls(
            spec,
            [np.zeros(spec.weight_shape(i)) for i in range(spec.n_layers)],
            [np.zeros(spec.bias_shape(i)) for i in range(spec.n_layers)],
        )

    @property
    def n_parameters(self) -> int:
        """Total count of weights and biases (equals ``spec.n_parameters``)."""
        return self.spec.n_parameters

    def clone(self) -> "Network":
        """An independent copy: changing the clone's numbers never changes this network."""
        return Network(self.spec, self.weights, self.biases)

    def flatten(self) -> NDArray[np.float64]:
        """The genome: all parameters as one 1-D vector of length ``n_parameters``.

        Order: for each layer, the weight matrix row by row (all of neuron 0's weights, then
        neuron 1's, ...), then that layer's biases. The result is a copy.
        """
        parts: list[NDArray[np.float64]] = []
        for w, b in zip(self.weights, self.biases, strict=True):
            parts.append(w.ravel())  # row-major: row by row
            parts.append(b)
        return np.concatenate(parts)

    @classmethod
    def unflatten(cls, spec: NetworkSpec, vector: ArrayLike) -> "Network":
        """Rebuild a network from a genome made by :meth:`flatten` (the exact inverse).

        ``vector`` must be 1-D with exactly ``spec.n_parameters`` numbers. For each layer the
        next n_out * n_in numbers become W (row by row), then the next n_out numbers become b.
        The network gets its own copy of the numbers.
        """
        v = np.asarray(vector, dtype=np.float64)
        if v.shape != (spec.n_parameters,):
            raise ValueError(f"genome must have shape ({spec.n_parameters},), got {v.shape}")
        weights: list[NDArray[np.float64]] = []
        biases: list[NDArray[np.float64]] = []
        pos = 0
        for layer in range(spec.n_layers):
            n_out, n_in = spec.weight_shape(layer)
            weights.append(v[pos : pos + n_out * n_in].reshape(n_out, n_in))
            pos += n_out * n_in
            biases.append(v[pos : pos + n_out])
            pos += n_out
        return cls(spec, weights, biases)  # the constructor copies

    def equals(self, other: "Network") -> bool:
        """True when the specs are equal and every number matches exactly (bit for bit)."""
        return self.spec == other.spec and np.array_equal(self.flatten(), other.flatten())
