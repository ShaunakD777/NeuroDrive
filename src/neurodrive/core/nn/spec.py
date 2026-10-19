"""The network spec: the *shape* of a network, without any of its numbers.

A multilayer network is a stack of layers. The spec lists how many numbers flow through each
stage: ``layer_sizes = (7, 16, 16, 2)`` means 7 inputs, two hidden layers of 16 neurons and 2
outputs. Layer ``l`` (counting from 0) turns ``n_in = layer_sizes[l]`` numbers into
``n_out = layer_sizes[l + 1]`` numbers using

* a weight matrix ``W_l`` with ``n_out`` rows and ``n_in`` columns (row j holds the weights of
  neuron j), and
* a bias vector ``b_l`` with ``n_out`` entries (one per neuron).

So layer ``l`` owns ``n_out * n_in + n_out`` numbers and the whole network owns their sum. The
spec also names the activation function of the hidden layers and of the output layer, and a
format version so that saved files can be recognised later (docs/ARCHITECTURE.md §6.8).
"""

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from neurodrive.core.activations import get_activation

SPEC_VERSION = "v1"


@dataclass(frozen=True)
class NetworkSpec:
    """Layer sizes, activation names and a version. Immutable, so one spec can be shared freely.

    ``layer_sizes`` needs at least two whole numbers >= 1 (inputs, then each layer's neuron
    count, ending with the outputs). ``hidden_activation`` is used by every layer except the
    last, ``output_activation`` by the last; both are names from
    ``neurodrive.core.activations.ACTIVATIONS``. Invalid values raise ``ValueError``.
    """

    layer_sizes: tuple[int, ...] = (7, 16, 16, 2)
    hidden_activation: str = "tanh"
    output_activation: str = "tanh"
    version: str = SPEC_VERSION

    def __post_init__(self) -> None:
        sizes = tuple(self.layer_sizes)
        if len(sizes) < 2:
            raise ValueError(f"layer_sizes needs at least 2 entries, got {sizes}")
        for n in sizes:
            if isinstance(n, bool) or not isinstance(n, int) or n < 1:
                raise ValueError(f"every layer size must be a whole number >= 1, got {sizes}")
        object.__setattr__(self, "layer_sizes", sizes)  # lists become tuples; frozen otherwise
        get_activation(self.hidden_activation)  # raise early on an unknown name
        get_activation(self.output_activation)
        if self.version != SPEC_VERSION:
            raise ValueError(f"unsupported network version {self.version!r}; need {SPEC_VERSION!r}")

    @property
    def n_layers(self) -> int:
        """Number of weight layers (one fewer than the number of sizes): 3 for (7, 16, 16, 2)."""
        return len(self.layer_sizes) - 1

    @property
    def n_inputs(self) -> int:
        """How many numbers the network reads (the first size)."""
        return self.layer_sizes[0]

    @property
    def n_outputs(self) -> int:
        """How many numbers the network produces (the last size)."""
        return self.layer_sizes[-1]

    def weight_shape(self, layer: int) -> tuple[int, int]:
        """Shape (n_out, n_in) of the weight matrix of ``layer`` (0-based)."""
        self._check_layer(layer)
        return (self.layer_sizes[layer + 1], self.layer_sizes[layer])

    def bias_shape(self, layer: int) -> tuple[int]:
        """Shape (n_out,) of the bias vector of ``layer`` (0-based)."""
        self._check_layer(layer)
        return (self.layer_sizes[layer + 1],)

    def activation_name(self, layer: int) -> str:
        """Activation name of ``layer``: the output activation for the last layer, else hidden."""
        self._check_layer(layer)
        return self.output_activation if layer == self.n_layers - 1 else self.hidden_activation

    @property
    def n_parameters(self) -> int:
        """Total count of weights and biases: sum over layers of n_out * n_in + n_out."""
        return sum(
            n_out * n_in + n_out
            for n_in, n_out in zip(self.layer_sizes, self.layer_sizes[1:], strict=False)
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-data form (lists, strings) that survives JSON."""
        return {
            "layer_sizes": list(self.layer_sizes),
            "hidden_activation": self.hidden_activation,
            "output_activation": self.output_activation,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NetworkSpec":
        """Rebuild a spec from :meth:`to_dict` output; missing keys take the defaults."""
        return cls(**{**cls().to_dict(), **data})

    def to_json(self) -> str:
        """The spec as a JSON string (stored inside network files)."""
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "NetworkSpec":
        """Rebuild a spec from :meth:`to_json` output."""
        return cls.from_dict(json.loads(text))

    def _check_layer(self, layer: int) -> None:
        if not 0 <= layer < self.n_layers:
            raise IndexError(f"layer {layer} out of range for {self.n_layers} layers")
