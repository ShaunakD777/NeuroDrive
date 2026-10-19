"""Contributions: how much each incoming connection pushed one neuron, right now.

Take neuron ``j`` of a layer. Its inputs are the previous layer's values ``a_i`` (for the first
layer these are the observation), each arriving over a connection with weight ``W[j, i]``. The
*contribution* of input ``i`` is

    c_i = W[j, i] * a_i

so a positive contribution pushes the neuron's sum up and a negative one pushes it down. The
contributions plus the bias ``b_j`` add up to the weighted sum the forward pass computed,

    z_j = c_1 + c_2 + ... + c_n + b_j,

and the neuron's output is ``f(z_j)``. That makes the contributions a complete receipt for z: no
part of the sum is left unexplained. (The forward pass adds the same numbers in a different
order, so the two totals can differ in the last bit, about 1e-16; never more than rounding.)

Layers are counted as in :class:`~neurodrive.core.nn.forward.ForwardTrace`: weight layer ``l``
reads ``trace.activations[l]`` and produces ``trace.weighted_sums[l]`` and
``trace.activations[l + 1]``.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrive.core.nn.forward import ForwardTrace
from neurodrive.core.nn.network import Network

Array = NDArray[np.float64]


@dataclass(frozen=True)
class NeuronContributions:
    """The receipt for one neuron's weighted sum, for one observation.

    ``layer`` is the weight layer (0 = the first hidden layer, reading the observation) and
    ``neuron`` the neuron's position in it. ``inputs`` are the incoming values a_i, ``weights``
    the row W[j, :], ``contributions`` their products c_i, ``bias`` b_j, ``z`` the weighted sum
    the forward pass computed, ``activation`` the activation's name and ``output`` f(z).
    """

    layer: int
    neuron: int
    inputs: Array
    weights: Array
    contributions: Array
    bias: float
    z: float
    activation: str
    output: float

    @property
    def total(self) -> float:
        """Sum of the contributions, c_1 + ... + c_n (equals z - bias up to rounding)."""
        return float(np.sum(self.contributions))

    def ranked(self) -> list[int]:
        """Input indices, largest |contribution| first (ties keep the input order)."""
        return [int(i) for i in np.argsort(-np.abs(self.contributions), kind="stable")]


def _check_single_trace(network: Network, trace: ForwardTrace) -> None:
    sizes = list(network.spec.layer_sizes)
    shapes = [a.shape for a in trace.activations]
    if shapes != [(n,) for n in sizes]:
        raise ValueError(
            f"need the trace of one observation through this network, with activation shapes "
            f"{[(n,) for n in sizes]}; got {shapes}"
        )


def layer_contributions(network: Network, trace: ForwardTrace, layer: int) -> Array:
    """Every contribution of weight layer ``layer`` at once, shape (n_out, n_in).

    Entry [j, i] is c_ji = W[j, i] * a_i: row j is the receipt of neuron j, so the sum of row j
    plus b_j is z_j. ``trace`` is the trace of a single observation through ``network``.
    """
    _check_single_trace(network, trace)
    network.spec.weight_shape(layer)  # raises IndexError for a layer that does not exist
    return network.weights[layer] * trace.activations[layer][np.newaxis, :]


def neuron_contributions(
    network: Network, trace: ForwardTrace, layer: int, neuron: int
) -> NeuronContributions:
    """The receipt of neuron ``neuron`` in weight layer ``layer`` (see the module docstring).

    For the last layer these are the outputs: neuron 0 is steer and neuron 1 is throttle.
    Raises ``IndexError`` for a layer or neuron that does not exist.
    """
    contributions = layer_contributions(network, trace, layer)
    n_out = contributions.shape[0]
    if not 0 <= neuron < n_out:
        raise IndexError(f"neuron {neuron} out of range for a layer of {n_out}")
    return NeuronContributions(
        layer=layer,
        neuron=neuron,
        inputs=trace.activations[layer].copy(),
        weights=network.weights[layer][neuron].copy(),
        contributions=contributions[neuron].copy(),
        bias=float(network.biases[layer][neuron]),
        z=float(trace.weighted_sums[layer][neuron]),
        activation=network.spec.activation_name(layer),
        output=float(trace.activations[layer + 1][neuron]),
    )
