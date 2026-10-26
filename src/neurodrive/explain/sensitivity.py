"""Sensitivity: how much an output moves when one input is nudged a little.

For output ``m`` and input ``k`` the sensitivity is the slope of the output as that one input
changes, with every other input held still (in calculus terms, the partial derivative
d output_m / d input_k). The whole table, one row per output and one column per input, is the
network's *Jacobian* ``J`` at the observation ``x``.

:func:`sensitivity` computes ``J`` exactly with backpropagation (``core/nn/backward.py``, task 3.6).
Backprop turns an upstream gradient G at the outputs into the gradient for the inputs. With
``G = e_m`` (1 for output m, 0 for the others) that input gradient is exactly row m of J. All rows
are found in one batched backward pass: n copies of ``x`` with G = the identity matrix.

:func:`sensitivity_finite_difference` is the original *measured* version (task 2.9), kept as an
independent check. It nudges input ``k`` up and down by a small step ``h`` and divides the change
by the distance:

    s_mk = (f_m(x + h e_k) - f_m(x - h e_k)) / (2 h)

where ``e_k`` is 1 in position k and 0 elsewhere and ``f_m`` is output m. Its error shrinks like
h^2, so h = 1e-4 is accurate to about 1e-8 for smooth activations; the two agree within 1e-5.

Reading it: s = +0.4 means "if this input rose by 0.1, the output would rise by about 0.04". A
sensitivity near 0 means the output barely reacts to that input right now, for instance because
the neurons in between are saturated (tanh stuck at -1 or +1).

ReLU's corner at z = 0: backprop uses the convention that the slope there is 0, while a finite
difference across the corner gives the average of the slopes on either side.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.nn.backward import backward
from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec

Array = NDArray[np.float64]

DEFAULT_STEP = 1e-4


def _check(network: Network, observation: ArrayLike, layer: int) -> tuple[Array, int]:
    """Validate the arguments; return the observation as floats and ``layer`` as 0..L."""
    x = np.asarray(observation, dtype=np.float64)
    n_in = network.spec.n_inputs
    if x.shape != (n_in,):
        raise ValueError(f"observation must have shape ({n_in},), got {x.shape}")
    n_layers = network.spec.n_layers + 1  # activations a_0 .. a_L
    if not -n_layers <= layer < n_layers:
        raise IndexError(f"layer {layer} out of range for {n_layers} activation layers")
    return x, layer % n_layers


def sensitivity(network: Network, observation: ArrayLike, layer: int = -1) -> Array:
    """The exact sensitivity table of ``network`` at ``observation``, shape (n_neurons, n_inputs).

    Entry [m, k] is d(neuron m) / d(input k), computed by backpropagation (see the module
    docstring). ``layer`` picks whose neurons: an index into the trace's activations, so -1 (the
    default) is the outputs (row 0 steer, row 1 throttle), 1 the first hidden layer and 0 the
    inputs themselves (the identity matrix). One forward and one backward pass.
    """
    x, layer = _check(network, observation, layer)
    spec = network.spec
    if layer == 0:
        return np.eye(spec.n_inputs)
    # The network cut off after `layer`: same weights, each layer keeps its own activation.
    sub = network
    if layer < spec.n_layers:
        sub = Network(
            NetworkSpec(
                spec.layer_sizes[: layer + 1],
                hidden_activation=spec.hidden_activation,
                output_activation=spec.activation_name(layer - 1),
            ),
            network.weights[:layer],
            network.biases[:layer],
        )
    n_neurons = spec.layer_sizes[layer]
    batch = np.tile(x, (n_neurons, 1))  # one copy of x per neuron asked about
    trace = forward(sub, batch)
    return backward(sub, trace, np.eye(n_neurons)).inputs  # row m: G = e_m


def sensitivity_finite_difference(
    network: Network,
    observation: ArrayLike,
    step: float = DEFAULT_STEP,
    layer: int = -1,
) -> Array:
    """The sensitivity table *measured* by central differences (task 2.9), the test oracle.

    Same shape and meaning as :func:`sensitivity`; ``step`` is h. All 2 * n_inputs nudged
    observations go through the network in one batch.
    """
    x, layer = _check(network, observation, layer)
    if not (np.isfinite(step) and step > 0):
        raise ValueError(f"step must be a positive number, got {step}")
    n_in = network.spec.n_inputs
    nudges = step * np.eye(n_in)
    batch = np.concatenate([x + nudges, x - nudges])  # rows 0..n-1 up, rows n..2n-1 down
    values = forward(network, batch).activations[layer]  # shape (2 n_in, n_neurons)
    slopes = (values[:n_in] - values[n_in:]) / (2.0 * step)  # shape (n_in, n_neurons)
    return slopes.T
