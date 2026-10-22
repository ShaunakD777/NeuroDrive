"""Sensitivity: how much an output moves when one input is nudged a little.

For output ``m`` and input ``k`` the sensitivity is the slope of the output as that one input
changes, with every other input held still (in calculus terms, the partial derivative
d output_m / d input_k). Here it is *measured*, not derived: nudge input ``k`` up by a small step
``h`` and down by ``h``, run the network both times, and divide the change by the distance:

    s_mk = (f_m(x + h e_k) - f_m(x - h e_k)) / (2 h)

where ``x`` is the observation, ``e_k`` is 1 in position k and 0 elsewhere, and ``f_m`` is output m
of the network. This is a *central difference*. Its error shrinks like h^2 (halve h, quarter the
error), so h = 1e-4 is already accurate to about 1e-8 for smooth activations; a much smaller h
would let rounding errors (about 1e-16 / h) take over.

Reading it: s = +0.4 means "if this input rose by 0.1, the output would rise by about 0.04". A
sensitivity near 0 means the output barely reacts to that input right now, for instance because
the neurons in between are saturated (tanh stuck at -1 or +1). Phase 3 computes the same numbers
exactly with backpropagation and checks them against these.

The nudged inputs are not clipped to their usual ranges (a ray reading of 1 + h is fine for the
network), since only the slope at ``x`` matters. For ReLU a nudge that crosses a kink gives the
average of the slopes on either side.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.network import Network

Array = NDArray[np.float64]

DEFAULT_STEP = 1e-4


def sensitivity(
    network: Network,
    observation: ArrayLike,
    step: float = DEFAULT_STEP,
    layer: int = -1,
) -> Array:
    """The sensitivity table of ``network`` at ``observation``, shape (n_neurons, n_inputs).

    Entry [m, k] is the central-difference slope of neuron m with respect to input k (see the
    module docstring). ``layer`` picks whose neurons: an index into the trace's activations, so
    -1 (the default) is the outputs (row 0 steer, row 1 throttle) and 1 the first hidden layer.
    ``step`` is h. All 2 * n_inputs nudged observations go through the network in one batch.
    """
    x = np.asarray(observation, dtype=np.float64)
    n_in = network.spec.n_inputs
    if x.shape != (n_in,):
        raise ValueError(f"observation must have shape ({n_in},), got {x.shape}")
    if not (np.isfinite(step) and step > 0):
        raise ValueError(f"step must be a positive number, got {step}")
    n_layers = network.spec.n_layers + 1  # activations a_0 .. a_L
    if not -n_layers <= layer < n_layers:
        raise IndexError(f"layer {layer} out of range for {n_layers} activation layers")

    nudges = step * np.eye(n_in)
    batch = np.concatenate([x + nudges, x - nudges])  # rows 0..n-1 up, rows n..2n-1 down
    values = forward(network, batch).activations[layer]  # shape (2 n_in, n_neurons)
    slopes = (values[:n_in] - values[n_in:]) / (2.0 * step)  # shape (n_in, n_neurons)
    return slopes.T
