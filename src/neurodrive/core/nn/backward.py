"""Backpropagation: how the loss changes when each weight, bias or input is nudged.

The forward pass goes inputs → weighted sum → output → loss. Backpropagation walks the same
chain backwards with the *chain rule*: the effect of a weight on the loss is the product of the
effects along the chain. For one neuron with inputs ``x_i``, weights ``w_i``, bias ``b``, weighted
sum ``z = Σ w_i x_i + b`` and output ``a = f(z)``:

1. ``g = ∂L/∂a``, the *upstream gradient*: how the loss changes per unit of output. It comes from
   the loss's derivative (``core/losses.py``), or from the layer above in a network.
2. ``δ = ∂L/∂z = g · f'(z)``, the *error signal* of the neuron (``f'`` from ``core/activations``).
3. ``∂L/∂w_i = δ · x_i``, ``∂L/∂b = δ`` and ``∂L/∂x_i = δ · w_i``, because z changes by x_i per
   unit of w_i, by 1 per unit of b and by w_i per unit of x_i.

For a batch every example has its own δ; the weight and bias gradients are summed over the
examples (the loss's 1/M average is already inside g), the input gradients stay one row per
example. This task (3.3) covers a single neuron; the full network follows in task 3.4.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.activations import get_derivative
from neurodrive.core.neuron import Neuron


@dataclass(frozen=True)
class NeuronGradients:
    """The result of :func:`neuron_backward`.

    ``weights``: ∂L/∂w, shape (n_inputs,), summed over the batch.
    ``bias``: ∂L/∂b, one number, summed over the batch.
    ``inputs``: ∂L/∂x, the same shape as the inputs (one row per example).
    ``delta``: the error signal δ = ∂L/∂z, shape () for one example or (N,) for a batch.
    """

    weights: NDArray[np.float64]
    bias: float
    inputs: NDArray[np.float64]
    delta: NDArray[np.float64]


def neuron_backward(neuron: Neuron, inputs: ArrayLike, grad_output: ArrayLike) -> NeuronGradients:
    """Backpropagate the upstream gradient ``grad_output`` (g = ∂L/∂a) through one neuron.

    ``inputs`` has shape (n_inputs,) for one example, then ``grad_output`` is one number; or
    (N, n_inputs) for N examples, then ``grad_output`` has shape (N,). The weighted sum z is
    recomputed from ``inputs``, so the gradients always belong to the neuron's current weights.
    Raises ``ValueError`` if the shape of ``grad_output`` does not match one output per example.
    """
    x = np.asarray(inputs, dtype=np.float64)
    z, _ = neuron.forward_with_sum(x)  # also checks the shape of the inputs
    g = np.asarray(grad_output, dtype=np.float64)
    if g.shape != z.shape:
        raise ValueError(
            f"grad_output must have shape {z.shape} (one value per example), got {g.shape}"
        )
    delta = g * get_derivative(neuron.activation)(z)  # δ = g · f'(z)
    if x.ndim == 1:
        grad_w = delta * x  # δ · x_i
        grad_b = float(delta)
        grad_x = delta * neuron.weights  # δ · w_i
    else:
        grad_w = delta @ x  # Σ over examples of δ · x_i
        grad_b = float(delta.sum())
        grad_x = np.outer(delta, neuron.weights)  # row k is δ_k · w
    return NeuronGradients(weights=grad_w, bias=grad_b, inputs=grad_x, delta=delta)
