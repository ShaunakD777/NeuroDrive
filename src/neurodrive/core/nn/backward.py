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
example. :func:`neuron_backward` does this for one neuron (the teaching path).

A whole network (:func:`backward`) repeats the step layer by layer from the top down, in matrix
form for N examples at once (rows are examples). With ``A_0`` the inputs, ``Z_l = A_{l-1} W_l^T +
b_l`` and ``A_l = f_l(Z_l)`` from the forward trace, and ``G_L = ∂L/∂A_L`` from the loss:

1. ``Δ_l = G_l ⊙ f_l'(Z_l)`` (⊙: entry by entry), the error signals of every neuron and example;
2. ``∂L/∂W_l = Δ_l^T A_{l-1}`` and ``∂L/∂b_l`` = the column sums of ``Δ_l``;
3. ``G_{l-1} = Δ_l W_l``: the push on layer l's inputs, which are layer l-1's outputs. A neuron
   feeding several neurons above it receives the *sum* of their pushes.

The final ``G_0`` is the gradient with respect to the network's inputs.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.activations import get_derivative
from neurodrive.core.neuron import Neuron
from neurodrive.core.nn.forward import ForwardTrace
from neurodrive.core.nn.network import Network


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


@dataclass(frozen=True)
class NetworkGradients:
    """The result of :func:`backward`: ∂L/∂(every parameter) and ∂L/∂(inputs).

    ``weights[l]`` and ``biases[l]`` have exactly the shapes of ``network.weights[l]`` and
    ``network.biases[l]`` (summed over the batch). ``inputs`` has the shape of the observations
    passed to the forward pass. ``deltas[l]`` is layer l's error signal Δ = ∂L/∂Z, shape
    (N, layer width), or (layer width,) for a single example.
    """

    weights: list[NDArray[np.float64]]
    biases: list[NDArray[np.float64]]
    inputs: NDArray[np.float64]
    deltas: list[NDArray[np.float64]]

    def flatten(self) -> NDArray[np.float64]:
        """All parameter gradients as one vector, in the order of ``Network.flatten()``.

        For each layer: the weight gradient row by row, then the bias gradient. Entry k is
        therefore ∂L/∂(entry k of the genome).
        """
        parts: list[NDArray[np.float64]] = []
        for w, b in zip(self.weights, self.biases, strict=True):
            parts.append(w.ravel())
            parts.append(b)
        return np.concatenate(parts)


def backward(network: Network, trace: ForwardTrace, grad_output: ArrayLike) -> NetworkGradients:
    """Backpropagate ``grad_output`` (G_L = ∂L/∂output) through ``network``.

    ``trace`` must be ``forward(network, observations)`` for the network's *current* weights; it
    is not recomputed. ``grad_output`` has the shape of ``trace.output``: (n_outputs,) for one
    example or (N, n_outputs) for a batch, usually ``mse_grad`` or ``huber_grad`` of the output.
    Raises ``ValueError`` if the trace does not fit the network or ``grad_output`` has the wrong
    shape.
    """
    spec = network.spec
    if len(trace.weighted_sums) != spec.n_layers or len(trace.activations) != spec.n_layers + 1:
        raise ValueError(
            f"trace has {len(trace.weighted_sums)} layers but the network has {spec.n_layers}"
        )
    single = trace.output.ndim == 1
    g = np.asarray(grad_output, dtype=np.float64)
    if g.shape != trace.output.shape:
        raise ValueError(
            f"grad_output must have the output's shape {trace.output.shape}, got {g.shape}"
        )
    # Work in batch form: a single example is a batch of one.
    acts = [a.reshape(1, -1) if single else a for a in trace.activations]
    sums = [z.reshape(1, -1) if single else z for z in trace.weighted_sums]
    for layer, z in enumerate(sums):
        if (
            z.shape[1] != spec.layer_sizes[layer + 1]
            or acts[layer].shape[1] != spec.layer_sizes[layer]
        ):
            raise ValueError(f"trace layer {layer} does not match the network's layer sizes")
    g = g.reshape(1, -1) if single else g

    n = spec.n_layers
    grad_w: list[NDArray[np.float64]] = [np.empty(0)] * n
    grad_b: list[NDArray[np.float64]] = [np.empty(0)] * n
    deltas: list[NDArray[np.float64]] = [np.empty(0)] * n
    for layer in reversed(range(n)):
        delta = g * get_derivative(spec.activation_name(layer))(sums[layer])  # Δ = G ⊙ f'(Z)
        grad_w[layer] = delta.T @ acts[layer]  # Δ^T A_{l-1}: summed over examples
        grad_b[layer] = delta.sum(axis=0)
        deltas[layer] = delta[0] if single else delta
        g = delta @ network.weights[layer]  # G_{l-1} = Δ W: push on this layer's inputs
    return NetworkGradients(grad_w, grad_b, g[0] if single else g, deltas)
