"""Gradient checking: verify backpropagation against slow, simple finite differences.

Backpropagation (``core/nn/backward.py``) is fast but intricate; a forgotten slope or a transposed
matrix gives gradients that look plausible and are wrong. A *finite-difference* estimate uses only
the forward pass, so it is slow but hard to get wrong. Symbols:

* ``θ`` (theta): all parameters as one vector (the genome, ``Network.flatten()``).
* ``L(θ)``: the loss on a fixed batch of observations and targets.
* ``h``: the nudge size (default 1e-5).
* ``b``: backprop's gradient; ``n``: the finite-difference estimate.

Central difference for entry k: ``n_k = [L(θ_k + h) − L(θ_k − h)] / (2h)``. Its truncation error
shrinks like h², but rounding noise grows like (machine precision) / h, about 1e-16 / h, so h
cannot be tiny; 1e-5 balances the two.

Relative error: ``r = ‖b − n‖ / (‖b‖ + ‖n‖)`` with ‖v‖ = √(Σ v_k²), defined as 0 when both are
zero. It is 0 for a perfect match and about 1 for unrelated vectors; backprop passes when
r < 1e-6 (docs/EXPERIMENT_PROTOCOL.md). One norm-based number is used rather than a per-entry
ratio, because per-entry ratios blow up on tiny gradients that agree in absolute terms.

ReLU has a corner at z = 0; a finite difference across it is meaningless. ``GradCheckResult``
reports how close any ReLU weighted sum came to 0 so such cases can be recognised.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.losses import mse, mse_grad
from neurodrive.core.nn.backward import backward
from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec

DEFAULT_H = 1e-5
DEFAULT_TOL = 1e-6

LossFn = Callable[[ArrayLike, ArrayLike], float]
LossGradFn = Callable[[ArrayLike, ArrayLike], NDArray[np.float64]]


def numeric_gradient(
    f: Callable[[NDArray[np.float64]], float], x: ArrayLike, h: float = DEFAULT_H
) -> NDArray[np.float64]:
    """Central-difference estimate of the gradient of the scalar function ``f`` at ``x``.

    For every entry i of ``x`` (any shape): nudge it up by ``h`` and down by ``h``, evaluate
    ``f`` both times and divide the difference by 2h. ``x`` itself is not changed. Returns an
    array shaped like ``x``.
    """
    x = np.array(x, dtype=np.float64)  # a private copy
    grad = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        original = x[i]
        x[i] = original + h
        up = f(x)
        x[i] = original - h
        down = f(x)
        x[i] = original
        grad[i] = (up - down) / (2.0 * h)
    return grad


def relative_error(a: ArrayLike, b: ArrayLike) -> float:
    """``‖a − b‖ / (‖a‖ + ‖b‖)``: 0 for identical vectors (also when both are zero), up to 1."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"cannot compare shapes {a.shape} and {b.shape}")
    denom = float(np.linalg.norm(a) + np.linalg.norm(b))
    return 0.0 if denom == 0.0 else float(np.linalg.norm(a - b)) / denom


def parameter_name(spec: NetworkSpec, k: int) -> str:
    """Readable name of genome entry ``k``, e.g. ``"W2[0,1]"`` or ``"b1[3]"``.

    Layers are counted from 1 as in the docs (W1 is the first weight matrix); ``W2[j,i]`` is the
    weight from input i to neuron j of layer 2. Follows the order of ``Network.flatten()``.
    """
    if not 0 <= k < spec.n_parameters:
        raise ValueError(f"index {k} outside 0..{spec.n_parameters - 1}")
    sizes = spec.layer_sizes
    for layer in range(spec.n_layers):
        n_in, n_out = sizes[layer], sizes[layer + 1]
        if k < n_out * n_in:
            return f"W{layer + 1}[{k // n_in},{k % n_in}]"
        k -= n_out * n_in
        if k < n_out:
            return f"b{layer + 1}[{k}]"
        k -= n_out
    raise AssertionError("unreachable")  # pragma: no cover


@dataclass(frozen=True)
class GradCheckResult:
    """Backprop's gradients next to the finite-difference estimates, and how far apart they are.

    ``analytic`` / ``numeric``: parameter gradients in genome order. ``input_analytic`` /
    ``input_numeric``: gradients for the observations (their shape). ``relative_error`` /
    ``input_relative_error``: r for each. ``worst_index`` / ``worst_name`` / ``worst_gap``: the
    parameter with the largest |b − n|. ``relu_margin``: the smallest |z| in any ReLU layer
    (infinity if there is none); below about ``h`` times the inputs' size, estimates there are
    unreliable.
    """

    analytic: NDArray[np.float64]
    numeric: NDArray[np.float64]
    relative_error: float
    input_analytic: NDArray[np.float64]
    input_numeric: NDArray[np.float64]
    input_relative_error: float
    worst_index: int
    worst_name: str
    worst_gap: float
    relu_margin: float

    def passed(self, tol: float = DEFAULT_TOL) -> bool:
        """True when both relative errors (parameters and inputs) are below ``tol``."""
        return self.relative_error < tol and self.input_relative_error < tol


def gradient_check(
    network: Network,
    observations: ArrayLike,
    targets: ArrayLike,
    loss: LossFn = mse,
    loss_grad: LossGradFn = mse_grad,
    h: float = DEFAULT_H,
) -> GradCheckResult:
    """Compare ``backward`` with central differences for every parameter and every input.

    ``observations`` has shape (n_inputs,) or (N, n_inputs); ``targets`` has the shape of the
    network's output for them. ``loss`` and ``loss_grad`` are a loss and its derivative from
    ``core/losses.py``. The network is not changed. No randomness is used.
    """
    spec = network.spec
    x = np.asarray(observations, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    trace = forward(network, x)
    grads = backward(network, trace, loss_grad(trace.output, y))
    analytic = grads.flatten()

    def loss_of_params(theta: NDArray[np.float64]) -> float:
        return loss(forward(Network.unflatten(spec, theta), x).output, y)

    def loss_of_inputs(inputs: NDArray[np.float64]) -> float:
        return loss(forward(network, inputs).output, y)

    numeric = numeric_gradient(loss_of_params, network.flatten(), h)
    input_numeric = numeric_gradient(loss_of_inputs, x, h)

    gaps = np.abs(analytic - numeric)
    worst = int(np.argmax(gaps))
    relu_z = [
        np.abs(z).min()
        for layer, z in enumerate(trace.weighted_sums)
        if spec.activation_name(layer) == "relu"
    ]
    return GradCheckResult(
        analytic=analytic,
        numeric=numeric,
        relative_error=relative_error(analytic, numeric),
        input_analytic=grads.inputs,
        input_numeric=input_numeric,
        input_relative_error=relative_error(grads.inputs, input_numeric),
        worst_index=worst,
        worst_name=parameter_name(spec, worst),
        worst_gap=float(gaps[worst]),
        relu_margin=float(min(relu_z)) if relu_z else float("inf"),
    )
