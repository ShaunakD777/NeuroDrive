"""Loss functions: one number that says how wrong a batch of predictions is, plus its derivative.

Symbols used throughout:

* ``y_hat`` (ŷ): the predictions, an array of any shape, usually (examples, outputs).
* ``y``: the targets, the values the predictions should have been. Same shape as ``y_hat``.
* ``r = y_hat - y``: the residual, each prediction's miss with its sign.
* ``M``: the total number of entries in ``y_hat`` (examples × outputs). Losses are averaged over
  all of them, so the loss does not grow just because the batch is bigger.

Each loss ``L`` comes with ``*_grad``, the derivative ∂L/∂ŷ: an array shaped like ``y_hat`` whose
entry says how much ``L`` changes when that one prediction is nudged up a tiny bit. Backpropagation
starts from this array.

Conventions match PyTorch's ``MSELoss`` and ``HuberLoss`` with mean reduction: no ½ in MSE, the
standard ½ inside Huber's quadratic part. Shapes must match exactly; NumPy broadcasting is refused
because predictions of shape (N, 1) against targets of shape (N,) would silently become an N × N
table of misses.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _residual(y_hat: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Return r = y_hat - y as float64, after checking both have the same, non-empty shape."""
    y_hat = np.asarray(y_hat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if y_hat.shape != y.shape:
        raise ValueError(f"predictions have shape {y_hat.shape} but targets have shape {y.shape}")
    if y_hat.size == 0:
        raise ValueError("cannot compute a loss over zero predictions")
    return y_hat - y


def _check_delta(delta: float) -> float:
    delta = float(delta)
    if not (np.isfinite(delta) and delta > 0.0):
        raise ValueError(f"Huber delta must be a positive finite number, got {delta}")
    return delta


def mse(y_hat: ArrayLike, y: ArrayLike) -> float:
    """Mean squared error: L = (1/M) · Σ r², with r = y_hat - y and M the number of entries.

    Squaring makes every miss positive and makes big misses count much more than small ones
    (a miss of 3 costs 9, a miss of 1 costs 1).
    """
    r = _residual(y_hat, y)
    return float(np.mean(r * r))


def mse_grad(y_hat: ArrayLike, y: ArrayLike) -> NDArray[np.float64]:
    """Derivative of ``mse`` with respect to each prediction: ∂L/∂ŷ = 2r / M.

    The push on each prediction is proportional to its own miss, so the biggest miss gets the
    biggest push. Returns an array shaped like ``y_hat``.
    """
    r = _residual(y_hat, y)
    return 2.0 * r / r.size


def huber(y_hat: ArrayLike, y: ArrayLike, delta: float = 1.0) -> float:
    """Huber loss: L = (1/M) · Σ h(r), squared for small misses and straight-line for big ones.

    For each residual r and threshold ``delta`` (δ > 0):
    h(r) = ½ r²              when |r| ≤ δ,
    h(r) = δ · (|r| − ½ δ)   when |r| > δ.
    The two pieces meet at |r| = δ with the same value (½ δ²) and the same slope (δ), so the
    curve has no jump and no corner. Large misses (outliers) count only in proportion to their size.
    """
    delta = _check_delta(delta)
    a = np.abs(_residual(y_hat, y))
    h = np.where(a <= delta, 0.5 * a * a, delta * (a - 0.5 * delta))
    return float(np.mean(h))


def huber_grad(y_hat: ArrayLike, y: ArrayLike, delta: float = 1.0) -> NDArray[np.float64]:
    """Derivative of ``huber`` with respect to each prediction: ∂L/∂ŷ = clip(r, −δ, δ) / M.

    Inside the threshold the push equals the miss r (divided by M); outside it is capped at ±δ,
    so a wild outlier pushes no harder than a miss of exactly δ. Returns an array shaped like
    ``y_hat``.
    """
    delta = _check_delta(delta)
    r = _residual(y_hat, y)
    return np.clip(r, -delta, delta) / r.size
