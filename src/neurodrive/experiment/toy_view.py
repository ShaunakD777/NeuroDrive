"""What the toy lab shows for one checkpoint of a toy run (task 3.8). All numbers come from here;
the browser only draws them (ADR-012).

For checkpoint u with parameters θ_u, the run's dataset (x, y) and the run's loss L:

* ``loss``: L(θ_u) on the whole dataset (the same number the training metrics hold);
* ``weight_gradients`` / ``bias_gradients``: ∂L/∂W and ∂L/∂b at θ_u, from ``backward``: how hard
  learning pushes each parameter at that moment, i.e. what the next gradient-descent step uses;
* ``glow[l][j][i] = |∂L/∂W_l[j, i]| / max |∂L/∂W|``: each weight's gradient size relative to the
  largest weight gradient of the checkpoint (1 for the most-pushed weight, 0 if all are 0);
* predictions on a grid: for a 1-input task, ŷ at ``GRID_1D`` evenly spaced x values covering the
  data plus ``MARGIN`` of its range on each side; for a 2-input task, ŷ on a ``GRID_2D`` ×
  ``GRID_2D`` grid over the data's box (``prediction_map[i][k]`` is at x1 = grid_x[k],
  x2 = grid_y[i]). Toy tasks have one output, so ŷ is a single number per grid point.
"""

from dataclasses import dataclass

import numpy as np

from neurodrive.core.nn.backward import backward
from neurodrive.core.nn.forward import forward, predict
from neurodrive.core.nn.network import Network
from neurodrive.experiment.config import toy_config_from_run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.supervised.train_toy import LOSSES

Array = np.ndarray

GRID_1D = 201
GRID_2D = 41
MARGIN = 0.1


@dataclass(frozen=True)
class ToyCheckpointView:
    """The toy lab's numbers for one checkpoint (see the module docstring)."""

    unit: int
    network: Network
    loss: float
    data_x: Array
    data_y: Array
    grid_x: Array
    grid_y: Array | None
    prediction_curve: Array | None
    prediction_map: Array | None
    weight_gradients: list[Array]
    bias_gradients: list[Array]
    glow: list[Array]
    max_gradient: float


def _axis(values: Array, n: int) -> Array:
    """``n`` evenly spaced points covering ``values`` plus MARGIN of their range on each side."""
    lo, hi = float(values.min()), float(values.max())
    pad = MARGIN * (hi - lo) if hi > lo else 1.0
    return np.linspace(lo - pad, hi + pad, n)


def toy_checkpoint_view(store: RunStore, unit: int) -> ToyCheckpointView:
    """Compute the toy lab's view of checkpoint ``unit`` of the toy run in ``store``.

    Raises ``KeyError`` for a missing checkpoint and ``ValueError`` if the run is not a toy run
    with a stored dataset and a single output.
    """
    config = toy_config_from_run(store.read_config())
    network = store.load_checkpoint(unit)
    data = store.load_extra("dataset")
    x, y = np.asarray(data["x"], dtype=np.float64), np.asarray(data["y"], dtype=np.float64)
    if y.shape[1] != 1 or x.shape[1] not in (1, 2):
        raise ValueError(f"toy view needs 1 or 2 inputs and 1 output, got {x.shape}, {y.shape}")
    loss_fn, loss_grad = LOSSES[config.loss]

    trace = forward(network, x)
    grads = backward(network, trace, loss_grad(trace.output, y))
    max_gradient = max(float(np.abs(g).max()) for g in grads.weights)
    glow = [
        np.abs(g) / max_gradient if max_gradient > 0 else np.zeros_like(g) for g in grads.weights
    ]

    if x.shape[1] == 1:
        grid_x, grid_y = _axis(x[:, 0], GRID_1D), None
        curve = predict(network, grid_x.reshape(-1, 1))[:, 0]
        surface = None
    else:
        grid_x, grid_y = _axis(x[:, 0], GRID_2D), _axis(x[:, 1], GRID_2D)
        gx, gy = np.meshgrid(grid_x, grid_y)  # row i: x2 = grid_y[i]; column k: x1 = grid_x[k]
        points = np.stack([gx.ravel(), gy.ravel()], axis=1)
        curve = None
        surface = predict(network, points)[:, 0].reshape(GRID_2D, GRID_2D)

    return ToyCheckpointView(
        unit=unit,
        network=network,
        loss=float(loss_fn(trace.output, y)),
        data_x=x,
        data_y=y[:, 0],
        grid_x=grid_x,
        grid_y=grid_y,
        prediction_curve=curve,
        prediction_map=surface,
        weight_gradients=list(grads.weights),
        bias_gradients=list(grads.biases),
        glow=glow,
        max_gradient=max_gradient,
    )
