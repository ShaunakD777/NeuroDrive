"""Toy datasets: tiny supervised problems where learning can be seen and checked by hand.

Each task is a set of inputs ``x`` (shape (N, n_inputs)) with targets ``y`` (shape (N, n_outputs)):

* ``line``: x uniform in [-1, 1], y = 2x - 0.5. One linear neuron can fit it exactly, so training
  must recover the slope 2 and the intercept -0.5.
* ``sine``: x uniform in [-pi, pi], y = sin(x). Needs a hidden layer to bend.
* ``xor``: the four corners of the unit square, y = 1 when exactly one input is 1, else 0. No
  single neuron can separate it (no straight line splits the 1s from the 0s); one hidden layer can.
* ``two_spirals``: two interleaved spirals, labelled 0 and 1. A classic hard case for small
  networks; provided as data, solving it is optional.
* ``noisy_sine``: 64 points of y = sin(x) + noise (Gaussian, standard deviation 0.2), x uniform in
  [-pi, pi]. Few, noisy points: a big network can fit the noise itself, which is overfitting
  (Study S3, task 4.8). A perfect sine curve has a loss of about 0.2² = 0.04 on these points.

Random draws come from the generator passed in, so the same seed always gives the same data.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrive.core.rng import Rng

Array = NDArray[np.float64]

LINE_SLOPE = 2.0
LINE_INTERCEPT = -0.5


@dataclass(frozen=True)
class ToyDataset:
    """A toy problem: ``x`` (N, n_inputs), ``y`` (N, n_outputs) and a one-line description."""

    name: str
    x: Array
    y: Array
    description: str

    @property
    def n_inputs(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_outputs(self) -> int:
        return int(self.y.shape[1])


def line(rng: Rng, n: int = 64) -> ToyDataset:
    """``n`` points on the line y = 2x - 0.5, with x uniform in [-1, 1]."""
    x = rng.uniform(-1.0, 1.0, size=(n, 1))
    return ToyDataset("line", x, LINE_SLOPE * x + LINE_INTERCEPT, "y = 2x - 0.5 on [-1, 1]")


def sine(rng: Rng, n: int = 128) -> ToyDataset:
    """``n`` points of y = sin(x), with x uniform in [-pi, pi]."""
    x = rng.uniform(-np.pi, np.pi, size=(n, 1))
    return ToyDataset("sine", x, np.sin(x), "y = sin(x) on [-pi, pi]")


def xor(rng: Rng | None = None) -> ToyDataset:
    """The four XOR examples (no randomness; ``rng`` is accepted for a uniform signature)."""
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([[0.0], [1.0], [1.0], [0.0]])
    return ToyDataset("xor", x, y, "y = 1 when exactly one input is 1")


def two_spirals(rng: Rng, n_per_class: int = 100, noise: float = 0.02) -> ToyDataset:
    """Two interleaved spirals: class 0 and class 1 (the second is the first turned by 180°).

    For t from 0 to 1, the angle is 3π·t (one and a half turns) and the radius t, so a point is
    (t·cos(3πt), t·sin(3πt)); class 1 is the same point times -1. ``noise`` adds Gaussian jitter
    of that size. Returns 2·n_per_class points, classes alternating so both are evenly mixed.
    """
    t = np.linspace(0.05, 1.0, n_per_class)
    angle = 3.0 * np.pi * t
    arm = np.stack([t * np.cos(angle), t * np.sin(angle)], axis=1)
    x = np.empty((2 * n_per_class, 2))
    x[0::2], x[1::2] = arm, -arm
    x += noise * rng.standard_normal(x.shape)
    y = np.empty((2 * n_per_class, 1))
    y[0::2], y[1::2] = 0.0, 1.0
    return ToyDataset("two_spirals", x, y, "two interleaved spirals, labels 0 and 1")


NOISY_SINE_NOISE = 0.2


def noisy_sine(rng: Rng, n: int = 64, noise: float = NOISY_SINE_NOISE) -> ToyDataset:
    """``n`` points of y = sin(x) + noise: x uniform in [-pi, pi], Gaussian noise of std ``noise``.

    The inputs are drawn first, then the noise, both from ``rng``.
    """
    x = rng.uniform(-np.pi, np.pi, size=(n, 1))
    y = np.sin(x) + noise * rng.standard_normal((n, 1))
    return ToyDataset("noisy_sine", x, y, f"y = sin(x) + noise (std {noise}) on [-pi, pi]")


TOY_TASKS = {
    "line": line,
    "sine": sine,
    "xor": xor,
    "two_spirals": two_spirals,
    "noisy_sine": noisy_sine,
}


def make_toy_dataset(name: str, rng: Rng) -> ToyDataset:
    """Build the toy task called ``name`` from ``rng``. Raises ``ValueError`` for unknown names."""
    try:
        maker = TOY_TASKS[name]
    except KeyError:
        valid = ", ".join(sorted(TOY_TASKS))
        raise ValueError(f"unknown toy task {name!r}; valid names: {valid}") from None
    return maker(rng)
