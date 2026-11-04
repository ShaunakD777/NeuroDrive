"""What the dataset inspector shows for one demonstration dataset (task 6.2, ADR-045). All numbers
come from here; the browser only draws them (ADR-012).

A dataset (dataset v1, ARCHITECTURE §6.16) has F frames, each an observation o (7 numbers) and a
label y (steer, throttle). The inspector asks two questions: what did the teacher do, and which
situations did the student get to see?

* **Histogram** of one quantity over its full contract range [lo, hi] cut into B equal bins of
  width w = (hi - lo) / B: count_b = the number of frames with a value in
  [lo + b·w, lo + (b + 1)·w); the last bin also takes the value hi itself, so the counts add up
  to F. Example: steers [0.02, -0.05, 0.43, 0.01, -0.62] on [-1, 1] with B = 20 (w = 0.1) give
  bin 3 [-0.7, -0.6): 1, bin 9 [-0.1, 0.0): 1, bin 10 [0.0, 0.1): 2, bin 14 [0.4, 0.5): 1.
* **Covered range** c = (max - min) / (hi - lo): the share of the range the dataset's values
  reach. Ray readings [0.2, 0.5, 0.3] on [0, 1] give c = 0.3. A small c means the student never
  saw most of what that sensor can report.
* **Map**: every k-th frame, k = ceil(F / M) with M = ``MAP_POINTS``, at its recorded position
  with its label, so a dataset of any size draws as at most M points. Rows are stored episode by
  episode, so the kept frames lie evenly along every drive. Example: F = 25, M = 10 give k = 3 and
  rows 0, 3, ..., 24 (9 points).

The ranges are the observation and action contracts' (observation v1, action v1): rays and speed
in [0, 1], heading error in (-1, 1], steer and throttle in [-1, 1].
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrive.env.car import ACTION_NAMES
from neurodrive.env.sensors import OBSERVATION_NAMES
from neurodrive.learn.supervised.demos import DemoDataset

Array = NDArray[np.float64]

BINS = 20
MAP_POINTS = 4000
LABEL_RANGES = {"steer": (-1.0, 1.0), "throttle": (-1.0, 1.0)}
OBSERVATION_RANGES = {
    "ray_far_left": (0.0, 1.0),
    "ray_left": (0.0, 1.0),
    "ray_centre": (0.0, 1.0),
    "ray_right": (0.0, 1.0),
    "ray_far_right": (0.0, 1.0),
    "speed": (0.0, 1.0),
    "heading_error": (-1.0, 1.0),
}


@dataclass(frozen=True)
class Histogram:
    """One quantity's histogram over [lo, hi] (see the module docstring)."""

    name: str
    lo: float
    hi: float
    edges: Array  # B + 1 bin edges, edges[0] = lo, edges[B] = hi
    counts: NDArray[np.int64]  # B counts that add up to F
    min: float | None  # None for an empty dataset
    max: float | None
    covered: float  # c = (max - min) / (hi - lo), 0 for an empty dataset


@dataclass(frozen=True)
class DatasetView:
    """The inspector's numbers for one dataset: the histograms and the thinned map rows."""

    labels: list[Histogram]
    observations: list[Histogram]
    stride: int  # k: every k-th row is on the map
    rows: NDArray[np.int64]  # the rows on the map: 0, k, 2k, ...


def histogram(values: Array, lo: float, hi: float, bins: int = BINS) -> tuple[Array, NDArray]:
    """Bin edges and counts of ``values`` over [lo, hi] in ``bins`` equal bins.

    Every bin is [edge_b, edge_b+1) except the last, which also takes hi, so the counts add up to
    the number of values. A value outside [lo, hi] cannot happen in a valid dataset (the contracts
    keep observations and labels inside their ranges) and is refused.
    """
    values = np.asarray(values, dtype=np.float64)
    edges = np.linspace(lo, hi, bins + 1)
    counts, _ = np.histogram(values, bins=edges)
    if int(counts.sum()) != len(values):
        raise ValueError(f"{len(values) - int(counts.sum())} value(s) outside [{lo}, {hi}]")
    return edges, counts.astype(np.int64)


def covered_range(values: Array, lo: float, hi: float) -> float:
    """c = (max - min) / (hi - lo), the share of [lo, hi] that ``values`` span (0 when empty)."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    return float((values.max() - values.min()) / (hi - lo))


def thin(n_frames: int, max_points: int = MAP_POINTS) -> tuple[int, NDArray[np.int64]]:
    """k = ceil(F / M) (at least 1) and the rows 0, k, 2k, ... below F: at most M of them."""
    stride = max(1, math.ceil(n_frames / max_points))
    return stride, np.arange(0, n_frames, stride, dtype=np.int64)


def _histogram(name: str, values: Array, lo: float, hi: float) -> Histogram:
    edges, counts = histogram(values, lo, hi)
    empty = len(values) == 0
    return Histogram(
        name=name,
        lo=lo,
        hi=hi,
        edges=edges,
        counts=counts,
        min=None if empty else float(np.min(values)),
        max=None if empty else float(np.max(values)),
        covered=covered_range(values, lo, hi),
    )


def dataset_view(dataset: DemoDataset, max_points: int = MAP_POINTS) -> DatasetView:
    """The histograms of both label columns and all 7 observations, and the map rows."""
    labels = [
        _histogram(name, dataset.label[:, m], *LABEL_RANGES[name])
        for m, name in enumerate(ACTION_NAMES)
    ]
    observations = [
        _histogram(name, dataset.observation[:, k], *OBSERVATION_RANGES[name])
        for k, name in enumerate(OBSERVATION_NAMES)
    ]
    stride, rows = thin(dataset.n_frames, max_points)
    return DatasetView(labels=labels, observations=observations, stride=stride, rows=rows)
