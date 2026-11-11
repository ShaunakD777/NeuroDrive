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
* **Dense coverage** d (6.2 follow-up, ADR-046): the reach is set by the two most extreme frames,
  so a few outliers make a range look covered. A bin is *dense* when it holds at least the share
  tau of the frames (count_b >= tau·F, tau = ``DENSE_SHARE``); d = (the summed length of every
  dense bin's part inside [min, max]) / (hi - lo). Clipping to [min, max] keeps d <= c. Example:
  [0.1, 0.15, 0.2, 0.3, 0.9] on [0, 1] in 4 bins with tau = 0.3 (a dense bin needs 1.5 frames):
  only [0, 0.25) is dense (3 frames); clipped to [0.1, 0.9] it is [0.1, 0.25], so d = 0.15
  while c = 0.8.
* **Map**: every k-th frame, k = ceil(F / M) with M = ``MAP_POINTS``, at its recorded position
  with its label, so a dataset of any size draws as at most M points. Rows are stored episode by
  episode, so the kept frames lie evenly along every drive. Example: F = 25, M = 10 give k = 3 and
  rows 0, 3, ..., 24 (9 points).

The ranges are the observation and action contracts' (observation v1, action v1): rays and speed
in [0, 1], heading error in (-1, 1], steer and throttle in [-1, 1].

**Novelty** (task 6.5, Study S5b): how far a state is from everything the dataset holds. With o
the 7 observations of a state and o_j those of frame j, d(o) = min_j ||o - o_j||, the ordinary
straight-line distance in the network's own inputs. It sees combinations the histograms above
cannot: frames A = (0.98, 0.3), B = (0.5, 1.0), C = (0.7, 0.3) (speed, left ray) and the state
q = (0.98, 1.0) give distances 0.70, 0.48 and 0.754, so d(q) = 0.48, although each of q's values
occurs in the data on its own. The threshold tau of "never visited" is the q-th percentile of
every frame's distance to the nearest frame of *another* episode (frames 1/30 s apart are near
twins, so its own episode would make tau tiny). Frames 0.10, 0.101 (episode 1) and 0.20, 0.201
(episode 2) have other-episode distances 0.1, 0.099, 0.099, 0.1, so the 99th percentile is
tau = 0.1; with the own episode allowed every distance would be 0.001.
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from neurodrive.env.car import ACTION_NAMES
from neurodrive.env.sensors import OBSERVATION_NAMES
from neurodrive.experiment.stats import percentile
from neurodrive.learn.supervised.demos import DemoDataset

Array = NDArray[np.float64]

BINS = 20
DENSE_SHARE = 0.01  # tau: a bin is dense when it holds at least 1 % of the frames (ADR-046)
MAP_POINTS = 4000
NOVELTY_PERCENTILE = 99.0  # tau: the 99th percentile of the other-episode distances
_CHUNK_ENTRIES = 1 << 22  # distances held in memory at once by nearest_distances (32 MB)
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
    dense: float  # d, the dense coverage (<= covered), 0 for an empty dataset
    dense_threshold: float  # tau·F, the frames a bin needs to be dense


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


def dense_coverage(
    edges: Array,
    counts: NDArray,
    vmin: float,
    vmax: float,
    lo: float,
    hi: float,
    tau: float = DENSE_SHARE,
) -> float:
    """d = (summed length of the dense bins' parts inside [vmin, vmax]) / (hi - lo).

    A bin is dense when count_b >= tau·F, F = the sum of the counts. Each dense bin
    [edge_b, edge_b+1] is clipped to [vmin, vmax] (the values' min and max), so d never exceeds
    the covered range c. An empty dataset gives 0.
    """
    counts = np.asarray(counts)
    n_frames = int(counts.sum())
    if n_frames == 0:
        return 0.0
    dense = counts >= tau * n_frames
    overlap = np.minimum(edges[1:], vmax) - np.maximum(edges[:-1], vmin)
    return float(np.sum(np.where(dense, np.clip(overlap, 0.0, None), 0.0)) / (hi - lo))


def thin(n_frames: int, max_points: int = MAP_POINTS) -> tuple[int, NDArray[np.int64]]:
    """k = ceil(F / M) (at least 1) and the rows 0, k, 2k, ... below F: at most M of them."""
    stride = max(1, math.ceil(n_frames / max_points))
    return stride, np.arange(0, n_frames, stride, dtype=np.int64)


def _histogram(name: str, values: Array, lo: float, hi: float) -> Histogram:
    edges, counts = histogram(values, lo, hi)
    empty = len(values) == 0
    vmin = None if empty else float(np.min(values))
    vmax = None if empty else float(np.max(values))
    return Histogram(
        name=name,
        lo=lo,
        hi=hi,
        edges=edges,
        counts=counts,
        min=vmin,
        max=vmax,
        covered=covered_range(values, lo, hi),
        dense=0.0 if empty else dense_coverage(edges, counts, vmin, vmax, lo, hi),
        dense_threshold=DENSE_SHARE * len(values),
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


def nearest_distances(states: Array, frames: Array) -> Array:
    """d_i = min_j ||states_i - frames_j||, the distance from each state to its nearest frame.

    ``states`` is (Q, n) and ``frames`` (F, n). The nearest frame is found in chunks of states
    with ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b (one matrix product per chunk), then the
    distance to it is computed again directly, so the result has no cancellation error.
    """
    states = np.asarray(states, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float64)
    if states.ndim != 2 or frames.ndim != 2 or states.shape[1] != frames.shape[1]:
        raise ValueError(f"need (Q, n) states and (F, n) frames: {states.shape}, {frames.shape}")
    if len(frames) == 0:
        raise ValueError("need at least one frame")
    out = np.empty(len(states))
    frame_sq = np.einsum("ij,ij->i", frames, frames)
    chunk = max(1, _CHUNK_ENTRIES // len(frames))
    for lo in range(0, len(states), chunk):
        part = states[lo : lo + chunk]
        sq = frame_sq[np.newaxis, :] - 2.0 * (part @ frames.T)  # ||a||^2 is the same in a row
        nearest = np.argmin(sq, axis=1)
        out[lo : lo + chunk] = np.linalg.norm(part - frames[nearest], axis=1)
    return out


def other_episode_distances(observations: Array, episodes: NDArray) -> Array:
    """Each frame's distance to the nearest frame of a different episode (needs 2+ episodes)."""
    observations = np.asarray(observations, dtype=np.float64)
    episodes = np.asarray(episodes)
    out = np.empty(len(observations))
    labels = np.unique(episodes)
    if len(labels) < 2:
        raise ValueError("the novelty threshold needs at least two episodes")
    for e in labels:
        own = episodes == e
        out[own] = nearest_distances(observations[own], observations[~own])
    return out


def novelty_threshold(
    observations: Array, episodes: NDArray, q: float = NOVELTY_PERCENTILE
) -> float:
    """tau: the q-th percentile of :func:`other_episode_distances` (see the module docstring)."""
    return percentile(other_episode_distances(observations, episodes), q)
