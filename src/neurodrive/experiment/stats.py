"""Statistics across seeds: median, interquartile range, curve bands and bootstrap intervals.

One training run is one roll of the dice, so results are summarised over several seeds
(docs/EXPERIMENT_PROTOCOL.md §2.5, §4):

* **Percentile p** of n values: sort them and go to position h = (n − 1) · p / 100, counted from
  0; when h falls between two values, interpolate linearly between them. The *median* is p 50,
  *Q1* p 25, *Q3* p 75, and the *interquartile range* IQR = Q3 − Q1 is the spread of the middle
  half. Example: [0.04, 0.01, 0.03, 0.10, 0.02] sorts to [0.01, 0.02, 0.03, 0.04, 0.10]; median
  0.03 (h = 2), Q1 0.02, Q3 0.04, IQR 0.02.
* A diverged run counts as **+∞** (Study S2b convention): it sorts to the top, and a percentile
  that lands on it, or between it and a finite value, is +∞. So "most seeds diverged" shows in the
  median instead of those seeds silently disappearing.
* **Bootstrap 95 % interval** of a median: draw R samples of n values *with replacement* from the
  n results, take each sample's median, and report the 2.5th and 97.5th percentiles of those R
  medians. For a difference "A − B" each draw records median(A) − median(B). The generator is
  passed in, so an interval is reproducible.

Same rules as the study scripts, so a report and the web view never disagree (ADR-033).
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.rng import Rng

BOOTSTRAP_RESAMPLES = 10_000


def _values(values: ArrayLike) -> NDArray[np.float64]:
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise ValueError("need at least one value")
    if np.isnan(v).any():
        raise ValueError("values must not be NaN (use +inf for a diverged run)")
    return v


def _percentile_sorted(v: NDArray[np.float64], p: float) -> float:
    """Percentile ``p`` of already sorted values ``v`` (linear interpolation, +inf allowed)."""
    h = (v.size - 1) * p / 100.0
    lo, hi = int(np.floor(h)), int(np.ceil(h))
    a, b = float(v[lo]), float(v[hi])
    if lo == hi or a == b:  # also covers inf == inf, which would otherwise give inf - inf
        return a
    if np.isinf(b):
        return b
    return a + (h - lo) * (b - a)


def percentile(values: ArrayLike, p: float) -> float:
    """The ``p``-th percentile (0 to 100) of ``values``, by linear interpolation.

    Agrees with ``numpy.percentile`` for finite values; +inf values sort to the top (see the
    module docstring). NaN values and empty input raise ``ValueError``.
    """
    if not 0.0 <= p <= 100.0:
        raise ValueError(f"p must be in [0, 100], got {p}")
    return _percentile_sorted(np.sort(_values(values)), p)


@dataclass(frozen=True)
class Summary:
    """Median, Q1, Q3 and the number of values n."""

    median: float
    q1: float
    q3: float
    n: int

    @property
    def iqr(self) -> float:
        return self.q3 - self.q1


def summarise(values: ArrayLike) -> Summary:
    """Median and quartiles of ``values`` (one value per seed)."""
    v = np.sort(_values(values))
    return Summary(
        median=_percentile_sorted(v, 50.0),
        q1=_percentile_sorted(v, 25.0),
        q3=_percentile_sorted(v, 75.0),
        n=int(v.size),
    )


@dataclass(frozen=True)
class CurveBands:
    """Per-position statistics of several curves (position i = epoch i for toy runs).

    ``median``, ``q1``, ``q3``: across the runs present at that position (+inf where they land
    on diverged runs). ``n_present``: runs that reach the position (diverged runs count as
    present, at +inf). ``n_finite``: runs with a finite value there.
    """

    median: list[float]
    q1: list[float]
    q3: list[float]
    n_present: list[int]
    n_finite: list[int]


def curve_bands(curves: Sequence[Sequence[float | None]], diverged: Sequence[bool]) -> CurveBands:
    """Median and IQR at every position of ``curves`` (one sequence of values per run).

    A value of None, and every position after the end of a run whose ``diverged`` flag is set,
    counts as +inf (the run broke down). A run that ends early without diverging (a shorter run)
    is simply absent from later positions.
    """
    if len(curves) != len(diverged):
        raise ValueError(f"{len(curves)} curves but {len(diverged)} diverged flags")
    if not curves:
        raise ValueError("need at least one curve")
    length = max(len(c) for c in curves)
    grid = np.full((len(curves), length), np.nan)
    for row, (curve, broke) in enumerate(zip(curves, diverged, strict=True)):
        values = [np.inf if x is None else float(x) for x in curve]
        grid[row, : len(values)] = values
        if broke:
            grid[row, len(values) :] = np.inf
    median, q1, q3, n_present, n_finite = [], [], [], [], []
    for col in grid.T:
        present = np.sort(col[~np.isnan(col)])
        if present.size == 0:
            raise ValueError("every curve must start at position 0")
        median.append(_percentile_sorted(present, 50.0))
        q1.append(_percentile_sorted(present, 25.0))
        q3.append(_percentile_sorted(present, 75.0))
        n_present.append(int(present.size))
        n_finite.append(int(np.isfinite(present).sum()))
    return CurveBands(median, q1, q3, n_present, n_finite)


@dataclass(frozen=True)
class Interval:
    """A point estimate and its bootstrap confidence interval [low, high] at ``level``."""

    estimate: float
    low: float
    high: float
    level: float

    @property
    def contains_zero(self) -> bool:
        return self.low <= 0.0 <= self.high


def _interval_bounds(samples: NDArray[np.float64], level: float) -> tuple[float, float]:
    samples = np.sort(samples[~np.isnan(samples)])  # inf - inf (both diverged) is dropped
    if samples.size == 0:
        raise ValueError("every bootstrap sample was undefined")
    tail = (1.0 - level) / 2.0 * 100.0
    return _percentile_sorted(samples, tail), _percentile_sorted(samples, 100.0 - tail)


def _check_level(level: float, resamples: int) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if resamples < 1:
        raise ValueError(f"resamples must be at least 1, got {resamples}")


def bootstrap_median_ci(
    values: ArrayLike, rng: Rng, resamples: int = BOOTSTRAP_RESAMPLES, level: float = 0.95
) -> Interval:
    """The median of ``values`` and its bootstrap interval (``resamples`` draws from ``rng``)."""
    _check_level(level, resamples)
    v = _values(values)
    idx = rng.integers(0, v.size, size=(resamples, v.size))
    medians = np.median(v[idx], axis=1)
    low, high = _interval_bounds(medians, level)
    return Interval(summarise(v).median, low, high, level)


def bootstrap_median_difference(
    a: ArrayLike,
    b: ArrayLike,
    rng: Rng,
    resamples: int = BOOTSTRAP_RESAMPLES,
    level: float = 0.95,
    paired: bool = False,
) -> Interval:
    """median(a) − median(b) and its bootstrap interval.

    ``paired``: a[i] and b[i] come from the same seed, so each draw uses the same indices for
    both (the lengths must match). Otherwise a and b are resampled independently. If the interval
    contains 0, the data show no clear difference.
    """
    _check_level(level, resamples)
    va, vb = _values(a), _values(b)
    if paired and va.size != vb.size:
        raise ValueError(f"paired samples need equal lengths, got {va.size} and {vb.size}")
    idx_a = rng.integers(0, va.size, size=(resamples, va.size))
    idx_b = idx_a if paired else rng.integers(0, vb.size, size=(resamples, vb.size))
    with np.errstate(invalid="ignore"):
        diffs = np.median(va[idx_a], axis=1) - np.median(vb[idx_b], axis=1)
        estimate = summarise(va).median - summarise(vb).median
    low, high = _interval_bounds(diffs, level)
    return Interval(float(estimate), low, high, level)
