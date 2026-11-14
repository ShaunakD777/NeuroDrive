"""What the evolution charts show for one GA run (task 7.7, ADR-058). All numbers come from here;
the browser only draws them (ADR-012).

A GA run (task 7.5) scores one generation of N networks per unit. ``metrics.jsonl`` holds each
generation's summary (best so far, diversity, finishers, ...) and ``extra/population.jsonl`` every
network's fitness F_i (the mean of f = p + c(1 - t/T) over the S evaluation starts, task 7.2).

* **Statistics of one generation:** best = max F_i, worst = min F_i, mean = (sum of F_i) / N,
  median = the middle value of the sorted F (the mean of the two middle ones for even N), and the
  quartiles q1, q3 = the 25th and 75th percentiles by linear interpolation (position
  0.25·(N - 1) and 0.75·(N - 1) in the sorted list, counting from 0). Example: F = (0, 0, 0.12,
  0.31, 1.27) gives best 1.27, q3 0.31, median 0.12, mean 0.34, q1 0, worst 0. The mean is almost
  three times the median: one good network pulls it up.
* **Fitness bands:** edges e_k = k·w with the band width w = ``BAND_WIDTH`` = 0.05, from 0 to 2
  by default (fitness v2 stays below 2: p <= 1 and the time bonus is below 1). Band k holds the
  networks with e_k <= F_i < e_(k+1), so a value on an edge always goes into the band above it.
  The band of a value is found by searching the stored edges, not by rounding F / w down:
  0.15 / 0.05 is 2.9999999999999996 in floating point, but 0.15 belongs in band 3, [0.15, 0.20).
  If any fitness of the run lies outside [0, 2), whole bands are added on that side (never
  removed), so every generation of a run shares the same edges. The same example gives counts
  2, 1, 1, 1 in bands 0, 2, 6 and 25, and 0 elsewhere. 2 networks are at exactly 0 (they never
  moved forward); they are counted separately as well, because band 0 also holds networks that
  crawled a few metres.
* **Diversity** D, the mean Euclidean distance between every pair of genomes, as the GA stored it
  in ``metrics.jsonl`` (``learn/evolution/ga.py: diversity``): (0, 0), (3, 4), (0, 4) give
  (5 + 3 + 4) / 3 = 4.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.experiment.stats import percentile

Array = NDArray[np.float64]

BAND_WIDTH = 0.05  # w
BAND_TOP = 2.0  # the default top edge: fitness v2 stays below 2


@dataclass(frozen=True)
class GenerationStats:
    """The six statistics of one generation's fitness (see the module docstring)."""

    best: float
    q3: float
    median: float
    mean: float
    q1: float
    worst: float


@dataclass(frozen=True)
class EvolutionData:
    """The evolution charts' numbers for one GA run, one entry per generation (``unit``).

    ``edges`` has B + 1 band edges shared by every generation; ``counts`` is (G, B), row g the
    networks per band of generation ``unit[g]``, each row adding up to ``population``.
    """

    population: int
    elite: int
    unit: list[int]
    best_so_far: list[float]
    stats: list[GenerationStats]
    zero: list[int]
    diversity: list[float | None]
    finished_share: list[float | None]
    finishers: list[int | None]
    champion: list[int]
    band_width: float
    edges: Array
    counts: NDArray[np.int64]


def generation_stats(fitness: ArrayLike) -> GenerationStats:
    """Best, q3, median, mean, q1 and worst of one generation's fitness values."""
    f = np.asarray(fitness, dtype=np.float64).ravel()
    if f.size == 0:
        raise ValueError("a generation needs at least one network")
    if not np.isfinite(f).all():
        raise ValueError("fitness values must be finite")
    return GenerationStats(
        best=float(f.max()),
        q3=percentile(f, 75.0),
        median=percentile(f, 50.0),
        mean=float(f.mean()),
        q1=percentile(f, 25.0),
        worst=float(f.min()),
    )


def band_edges(
    fitness_rows: Sequence[ArrayLike], width: float = BAND_WIDTH, top: float = BAND_TOP
) -> Array:
    """The band edges for every generation of a run: from 0 to ``top`` in steps of ``width``,
    with whole bands added below 0 or at and above ``top`` if any fitness lies there.

    Each edge is k·w rounded to 12 decimals, so 3 × 0.05 is stored as 0.15 and not as
    0.15000000000000002 (which would put a fitness of exactly 0.15 into the band below it).
    """
    if not width > 0:
        raise ValueError(f"the band width must be positive, got {width}")
    k_lo, k_hi = 0, int(round(top / width))
    for row in fitness_rows:
        f = np.asarray(row, dtype=np.float64).ravel()
        if f.size == 0:
            continue
        lo, hi = float(f.min()), float(f.max())
        # floor can be off by one band at an edge (floating point); the loops settle it.
        k = math.floor(lo / width)  # the last edge at or below lo
        while round(k * width, 12) > lo:
            k -= 1
        while round((k + 1) * width, 12) <= lo:
            k += 1
        k_lo = min(k_lo, k)
        k = math.floor(hi / width) + 1  # the first edge above hi
        while round((k - 1) * width, 12) > hi:
            k -= 1
        while round(k * width, 12) <= hi:
            k += 1
        k_hi = max(k_hi, k)
    return np.round(np.arange(k_lo, k_hi + 1) * width, 12)


def band_counts(fitness: ArrayLike, edges: ArrayLike) -> NDArray[np.int64]:
    """The networks per band: count k = the values with edges[k] <= F < edges[k + 1].

    The band of a value is the last edge at or below it (a search of the edges), so a value on
    an edge goes into the band above. Every value must lie in [edges[0], edges[-1]).
    """
    f = np.asarray(fitness, dtype=np.float64).ravel()
    e = np.asarray(edges, dtype=np.float64)
    if e.ndim != 1 or e.size < 2:
        raise ValueError("need at least two band edges")
    if f.size and (f.min() < e[0] or f.max() >= e[-1]):
        raise ValueError(f"fitness outside the bands [{e[0]}, {e[-1]})")
    band = np.searchsorted(e, f, side="right") - 1
    return np.bincount(band, minlength=e.size - 1).astype(np.int64)


def evolution_view(
    metrics: Sequence[dict[str, Any]],
    population: Sequence[dict[str, Any]],
    elite: int = 0,
    width: float = BAND_WIDTH,
) -> EvolutionData:
    """The evolution charts for one GA run from its metrics lines and ``population.jsonl`` lines.

    1. Match every metrics line (one per generation) with the population line of the same unit;
       a generation without one, or with a different number of networks, is an error.
    2. Per generation: the six statistics, the networks at exactly 0, the GA's own best so far,
       diversity, finished share and finishers (from the metrics line), and the champion row.
    3. One set of band edges for the whole run, then the networks per band of every generation.
    """
    lines = {int(line["unit"]): line for line in population}
    units: list[int] = []
    rows: list[Array] = []
    n: int | None = None
    for record in metrics:
        u = int(record["unit"])
        if u not in lines:
            raise ValueError(f"generation {u} has a metrics line but no population line")
        f = np.asarray(lines[u]["fitness"], dtype=np.float64)
        if n is None:
            n = f.size
        if f.size != n or n == 0:
            raise ValueError(f"generation {u} has {f.size} networks, generation {units[0]} {n}")
        units.append(u)
        rows.append(f)
    edges = band_edges(rows, width=width)
    by_unit = {int(r["unit"]): r for r in metrics}
    return EvolutionData(
        population=n or 0,
        elite=elite,
        unit=units,
        best_so_far=[float(by_unit[u]["best_fitness"]) for u in units],
        stats=[generation_stats(f) for f in rows],
        zero=[int((f == 0.0).sum()) for f in rows],
        diversity=[_number(by_unit[u].get("diversity")) for u in units],
        finished_share=[_number(by_unit[u].get("finished_share")) for u in units],
        finishers=[_whole(by_unit[u].get("finishers")) for u in units],
        champion=[int(lines[u]["champion"]) for u in units],
        band_width=width,
        edges=edges,
        counts=(
            np.stack([band_counts(f, edges) for f in rows])
            if rows
            else np.zeros((0, edges.size - 1), dtype=np.int64)
        ),
    )


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _whole(value: Any) -> int | None:
    return None if value is None else int(value)
