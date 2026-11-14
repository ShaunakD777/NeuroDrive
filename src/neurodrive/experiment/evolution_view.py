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

**One start at a time** (task 7.7 follow-up, ADR-059). Since then every generation's file
``extra/generations/unit_<g>.npz`` also holds f_ik, network i's fitness on start k, as an array
(N, S), and the outcome of every drive. F_i = (f_i1 + ... + f_iS) / S is an average over tests,
and an average hides which tests fail. With start k chosen, every statistic, band count and
"best so far" above is computed from the column f_.k instead of F (the same functions). The
*finish share* of start k in generation g is (networks whose drive from start k finished) / N.

Example, N = 3, S = 2: A f = (1.3, 0.2), F = 0.75; B f = (1.2, 1.25), F = 1.225; C f = (0, 0.4),
F = 0.2. Ranked by F: B, A, C, yet A is the best on start 1. Start 1 alone: best 1.3, median 1.2,
worst 0; start 2 alone: best 1.25, median 0.4, worst 0.2. If A and B finish start 1 and only B
finishes start 2, the finish shares are 2/3 and 1/3: start 2 is the hard one.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.env.world import Status
from neurodrive.experiment.stats import percentile

Array = NDArray[np.float64]

BAND_WIDTH = 0.05  # w
BAND_TOP = 2.0  # the default top edge: fitness v2 stays below 2
START_ARRAYS = ("start_fitness", "status", "progress", "finish_time")  # (N, S) each, ADR-059


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

    ``start`` is the start the statistics describe (None: F, every start). ``n_starts`` is S when
    every generation has its per-start results, else None (runs trained before ADR-059); then
    ``start_finish_share`` and ``start_mean`` (G, S) give each start's finish share and mean f.
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
    start: int | None = None
    n_starts: int | None = None
    start_finish_share: Array | None = None
    start_mean: Array | None = None


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
    starts: Mapping[int, Mapping[str, ArrayLike]] | None = None,
    start: int | None = None,
) -> EvolutionData:
    """The evolution charts for one GA run from its metrics lines and ``population.jsonl`` lines.

    1. Match every metrics line (one per generation) with the population line of the same unit;
       a generation without one, or with a different number of networks, is an error.
    2. If ``starts`` (unit -> that generation's arrays, ADR-059) covers every generation, check
       the arrays are (N, S) and note each start's finish share and mean f. With ``start`` = k,
       the fitness of generation g becomes column k of its ``start_fitness``.
    3. Per generation: the six statistics, the networks at exactly 0, the best so far (the GA's
       own for F; for one start the highest best on that start up to g), diversity, finished
       share and finishers (from the metrics line, for every start), and the champion row.
    4. One set of band edges for the whole run, then the networks per band of every generation.

    Raises ``ValueError`` for files that disagree, for ``start`` when a generation has no
    per-start results, and for a start that does not exist.
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
    columns = _start_columns(units, n or 0, starts)
    share = mean = None
    if columns is not None:
        share = np.stack([(status == Status.FINISHED).mean(axis=0) for _, status in columns])
        mean = np.stack([f.mean(axis=0) for f, _ in columns])
    by_unit = {int(r["unit"]): r for r in metrics}
    best_so_far = [float(by_unit[u]["best_fitness"]) for u in units]
    if start is not None:
        if columns is None:
            raise ValueError(
                "this run has no per-start results (trained before they were saved, ADR-059): "
                "train it again to see one start at a time"
            )
        n_starts = columns[0][0].shape[1]
        if not 0 <= start < n_starts:
            raise ValueError(
                f"start {start} does not exist: the run has starts 0 to {n_starts - 1}"
            )
        rows = [f[:, start] for f, _ in columns]
        best_so_far = np.maximum.accumulate([float(f.max()) for f in rows]).tolist()
    edges = band_edges(rows, width=width)
    return EvolutionData(
        population=n or 0,
        elite=elite,
        unit=units,
        best_so_far=best_so_far,
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
        start=start,
        n_starts=None if columns is None else int(columns[0][0].shape[1]),
        start_finish_share=share,
        start_mean=mean,
    )


def _start_columns(
    units: Sequence[int], n: int, starts: Mapping[int, Mapping[str, ArrayLike]] | None
) -> list[tuple[Array, NDArray[np.int8]]] | None:
    """(f, status) of every generation, each (N, S); None unless every generation has them."""
    if starts is None or not units or any(u not in starts for u in units):
        return None
    out: list[tuple[Array, NDArray[np.int8]]] = []
    for u in units:
        arrays = starts[u]
        if "start_fitness" not in arrays or "status" not in arrays:
            return None
        f = np.asarray(arrays["start_fitness"], dtype=np.float64)
        status = np.asarray(arrays["status"]).astype(np.int8)
        if f.ndim != 2 or f.shape[0] != n or f.shape[1] == 0 or status.shape != f.shape:
            raise ValueError(
                f"generation {u}'s per-start results have shapes {f.shape} and {status.shape}, "
                f"not ({n}, S)"
            )
        if out and f.shape != out[0][0].shape:
            raise ValueError(
                f"generation {u} has {f.shape[1]} starts, generation {units[0]} "
                f"{out[0][0].shape[1]}"
            )
        out.append((f, status))
    return out


STATUS_NAMES = {s.value: s.name.lower() for s in Status}  # 2 -> "finished", 4 -> "timeout"


@dataclass(frozen=True)
class StartGrid:
    """How every network of one generation drove from every start (ADR-059).

    ``order`` lists the rows best first by F (the GA's ranking, ties to the lower row); the
    arrays are (N, S) in row order: ``start_fitness`` f, ``status`` the outcome's name
    (finished, crashed, stalled or timeout), ``progress`` the share of the target distance and
    ``finish_time`` seconds or None. ``finish_share`` and ``start_mean`` (S,): per start, the
    share of the N networks that finished it and their mean f.
    """

    fitness: Array
    order: list[int]
    start_fitness: Array
    status: list[list[str]]
    progress: Array
    finish_time: list[list[float | None]]
    finish_share: Array
    start_mean: Array


def generation_starts(line: Mapping[str, Any], arrays: Mapping[str, ArrayLike]) -> StartGrid:
    """One generation's outcome grid from its ``population.jsonl`` line and its saved arrays.

    1. Check the four arrays are there and (N, S) for the line's N networks.
    2. Check each row's mean is the line's F_i (the file and the line belong together).
    3. Rank the rows by F, best first; name every outcome; count each start's finishers.

    Raises ``ValueError`` (in plain words) if the arrays are missing or disagree with the line.
    """
    missing = [name for name in START_ARRAYS if name not in arrays]
    if missing:
        raise ValueError(
            f"this generation has no per-start results ({', '.join(missing)} missing; trained "
            "before they were saved, ADR-059): train the run again to see them"
        )
    fitness = np.asarray(line["fitness"], dtype=np.float64)
    f = np.asarray(arrays["start_fitness"], dtype=np.float64)
    status = np.asarray(arrays["status"]).astype(np.int8)
    progress = np.asarray(arrays["progress"], dtype=np.float64)
    finish = np.asarray(arrays["finish_time"], dtype=np.float64)
    n = fitness.size
    if f.ndim != 2 or f.shape[0] != n or f.shape[1] == 0:
        raise ValueError(f"the per-start fitness has shape {f.shape}, not ({n}, S)")
    if not (status.shape == progress.shape == finish.shape == f.shape):
        raise ValueError("the per-start arrays have different shapes")
    if not np.allclose(f.mean(axis=1), fitness, rtol=0.0, atol=1e-9):
        raise ValueError("the per-start fitness does not average to the generation's F")
    order = np.argsort(-fitness, kind="stable")  # the GA's ranking (ga.ranking)
    return StartGrid(
        fitness=fitness,
        order=[int(i) for i in order],
        start_fitness=f,
        status=[[STATUS_NAMES.get(int(c), "running") for c in row] for row in status],
        progress=progress,
        finish_time=[[None if np.isnan(t) else float(t) for t in row] for row in finish],
        finish_share=(status == Status.FINISHED).mean(axis=0),
        start_mean=f.mean(axis=0),
    )


def _number(value: Any) -> float | None:
    return None if value is None else float(value)


def _whole(value: Any) -> int | None:
    return None if value is None else int(value)
