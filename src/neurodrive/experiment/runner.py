"""The runner, version 0 (ADR-030): the loop that trains a learner and fills its run folder.

The learner decides *how* to learn; the runner decides *how long*, *what to record* and *when
to stop* (docs/ARCHITECTURE.md §6.12). Steps:

1. save checkpoint 0 (the network before any learning);
2. for unit = 1, 2, ..., ``units``: call ``learner.train_unit()``, add its budget increments to
   the running totals and append one metrics line (unit, totals, wall-clock seconds since the
   start, the learner's own metrics); stop early if the unit diverged;
3. save a checkpoint at every unit of the schedule and at the last completed unit;
4. mark the run completed, diverged, or failed (on an exception, which is then re-raised).

Version 1 (task 5.5) adds stop conditions on budgets and time, exam schedules and resuming.

``train_toy_run`` starts a whole toy run from a validated config (``nd train``, task 4.6b);
``sweep_seeds`` runs one config once per seed (``nd sweep``, task 4.7a, ADR-033).
"""

import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from neurodrive.core.rng import make_rng
from neurodrive.experiment.config import ToyRunConfig, parse_config, resolved, to_toy_config
from neurodrive.experiment.store import RunStore
from neurodrive.learn.base import Learner
from neurodrive.learn.supervised.train_toy import ToyLearner


def checkpoint_schedule(units: int) -> list[int]:
    """Units to checkpoint: 0, then 1, 2, 5, 10, 20, 50, 100, ... up to ``units``, plus ``units``.

    Dense at the start, where learning changes the network fastest, sparse later; about
    3 checkpoints per factor of 10. E.g. 3000 units → 0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
    1000, 2000, 3000.
    """
    if units < 0:
        raise ValueError(f"units must be >= 0, got {units}")
    marks = {0, units}
    scale = 1
    while scale <= units:
        marks.update(m * scale for m in (1, 2, 5) if m * scale <= units)
        scale *= 10
    return sorted(marks)


def run(learner: Learner, store: RunStore, units: int) -> str:
    """Train ``learner`` for up to ``units`` units, writing everything to ``store``.

    Returns the final status: "completed", or "diverged" when a unit reports divergence (the
    run then stops and the last finite network is checkpointed at the last completed unit).
    """
    schedule = set(checkpoint_schedule(units))
    store.save_checkpoint(0, learner.current_networks()[0])
    totals = {"env_steps_total": 0, "demo_frames_total": 0, "teacher_queries_total": 0}
    start = time.perf_counter()
    last_done, last_saved, status = 0, 0, "completed"
    try:
        for unit in range(1, units + 1):
            result = learner.train_unit()
            totals["env_steps_total"] += result.env_steps
            totals["demo_frames_total"] += result.demo_frames
            totals["teacher_queries_total"] += result.teacher_queries
            record = {"unit": unit, **totals, "wall_clock_s": time.perf_counter() - start}
            store.append_metrics({**record, **result.metrics, "diverged": result.diverged})
            if result.diverged:
                status = "diverged"
                break
            last_done = unit
            if unit in schedule:
                store.save_checkpoint(unit, learner.current_networks()[0])
                last_saved = unit
        if last_saved != last_done:
            store.save_checkpoint(last_done, learner.current_networks()[0])
    except BaseException:
        store.finish("failed")
        raise
    store.finish(status)
    return status


@dataclass(frozen=True)
class ToyRunResult:
    """What ``train_toy_run`` reports: the run folder, its final status, the first and last loss."""

    store: RunStore
    status: str
    epochs_done: int
    start_loss: float
    end_loss: float | None  # the last finite loss; None if the first epoch already diverged


def train_toy_run(
    config: ToyRunConfig, runs_dir: str | Path, extra_meta: dict | None = None
) -> ToyRunResult:
    """Train one toy run described by ``config`` and write its run folder under ``runs_dir``.

    Steps: convert the config to the learner's ``ToyConfig``; build the learner from
    ``make_rng(seed)``; create the run folder with the *resolved* config (every default written
    out, ADR-032), ``extra_meta`` in its ``meta.json``, and the dataset; then :func:`run` for
    ``training.epochs`` units.
    """
    toy = to_toy_config(config)
    learner = ToyLearner(toy, make_rng(toy.seed))
    store = RunStore.create(
        runs_dir, learner.name, toy.tag, resolved(config), extra_meta=extra_meta
    )
    store.save_extra("dataset", x=learner.dataset.x, y=learner.dataset.y)
    start_loss = learner.loss()
    status = run(learner, store, toy.epochs)
    metrics = store.read_metrics()
    finite = [m["loss"] for m in metrics if m["loss"] is not None]
    return ToyRunResult(store, status, len(metrics), start_loss, finite[-1] if finite else None)


_SEED_PART = re.compile(r"^(\d+)(?:-(\d+))?$")


def parse_seeds(spec: str) -> list[int]:
    """Seeds from a list such as ``0-4``, ``0,2,7`` or ``0-2,5`` (ranges include both ends).

    Raises ``ValueError`` for an empty list, a reversed range, a repeated seed or anything that
    is not whole numbers >= 0.
    """
    seeds: list[int] = []
    for part in spec.replace(" ", "").split(","):
        match = _SEED_PART.match(part)
        if not match:
            raise ValueError(f"bad seed list {spec!r}: use e.g. 0-4 or 0,2,7")
        first = int(match.group(1))
        last = int(match.group(2)) if match.group(2) is not None else first
        if last < first:
            raise ValueError(f"bad seed range {part!r}: the end is before the start")
        seeds.extend(range(first, last + 1))
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"seed list {spec!r} repeats a seed")
    return seeds


def sweep_seeds(
    config: ToyRunConfig,
    seeds: list[int],
    runs_dir: str | Path,
    when: datetime | None = None,
) -> tuple[str, list[ToyRunResult]]:
    """Train ``config`` once per seed; returns the sweep id and one result per seed.

    The sweep id is ``<YYYYMMDD-HHMMSS>-toy_<task>-<tag>`` (formed like a run id). Run k gets
    seed k, tag ``<tag>-s<k>`` (so its resolved config re-creates it on its own) and
    ``"sweep": <id>`` in its ``meta.json``. Runs are trained one after another.
    """
    if not seeds:
        raise ValueError("need at least one seed")
    when = when or datetime.now(UTC)
    sweep_id = f"{when:%Y%m%d-%H%M%S}-toy_{config.task}-{config.tag}"
    results = []
    for seed in seeds:
        one = parse_config(resolved(config), seed=seed, tag=f"{config.tag}-s{seed}")
        results.append(train_toy_run(one, runs_dir, extra_meta={"sweep": sweep_id}))
    return sweep_id, results
