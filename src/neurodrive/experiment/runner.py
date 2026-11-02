"""Runner v1 (ADR-030, ADR-037): the loop that trains a learner and fills its run folder.

The learner decides *how* to learn; the runner decides *how long*, *what to record* and *when
to stop* (docs/ARCHITECTURE.md §6.12). Each unit, in this order:

1. ``learner.train_unit()``; add its budget increments (environment steps, demonstration
   frames, teacher queries) to the running totals;
2. append one metrics line (unit, totals, training wall-clock seconds, the learner's metrics);
   stop with status "diverged" if the unit diverged;
3. if the unit is on the checkpoint schedule (0, 1, 2, 5, 10, 20, 50, ...): save a checkpoint
   and, when exams are on and the network drives (7 inputs, 2 outputs), take the Driving Exam
   and save the card;
4. save the learner's state (a :class:`~neurodrive.learn.base.Resumable` learner), last;
5. stop as soon as any :class:`Budget` limit is reached. A unit is never cut in half, so a run
   can overshoot by at most one unit: 300 steps per unit and a 1,000-step limit stop after
   unit 4, at 1,200 steps.

The last completed unit always gets a checkpoint (and an exam). Exam time is measured
separately and kept out of the training clock and the budget.

**Resuming** (:func:`resume`): the saved state says which unit U it belongs to; everything later
(metrics lines, checkpoints, exam cards) is cut away, the learner is restored, random generators
included, and the loop continues with unit U + 1. Because the state is saved last, a kill at any
point loses at most the unit in progress, and that unit is recomputed with the same random
numbers: the run continues exactly as if it had never stopped, except for wall-clock times.

``train_toy_run`` starts a whole toy run from a validated config (``nd train``, task 4.6b);
``sweep_seeds`` runs one config once per seed (``nd sweep``, task 4.7a, ADR-033).
"""

import re
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neurodrive.core.nn.network import Network
from neurodrive.core.rng import make_rng
from neurodrive.env.car import ACTION_SIZE
from neurodrive.env.policies import MLPPolicy
from neurodrive.env.sensors import OBSERVATION_SIZE
from neurodrive.experiment.config import ToyRunConfig, parse_config, resolved, to_toy_config
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, run_exam
from neurodrive.experiment.store import RunStore
from neurodrive.learn.base import Learner, Resumable
from neurodrive.learn.supervised.train_toy import ToyLearner

_TOTALS = ("env_steps_total", "demo_frames_total", "teacher_queries_total")


def checkpoint_schedule(units: int) -> list[int]:
    """Units to checkpoint: 0, then 1, 2, 5, 10, 20, 50, 100, ... up to ``units``, plus ``units``.

    Dense at the start, where learning changes the network fastest, sparse later; about
    3 checkpoints per factor of 10. E.g. 3000 units → 0, 1, 2, 5, 10, 20, 50, 100, 200, 500,
    1000, 2000, 3000.
    """
    if units < 0:
        raise ValueError(f"units must be >= 0, got {units}")
    return sorted({u for u in range(units + 1) if on_checkpoint_schedule(u)} | {units})


def on_checkpoint_schedule(unit: int) -> bool:
    """True for 0 and for 1, 2, 5 times a power of 10 (the schedule without a known end)."""
    if unit == 0:
        return True
    while unit % 10 == 0:
        unit //= 10
    return unit in (1, 2, 5)


@dataclass(frozen=True)
class Budget:
    """When a run stops: after this many units, environment steps or training seconds, whichever
    comes first. None means no limit on that currency; at least one limit is needed."""

    units: int | None = None
    env_steps: int | None = None
    wall_clock_s: float | None = None

    def __post_init__(self) -> None:
        limits = (self.units, self.env_steps, self.wall_clock_s)
        if all(limit is None for limit in limits):
            raise ValueError("a budget needs at least one limit")
        if any(limit is not None and limit < 0 for limit in limits):
            raise ValueError("budget limits must not be negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def reached(self, unit: int, totals: dict[str, int], wall_clock_s: float) -> str | None:
        """The first limit reached ("units", "env_steps" or "wall_clock"), or None."""
        if self.units is not None and unit >= self.units:
            return "units"
        if self.env_steps is not None and totals["env_steps_total"] >= self.env_steps:
            return "env_steps"
        if self.wall_clock_s is not None and wall_clock_s >= self.wall_clock_s:
            return "wall_clock"
        return None


def drives(network: Network) -> bool:
    """True when ``network`` can drive: 7 observation inputs and 2 action outputs."""
    return network.spec.n_inputs == OBSERVATION_SIZE and network.spec.n_outputs == ACTION_SIZE


def run(
    learner: Learner,
    store: RunStore,
    budget: Budget | int,
    exams: bool = True,
    tracks_dir: str | Path = DEFAULT_TRACKS_DIR,
) -> str:
    """Train ``learner`` until ``budget`` is used up, writing everything to ``store``.

    ``budget`` may be a plain number of units. ``exams`` turns the Driving Exam at checkpoints on
    or off (it only ever runs for networks that drive). Returns the final status: "completed", or
    "diverged" when a unit reports divergence (the last finite network is then checkpointed at
    the last completed unit).
    """
    budget = Budget(units=budget) if isinstance(budget, int) else budget
    store.update_meta(
        budget=budget.to_dict(), exams=exams, stop_reason=None, resumed_at=[], exam_wall_clock_s=0.0
    )
    totals = dict.fromkeys(_TOTALS, 0)
    return _loop(learner, store, budget, exams, tracks_dir, 0, totals, 0.0, 0.0)


def resume(learner: Learner, store: RunStore, tracks_dir: str | Path = DEFAULT_TRACKS_DIR) -> str:
    """Continue a killed (status "running") or failed run of ``store`` exactly where it stopped.

    ``learner`` must be built exactly as for the original run (same config and seed): with no
    saved state yet it *is* the unit-0 learner; otherwise its ``restore`` replaces its state with
    the saved one. The budget and the exams setting come from ``meta.json``.
    """
    meta = store.read_meta()
    if meta.get("run_version", 0) < 1:
        raise ValueError("only run folders of version 1 can be resumed")
    if meta["status"] not in ("running", "failed"):
        raise ValueError(f"the run is already {meta['status']}; nothing to resume")
    if not isinstance(learner, Resumable):
        raise ValueError(f"learner {learner.name!r} cannot save its state, so it cannot resume")
    saved = store.load_state()
    if saved is None:
        unit, totals, clock, exam_clock = 0, dict.fromkeys(_TOTALS, 0), 0.0, 0.0
    else:
        state, info = saved
        learner.restore(state)
        unit, totals = info["unit"], info["totals"]
        clock, exam_clock = info["wall_clock_s"], info["exam_wall_clock_s"]
    store.cut_back(unit)
    store.update_meta(
        status="running",
        ended_at=None,
        resumed_at=[*meta["resumed_at"], datetime.now(UTC).isoformat(timespec="seconds")],
    )
    budget = Budget(**meta["budget"])
    return _loop(learner, store, budget, meta["exams"], tracks_dir, unit, totals, clock, exam_clock)


def _loop(
    learner: Learner,
    store: RunStore,
    budget: Budget,
    exams: bool,
    tracks_dir: str | Path,
    unit: int,
    totals: dict[str, int],
    clock_offset: float,
    exam_clock: float,
) -> str:
    """The unit loop shared by :func:`run` (from unit 0) and :func:`resume` (from unit U)."""
    started = time.perf_counter()
    exam_here = 0.0  # exam seconds spent in this session (kept out of the training clock)

    def wall_clock() -> float:
        return clock_offset + time.perf_counter() - started - exam_here

    def checkpoint(at: int) -> None:
        nonlocal exam_here
        network = learner.current_networks()[0]
        store.save_checkpoint(at, network)
        if exams and drives(network):
            t0 = time.perf_counter()
            ref = {"kind": "checkpoint", "run": store.path.name, "unit": at}
            store.save_exam(at, run_exam(lambda world: MLPPolicy(network), ref, tracks_dir))
            exam_here += time.perf_counter() - t0

    def save_state(at: int) -> None:
        if isinstance(learner, Resumable):
            runner = {
                "unit": at,
                "totals": totals,
                "wall_clock_s": wall_clock(),
                "exam_wall_clock_s": exam_clock + exam_here,
            }
            store.save_state(learner.state(), runner)

    status, reason = "completed", None
    last_done = unit
    last_saved = max((u for u in store.checkpoint_units() if u <= unit), default=-1)
    try:
        if unit == 0:
            checkpoint(0)  # the network before any learning (again, after a very early kill)
            last_saved = 0
        while (reason := budget.reached(unit, totals, wall_clock())) is None:
            unit += 1
            result = learner.train_unit()
            totals["env_steps_total"] += result.env_steps
            totals["demo_frames_total"] += result.demo_frames
            totals["teacher_queries_total"] += result.teacher_queries
            record = {"unit": unit, **totals, "wall_clock_s": wall_clock()}
            store.append_metrics({**record, **result.metrics, "diverged": result.diverged})
            if result.diverged:
                status, reason = "diverged", "diverged"
                break
            last_done = unit
            if on_checkpoint_schedule(unit):
                checkpoint(unit)
                last_saved = unit
            save_state(unit)
        if last_saved != last_done:
            checkpoint(last_done)
    except BaseException:
        store.update_meta(exam_wall_clock_s=exam_clock + exam_here)
        store.finish("failed", "error")
        raise
    store.update_meta(exam_wall_clock_s=exam_clock + exam_here)
    store.finish(status, reason)
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
    status = run(learner, store, Budget(units=toy.epochs), exams=False)
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
