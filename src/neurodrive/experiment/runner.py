"""Runner v1 (ADR-030, ADR-037): the loop that trains a learner and fills its run folder.

The learner decides *how* to learn; the runner decides *how long*, *what to record* and *when
to stop* (docs/ARCHITECTURE.md §6.12). Each unit, in this order:

1. ``learner.train_unit()``; add its budget increments (environment steps, demonstration
   frames, teacher queries) to the running totals;
2. append one metrics line (unit, totals, training wall-clock seconds, the learner's metrics);
   stop with status "diverged" if the unit diverged;
3. if the unit is on the checkpoint schedule (0, 1, 2, 5, 10, 20, 50, ...), or the learner
   reports a new best network in it (``UnitResult.new_best``, task 5.9, ADR-042), or it is a
   multiple of ``checkpoint_every`` k when one is set (task 6.3, ADR-047; k = 5 adds 15, 25,
   ...): save a checkpoint and, when exams are on and the network drives (7 inputs, 2 outputs),
   take the Driving Exam and save the card;
4. save the learner's state (a :class:`~neurodrive.learn.base.Resumable` learner), last in
   the folder;
5. copy the unit into the database mirror ``<runs dir>/neurodrive.db`` (task 5.5b, ADR-038):
   one transaction with its metrics rows and any checkpoint and exam card just saved. The
   folder is written first and is the master copy; a mirror that cannot be written only warns;
6. stop as soon as any :class:`Budget` limit is reached, or when another process asked the run
   to stop (the file ``stop_requested`` in the run folder, task 5.8, ADR-041: stop reason
   "stopped", status "completed"). A unit is never cut in half, so a run can overshoot by at
   most one unit: 300 steps per unit and a 1,000-step limit stop after unit 4, at 1,200 steps.

The last completed unit always gets a checkpoint (and an exam). Exam time and the mirror's
time are measured separately and kept out of the training clock and the budget.

**Resuming** (:func:`resume`): the saved state says which unit U it belongs to; everything later
(metrics lines, checkpoints, exam cards) is cut away, the learner is restored, random generators
included, and the loop continues with unit U + 1. Because the state is saved last, a kill at any
point loses at most the unit in progress, and that unit is recomputed with the same random
numbers: the run continues exactly as if it had never stopped, except for wall-clock times.
The mirror's rows of that run are then copied again from the cut-back folder.

``train_toy_run`` starts a whole toy run from a validated config (``nd train``, task 4.6b),
``train_random_search_run`` a random-search run (task 5.7), ``train_imitation_run`` a
behavioural-cloning run (task 6.3); ``sweep_seeds`` runs one config once per seed (``nd sweep``,
task 4.7a, ADR-033); ``learner_from_config`` rebuilds the learner a run folder's config
describes, for ``nd resume`` (task 5.7).

An imitation config names its dataset by id or by a ``<teacher>-<tag>`` selector
(:func:`pin_dataset`, ADR-047). The run folder's config names the exact dataset found and its
SHA-256 checksum, so training again from that file, or resuming, uses the very same frames or
refuses to start.
"""

import re
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from neurodrive.core.nn.network import Network
from neurodrive.core.rng import make_rng
from neurodrive.env.car import ACTION_SIZE
from neurodrive.env.policies import MLPPolicy
from neurodrive.env.sensors import OBSERVATION_SIZE
from neurodrive.env.track import Track
from neurodrive.experiment.config import (
    ImitationRunConfig,
    RandomSearchRunConfig,
    RunConfig,
    ToyRunConfig,
    parse_config,
    resolved,
    to_imitation_settings,
    to_random_search_settings,
    to_toy_config,
)
from neurodrive.experiment.db import DB_NAME, Mirror
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, ExamResult, run_exam
from neurodrive.experiment.store import RunStore
from neurodrive.learn.base import Learner, Resumable
from neurodrive.learn.evolution.random_search import RandomSearchLearner
from neurodrive.learn.supervised.demos import (
    DEFAULT_DATASETS_DIR,
    DemoDataset,
    dataset_sha256,
    find_dataset,
)
from neurodrive.learn.supervised.imitation import ImitationLearner
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


def checkpoint_due(unit: int, every: int | None = None) -> bool:
    """True when ``unit`` is on the schedule or, with ``every`` = k, a multiple of k.

    k = 1 gives every unit; k = 5 over 20 units gives 0, 1, 2, 5, 10, 15, 20.
    """
    return on_checkpoint_schedule(unit) or (every is not None and unit % every == 0)


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
    mirror: bool = True,
    checkpoint_every: int | None = None,
) -> str:
    """Train ``learner`` until ``budget`` is used up, writing everything to ``store``.

    ``budget`` may be a plain number of units. ``exams`` turns the Driving Exam at checkpoints on
    or off (it only ever runs for networks that drive). ``mirror`` keeps the database
    ``<runs dir>/neurodrive.db`` up to date as the run goes (ADR-038). ``checkpoint_every`` k
    also checkpoints every k-th unit (None: the schedule alone); it is kept in ``meta.json``, so
    a resumed run keeps it. Returns the final status: "completed", or "diverged" when a unit
    reports divergence (the last finite network is then checkpointed at the last completed unit).
    """
    budget = Budget(units=budget) if isinstance(budget, int) else budget
    if checkpoint_every is not None and checkpoint_every < 1:
        raise ValueError(f"checkpoint_every must be at least 1, got {checkpoint_every}")
    store.update_meta(
        budget=budget.to_dict(),
        exams=exams,
        checkpoint_every=checkpoint_every,
        stop_reason=None,
        resumed_at=[],
        exam_wall_clock_s=0.0,
    )
    totals = dict.fromkeys(_TOTALS, 0)
    db = _mirror_of(store) if mirror else None
    if db is not None:
        db.sync_run(store)
    return _loop(
        learner, store, budget, exams, tracks_dir, 0, totals, 0.0, 0.0, db, checkpoint_every
    )


def _mirror_of(store: RunStore) -> Mirror:
    """The database mirror next to the run folder: ``<runs dir>/neurodrive.db``."""
    return Mirror(store.path.parent / DB_NAME)


def resume(
    learner: Learner,
    store: RunStore,
    tracks_dir: str | Path = DEFAULT_TRACKS_DIR,
    mirror: bool = True,
) -> str:
    """Continue a killed (status "running") or failed run of ``store`` exactly where it stopped.

    ``learner`` must be built exactly as for the original run (same config and seed): with no
    saved state yet it *is* the unit-0 learner; otherwise its ``restore`` replaces its state with
    the saved one. The budget, the exams setting and ``checkpoint_every`` come from
    ``meta.json``. With ``mirror``,
    the database's rows of this run are replaced by the cut-back folder's before continuing.
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
    store.clear_stop_request()  # resuming is a new decision to train; an old request is void
    store.update_meta(
        status="running",
        ended_at=None,
        resumed_at=[*meta["resumed_at"], datetime.now(UTC).isoformat(timespec="seconds")],
    )
    budget = Budget(**meta["budget"])
    db = _mirror_of(store) if mirror else None
    if db is not None:
        db.sync_run(store)
    return _loop(
        learner,
        store,
        budget,
        meta["exams"],
        tracks_dir,
        unit,
        totals,
        clock,
        exam_clock,
        db,
        meta.get("checkpoint_every"),  # absent in runs made before task 6.3
    )


def _stop_reason(
    budget: Budget, store: RunStore, unit: int, totals: dict[str, int], wall_clock_s: float
) -> str | None:
    """Why the run stops before the next unit: a budget limit, a stop request, or None."""
    reason = budget.reached(unit, totals, wall_clock_s)
    if reason is None and store.stop_requested():
        reason = "stopped"
    return reason


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
    db: Mirror | None,
    checkpoint_every: int | None = None,
) -> str:
    """The unit loop shared by :func:`run` (from unit 0) and :func:`resume` (from unit U)."""
    started = time.perf_counter()
    exam_here = 0.0  # exam seconds spent in this session (kept out of the training clock)
    mirror_here = 0.0  # database seconds in this session (also kept out, ADR-038)
    saved: list[int] = []  # checkpoints and exam cards saved since the mirror last copied
    cards: list[tuple[int, ExamResult]] = []

    def mirror(record: dict[str, Any] | None) -> None:
        nonlocal mirror_here
        if db is not None:
            t0 = time.perf_counter()
            db.add_unit(store, record, saved, cards)
            mirror_here += time.perf_counter() - t0
        saved.clear()
        cards.clear()

    def wall_clock() -> float:
        return clock_offset + time.perf_counter() - started - exam_here - mirror_here

    def checkpoint(at: int) -> None:
        nonlocal exam_here
        network = learner.current_networks()[0]
        store.save_checkpoint(at, network)
        saved.append(at)
        if exams and drives(network):
            t0 = time.perf_counter()
            ref = {"kind": "checkpoint", "run": store.path.name, "unit": at}
            card = run_exam(lambda world: MLPPolicy(network), ref, tracks_dir)
            store.save_exam(at, card)
            cards.append((at, card))
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
    # The stop rule reads the clock as recorded in the last metrics line, like the step totals,
    # so whether a run stopped at the right unit can be checked from its folder alone.
    recorded = clock_offset
    try:
        if unit == 0:
            checkpoint(0)  # the network before any learning (again, after a very early kill)
            last_saved = 0
            mirror(None)
        while (reason := _stop_reason(budget, store, unit, totals, recorded)) is None:
            unit += 1
            result = learner.train_unit()
            totals["env_steps_total"] += result.env_steps
            totals["demo_frames_total"] += result.demo_frames
            totals["teacher_queries_total"] += result.teacher_queries
            recorded = wall_clock()
            record = {"unit": unit, **totals, "wall_clock_s": recorded}
            record = {**record, **result.metrics, "diverged": result.diverged}
            store.append_metrics(record)
            if result.diverged:
                status, reason = "diverged", "diverged"
                mirror(record)
                break
            last_done = unit
            if checkpoint_due(unit, checkpoint_every) or result.new_best:
                checkpoint(unit)
                last_saved = unit
            save_state(unit)
            mirror(record)
        if last_saved != last_done:
            checkpoint(last_done)
            mirror(None)
    except BaseException:
        store.update_meta(exam_wall_clock_s=exam_clock + exam_here)
        store.finish("failed", "error")
        if db is not None:
            db.update_run(store)
            db.close()
        raise
    store.update_meta(exam_wall_clock_s=exam_clock + exam_here)
    store.clear_stop_request()  # answered (or the budget ended the run first), before the status
    store.finish(status, reason)
    if db is not None:
        db.update_run(store)
        db.close()
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
    config: ToyRunConfig,
    runs_dir: str | Path,
    extra_meta: dict | None = None,
    on_start: Callable[[RunStore], None] | None = None,
) -> ToyRunResult:
    """Train one toy run described by ``config`` and write its run folder under ``runs_dir``.

    Steps: convert the config to the learner's ``ToyConfig``; build the learner from
    ``make_rng(seed)``; create the run folder with the *resolved* config (every default written
    out, ADR-032), ``extra_meta`` in its ``meta.json``, and the dataset; then :func:`run` for
    ``training.epochs`` units. ``on_start(store)`` is called as soon as the folder exists (``nd
    train`` prints its name then, which is how the server learns the id of a run it launched).
    """
    toy = to_toy_config(config)
    learner = ToyLearner(toy, make_rng(toy.seed))
    store = RunStore.create(
        runs_dir, learner.name, toy.tag, resolved(config), extra_meta=extra_meta
    )
    store.save_extra("dataset", x=learner.dataset.x, y=learner.dataset.y)
    if on_start is not None:
        on_start(store)
    start_loss = learner.loss()
    status = run(learner, store, Budget(units=toy.epochs), exams=False)
    metrics = store.read_metrics()
    finite = [m["loss"] for m in metrics if m["loss"] is not None]
    return ToyRunResult(store, status, len(metrics), start_loss, finite[-1] if finite else None)


@dataclass(frozen=True)
class RandomSearchRunResult:
    """What ``train_random_search_run`` reports: the folder, the status, units, best fitness."""

    store: RunStore
    status: str
    units_done: int
    best_fitness: float | None  # None if no unit finished
    best_unit: int


@dataclass(frozen=True)
class ImitationRunResult:
    """What ``train_imitation_run`` reports: the folder, the status, epochs, the last losses."""

    store: RunStore
    status: str
    epochs_done: int
    end_loss: float | None  # training loss after the last finite epoch; None if none finished
    end_val_loss: float | None  # validation loss then; None without a validation set


def pin_dataset(
    config: ImitationRunConfig, datasets_dir: str | Path = DEFAULT_DATASETS_DIR
) -> tuple[ImitationRunConfig, DemoDataset]:
    """Find and load the dataset an imitation config names; return the config naming it exactly.

    1. :func:`find_dataset` turns ``data.dataset`` (an id or a ``<teacher>-<tag>`` selector) into
       one folder of ``datasets_dir``.
    2. Its ``dataset.npz`` checksum must equal ``data.sha256`` when the config gives one (a run
       folder's config always does); otherwise the frames changed and the run is refused.
    3. The dataset is loaded (``DemoDataset.load`` checks it), and the config is returned with
       ``data.dataset`` set to the folder name and ``data.sha256`` to the checksum.
    """
    folder = find_dataset(datasets_dir, config.data.dataset)
    checksum = dataset_sha256(folder)
    if config.data.sha256 is not None and checksum != config.data.sha256:
        raise ValueError(
            f"dataset {folder.name} has changed: its checksum is {checksum[:12]}..., the config "
            f"expects {config.data.sha256[:12]}...; a run must learn from the same frames"
        )
    data = {**config.data.model_dump(), "dataset": folder.name, "sha256": checksum}
    pinned = config.model_copy(update={"data": type(config.data).model_validate(data)})
    return pinned, DemoDataset.load(folder)


def learner_from_config(
    data: dict[str, Any],
    tracks_dir: str | Path = DEFAULT_TRACKS_DIR,
    datasets_dir: str | Path = DEFAULT_DATASETS_DIR,
) -> Learner:
    """The learner a (resolved) config describes, freshly built from its seed.

    For a run folder's ``config.yaml`` this is the learner exactly as the run started, which is
    what :func:`resume` needs (it then restores the saved state into it). An imitation config's
    dataset is looked up in ``datasets_dir`` (:func:`pin_dataset`).
    """
    config = parse_config(data, source="run config")
    if isinstance(config, RandomSearchRunConfig):
        track = Track.from_json(Path(tracks_dir) / f"{config.fitness.track}.json")
        return RandomSearchLearner(to_random_search_settings(config), track, make_rng(config.seed))
    if isinstance(config, ImitationRunConfig):
        _, dataset = pin_dataset(config, datasets_dir)
        return ImitationLearner(to_imitation_settings(config), dataset, make_rng(config.seed))
    toy = to_toy_config(config)
    return ToyLearner(toy, make_rng(toy.seed))


def train_imitation_run(
    config: ImitationRunConfig,
    runs_dir: str | Path,
    extra_meta: dict | None = None,
    tracks_dir: str | Path = DEFAULT_TRACKS_DIR,
    on_start: Callable[[RunStore], None] | None = None,
    datasets_dir: str | Path = DEFAULT_DATASETS_DIR,
) -> ImitationRunResult:
    """Train one behavioural-cloning run described by ``config`` under ``runs_dir``.

    Steps: pin the dataset (:func:`pin_dataset`); build the learner from ``make_rng(seed)``;
    create the run folder with the pinned, resolved config, and in ``meta.json`` the dataset's
    id, checksum and frame count and the held-back episodes (``on_start(store)`` is called then);
    then :func:`run` for ``training.epochs`` units with the config's exams and
    ``checkpoint_every`` settings.
    """
    pinned, dataset = pin_dataset(config, datasets_dir)
    learner = ImitationLearner(to_imitation_settings(pinned), dataset, make_rng(pinned.seed))
    meta = {
        **(extra_meta or {}),
        "dataset": {
            "id": pinned.data.dataset,
            "sha256": pinned.data.sha256,
            "demo_frames": dataset.n_frames,
            "train_frames": int(learner.x_train.shape[0]),
            "val_episodes": learner.val_episodes,
        },
    }
    store = RunStore.create(runs_dir, learner.name, pinned.tag, resolved(pinned), extra_meta=meta)
    if on_start is not None:
        on_start(store)
    status = run(
        learner,
        store,
        Budget(units=pinned.training.epochs),
        exams=pinned.exams,
        tracks_dir=tracks_dir,
        checkpoint_every=pinned.checkpoint_every,
    )
    metrics = store.read_metrics()
    finite = [m for m in metrics if m["loss"] is not None]
    last = finite[-1] if finite else {}
    return ImitationRunResult(store, status, len(metrics), last.get("loss"), last.get("val_loss"))


def train_random_search_run(
    config: RandomSearchRunConfig,
    runs_dir: str | Path,
    extra_meta: dict | None = None,
    tracks_dir: str | Path = DEFAULT_TRACKS_DIR,
    on_start: Callable[[RunStore], None] | None = None,
) -> RandomSearchRunResult:
    """Train one random-search run described by ``config`` under ``runs_dir``.

    Steps: build the learner from ``make_rng(seed)``; create the run folder with the resolved
    config (``on_start(store)`` is called then, as in :func:`train_toy_run`); then :func:`run`
    with the config's budget and exams setting.
    """
    data = resolved(config)
    learner = learner_from_config(data, tracks_dir)
    store = RunStore.create(runs_dir, learner.name, config.tag, data, extra_meta=extra_meta)
    if on_start is not None:
        on_start(store)
    budget = Budget(**config.budget.model_dump())
    status = run(learner, store, budget, exams=config.exams, tracks_dir=tracks_dir)
    metrics = store.read_metrics()
    last = metrics[-1] if metrics else {}
    return RandomSearchRunResult(
        store, status, len(metrics), last.get("best_fitness"), last.get("best_unit", 0)
    )


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
    config: RunConfig,
    seeds: list[int],
    runs_dir: str | Path,
    when: datetime | None = None,
    study: str | None = None,
    datasets_dir: str | Path = DEFAULT_DATASETS_DIR,
) -> tuple[str, list[ToyRunResult] | list[RandomSearchRunResult] | list[ImitationRunResult]]:
    """Train ``config`` once per seed; returns the sweep id and one result per seed.

    The sweep id is ``<YYYYMMDD-HHMMSS>-<method>-<tag>`` (formed like a run id; the method is
    ``toy_<task>``, ``random_search`` or ``imitation``). Run k gets seed k, tag ``<tag>-s<k>`` (so
    its resolved config re-creates it on its own) and ``"sweep": <id>`` in its ``meta.json``, plus
    ``"study": <study>`` when ``study`` is given (ADR-043). Runs are trained one after another.
    An imitation config's dataset is pinned once, before the first run, so every seed learns from
    the same frames even if a newer dataset is recorded meanwhile.
    """
    if not seeds:
        raise ValueError("need at least one seed")
    when = when or datetime.now(UTC)
    label = f"toy_{config.task}" if isinstance(config, ToyRunConfig) else config.method
    sweep_id = f"{when:%Y%m%d-%H%M%S}-{label}-{config.tag}"
    if isinstance(config, ImitationRunConfig):
        config, _ = pin_dataset(config, datasets_dir)
    results = []
    for seed in seeds:
        one = parse_config(resolved(config), seed=seed, tag=f"{config.tag}-s{seed}")
        meta = {"sweep": sweep_id} if study is None else {"sweep": sweep_id, "study": study}
        if isinstance(one, ToyRunConfig):
            results.append(train_toy_run(one, runs_dir, extra_meta=meta))
        elif isinstance(one, ImitationRunConfig):
            results.append(
                train_imitation_run(one, runs_dir, extra_meta=meta, datasets_dir=datasets_dir)
            )
        else:
            results.append(train_random_search_run(one, runs_dir, extra_meta=meta))
    return sweep_id, results
