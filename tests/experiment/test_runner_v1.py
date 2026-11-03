"""Tests for runner v1 and run folder v1 (task 5.5): budgets, exams at checkpoints, and a run
killed at different moments that resumes exactly as if it had never stopped."""

from dataclasses import replace

import numpy as np
import pytest
from walk_learner import KILLS, SEED, UNITS, WalkLearner, run_killed

from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.experiment.runner import Budget, on_checkpoint_schedule, resume, run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.base import Resumable, UnitResult

SCHEDULE = [0, 1, 2, 5, 10, 12]  # 0, 1, 2, 5, 10 from the schedule, and the last unit, 12


def walk_run(tmp_path, budget, exams=True, **learner_kwargs) -> RunStore:
    store = RunStore.create(tmp_path, "walk", "t", {"seed": SEED})
    run(WalkLearner(SEED, **learner_kwargs), store, budget, exams=exams)
    return store


@pytest.fixture(scope="module")
def reference(tmp_path_factory) -> RunStore:
    """The uninterrupted 12-unit run with exams that every resumed run must equal."""
    return walk_run(tmp_path_factory.mktemp("reference"), Budget(units=UNITS))


def same_run(a: RunStore, b: RunStore) -> None:
    """Everything except wall-clock times and the run's own name is identical."""

    def metrics(store):
        return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]

    assert metrics(a) == metrics(b)
    assert a.checkpoint_units() == b.checkpoint_units() == SCHEDULE
    for unit in SCHEDULE:
        assert a.load_checkpoint(unit).equals(b.load_checkpoint(unit))
    assert a.exam_units() == b.exam_units() == SCHEDULE
    for unit in SCHEDULE:
        card_a, card_b = a.load_exam(unit).to_dict(), b.load_exam(unit).to_dict()
        assert card_a.pop("policy")["unit"] == card_b.pop("policy")["unit"] == unit
        assert card_a == card_b
    (state_a, runner_a), (state_b, runner_b) = a.load_state(), b.load_state()
    assert runner_a["unit"] == runner_b["unit"] == UNITS
    assert runner_a["totals"] == runner_b["totals"]
    assert np.array_equal(state_a.arrays["genome"], state_b.arrays["genome"])
    assert state_a.info == state_b.info  # the generators ended in the same state


# ── a fresh run fills a valid v1 folder ─────────────────────────────────────────────────────


def test_schedule_without_a_known_end():
    assert [u for u in range(2001) if on_checkpoint_schedule(u)] == [
        0,
        1,
        2,
        5,
        10,
        20,
        50,
        100,
        200,
        500,
        1000,
        2000,
    ]


def test_a_run_folder_v1_is_complete(reference):
    meta = reference.read_meta()
    assert meta["run_version"] == 1 and meta["status"] == "completed"
    assert meta["stop_reason"] == "units" and meta["resumed_at"] == [] and meta["exams"] is True
    assert meta["budget"] == {"units": 12, "env_steps": None, "wall_clock_s": None}
    assert meta["exam_wall_clock_s"] > 0
    metrics = reference.read_metrics()
    assert [m["unit"] for m in metrics] == list(range(1, 13))
    assert [m["env_steps_total"] for m in metrics] == [300 * u for u in range(1, 13)]
    assert reference.checkpoint_units() == reference.exam_units() == SCHEDULE
    card = reference.load_exam(5)
    assert card.policy == {"kind": "checkpoint", "run": reference.path.name, "unit": 5}
    assert len(card.tracks) == 5  # a real Driving Exam of the unit-5 network
    assert (reference.path / "state" / "latest.npz").is_file()
    assert not (reference.path / "state" / "latest.tmp.npz").exists()
    _, runner = reference.load_state()
    assert runner["unit"] == 12 and runner["totals"]["env_steps_total"] == 3600


def test_env_step_budget_overshoots_by_at_most_one_unit(tmp_path):
    store = walk_run(tmp_path, Budget(env_steps=1000), exams=False)
    # 300 steps a unit: 900 after unit 3 (go on), 1,200 after unit 4 (stop).
    assert [m["env_steps_total"] for m in store.read_metrics()] == [300, 600, 900, 1200]
    assert store.read_meta()["stop_reason"] == "env_steps"
    assert store.checkpoint_units() == [0, 1, 2, 4]  # the last unit is always checkpointed
    assert store.exam_units() == []  # exams off


def test_wall_clock_budget_stops_on_training_time(tmp_path):
    store = walk_run(tmp_path, Budget(units=1000, wall_clock_s=0.3), exams=False, sleep_s=0.05)
    metrics = store.read_metrics()
    assert store.read_meta()["stop_reason"] == "wall_clock"
    assert 2 <= len(metrics) < 1000 and metrics[-1]["wall_clock_s"] >= 0.3
    assert metrics[-2]["wall_clock_s"] < 0.3  # stopped after the first unit that reached it


def test_budgets_need_a_limit_and_no_negatives():
    for bad in ({}, {"units": -1}, {"env_steps": -5}):
        with pytest.raises(ValueError):
            Budget(**bad)


# ── killed and resumed: exactly the same run ────────────────────────────────────────────────


@pytest.mark.parametrize("kill", list(KILLS))
def test_a_killed_run_resumes_exactly(kill, reference, tmp_path):
    store = run_killed(tmp_path, kill)
    assert store.read_meta()["status"] == "running"  # killed: nobody marked it finished
    unit, _ = KILLS[kill]
    assert max(store.checkpoint_units()) < unit or kill == "before_state"
    status = resume(WalkLearner(SEED), store)
    assert status == "completed"
    meta = store.read_meta()
    assert meta["stop_reason"] == "units" and len(meta["resumed_at"]) == 1
    assert not (store.path / "state" / "latest.tmp.npz").exists()
    same_run(store, reference)


def test_resume_refuses_finished_old_and_stateless_runs(reference, tmp_path):
    with pytest.raises(ValueError, match="already completed"):
        resume(WalkLearner(SEED), reference)
    old = RunStore.create(tmp_path, "walk", "old", {})
    old.update_meta(run_version=0)
    with pytest.raises(ValueError, match="version 1"):
        resume(WalkLearner(SEED), old)
    killed = RunStore.create(tmp_path, "walk", "nostate", {})
    with pytest.raises(ValueError, match="cannot resume"):
        resume(Stateless(), killed)


class Stateless:
    """A learner without state() / restore(): it trains, but cannot be resumed."""

    name, feedback_kind = "stateless", "none"

    def __init__(self):
        self.network = init_network(NetworkSpec(), make_rng(0))

    def train_unit(self) -> UnitResult:
        return UnitResult(metrics={}, env_steps=1)

    def current_networks(self):
        return [self.network]


def test_a_learner_without_state_still_trains_but_saves_none(tmp_path):
    assert isinstance(WalkLearner(SEED), Resumable) and not isinstance(Stateless(), Resumable)
    store = RunStore.create(tmp_path, "stateless", "t", {})
    assert run(Stateless(), store, Budget(units=3), exams=False) == "completed"
    assert len(store.read_metrics()) == 3 and store.load_state() is None


# ── a stop request (task 5.8, ADR-041) ──────────────────────────────────────────────────────


class StopsAt(WalkLearner):
    """A walk learner that, during unit ``at``, asks its own run to stop (as the server would)."""

    def __init__(self, store: RunStore, at: int) -> None:
        super().__init__(SEED)
        self.store, self.at = store, at

    def train_unit(self) -> UnitResult:
        result = super().train_unit()
        if self.units == self.at:
            self.store.request_stop()
        return result


def test_a_stop_request_ends_the_run_cleanly_after_the_current_unit(reference, tmp_path):
    store = RunStore.create(tmp_path, "walk", "t", {"seed": SEED})
    assert run(StopsAt(store, 3), store, Budget(units=UNITS)) == "completed"
    meta = store.read_meta()
    assert (meta["status"], meta["stop_reason"]) == ("completed", "stopped")
    lines = [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]
    first_three = [
        {k: v for k, v in m.items() if k != "wall_clock_s"} for m in reference.read_metrics()[:3]
    ]
    assert lines == first_three  # unit 3 finished, no unit 4 started
    assert store.checkpoint_units() == [0, 1, 2, 3] == store.exam_units()  # 3: the last unit
    assert not store.stop_requested()  # answered and removed


def test_resume_ignores_an_old_stop_request(reference, tmp_path):
    store = run_killed(tmp_path, "during_unit")
    store.request_stop()  # e.g. asked for while the process was already dead
    assert resume(WalkLearner(SEED), store) == "completed"
    assert store.read_meta()["stop_reason"] == "units"
    same_run(store, reference)


# ── new bests get a checkpoint and an exam (task 5.9, ADR-042) ──────────────────────────────


class NewBestAt(WalkLearner):
    """A walk learner that reports a new best network in units 3 and 7."""

    def train_unit(self) -> UnitResult:
        return replace(super().train_unit(), new_best=self.units in (3, 7))


def test_units_with_a_new_best_are_checkpointed_and_examined_too(reference, tmp_path):
    store = RunStore.create(tmp_path, "walk", "t", {"seed": SEED})
    run(NewBestAt(SEED), store, Budget(units=UNITS))
    assert store.checkpoint_units() == [0, 1, 2, 3, 5, 7, 10, 12] == store.exam_units()
    for unit in SCHEDULE:  # the same learner, so the scheduled checkpoints are unchanged
        assert store.load_checkpoint(unit).equals(reference.load_checkpoint(unit))
    lines = [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]
    assert lines == [
        {k: v for k, v in m.items() if k != "wall_clock_s"} for m in reference.read_metrics()
    ]
