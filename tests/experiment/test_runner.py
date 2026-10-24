"""Tests for experiment/runner.py: the v0 loop fills a valid run folder and stops correctly."""

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.experiment.runner import checkpoint_schedule, run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.base import UnitResult
from neurodrive.learn.supervised.train_toy import ToyConfig, ToyLearner


def toy_run(tmp_path, tag: str = "t", **overrides) -> tuple[RunStore, str, ToyLearner]:
    config = ToyConfig.for_task(overrides.pop("task", "xor"), tag=tag, **overrides)
    learner = ToyLearner(config, make_rng(config.seed))
    store = RunStore.create(tmp_path, learner.name, config.tag, config.to_dict())
    store.save_extra("dataset", x=learner.dataset.x, y=learner.dataset.y)
    return store, run(learner, store, config.epochs), learner


def test_checkpoint_schedule():
    assert checkpoint_schedule(3000) == [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 3000]
    assert checkpoint_schedule(1) == [0, 1]
    assert checkpoint_schedule(0) == [0]
    assert checkpoint_schedule(7) == [0, 1, 2, 5, 7]
    with pytest.raises(ValueError):
        checkpoint_schedule(-1)


def test_xor_run_folder_is_complete_and_consistent(tmp_path):
    store, status, learner = toy_run(tmp_path, epochs=500)
    assert status == "completed" and store.read_meta()["status"] == "completed"
    metrics = store.read_metrics()
    assert [m["unit"] for m in metrics] == list(range(1, 501))
    assert all(m["env_steps_total"] == 0 and m["diverged"] is False for m in metrics)
    times = [m["wall_clock_s"] for m in metrics]
    assert times == sorted(times)
    assert store.checkpoint_units() == checkpoint_schedule(500)
    final = store.load_checkpoint(500)
    assert final.equals(learner.network)
    assert learner.loss(final) == pytest.approx(metrics[-1]["loss"], abs=0)  # same network
    assert store.read_config() == learner.config.to_dict()
    assert ToyConfig.from_dict(store.read_config()) == learner.config


def test_checkpoint_zero_is_the_untrained_network(tmp_path):
    store, _, _ = toy_run(tmp_path, epochs=5)
    config = ToyConfig.from_dict(store.read_config())
    fresh = ToyLearner(config, make_rng(config.seed))
    assert store.load_checkpoint(0).equals(fresh.network)
    assert fresh.loss() == store.read_metrics()[0]["loss_before"]  # unit 0's loss


def test_full_xor_run_reaches_loss_below_0_01(tmp_path):
    store, status, _ = toy_run(tmp_path)
    assert status == "completed" and store.read_metrics()[-1]["loss"] < 0.01


def test_diverging_run_stops_and_keeps_finite_checkpoints(tmp_path):
    store, status, _ = toy_run(tmp_path, task="sine", learning_rate=50.0)
    assert status == "diverged" and store.read_meta()["status"] == "diverged"
    metrics = store.read_metrics()
    assert metrics[-1]["diverged"] is True and metrics[-1]["loss"] is None
    assert len(metrics) < 3000
    last = store.checkpoint_units()[-1]
    assert last == len(metrics) - 1  # the last completed unit is saved
    for unit in store.checkpoint_units():
        assert np.all(np.isfinite(store.load_checkpoint(unit).flatten()))


def test_same_config_and_seed_give_identical_runs(tmp_path):
    a, _, _ = toy_run(tmp_path, tag="a", task="sine", epochs=200)
    b, _, _ = toy_run(tmp_path, tag="b", task="sine", epochs=200)
    c, _, _ = toy_run(tmp_path, tag="c", task="sine", epochs=200, seed=1)

    def strip(store):  # wall-clock time is the only thing allowed to differ
        return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]

    assert strip(a) == strip(b) and strip(a) != strip(c)
    assert a.checkpoint_units() == b.checkpoint_units()
    for unit in a.checkpoint_units():
        np.testing.assert_array_equal(
            a.load_checkpoint(unit).flatten(), b.load_checkpoint(unit).flatten()
        )
    np.testing.assert_array_equal(a.load_extra("dataset")["x"], b.load_extra("dataset")["x"])


class Exploding:
    """A learner whose third unit raises an error."""

    name, feedback_kind = "boom", "none"

    def __init__(self, network):
        self.network, self.units = network, 0

    def train_unit(self) -> UnitResult:
        self.units += 1
        if self.units == 3:
            raise RuntimeError("boom")
        return UnitResult(metrics={"loss": 1.0}, env_steps=10)

    def current_networks(self):
        return [self.network]


def test_an_exception_marks_the_run_failed(tmp_path):
    learner = ToyLearner(ToyConfig.for_task("xor"), make_rng(0))
    store = RunStore.create(tmp_path, "boom", "t", {})
    with pytest.raises(RuntimeError, match="boom"):
        run(Exploding(learner.network), store, 10)
    assert store.read_meta()["status"] == "failed"
    assert [m["env_steps_total"] for m in store.read_metrics()] == [10, 20]  # budgets add up
