"""Tests for learn/supervised/train_toy.py: plain gradient descent learns the toy tasks."""

import re
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.nn.forward import predict
from neurodrive.core.rng import make_rng
from neurodrive.learn.base import Learner, UnitResult
from neurodrive.learn.supervised.toy_tasks import ToyDataset
from neurodrive.learn.supervised.train_toy import DEFAULTS, ToyConfig, ToyLearner

LEARN_ROOT = Path(__file__).resolve().parents[3] / "src" / "neurodrive" / "learn"


def train(learner: ToyLearner, epochs: int | None = None) -> list[UnitResult]:
    return [learner.train_unit() for _ in range(epochs or learner.config.epochs)]


def hand_learner() -> ToyLearner:
    """The Brief's example: y = w x + b from w = b = 0, x = [0, 1, 2], y = [1, 3, 5], η = 0.1."""
    config = ToyConfig.for_task("line", init_scheme="zeros", learning_rate=0.1)
    data = ToyDataset("line", np.array([[0.0], [1.0], [2.0]]), np.array([[1.0], [3.0], [5.0]]), "")
    return ToyLearner(config, make_rng(0), dataset=data)


def test_hand_example_two_epochs():
    learner = hand_learner()
    first = learner.train_unit()
    # loss (1 + 9 + 25) / 3; dL/dw = -8.6667, dL/db = -6; w = 0.8667, b = 0.6
    assert first.metrics["loss_before"] == pytest.approx(11.666667, abs=1e-6)
    assert learner.network.weights[0][0, 0] == pytest.approx(0.866667, abs=1e-6)
    assert learner.network.biases[0][0] == pytest.approx(0.6, abs=1e-12)
    assert first.metrics["loss"] == pytest.approx(3.207407, abs=1e-6)
    assert first.metrics["grad_norm"][0] == pytest.approx(np.hypot(8.666667, 6.0), abs=1e-5)
    second = learner.train_unit()
    assert second.metrics["loss_before"] == first.metrics["loss"]
    assert learner.network.weights[0][0, 0] == pytest.approx(1.324444, abs=1e-6)
    assert learner.network.biases[0][0] == pytest.approx(0.906667, abs=1e-6)
    assert second.metrics["loss"] == pytest.approx(0.895440, abs=1e-6)


def test_line_fit_recovers_slope_and_intercept():
    learner = ToyLearner(ToyConfig.for_task("line"), make_rng(0))
    train(learner)
    assert learner.network.weights[0][0, 0] == pytest.approx(2.0, abs=1e-3)
    assert learner.network.biases[0][0] == pytest.approx(-0.5, abs=1e-3)


@pytest.mark.parametrize("seed", range(5))
def test_xor_is_solved(seed):
    learner = ToyLearner(ToyConfig.for_task("xor", seed=seed), make_rng(seed))
    results = train(learner)
    assert results[-1].metrics["loss"] < 0.01
    out = predict(learner.network, learner.dataset.x)[:, 0]  # outputs round to the labels
    np.testing.assert_array_equal(np.round(out), [0, 1, 1, 0])


def test_sine_loss_falls_below_0_01():
    learner = ToyLearner(ToyConfig.for_task("sine"), make_rng(0))
    start = learner.loss()
    results = train(learner)
    assert start > 0.3 and results[-1].metrics["loss"] < 0.01


def test_loss_mostly_falls_and_reports_gradient_sizes():
    learner = ToyLearner(ToyConfig.for_task("xor"), make_rng(1))
    results = train(learner)
    losses = [r.metrics["loss"] for r in results]
    assert losses[-1] < losses[0] / 10
    assert all(len(r.metrics["grad_norm"]) == 2 for r in results)
    pairs = zip(results[:-1], results[1:], strict=True)
    assert all(r.metrics["loss_before"] == p.metrics["loss"] for p, r in pairs)


def test_too_large_learning_rate_diverges_cleanly():
    learner = ToyLearner(ToyConfig.for_task("sine", learning_rate=50.0), make_rng(0))
    for _ in range(200):
        result = learner.train_unit()
        if result.diverged:
            break
    assert result.diverged, "expected divergence within 200 epochs"
    assert result.metrics["loss"] is None
    assert np.all(np.isfinite(learner.network.flatten()))  # the last finite network is kept


def test_same_seed_is_bit_identical_and_other_seed_differs():
    def losses_and_params(seed: int):
        learner = ToyLearner(ToyConfig.for_task("sine", epochs=50, seed=seed), make_rng(seed))
        return [r.metrics["loss"] for r in train(learner)], learner.network.flatten()

    (la, pa), (lb, pb), (lc, _) = losses_and_params(0), losses_and_params(0), losses_and_params(1)
    assert la == lb
    np.testing.assert_array_equal(pa, pb)
    assert la != lc


def test_data_and_initialisation_use_separate_streams():
    # changing the data size must not change the starting weights
    a = ToyLearner(ToyConfig.for_task("sine"), make_rng(0))
    small = ToyDataset("sine", np.zeros((3, 1)), np.zeros((3, 1)), "")
    b = ToyLearner(ToyConfig.for_task("sine"), make_rng(0), dataset=small)
    np.testing.assert_array_equal(a.network.flatten(), b.network.flatten())


def test_follows_the_learner_protocol():
    learner = ToyLearner(ToyConfig.for_task("xor"), make_rng(0))
    assert isinstance(learner, Learner)
    assert learner.name == "toy_xor" and learner.feedback_kind == "labels"
    assert learner.current_networks() == [learner.network]
    result = learner.train_unit()
    assert (result.env_steps, result.demo_frames, result.teacher_queries) == (0, 0, 0)


@pytest.mark.parametrize("task", sorted(DEFAULTS))
def test_defaults_fit_their_task_and_round_trip(task):
    config = ToyConfig.for_task(task)
    ToyLearner(config, make_rng(0)).train_unit()  # shapes fit the data
    assert ToyConfig.from_dict(config.to_dict()) == config


def test_bad_configs_are_refused():
    with pytest.raises(ValueError, match="toy task"):
        ToyConfig("circle", (1, 1))
    with pytest.raises(ValueError, match="learning_rate"):
        ToyConfig.for_task("line", learning_rate=0.0)
    with pytest.raises(ValueError, match="epochs"):
        ToyConfig.for_task("line", epochs=0)
    with pytest.raises(ValueError, match="loss"):
        ToyConfig.for_task("line", loss="hinge")
    with pytest.raises(ValueError, match="do not fit"):
        ToyLearner(ToyConfig.for_task("xor", layer_sizes=(3, 4, 1)), make_rng(0))


def test_learn_never_imports_experiment_or_server():
    pattern = re.compile(r"^\s*(from|import)\s+neurodrive\.(experiment|server)\b", re.MULTILINE)
    offenders = [p.name for p in LEARN_ROOT.rglob("*.py") if pattern.search(p.read_text("utf-8"))]
    assert offenders == []
