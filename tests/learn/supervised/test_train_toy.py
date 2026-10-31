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


# --- task 4.6a: optimizers, minibatches, validation split ---


def phase3_epoch(net, x, y, lr):
    """The Phase 3 update, written out by hand: full batch, MSE, θ ← θ − η ∂L/∂θ."""
    from neurodrive.core.losses import mse_grad
    from neurodrive.core.nn.backward import backward
    from neurodrive.core.nn.forward import forward
    from neurodrive.core.nn.network import Network

    trace = forward(net, x)
    grads = backward(net, trace, mse_grad(trace.output, y))
    new_w = [w - lr * g for w, g in zip(net.weights, grads.weights, strict=True)]
    new_b = [b - lr * g for b, g in zip(net.biases, grads.biases, strict=True)]
    return Network(net.spec, new_w, new_b)


@pytest.mark.parametrize("task", ["sine", "xor"])
def test_defaults_reproduce_the_phase3_update_bit_for_bit(task):
    learner = ToyLearner(ToyConfig.for_task(task), make_rng(0))
    net = learner.network
    for _ in range(30):
        learner.train_unit()
        net = phase3_epoch(net, learner.dataset.x, learner.dataset.y, learner.config.learning_rate)
        np.testing.assert_array_equal(learner.network.flatten(), net.flatten())


@pytest.mark.parametrize(
    "optimizer,lr",
    [("sgd", 0.05), ("momentum", 0.05), ("rmsprop", 0.005), ("adam", 0.01)],
)
def test_every_optimizer_lowers_the_sine_loss(optimizer, lr):
    config = ToyConfig.for_task("sine", optimizer=optimizer, learning_rate=lr)
    learner = ToyLearner(config, make_rng(0))
    start = learner.loss()
    results = train(learner, 300)
    assert not results[-1].diverged
    assert results[-1].metrics["loss"] < 0.5 * start


class CountingOptimizer:
    """Wraps an optimizer and counts its steps."""

    def __init__(self, inner):
        self.inner, self.steps = inner, 0

    def step(self, params, grads):
        self.steps += 1
        return self.inner.step(params, grads)


@pytest.mark.parametrize(
    "batch_size,val_fraction,expected",
    [(32, 0.0, 4), (32, 0.25, 3), (50, 0.0, 3), (1000, 0.0, 1), (None, 0.25, 1), (1, 0.0, 128)],
)
def test_an_epoch_takes_ceil_n_over_b_steps(batch_size, val_fraction, expected):
    config = ToyConfig.for_task("sine", batch_size=batch_size, val_fraction=val_fraction)
    learner = ToyLearner(config, make_rng(0))  # sine has N = 128 examples
    counter = CountingOptimizer(learner.optimizer)
    learner.optimizer = counter
    learner.train_unit()
    assert counter.steps == expected


def test_split_sizes_and_val_loss_only_with_a_split():
    plain = ToyLearner(ToyConfig.for_task("sine"), make_rng(0))
    assert plain.x_val is None and plain.val_loss() is None
    assert "val_loss" not in plain.train_unit().metrics
    split = ToyLearner(ToyConfig.for_task("sine", val_fraction=0.25), make_rng(0))
    assert split.x_train.shape == (96, 1) and split.x_val.shape == (32, 1)
    metrics = split.train_unit().metrics
    assert metrics["val_loss"] == pytest.approx(split.val_loss())
    assert metrics["loss"] == pytest.approx(split.loss())


def test_minibatch_runs_are_seeded():
    def run(seed: int):
        config = ToyConfig.for_task(
            "sine",
            seed=seed,
            optimizer="adam",
            learning_rate=0.01,
            batch_size=16,
            val_fraction=0.25,
        )
        learner = ToyLearner(config, make_rng(seed))
        results = train(learner, 20)
        return [r.metrics["val_loss"] for r in results], learner.network.flatten()

    (va, pa), (vb, pb), (vc, _) = run(0), run(0), run(1)
    assert va == vb
    np.testing.assert_array_equal(pa, pb)
    assert va != vc


def test_batches_and_split_do_not_change_the_starting_weights():
    a = ToyLearner(ToyConfig.for_task("sine"), make_rng(0))
    b = ToyLearner(ToyConfig.for_task("sine", batch_size=8, val_fraction=0.5), make_rng(0))
    np.testing.assert_array_equal(a.network.flatten(), b.network.flatten())
    np.testing.assert_array_equal(a.dataset.x, b.dataset.x)


def test_divergence_with_minibatches_keeps_the_last_epoch_network():
    config = ToyConfig.for_task("sine", learning_rate=50.0, batch_size=16)
    learner = ToyLearner(config, make_rng(0))
    for _ in range(200):
        before = learner.network.flatten()
        result = learner.train_unit()
        if result.diverged:
            break
    assert result.diverged
    np.testing.assert_array_equal(learner.network.flatten(), before)


def test_new_fields_are_validated_and_old_dicts_still_load():
    with pytest.raises(ValueError, match="unknown optimizer"):
        ToyConfig.for_task("sine", optimizer="adma")
    with pytest.raises(ValueError, match="no setting"):
        ToyConfig.for_task("sine", optimizer="sgd", optimizer_settings={"beta1": 0.9})
    with pytest.raises(ValueError, match="batch_size"):
        ToyConfig.for_task("sine", batch_size=0)
    with pytest.raises(ValueError, match="val_fraction"):
        ToyConfig.for_task("sine", val_fraction=1.0)
    old = ToyConfig.for_task("xor").to_dict()
    for key in ("optimizer", "optimizer_settings", "batch_size", "val_fraction"):
        del old[key]  # a Phase 3 run folder's config.yaml
    assert ToyConfig.from_dict(old) == ToyConfig.for_task("xor")
    adam = ToyConfig.for_task("xor", optimizer="adam", optimizer_settings={"beta1": 0.8})
    assert ToyConfig.from_dict(adam.to_dict()) == adam
