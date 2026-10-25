"""Guards for Study S2b (docs/studies/S2b.md): its measurements mean what the report says, and
its failure cases still behave as described."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.learn.supervised.train_toy import ToyLearner

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def study():
    spec = importlib.util.spec_from_file_location("study_s2b", ROOT / "scripts" / "study_s2b.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_dead_relu_fraction_by_hand(study):
    # Hidden neuron 0 has z = x0 + x1 >= 0 for these inputs; neuron 1 has z = -(x0 + x1) - 1 < 0
    # for all of them, so it is dead: 1 of 2 hidden neurons.
    spec = NetworkSpec((2, 2, 1), hidden_activation="relu", output_activation="linear")
    net = Network(spec, [[[1.0, 1.0], [-1.0, -1.0]], [[1.0, 1.0]]], [[0.0, -1.0], [0.0]])
    x = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    assert study.hidden_stats(net, x) == {"saturated": None, "dead": 0.5}


def test_saturation_fraction_by_hand(study):
    # sigmoid'(z) < 0.1 * 0.25 when |z| > about 3.66: z = 0 and 10 for the one hidden neuron.
    spec = NetworkSpec((1, 1, 1), hidden_activation="sigmoid", output_activation="linear")
    net = Network(spec, [[[10.0]], [[1.0]]], [[0.0], [0.0]])
    stats = study.hidden_stats(net, np.array([[0.0], [1.0]]))
    assert stats == {"saturated": 0.5, "dead": None}


def test_a_study_run_is_a_plain_toy_learner_run(study, monkeypatch):
    monkeypatch.setitem(study.TASKS["sine"], "epochs", 20)
    result = study.run_one(("sine", 3, "tanh", 0.03, 2))
    learner = ToyLearner(study.make_config("sine", 3, "tanh", 0.03, 2), make_rng(2))
    losses = [learner.train_unit().metrics["loss"] for _ in range(20)]
    assert result["final_loss"] == losses[-1]
    assert learner.network.spec.layer_sizes == (1, 16, 16, 16, 1)
    first = result["grad_norm_start"]
    assert result["rho"] == first[0] / first[-1]


def test_rho_does_not_depend_on_the_learning_rate(study, monkeypatch):
    monkeypatch.setitem(study.TASKS["xor"], "epochs", 3)
    a = study.run_one(("xor", 5, "sigmoid", 0.1, 0))
    b = study.run_one(("xor", 5, "sigmoid", 3.0, 0))
    assert a["grad_norm_start"] == b["grad_norm_start"] and a["rho"] == b["rho"]


def test_deep_sigmoid_case_is_stuck_predicting_one_half(study):
    result = study.run_one(("xor", 5, "sigmoid", 3.0, 0))
    assert result["final_loss"] == pytest.approx(0.25, abs=1e-4) and not result["solved"]
    assert result["rho"] < 1e-2  # vanishing at the start
    grads = result["grad_norm_start"]
    assert all(a < b for a, b in zip(grads, grads[1:], strict=False))  # weaker towards the input


def test_relu_case_ends_with_a_dead_last_hidden_layer(study):
    learner = ToyLearner(study.make_config("xor", 5, "relu", 1.0, 4), make_rng(4))
    for _ in range(3000):
        learner.train_unit()
    trace = forward(learner.network, learner.dataset.x)
    assert np.all(trace.weighted_sums[-2] <= 0)  # all 8 neurons of hidden layer 5 are off
    np.testing.assert_allclose(trace.output[:, 0], 0.5)  # so the output is a constant 0.5


def test_tanh_contrast_case_solves_xor(study):
    result = study.run_one(("xor", 5, "tanh", 3.0, 1))
    assert result["solved"] and result["epochs_to_threshold"] < 150


def test_bootstrap_and_quartiles(study):
    a = [1.0, 2.0, 3.0, 4.0, 5.0]
    same = study.bootstrap_median_difference(a, a)
    assert same["difference"] == 0 and same["ci95"] == [0.0, 0.0]
    shifted = study.bootstrap_median_difference([v + 2 for v in a], a)
    assert shifted["difference"] == 2 and shifted["ci95"] == [2.0, 2.0]  # paired: always +2
    assert study.quartiles([1.0, 2.0, math.inf])[0] == 2.0
    assert study.quartiles([1.0, math.inf, math.inf])[0] == math.inf
