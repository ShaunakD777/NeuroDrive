"""Tests for experiment/toy_view.py: the toy lab's numbers equal the network's own maths."""

import numpy as np
import pytest

from neurodrive.core.losses import mse_grad
from neurodrive.core.nn.backward import backward
from neurodrive.core.nn.forward import forward, predict
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.experiment.runner import run
from neurodrive.experiment.store import RunStore
from neurodrive.experiment.toy_view import GRID_1D, GRID_2D, MARGIN, toy_checkpoint_view
from neurodrive.learn.supervised.train_toy import ToyConfig, ToyLearner


def toy_run(tmp_path, task: str, epochs: int) -> RunStore:
    config = ToyConfig.for_task(task, epochs=epochs)
    learner = ToyLearner(config, make_rng(config.seed))
    store = RunStore.create(tmp_path, learner.name, config.tag, config.to_dict())
    store.save_extra("dataset", x=learner.dataset.x, y=learner.dataset.y)
    run(learner, store, epochs)
    return store


def test_hand_network_glow_matches_the_brief(tmp_path):
    # The 2-2-1 network of task 3.4 with x = [1, 2], target 0, stored as a one-example toy run.
    spec = NetworkSpec((2, 2, 1), hidden_activation="tanh", output_activation="linear")
    net = Network(spec, [[[0.5, -1.0], [1.0, 0.0]], [[2.0, 1.0]]], [[0.0, 0.5], [0.1]])
    config = ToyConfig("xor", (2, 2, 1), "tanh", "linear")
    store = RunStore.create(tmp_path, "toy_xor", "hand", config.to_dict())
    store.save_extra("dataset", x=np.array([[1.0, 2.0]]), y=np.array([[0.0]]))
    store.save_checkpoint(0, net)
    view = toy_checkpoint_view(store, 0)
    assert view.max_gradient == pytest.approx(1.457557, abs=1e-6)
    np.testing.assert_allclose(view.glow[1], [[1.0, 1.0]], atol=1e-9)
    np.testing.assert_allclose(view.glow[0], [[0.399, 0.799], [0.2, 0.399]], atol=1e-3)
    assert view.loss == pytest.approx(0.648264, abs=1e-6)


@pytest.mark.parametrize("task", ["sine", "xor"])
def test_view_equals_forward_and_backward(tmp_path, task):
    store = toy_run(tmp_path, task, epochs=20)
    for unit in store.checkpoint_units():
        view = toy_checkpoint_view(store, unit)
        net = store.load_checkpoint(unit)
        x, y = view.data_x, view.data_y.reshape(-1, 1)
        trace = forward(net, x)
        grads = backward(net, trace, mse_grad(trace.output, y))
        for got, want in zip(view.weight_gradients, grads.weights, strict=True):
            np.testing.assert_array_equal(got, want)
        for got, want in zip(view.bias_gradients, grads.biases, strict=True):
            np.testing.assert_array_equal(got, want)
        assert view.glow[0].max() <= 1.0 and max(g.max() for g in view.glow) == 1.0


def test_loss_matches_the_training_metrics(tmp_path):
    store = toy_run(tmp_path, "sine", epochs=20)
    metrics = store.read_metrics()
    assert toy_checkpoint_view(store, 0).loss == metrics[0]["loss_before"]  # untrained network
    for unit in store.checkpoint_units()[1:]:
        assert toy_checkpoint_view(store, unit).loss == metrics[unit - 1]["loss"]


def test_one_input_curve(tmp_path):
    store = toy_run(tmp_path, "sine", epochs=5)
    view = toy_checkpoint_view(store, 5)
    assert view.grid_y is None and view.prediction_map is None
    assert view.grid_x.shape == (GRID_1D,) and view.prediction_curve.shape == (GRID_1D,)
    lo, hi = view.data_x.min(), view.data_x.max()
    assert view.grid_x[0] == pytest.approx(lo - MARGIN * (hi - lo))
    assert view.grid_x[-1] == pytest.approx(hi + MARGIN * (hi - lo))
    want = predict(store.load_checkpoint(5), view.grid_x.reshape(-1, 1))[:, 0]
    np.testing.assert_array_equal(view.prediction_curve, want)


def test_two_input_map(tmp_path):
    store = toy_run(tmp_path, "xor", epochs=5)
    view = toy_checkpoint_view(store, 5)
    assert view.prediction_curve is None
    assert view.prediction_map.shape == (GRID_2D, GRID_2D)
    net = store.load_checkpoint(5)
    i, k = 7, 30  # row i: x2 = grid_y[i]; column k: x1 = grid_x[k]
    point = [[view.grid_x[k], view.grid_y[i]]]
    assert view.prediction_map[i, k] == pytest.approx(predict(net, point)[0, 0], abs=1e-12)
    np.testing.assert_array_equal(view.data_x, [[0, 0], [0, 1], [1, 0], [1, 1]])


def test_predictions_improve_as_training_goes_on(tmp_path):
    store = toy_run(tmp_path, "sine", epochs=500)
    losses = [toy_checkpoint_view(store, u).loss for u in store.checkpoint_units()]
    assert losses[-1] < losses[0] / 10


def test_missing_checkpoint(tmp_path):
    store = toy_run(tmp_path, "xor", epochs=2)
    with pytest.raises(KeyError):
        toy_checkpoint_view(store, 7)
