"""Tests for core/optim: the SGD step, the optimizer contract checks and the network helpers."""

import numpy as np
import pytest

from neurodrive.core.losses import mse, mse_grad
from neurodrive.core.nn.backward import backward
from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.optim.base import grads_of, network_from_params, params_of
from neurodrive.core.optim.sgd import SGD
from neurodrive.core.rng import make_rng


def quadratic_walk(lr: float, steps: int) -> list[float]:
    """Minimise f(w) = (w - 3)^2 from w = 0; returns w after each step."""
    opt, w, out = SGD(lr), [np.array([0.0])], []
    for _ in range(steps):
        w = opt.step(w, [2.0 * (w[0] - 3.0)])
        out.append(float(w[0][0]))
    return out


def test_roadmap_example_first_two_steps():
    first, second = quadratic_walk(0.1, 2)
    assert first == pytest.approx(0.6, abs=1e-12)
    assert second == pytest.approx(1.08, abs=1e-12)


def test_converges_to_three():
    assert quadratic_walk(0.1, 50)[-1] == pytest.approx(3.0, abs=1e-4)


def test_distance_shrinks_by_one_minus_two_eta():
    walk = quadratic_walk(0.1, 5)
    for n, w in enumerate(walk, start=1):
        assert 3.0 - w == pytest.approx(3.0 * 0.8**n, rel=1e-9)


def test_too_high_learning_rate_diverges():
    walk = quadratic_walk(1.1, 20)  # factor 1 - 2*1.1 = -1.2 per step
    assert abs(walk[-1] - 3.0) > abs(walk[0] - 3.0) * 10


def test_hand_update_on_several_arrays():
    params = [np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([0.5, -0.5])]
    grads = [np.array([[1.0, -1.0], [0.0, 2.0]]), np.array([1.0, 1.0])]
    new = SGD(0.5).step(params, grads)
    assert np.array_equal(new[0], [[0.5, 2.5], [3.0, 3.0]])
    assert np.array_equal(new[1], [0.0, -1.0])


def test_inputs_not_modified_and_results_are_new_arrays():
    p, g = np.array([1.0, 2.0]), np.array([0.5, 0.5])
    p0, g0 = p.copy(), g.copy()
    (new,) = SGD(0.1).step([p], [g])
    assert np.array_equal(p, p0) and np.array_equal(g, g0)
    assert new is not p


def test_zero_gradient_leaves_parameters_unchanged():
    p = np.array([1.0, -2.0, 3.0])
    (new,) = SGD(0.3).step([p], [np.zeros(3)])
    assert np.array_equal(new, p)


def test_length_and_shape_mismatch_rejected():
    opt = SGD(0.1)
    with pytest.raises(ValueError):
        opt.step([np.zeros(2)], [np.zeros(2), np.zeros(2)])
    with pytest.raises(ValueError):
        opt.step([np.zeros(2)], [np.zeros(3)])


@pytest.mark.parametrize("bad", [0.0, -0.1, float("nan"), float("inf")])
def test_bad_learning_rate_rejected(bad):
    with pytest.raises(ValueError):
        SGD(bad)


def make_net_and_grads():
    spec = NetworkSpec((2, 3, 1), "tanh", "linear")
    net = init_network(spec, make_rng(0))
    x = make_rng(1).uniform(-1, 1, size=(5, 2))
    y = make_rng(2).uniform(-1, 1, size=(5, 1))
    trace = forward(net, x)
    return net, backward(net, trace, mse_grad(trace.output, y)), x, y


def test_params_round_trip_and_order_matches_flatten():
    net, grads, _, _ = make_net_and_grads()
    params = params_of(net)
    assert len(params) == 4
    assert np.array_equal(np.concatenate([p.ravel() for p in params]), net.flatten())
    assert np.array_equal(np.concatenate([g.ravel() for g in grads_of(grads)]), grads.flatten())
    assert network_from_params(net, params).equals(net)
    with pytest.raises(ValueError):
        network_from_params(net, params[:3])


def test_sgd_step_on_network_equals_plain_update_and_lowers_loss():
    net, grads, x, y = make_net_and_grads()
    lr = 0.1
    stepped = network_from_params(net, SGD(lr).step(params_of(net), grads_of(grads)))
    by_hand = [w - lr * g for w, g in zip(net.weights, grads.weights, strict=True)]
    by_hand_b = [b - lr * g for b, g in zip(net.biases, grads.biases, strict=True)]
    for w, h in zip(stepped.weights, by_hand, strict=True):
        assert np.array_equal(w, h)
    for b, h in zip(stepped.biases, by_hand_b, strict=True):
        assert np.array_equal(b, h)
    assert mse(forward(stepped, x).output, y) < mse(forward(net, x).output, y)
