"""Tests for explain/sensitivity.py: exact slopes by backprop, checked by central differences."""

import numpy as np
import pytest

import neurodrive.explain.sensitivity as sensitivity_module
from neurodrive.core.activations import ACTIVATIONS
from neurodrive.core.nn.forward import forward, predict
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.explain.sensitivity import sensitivity, sensitivity_finite_difference

T = np.tanh(1.5)
DT = 1.0 - T * T  # tanh'(1.5) = tanh'(-1.5) = 0.180707...


def hand_network() -> Network:
    spec = NetworkSpec((2, 2, 1), hidden_activation="tanh", output_activation="linear")
    return Network(spec, [[[0.5, -1.0], [1.0, 0.0]], [[2.0, 1.0]]], [[0.0, 0.5], [0.1]])


def test_hand_example_matches_the_chain_rule_by_hand():
    s = sensitivity(hand_network(), [1.0, 2.0])
    assert s.shape == (1, 2)
    # d out/d x0 = 2*tanh'(z1_0)*0.5 + 1*tanh'(z1_1)*1 ; d out/d x1 = 2*tanh'(z1_0)*(-1) + 0
    assert DT == pytest.approx(0.180707, abs=1e-6)
    np.testing.assert_allclose(s[0], [2 * DT * 0.5 + DT, -2 * DT], atol=1e-8)
    np.testing.assert_allclose(s[0], [0.361414, -0.361414], atol=1e-6)


def test_one_layer_linear_network_sensitivity_equals_its_weights():
    rng = make_rng(0)
    spec = NetworkSpec((7, 2), output_activation="linear")
    net = Network(spec, [rng.uniform(-2, 2, (2, 7))], [rng.uniform(-1, 1, 2)])
    for obs in (np.zeros(7), rng.uniform(0, 1, 7), rng.uniform(-5, 5, 7)):
        np.testing.assert_allclose(sensitivity(net, obs), net.weights[0], atol=1e-8)


def test_two_layer_linear_network_sensitivity_equals_the_product_of_its_weights():
    rng = make_rng(1)
    spec = NetworkSpec((7, 5, 2), hidden_activation="linear", output_activation="linear")
    net = Network(
        spec, [rng.uniform(-1, 1, (5, 7)), rng.uniform(-1, 1, (2, 5))], [np.zeros(5), np.zeros(2)]
    )
    np.testing.assert_allclose(
        sensitivity(net, rng.uniform(0, 1, 7)), net.weights[1] @ net.weights[0], atol=1e-8
    )


def test_sensitivity_predicts_a_small_real_nudge_on_a_driving_network():
    rng = make_rng(2)
    net = init_network(NetworkSpec(), rng)
    obs = rng.uniform(0, 1, 7)
    s = sensitivity(net, obs)
    assert s.shape == (2, 7)
    base = predict(net, obs)
    for k in range(7):
        nudged = obs.copy()
        nudged[k] += 0.01
        np.testing.assert_allclose(predict(net, nudged) - base, 0.01 * s[:, k], atol=5e-5)


def test_finite_difference_smaller_step_gives_the_same_answer():
    net = init_network(NetworkSpec(), make_rng(3))
    obs = make_rng(4).uniform(0, 1, 7)
    np.testing.assert_allclose(
        sensitivity_finite_difference(net, obs, step=1e-3),
        sensitivity_finite_difference(net, obs, step=1e-5),
        atol=1e-6,
    )


def test_saturated_neurons_hide_the_inputs():
    # Huge first-layer weights push every hidden tanh to -1 or +1, where its slope is about 0:
    # small input changes then cannot reach the output at all.
    rng = make_rng(5)
    net = init_network(NetworkSpec(), rng)
    calm = sensitivity(net, rng.uniform(0, 1, 7))
    net.weights[0] *= 200.0
    net.biases[0][:] = 50.0
    saturated = sensitivity(net, make_rng(6).uniform(0, 1, 7))
    assert np.abs(calm).max() > 0.05
    assert np.abs(saturated).max() < 1e-6


def test_layer_argument_measures_hidden_neurons():
    net = hand_network()
    hidden = sensitivity(net, [1.0, 2.0], layer=1)  # the two tanh neurons
    np.testing.assert_allclose(hidden, DT * net.weights[0], atol=1e-8)  # tanh'(z) * W row by row
    np.testing.assert_allclose(sensitivity(net, [1.0, 2.0], layer=0), np.eye(2), atol=1e-8)
    np.testing.assert_allclose(sensitivity(net, [1.0, 2.0], layer=2), sensitivity(net, [1.0, 2.0]))


def test_does_not_change_the_network_and_is_deterministic():
    net = init_network(NetworkSpec(), make_rng(7))
    before = net.flatten()
    obs = np.full(7, 0.5)
    a, b = sensitivity(net, obs), sensitivity(net, obs)
    np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(net.flatten(), before)
    assert np.all(obs == 0.5)


def test_bad_arguments_rejected():
    net = hand_network()
    with pytest.raises(ValueError, match="observation"):
        sensitivity(net, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="observation"):
        sensitivity(net, [[1.0, 2.0]])
    for step in (0.0, -1e-4, float("nan")):
        with pytest.raises(ValueError, match="step"):
            sensitivity_finite_difference(net, [1.0, 2.0], step=step)
    for f in (sensitivity, sensitivity_finite_difference):
        with pytest.raises(IndexError):
            f(net, [1.0, 2.0], layer=3)
        with pytest.raises(IndexError):
            f(net, [1.0, 2.0], layer=-4)
        with pytest.raises(ValueError, match="observation"):
            f(net, [1.0, 2.0, 3.0])


def test_matches_the_trace_based_formula_for_one_hidden_layer():
    # For a [n, h, m] network with tanh hidden and linear output the slope is
    # W2 @ diag(1 - a1^2) @ W1; a cross-check written with the trace's own numbers.
    rng = make_rng(8)
    spec = NetworkSpec((7, 6, 2), hidden_activation="tanh", output_activation="linear")
    net = init_network(spec, rng)
    obs = rng.uniform(0, 1, 7)
    a1 = forward(net, obs).activations[1]
    exact = net.weights[1] @ np.diag(1.0 - a1**2) @ net.weights[0]
    np.testing.assert_allclose(sensitivity(net, obs), exact, atol=1e-8)


# --- Backprop (task 3.6) against the finite-difference oracle ---


def test_hand_example_is_exact_with_backprop():
    # J = δ1 W1 with δ1 = [2, 1] * tanh'(±1.5) = [0.361413, 0.180707]
    s = sensitivity(hand_network(), [1.0, 2.0])
    np.testing.assert_allclose(s, [[2 * DT * 0.5 + DT, -2 * DT]], atol=1e-12)
    np.testing.assert_allclose(s, [[0.361413, -0.361413]], atol=1e-6)
    oracle = sensitivity_finite_difference(hand_network(), [1.0, 2.0])
    assert 0 < np.abs(s - oracle).max() < 1e-8  # the oracle's own truncation error


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("sizes", [(7, 16, 16, 2), (7, 8, 2), (3, 5, 4, 3, 2)], ids=str)
@pytest.mark.parametrize("out_act", sorted(ACTIVATIONS))
@pytest.mark.parametrize("hidden_act", sorted(ACTIVATIONS))
def test_backprop_agrees_with_finite_differences_at_every_layer(hidden_act, out_act, sizes, seed):
    spec = NetworkSpec(sizes, hidden_activation=hidden_act, output_activation=out_act)
    rng = make_rng(seed)
    net = init_network(spec, rng)
    obs = rng.uniform(0, 1, sizes[0])
    trace = forward(net, obs)
    # A finite difference across a ReLU corner averages two slopes, so the oracle is only valid
    # when no nudge x ± h e_k switches any ReLU neuron on or off.
    nudges = sensitivity_module.DEFAULT_STEP * np.eye(sizes[0])
    nudged = forward(net, np.concatenate([obs + nudges, obs - nudges]))
    for layer, z in enumerate(trace.weighted_sums):
        if spec.activation_name(layer) == "relu":
            assert np.array_equal(
                nudged.weighted_sums[layer] > 0, np.broadcast_to(z > 0, (2 * sizes[0], z.size))
            )
    for layer in range(-1, len(sizes)):
        exact = sensitivity(net, obs, layer=layer)
        measured = sensitivity_finite_difference(net, obs, layer=layer)
        assert exact.shape == measured.shape == (trace.activations[layer].size, sizes[0])
        np.testing.assert_allclose(exact, measured, atol=1e-5)


def test_exact_at_relu_corner_where_finite_differences_average():
    # hidden neuron 0 sits exactly on its corner (z = 0): backprop uses slope 0 there,
    # the finite difference reports the average of 0 and 1
    spec = NetworkSpec((2, 2, 1), hidden_activation="relu", output_activation="linear")
    net = Network(spec, [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0]]], [[0.0, 0.0], [0.0]])
    np.testing.assert_allclose(sensitivity(net, [0.0, 2.0]), [[0.0, 1.0]], atol=1e-12)
    np.testing.assert_allclose(
        sensitivity_finite_difference(net, [0.0, 2.0]), [[0.5, 1.0]], atol=1e-9
    )


def test_one_forward_and_one_backward_pass(monkeypatch):
    calls = {"forward": 0, "backward": 0}
    real_forward, real_backward = sensitivity_module.forward, sensitivity_module.backward

    def counting_forward(*args):
        calls["forward"] += 1
        return real_forward(*args)

    def counting_backward(*args):
        calls["backward"] += 1
        return real_backward(*args)

    monkeypatch.setattr(sensitivity_module, "forward", counting_forward)
    monkeypatch.setattr(sensitivity_module, "backward", counting_backward)
    sensitivity(init_network(NetworkSpec(), make_rng(0)), np.full(7, 0.5))
    assert calls == {"forward": 1, "backward": 1}


def test_loss_gradient_for_inputs_is_upstream_gradient_times_sensitivity():
    # task 3.4: the loss gradient for x was [-0.581983, 0.581983] with g = -1.610297
    s = sensitivity(hand_network(), [1.0, 2.0])
    np.testing.assert_allclose(-1.610297 * s[0], [-0.581983, 0.581983], atol=1e-6)
