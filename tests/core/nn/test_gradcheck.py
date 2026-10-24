"""Tests for core/nn/gradcheck.py: backprop passes the gradient check everywhere, and the checker
catches broken backprop. The most important test in the project (docs/ROADMAP.md task 3.5)."""

import re

import numpy as np
import pytest

import neurodrive.core.nn.backward as backward_module
import neurodrive.core.nn.gradcheck as gradcheck_module
from neurodrive.core.activations import ACTIVATIONS
from neurodrive.core.losses import huber, huber_grad
from neurodrive.core.nn.backward import NetworkGradients
from neurodrive.core.nn.gradcheck import (
    gradient_check,
    numeric_gradient,
    parameter_name,
    relative_error,
)
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng

ARCHITECTURES = [(2, 1), (3, 4, 2), (7, 16, 16, 2), (7, 8, 2), (2, 3, 3, 3, 1)]
TOL = 1e-6


def hand_network() -> Network:
    """The 2-2-1 example of tasks 2.2 and 3.4: tanh hidden layer, linear output."""
    spec = NetworkSpec((2, 2, 1), hidden_activation="tanh", output_activation="linear")
    return Network(spec, [[[0.5, -1.0], [1.0, 0.0]], [[2.0, 1.0]]], [[0.0, 0.5], [0.1]])


def random_case(spec: NetworkSpec, seed: int, batch: int = 6):
    net = Network.unflatten(spec, make_rng(seed).normal(size=spec.n_parameters))
    rng = make_rng(1000 + seed)
    x = rng.normal(size=(batch, spec.n_inputs))
    y = rng.normal(size=(batch, spec.layer_sizes[-1]))
    return net, x, y


# --- The sweep: every activation pair, every architecture, several seeds ---


@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("sizes", ARCHITECTURES, ids=str)
@pytest.mark.parametrize("out_act", sorted(ACTIVATIONS))
@pytest.mark.parametrize("hidden_act", sorted(ACTIVATIONS))
def test_backprop_passes_gradient_check(hidden_act, out_act, sizes, seed):
    spec = NetworkSpec(sizes, hidden_activation=hidden_act, output_activation=out_act)
    net, x, y = random_case(spec, seed)
    result = gradient_check(net, x, y)
    assert result.relu_margin > 1e-4  # no finite difference straddles a ReLU corner
    assert result.relative_error < TOL, (result.relative_error, result.worst_name)
    assert result.input_relative_error < TOL, result.input_relative_error
    assert result.passed()


@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("sizes", [(3, 4, 2), (7, 16, 16, 2)], ids=str)
@pytest.mark.parametrize("hidden_act", sorted(ACTIVATIONS))
def test_backprop_passes_gradient_check_with_huber(hidden_act, sizes, seed):
    spec = NetworkSpec(sizes, hidden_activation=hidden_act, output_activation="linear")
    net, x, y = random_case(spec, seed)
    y = 3.0 * y  # large misses, so both pieces of Huber are used
    result = gradient_check(net, x, y, loss=huber, loss_grad=huber_grad)
    assert result.passed(), (result.relative_error, result.input_relative_error)


def test_single_observation_is_checked_too():
    net, x, y = random_case(NetworkSpec((7, 16, 16, 2)), 0)
    result = gradient_check(net, x[0], y[0])
    assert result.input_analytic.shape == (7,) and result.passed()


# --- Hand examples from the Brief ---


def test_hand_network_numbers():
    result = gradient_check(hand_network(), [[1.0, 2.0]], [[0.0]])
    # genome entry 6 is W2[0,0] = 2: n = (0.6482782860 - 0.6482491349) / 0.00002
    assert result.numeric[6] == pytest.approx(1.4575571, abs=1e-7)
    assert result.analytic[6] == pytest.approx(1.4575571, abs=1e-7)
    assert result.numeric[0] == pytest.approx(-0.5819825, abs=1e-7)
    assert result.relative_error < 1e-9 and result.passed()
    np.testing.assert_allclose(result.input_numeric, [[-0.581983, 0.581983]], atol=1e-6)
    assert result.relu_margin == float("inf")  # no ReLU layer


def test_relative_error_hand_example():
    # ‖[0, -0.001]‖ / (√5 + √(1 + 2.001²)) = 0.001 / 4.4730
    assert relative_error([1.0, 2.0], [1.0, 2.001]) == pytest.approx(2.2356e-4, rel=1e-4)


def test_relative_error_edge_cases():
    assert relative_error([0.0, 0.0], [0.0, 0.0]) == 0.0
    assert relative_error([1.0, -2.0], [1.0, -2.0]) == 0.0
    assert relative_error([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(1.0)  # opposite signs
    assert relative_error([5.0], [0.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="shape"):
        relative_error([1.0, 2.0], [1.0])


# --- The checker catches broken backprop ---


def _scaled_layer_backward(layer: int, factor: float):
    """A wrong backward: one layer's weight gradient multiplied by ``factor``."""
    real = backward_module.backward

    def broken(network, trace, grad_output) -> NetworkGradients:
        g = real(network, trace, grad_output)
        weights = list(g.weights)
        weights[layer] = weights[layer] * factor
        return NetworkGradients(weights, g.biases, g.inputs, g.deltas)

    return broken


def test_checker_catches_a_one_percent_error_and_names_the_layer(monkeypatch):
    net, x, y = random_case(NetworkSpec((3, 4, 4, 2)), 0)
    assert gradient_check(net, x, y).passed()
    monkeypatch.setattr(gradcheck_module, "backward", _scaled_layer_backward(1, 1.01))
    result = gradient_check(net, x, y)
    assert not result.passed()
    assert result.relative_error > 1e-4
    assert result.worst_name.startswith("W2[")  # layer index 1 is W2 in the docs' numbering
    assert result.input_relative_error < TOL  # the input gradient was not touched


def test_checker_catches_a_forgotten_activation_slope(monkeypatch):
    net, x, y = random_case(NetworkSpec((3, 4, 2), hidden_activation="tanh"), 1)
    monkeypatch.setattr(backward_module, "get_derivative", lambda name: np.ones_like)
    result = gradient_check(net, x, y)
    assert not result.passed()
    assert result.relative_error > 1e-2 and result.input_relative_error > 1e-2


# --- Building blocks ---


def test_numeric_gradient_is_exact_on_a_quadratic():
    c, d = np.array([[1.0, -2.0], [0.5, 3.0]]), np.array([[0.3, 0.0], [-1.0, 2.0]])
    x = np.array([[0.7, -1.2], [2.0, 0.1]])
    x_before = x.copy()
    grad = numeric_gradient(lambda v: float(np.sum(c * v * v + d * v)), x)
    np.testing.assert_allclose(grad, 2 * c * x + d, atol=1e-8)
    np.testing.assert_array_equal(x, x_before)  # input not changed


def test_parameter_names_for_the_hand_network():
    spec = hand_network().spec
    names = [parameter_name(spec, k) for k in range(spec.n_parameters)]
    assert names == [
        "W1[0,0]", "W1[0,1]", "W1[1,0]", "W1[1,1]", "b1[0]", "b1[1]", "W2[0,0]", "W2[0,1]", "b2[0]"
    ]  # fmt: skip
    with pytest.raises(ValueError):
        parameter_name(spec, 9)


def test_parameter_names_point_at_the_right_genome_entry():
    spec = NetworkSpec((7, 16, 16, 2))
    net = Network.unflatten(spec, np.arange(spec.n_parameters, dtype=np.float64))
    for k in range(spec.n_parameters):
        m = re.fullmatch(r"([Wb])(\d+)\[(\d+)(?:,(\d+))?\]", parameter_name(spec, k))
        assert m is not None
        layer = int(m.group(2)) - 1
        if m.group(1) == "W":
            assert net.weights[layer][int(m.group(3)), int(m.group(4))] == k
        else:
            assert net.biases[layer][int(m.group(3))] == k


def test_check_is_deterministic_and_leaves_the_network_unchanged():
    net, x, y = random_case(NetworkSpec((7, 8, 2)), 2)
    before = net.flatten()
    a, b = gradient_check(net, x, y), gradient_check(net, x, y)
    np.testing.assert_array_equal(a.numeric, b.numeric)
    np.testing.assert_array_equal(a.analytic, b.analytic)
    assert a.relative_error == b.relative_error
    np.testing.assert_array_equal(net.flatten(), before)


def test_relu_margin_is_reported():
    spec = NetworkSpec((2, 2, 1), hidden_activation="relu", output_activation="linear")
    net = Network(spec, [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0]]], [[0.0, 0.0], [0.0]])
    result = gradient_check(net, [[0.3, -0.05]], [[0.0]])
    assert result.relu_margin == pytest.approx(0.05)


def test_wrong_target_shape_is_refused():
    with pytest.raises(ValueError, match="shape"):
        gradient_check(hand_network(), [[1.0, 2.0]], [0.0])
