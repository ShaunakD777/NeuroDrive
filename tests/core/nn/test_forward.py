"""Tests for core/nn/forward.py: hand-computed example, batching, shapes and input checks."""

import numpy as np
import pytest

from neurodrive.core.nn.forward import forward, predict
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng

T = np.tanh(1.5)  # 0.905148...


def hand_network() -> Network:
    """The Brief's 2-2-1 example: tanh hidden layer, linear output."""
    spec = NetworkSpec((2, 2, 1), hidden_activation="tanh", output_activation="linear")
    return Network(spec, [[[0.5, -1.0], [1.0, 0.0]], [[2.0, 1.0]]], [[0.0, 0.5], [0.1]])


def random_network(spec: NetworkSpec, seed: int = 0) -> Network:
    return Network.unflatten(spec, make_rng(seed).standard_normal(spec.n_parameters))


def test_hand_example_every_intermediate_value():
    trace = forward(hand_network(), [1.0, 2.0])
    np.testing.assert_allclose(trace.weighted_sums[0], [-1.5, 1.5], atol=1e-9)
    np.testing.assert_allclose(trace.activations[1], [-T, T], atol=1e-9)
    np.testing.assert_allclose(trace.weighted_sums[1], [-2 * T + T + 0.1], atol=1e-9)
    np.testing.assert_allclose(trace.output, [-0.805148], atol=1e-6)
    np.testing.assert_array_equal(trace.activations[0], [1.0, 2.0])


def test_hand_example_matches_explicit_loops():
    net = hand_network()
    x = [1.0, 2.0]
    z1 = [sum(net.weights[0][j][i] * x[i] for i in range(2)) + net.biases[0][j] for j in range(2)]
    a1 = [np.tanh(z) for z in z1]
    z2 = sum(net.weights[1][0][j] * a1[j] for j in range(2)) + net.biases[1][0]
    assert predict(net, x)[0] == pytest.approx(z2, abs=1e-9)


def test_trace_lengths_and_shapes_single():
    net = random_network(NetworkSpec())
    trace = forward(net, np.zeros(7))
    assert [a.shape for a in trace.activations] == [(7,), (16,), (16,), (2,)]
    assert [z.shape for z in trace.weighted_sums] == [(16,), (16,), (2,)]


def test_trace_shapes_batch():
    net = random_network(NetworkSpec())
    trace = forward(net, np.zeros((5, 7)))
    assert [a.shape for a in trace.activations] == [(5, 7), (5, 16), (5, 16), (5, 2)]
    assert [z.shape for z in trace.weighted_sums] == [(5, 16), (5, 16), (5, 2)]


@pytest.mark.parametrize(
    "spec",
    [
        NetworkSpec(),
        NetworkSpec((3, 5, 4, 2), "relu", "sigmoid"),
        NetworkSpec((4, 2), "tanh", "linear"),
    ],
)
def test_batch_equals_row_by_row(spec):
    net = random_network(spec, seed=2)
    obs = make_rng(9).uniform(-1, 1, size=(20, spec.n_inputs))
    batch = forward(net, obs)
    for i, row in enumerate(obs):
        single = forward(net, row)
        for a_b, a_s in zip(batch.activations, single.activations, strict=True):
            np.testing.assert_allclose(a_b[i], a_s, atol=1e-9)
        for z_b, z_s in zip(batch.weighted_sums, single.weighted_sums, strict=True):
            np.testing.assert_allclose(z_b[i], z_s, atol=1e-9)


def test_activation_is_applied_to_the_weighted_sum_in_every_layer():
    net = random_network(NetworkSpec((3, 4, 2), "relu", "sigmoid"), seed=4)
    trace = forward(net, [0.3, -0.2, 0.9])
    np.testing.assert_array_equal(trace.activations[1], np.maximum(trace.weighted_sums[0], 0))
    np.testing.assert_allclose(
        trace.activations[2], 1 / (1 + np.exp(-trace.weighted_sums[1])), atol=1e-12
    )


def test_default_network_outputs_lie_in_tanh_range():
    net = random_network(NetworkSpec(), seed=1)
    out = predict(net, make_rng(3).uniform(-1, 1, size=(50, 7)))
    assert out.shape == (50, 2)
    assert np.all(np.abs(out) <= 1.0)


def test_zero_network_outputs_zero():
    np.testing.assert_array_equal(predict(Network.zeros(NetworkSpec()), np.ones(7)), [0.0, 0.0])


def test_forward_does_not_change_the_network_or_input():
    net = random_network(NetworkSpec(), seed=6)
    before = net.flatten()
    obs = np.full((3, 7), 0.5)
    forward(net, obs)
    np.testing.assert_array_equal(net.flatten(), before)
    assert np.all(obs == 0.5)


@pytest.mark.parametrize("shape", [(6,), (4, 6), (2, 3, 7), ()])
def test_wrong_observation_shape_rejected(shape):
    with pytest.raises(ValueError, match="expected observations"):
        forward(Network.zeros(NetworkSpec()), np.zeros(shape))
