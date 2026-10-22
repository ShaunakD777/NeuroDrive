"""Tests for explain/contributions.py: the receipt of a neuron's weighted sum."""

import numpy as np
import pytest

from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.explain.contributions import layer_contributions, neuron_contributions

T = np.tanh(1.5)  # 0.905148...


def hand_network() -> Network:
    """The 2-2-1 example of tasks 2.2 and 2.9: tanh hidden layer, linear output."""
    spec = NetworkSpec((2, 2, 1), hidden_activation="tanh", output_activation="linear")
    return Network(spec, [[[0.5, -1.0], [1.0, 0.0]], [[2.0, 1.0]]], [[0.0, 0.5], [0.1]])


def test_hand_example_hidden_neuron():
    net = hand_network()
    r = neuron_contributions(net, forward(net, [1.0, 2.0]), layer=0, neuron=0)
    np.testing.assert_allclose(r.contributions, [0.5, -2.0], atol=1e-12)  # 0.5*1, -1*2
    assert r.total == pytest.approx(-1.5, abs=1e-12)
    assert r.bias == 0.0 and r.z == pytest.approx(-1.5, abs=1e-12)
    assert r.output == pytest.approx(-T, abs=1e-12) and r.activation == "tanh"
    assert r.ranked() == [1, 0]  # the -2 from input 1 matters most


def test_hand_example_output_neuron():
    net = hand_network()
    r = neuron_contributions(net, forward(net, [1.0, 2.0]), layer=1, neuron=0)
    np.testing.assert_allclose(r.inputs, [-T, T], atol=1e-12)
    np.testing.assert_allclose(r.contributions, [-2 * T, T], atol=1e-12)  # -1.810297, 0.905148
    assert r.total == pytest.approx(-T, abs=1e-12)
    assert r.total + r.bias == pytest.approx(-0.805148, abs=1e-6)
    assert r.z == pytest.approx(r.total + r.bias, abs=1e-12) and r.output == r.z  # linear
    assert r.activation == "linear"


@pytest.mark.parametrize("seed", range(10))
def test_contributions_sum_to_z_minus_bias_for_every_neuron(seed):
    rng = make_rng(seed)
    net = init_network(NetworkSpec(), rng)
    net.biases[0][:] = rng.uniform(-1, 1, 16)  # non-zero biases, so the bias really matters
    net.biases[2][:] = rng.uniform(-1, 1, 2)
    trace = forward(net, rng.uniform(0, 1, 7))
    worst = 0.0
    for layer in range(net.spec.n_layers):
        c = layer_contributions(net, trace, layer)
        gap = np.abs(c.sum(axis=1) - (trace.weighted_sums[layer] - net.biases[layer]))
        worst = max(worst, float(gap.max()))
    assert worst < 1e-12  # the same numbers summed in another order: only rounding differs


def test_layer_contributions_rows_are_the_neuron_receipts():
    net = init_network(NetworkSpec(), make_rng(3))
    trace = forward(net, make_rng(4).uniform(0, 1, 7))
    matrix = layer_contributions(net, trace, 2)
    assert matrix.shape == (2, 16)
    for j in range(2):
        np.testing.assert_array_equal(
            matrix[j], neuron_contributions(net, trace, 2, j).contributions
        )


def test_a_zero_input_contributes_nothing_and_a_sign_flip_flips_it():
    net = hand_network()
    r = neuron_contributions(net, forward(net, [0.0, 2.0]), 0, 0)
    assert r.contributions[0] == 0.0
    flipped = Network(net.spec, [-net.weights[0], net.weights[1]], net.biases)
    r2 = neuron_contributions(flipped, forward(flipped, [1.0, 2.0]), 0, 0)
    np.testing.assert_allclose(r2.contributions, [-0.5, 2.0], atol=1e-12)


def test_ranked_orders_by_size_not_sign():
    net = init_network(NetworkSpec(), make_rng(5))
    r = neuron_contributions(net, forward(net, make_rng(6).uniform(0, 1, 7)), 0, 3)
    sizes = np.abs(r.contributions)[r.ranked()]
    assert list(sizes) == sorted(sizes, reverse=True)
    assert sorted(r.ranked()) == list(range(7))


def test_results_are_copies():
    net = hand_network()
    trace = forward(net, [1.0, 2.0])
    r = neuron_contributions(net, trace, 0, 0)
    r.weights[0] = 99.0
    r.inputs[0] = 99.0
    assert net.weights[0][0, 0] == 0.5 and trace.activations[0][0] == 1.0


def test_bad_layer_neuron_or_trace_rejected():
    net = hand_network()
    trace = forward(net, [1.0, 2.0])
    with pytest.raises(IndexError):
        neuron_contributions(net, trace, 2, 0)
    with pytest.raises(IndexError):
        neuron_contributions(net, trace, 0, 2)
    with pytest.raises(ValueError, match="one observation"):
        neuron_contributions(net, forward(net, [[1.0, 2.0], [0.0, 0.0]]), 0, 0)  # a batch
    other = Network.zeros(NetworkSpec((3, 2, 1)))
    with pytest.raises(ValueError, match="one observation"):
        layer_contributions(other, trace, 0)  # trace of a different network
