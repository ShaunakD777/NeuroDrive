"""Tests for core/nn/backward.py: one neuron and the full network, hand and finite differences."""

import numpy as np
import pytest

from neurodrive.core.activations import ACTIVATIONS
from neurodrive.core.losses import huber, huber_grad, mse, mse_grad
from neurodrive.core.neuron import Neuron
from neurodrive.core.nn.backward import backward, neuron_backward
from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng

LOSSES = {"mse": (mse, mse_grad), "huber": (huber, huber_grad)}


def hand_neuron() -> Neuron:
    """The Brief's example: tanh neuron, w = [0.5, -1], b = 0.5."""
    return Neuron([0.5, -1.0], 0.5, "tanh")


def test_hand_example_every_value():
    neuron, x, y = hand_neuron(), np.array([2.0, 1.0]), 1.0
    z, a = neuron.forward_with_sum(x)
    assert z == pytest.approx(0.5) and a == pytest.approx(0.462117, abs=1e-6)
    g = 2.0 * (a - y)  # MSE with M = 1
    assert g == pytest.approx(-1.075766, abs=1e-6)
    grads = neuron_backward(neuron, x, g)
    assert grads.delta == pytest.approx(-0.846033, abs=1e-6)  # g * (1 - a^2)
    np.testing.assert_allclose(grads.weights, [-1.692067, -0.846033], atol=1e-6)
    assert grads.bias == pytest.approx(-0.846033, abs=1e-6)
    np.testing.assert_allclose(grads.inputs, [-0.423017, 0.846033], atol=1e-6)


def test_one_gradient_descent_step_lowers_the_loss():
    neuron, x, y = hand_neuron(), np.array([2.0, 1.0]), np.array([1.0])
    a = neuron.forward(x.reshape(1, -1))
    assert mse(a, y) == pytest.approx(0.289318, abs=1e-6)
    grads = neuron_backward(neuron, x.reshape(1, -1), mse_grad(a, y))
    stepped = Neuron(neuron.weights - 0.1 * grads.weights, neuron.bias - 0.1 * grads.bias)
    np.testing.assert_allclose(stepped.weights, [0.669207, -0.915397], atol=1e-6)
    assert stepped.bias == pytest.approx(0.584603, abs=1e-6)
    assert mse(stepped.forward(x.reshape(1, -1)), y) == pytest.approx(0.055330, abs=1e-6)


def _loss_of(weights, bias, activation, x, y, loss) -> float:
    return loss(Neuron(weights, bias, activation).forward(x), y)


@pytest.mark.parametrize("loss_name", sorted(LOSSES))
@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("batch", [None, 5])
def test_gradients_match_central_finite_differences(loss_name, activation, seed, batch):
    loss, loss_grad = LOSSES[loss_name]
    rng = make_rng(seed)
    n_in = 3
    w, b = rng.normal(size=n_in), float(rng.normal())
    shape = (n_in,) if batch is None else (batch, n_in)
    x = rng.normal(size=shape)
    y = rng.normal(size=() if batch is None else (batch,))
    neuron = Neuron(w, b, activation)
    z, a = neuron.forward_with_sum(x)
    assert np.all(np.abs(z) > 1e-3)  # keep ReLU away from its corner
    grads = neuron_backward(neuron, x, loss_grad(a, y))

    h = 1e-6
    num_w = np.zeros(n_in)
    for i in range(n_in):
        up, down = w.copy(), w.copy()
        up[i] += h
        down[i] -= h
        num_w[i] = (
            _loss_of(up, b, activation, x, y, loss) - _loss_of(down, b, activation, x, y, loss)
        ) / (2 * h)
    num_b = (
        _loss_of(w, b + h, activation, x, y, loss) - _loss_of(w, b - h, activation, x, y, loss)
    ) / (2 * h)
    num_x = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        num_x[i] = (loss(neuron.forward(up), y) - loss(neuron.forward(down), y)) / (2 * h)

    np.testing.assert_allclose(grads.weights, num_w, rtol=1e-5, atol=1e-8)
    assert grads.bias == pytest.approx(num_b, rel=1e-5, abs=1e-8)
    np.testing.assert_allclose(grads.inputs, num_x, rtol=1e-5, atol=1e-8)


def test_batch_gradient_is_the_sum_of_per_example_gradients():
    rng = make_rng(7)
    neuron = Neuron(rng.normal(size=4), 0.3, "sigmoid")
    x, g = rng.normal(size=(6, 4)), rng.normal(size=6)
    batch = neuron_backward(neuron, x, g)
    singles = [neuron_backward(neuron, x[k], g[k]) for k in range(6)]
    np.testing.assert_allclose(batch.weights, sum(s.weights for s in singles), atol=1e-12)
    assert batch.bias == pytest.approx(sum(s.bias for s in singles), abs=1e-12)
    np.testing.assert_allclose(batch.inputs, np.stack([s.inputs for s in singles]), atol=1e-12)
    np.testing.assert_allclose(batch.delta, [float(s.delta) for s in singles], atol=1e-12)


def test_shapes_single_and_batched():
    neuron = hand_neuron()
    one = neuron_backward(neuron, [2.0, 1.0], 1.0)
    assert one.weights.shape == (2,) and one.inputs.shape == (2,) and one.delta.shape == ()
    assert isinstance(one.bias, float)
    many = neuron_backward(neuron, np.ones((3, 2)), np.ones(3))
    assert many.weights.shape == (2,) and many.inputs.shape == (3, 2) and many.delta.shape == (3,)


def test_zero_upstream_gradient_gives_zero_gradients():
    grads = neuron_backward(hand_neuron(), np.ones((3, 2)), np.zeros(3))
    assert not np.any(grads.weights) and grads.bias == 0.0 and not np.any(grads.inputs)


def test_saturated_neuron_passes_almost_nothing_back():
    # z = 10 for a tanh neuron: f'(z) is about 8e-9, so every gradient almost vanishes
    grads = neuron_backward(Neuron([1.0], 0.0, "tanh"), [10.0], 1.0)
    assert abs(grads.weights[0]) < 1e-7 and abs(grads.bias) < 1e-8


def test_wrong_grad_output_shape_is_refused():
    with pytest.raises(ValueError, match="grad_output"):
        neuron_backward(hand_neuron(), np.ones((3, 2)), np.ones(4))
    with pytest.raises(ValueError, match="grad_output"):
        neuron_backward(hand_neuron(), [2.0, 1.0], [1.0, 1.0])


def test_wrong_input_shape_is_refused():
    with pytest.raises(ValueError, match="inputs"):
        neuron_backward(hand_neuron(), [1.0, 2.0, 3.0], 1.0)


# --- Full network (task 3.4) ---

T = np.tanh(1.5)  # 0.905148...


def hand_network() -> Network:
    """The 2-2-1 example of task 2.2: tanh hidden layer, linear output."""
    spec = NetworkSpec((2, 2, 1), hidden_activation="tanh", output_activation="linear")
    return Network(spec, [[[0.5, -1.0], [1.0, 0.0]], [[2.0, 1.0]]], [[0.0, 0.5], [0.1]])


def random_network(spec: NetworkSpec, seed: int) -> Network:
    return Network.unflatten(spec, make_rng(seed).normal(size=spec.n_parameters))


def test_network_hand_example_every_value():
    net, x, y = hand_network(), np.array([1.0, 2.0]), np.array([0.0])
    trace = forward(net, x)
    np.testing.assert_allclose(trace.output, [-0.805148], atol=1e-6)
    g = mse_grad(trace.output, y)  # 2 * (output - 0)
    np.testing.assert_allclose(g, [-1.610297], atol=1e-6)
    grads = backward(net, trace, g)
    np.testing.assert_allclose(grads.deltas[1], [-1.610297], atol=1e-6)  # linear: δ2 = g
    np.testing.assert_allclose(grads.weights[1], [[1.457557, -1.457557]], atol=1e-6)
    np.testing.assert_allclose(grads.biases[1], [-1.610297], atol=1e-6)
    # pushed down: g1 = δ2 * W2 = [-3.220593, -1.610297]; f'(z1) = 1 - T^2 = 0.180707
    assert 1 - T * T == pytest.approx(0.180707, abs=1e-6)
    np.testing.assert_allclose(grads.deltas[0], [-0.581983, -0.290991], atol=1e-6)
    np.testing.assert_allclose(
        grads.weights[0], [[-0.581983, -1.163965], [-0.290991, -0.581983]], atol=1e-6
    )
    np.testing.assert_allclose(grads.biases[0], [-0.581983, -0.290991], atol=1e-6)
    np.testing.assert_allclose(grads.inputs, [-0.581983, 0.581983], atol=1e-6)
    # task 2.9's finite-difference sensitivity d(output)/dx = [0.361413, -0.361413], times g
    expected = -1.610297 * np.array([0.361413, -0.361413])
    np.testing.assert_allclose(grads.inputs, expected, atol=1e-5)


def test_network_one_gradient_descent_step_lowers_the_loss():
    net, x, y = hand_network(), np.array([[1.0, 2.0]]), np.array([[0.0]])
    trace = forward(net, x)
    assert mse(trace.output, y) == pytest.approx(0.648264, abs=1e-6)
    grads = backward(net, trace, mse_grad(trace.output, y))
    stepped = Network.unflatten(net.spec, net.flatten() - 0.1 * grads.flatten())
    out = forward(stepped, x).output
    np.testing.assert_allclose(out, [[-0.187760]], atol=1e-6)
    assert mse(out, y) == pytest.approx(0.035254, abs=1e-6)


def test_network_gradient_shapes_single_and_batched():
    spec = NetworkSpec((7, 16, 16, 2))
    net = random_network(spec, 0)
    for obs in (np.ones(7), np.ones((5, 7))):
        trace = forward(net, obs)
        grads = backward(net, trace, np.ones_like(trace.output))
        for gw, w, gb, b in zip(grads.weights, net.weights, grads.biases, net.biases, strict=True):
            assert gw.shape == w.shape and gb.shape == b.shape
        assert grads.inputs.shape == obs.shape
        assert [d.shape for d in grads.deltas] == [z.shape for z in trace.weighted_sums]
        assert grads.flatten().shape == (spec.n_parameters,) == (434,)


def test_flatten_follows_the_genome_order():
    net = hand_network()
    grads = backward(net, forward(net, [1.0, 2.0]), [1.0])
    # same layout as Network.flatten(): W1 row by row, b1, W2, b2
    parts = [grads.weights[0].ravel(), grads.biases[0], grads.weights[1].ravel(), grads.biases[1]]
    np.testing.assert_array_equal(grads.flatten(), np.concatenate(parts))
    assert grads.flatten().shape == net.flatten().shape


SHAPES = [(3, 4, 2), (2, 3, 3, 1), (4, 1)]


@pytest.mark.parametrize("loss_name", sorted(LOSSES))
@pytest.mark.parametrize("batch", [None, 4])
@pytest.mark.parametrize("seed", range(2))
@pytest.mark.parametrize("sizes", SHAPES, ids=str)
@pytest.mark.parametrize("out_act", ["tanh", "linear"])
@pytest.mark.parametrize("hidden_act", sorted(ACTIVATIONS))
def test_network_gradients_match_central_finite_differences(
    hidden_act, out_act, sizes, seed, batch, loss_name
):
    loss, loss_grad = LOSSES[loss_name]
    spec = NetworkSpec(sizes, hidden_activation=hidden_act, output_activation=out_act)
    net = random_network(spec, seed)
    rng = make_rng(100 + seed)
    x = rng.normal(size=(sizes[0],) if batch is None else (batch, sizes[0]))
    y = rng.normal(size=(sizes[-1],) if batch is None else (batch, sizes[-1]))
    trace = forward(net, x)
    if hidden_act == "relu":
        assert all(np.all(np.abs(z) > 1e-4) for z in trace.weighted_sums)  # away from corners
    grads = backward(net, trace, loss_grad(trace.output, y))

    def loss_at(params: np.ndarray, inputs: np.ndarray) -> float:
        return loss(forward(Network.unflatten(spec, params), inputs).output, y)

    h, theta = 1e-6, net.flatten()
    num_theta = np.zeros_like(theta)
    for k in range(theta.size):
        up, down = theta.copy(), theta.copy()
        up[k] += h
        down[k] -= h
        num_theta[k] = (loss_at(up, x) - loss_at(down, x)) / (2 * h)
    num_x = np.zeros_like(x)
    for i in np.ndindex(x.shape):
        up, down = x.copy(), x.copy()
        up[i] += h
        down[i] -= h
        num_x[i] = (loss_at(theta, up) - loss_at(theta, down)) / (2 * h)

    np.testing.assert_allclose(grads.flatten(), num_theta, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(grads.inputs, num_x, rtol=1e-5, atol=1e-8)


def test_network_batch_gradient_is_the_sum_of_single_example_gradients():
    spec = NetworkSpec((3, 5, 2), hidden_activation="sigmoid", output_activation="tanh")
    net = random_network(spec, 3)
    rng = make_rng(9)
    x, g = rng.normal(size=(6, 3)), rng.normal(size=(6, 2))
    batch = backward(net, forward(net, x), g)
    singles = [backward(net, forward(net, x[k]), g[k]) for k in range(6)]
    np.testing.assert_allclose(batch.flatten(), sum(s.flatten() for s in singles), atol=1e-12)
    np.testing.assert_allclose(batch.inputs, np.stack([s.inputs for s in singles]), atol=1e-12)
    for layer in range(spec.n_layers):
        stacked = np.stack([s.deltas[layer] for s in singles])
        np.testing.assert_allclose(batch.deltas[layer], stacked, atol=1e-12)


@pytest.mark.parametrize("activation", sorted(ACTIVATIONS))
def test_one_neuron_network_agrees_with_neuron_backward(activation):
    w, b = np.array([0.7, -0.4, 1.1]), 0.2
    spec = NetworkSpec((3, 1), hidden_activation="tanh", output_activation=activation)
    net = Network(spec, [[w]], [[b]])
    x, g = make_rng(4).normal(size=(5, 3)), make_rng(5).normal(size=5)
    full = backward(net, forward(net, x), g.reshape(5, 1))
    one = neuron_backward(Neuron(w, b, activation), x, g)
    np.testing.assert_allclose(full.weights[0][0], one.weights, atol=1e-12)
    assert full.biases[0][0] == pytest.approx(one.bias, abs=1e-12)
    np.testing.assert_allclose(full.inputs, one.inputs, atol=1e-12)
    np.testing.assert_allclose(full.deltas[0][:, 0], one.delta, atol=1e-12)


def test_linear_network_input_gradient_is_g_times_w2_w1():
    spec = NetworkSpec((3, 4, 2), hidden_activation="linear", output_activation="linear")
    net = random_network(spec, 6)
    g = np.array([0.5, -2.0])
    grads = backward(net, forward(net, [0.3, -1.0, 2.0]), g)
    np.testing.assert_allclose(grads.inputs, g @ net.weights[1] @ net.weights[0], atol=1e-12)


def test_network_zero_upstream_gradient_gives_zero_gradients():
    net = random_network(NetworkSpec((3, 4, 2)), 1)
    grads = backward(net, forward(net, np.ones((2, 3))), np.zeros((2, 2)))
    assert not np.any(grads.flatten()) and not np.any(grads.inputs)


def test_network_wrong_grad_output_shape_is_refused():
    net = hand_network()
    with pytest.raises(ValueError, match="grad_output"):
        backward(net, forward(net, np.ones((3, 2))), np.ones(3))  # (3,) instead of (3, 1)
    with pytest.raises(ValueError, match="grad_output"):
        backward(net, forward(net, [1.0, 2.0]), [1.0, 1.0])


def test_trace_from_a_different_network_shape_is_refused():
    net = hand_network()
    wider = random_network(NetworkSpec((2, 3, 1)), 0)
    with pytest.raises(ValueError, match="layer"):
        backward(net, forward(wider, [1.0, 2.0]), [1.0])
    deeper = random_network(NetworkSpec((2, 2, 2, 1)), 0)
    with pytest.raises(ValueError, match="layers"):
        backward(net, forward(deeper, [1.0, 2.0]), [1.0])
