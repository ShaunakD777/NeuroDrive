"""Tests for core/nn/backward.py (single neuron): hand example, finite differences, batching."""

import numpy as np
import pytest

from neurodrive.core.activations import ACTIVATIONS
from neurodrive.core.losses import huber, huber_grad, mse, mse_grad
from neurodrive.core.neuron import Neuron
from neurodrive.core.nn.backward import neuron_backward
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
