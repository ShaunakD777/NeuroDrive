"""Tests for core/neuron.py: the weighted sum, the activation, batching and input checks."""

import numpy as np
import pytest

from neurodrive.core.neuron import Neuron

# Roadmap example: inputs [2, 1], weights [0.5, -1], bias 0.2 -> z = 1.0 - 1.0 + 0.2 = 0.2
X = [2.0, 1.0]
W = [0.5, -1.0]
B = 0.2


def test_weighted_sum_matches_hand_calculation():
    assert Neuron(W, B).weighted_sum(X) == pytest.approx(0.2, abs=1e-9)


@pytest.mark.parametrize(
    ("activation", "expected"),
    [
        ("tanh", 0.19737532022490401),
        ("sigmoid", 0.54983399731247795),
        ("relu", 0.2),
        ("linear", 0.2),
    ],
)
def test_output_for_each_activation(activation, expected):
    assert Neuron(W, B, activation).forward(X) == pytest.approx(expected, abs=1e-9)


def test_forward_with_sum_returns_both_stages():
    z, a = Neuron(W, B).forward_with_sum(X)
    assert z == pytest.approx(0.2, abs=1e-9)
    assert a == pytest.approx(0.19737532022490401, abs=1e-9)


def test_default_activation_is_tanh():
    assert Neuron(W, B).activation == "tanh"


def test_batch_equals_single_calls():
    rng = np.random.default_rng(0)  # test-only generator, explicitly seeded
    batch = rng.normal(size=(5, 2))
    neuron = Neuron(W, B, "sigmoid")
    together = neuron.forward(batch)
    assert together.shape == (5,)
    for i in range(5):
        assert together[i] == pytest.approx(float(neuron.forward(batch[i])), abs=1e-12)


def test_sigmoid_neuron_can_never_output_negative():
    # The Milestone 1 lesson: whatever the weights, sigmoid output is >= 0.
    rng = np.random.default_rng(1)
    for _ in range(20):
        neuron = Neuron(rng.normal(scale=10, size=2), rng.normal(scale=10), "sigmoid")
        assert np.all(neuron.forward(rng.normal(scale=10, size=(50, 2))) >= 0.0)


def test_weights_are_copied_not_shared():
    w = np.array([1.0, 2.0])
    neuron = Neuron(w, 0.0)
    w[0] = 99.0
    assert neuron.weights[0] == 1.0


def test_wrong_input_length_is_rejected():
    neuron = Neuron(W, B)
    with pytest.raises(ValueError, match="expected inputs of shape"):
        neuron.forward([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="expected inputs of shape"):
        neuron.forward(np.zeros((4, 3)))


def test_bad_construction_is_rejected():
    with pytest.raises(ValueError, match="unknown activation"):
        Neuron(W, B, "gelu")
    with pytest.raises(ValueError, match="non-empty 1-D"):
        Neuron([], B)
    with pytest.raises(ValueError, match="non-empty 1-D"):
        Neuron([[1.0, 2.0]], B)
