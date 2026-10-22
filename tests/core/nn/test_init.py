"""Tests for core/nn/init.py: scale formulas, statistics, determinism and symmetry."""

import math

import numpy as np
import pytest

from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.init import init_network, init_population, weight_limit
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng

DEFAULT = NetworkSpec()


def test_weight_limit_matches_hand_values():
    assert weight_limit("xavier", 16, 16) == pytest.approx(math.sqrt(6 / 32), abs=1e-12)
    assert weight_limit("xavier", 16, 16) == pytest.approx(0.4330127, abs=1e-6)
    assert weight_limit("xavier", 7, 16) == pytest.approx(0.5108, abs=1e-4)
    assert weight_limit("uniform", 7, 16, scale=0.3) == 0.3
    assert weight_limit("zeros", 7, 16) == 0.0


def test_unknown_scheme_and_negative_scale_rejected():
    with pytest.raises(ValueError, match="unknown init scheme"):
        weight_limit("he", 3, 3)
    with pytest.raises(ValueError, match="scale"):
        weight_limit("uniform", 3, 3, scale=-1)


def test_xavier_variance_matches_the_formula():
    spec = NetworkSpec((300, 400, 200))
    net = init_network(spec, make_rng(0), "xavier")
    for layer in range(spec.n_layers):
        n_out, n_in = spec.weight_shape(layer)
        w = net.weights[layer]
        assert w.var() == pytest.approx(2 / (n_in + n_out), rel=0.03)
        assert w.mean() == pytest.approx(0.0, abs=0.005)
        assert np.abs(w).max() <= weight_limit("xavier", n_in, n_out)


def test_uniform_variance_is_scale_squared_over_three():
    net = init_network(NetworkSpec((300, 400)), make_rng(1), "uniform", scale=0.6)
    assert net.weights[0].var() == pytest.approx(0.6**2 / 3, rel=0.03)
    assert np.abs(net.weights[0]).max() <= 0.6


def test_wide_layers_get_smaller_weights_under_xavier():
    net = init_network(NetworkSpec((4, 4, 400)), make_rng(2), "xavier")
    assert net.weights[1].std() < net.weights[0].std()  # layer 1 is 4 -> 400, so is wider


def test_biases_are_zero_for_every_scheme():
    for scheme in ("zeros", "uniform", "xavier"):
        net = init_network(DEFAULT, make_rng(3), scheme)
        assert all(not b.any() for b in net.biases)


def test_zeros_scheme_gives_all_zero_network_without_drawing():
    rng = make_rng(4)
    net = init_network(DEFAULT, rng, "zeros")
    assert not net.flatten().any()
    assert rng.random() == make_rng(4).random()  # no numbers were consumed


def test_same_seed_same_network_different_seed_different_network():
    a = init_network(DEFAULT, make_rng(5))
    b = init_network(DEFAULT, make_rng(5))
    c = init_network(DEFAULT, make_rng(6))
    assert a.equals(b)
    assert not a.equals(c)
    assert a.spec == DEFAULT and a.n_parameters == 434


def test_population_member_i_is_the_ith_draw():
    pop = init_population(DEFAULT, 4, make_rng(7))
    rng = make_rng(7)
    for i in range(4):
        assert pop.network(i).equals(init_network(DEFAULT, rng))
    assert pop.size == 4
    assert len({tuple(pop.network(i).flatten()[:5]) for i in range(4)}) == 4


def test_population_size_must_be_positive():
    with pytest.raises(ValueError, match="size"):
        init_population(DEFAULT, 0, make_rng(0))


def test_zero_network_hidden_neurons_are_identical_and_inert():
    obs = make_rng(8).uniform(-1, 1, size=(10, 7))
    trace = forward(init_network(DEFAULT, make_rng(0), "zeros"), obs)
    for a in trace.activations[1:]:
        assert not a.any()  # every neuron outputs 0 for every input


def test_constant_weights_keep_hidden_neurons_identical_random_weights_do_not():
    spec = NetworkSpec((7, 16, 16, 2))
    constant = Network(
        spec,
        [np.full(spec.weight_shape(i), 0.1) for i in range(3)],
        [np.zeros(spec.bias_shape(i)) for i in range(3)],
    )
    obs = make_rng(9).uniform(-1, 1, size=(5, 7))
    same = forward(constant, obs)
    assert same.activations[1].std(axis=1).max() == 0.0  # 16 neurons, all equal, per input
    assert same.activations[1].any()  # ...but not zero: they do react to the input
    varied = forward(init_network(spec, make_rng(9), "xavier"), obs)
    assert varied.activations[1].std(axis=1).min() > 0.05


def test_xavier_keeps_outputs_off_the_saturation_walls():
    pop = init_population(DEFAULT, 200, make_rng(10))
    obs = make_rng(11).uniform(0, 1, size=(200, 7))
    out = pop.predict(obs)
    assert np.mean(np.abs(out) > 0.99) < 0.02
    assert out.std() > 0.05  # and they are not all squeezed to zero either
