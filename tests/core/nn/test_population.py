"""Tests for core/nn/population.py: stacking, genome rows, and equality with single passes."""

import numpy as np
import pytest

from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.population import Population
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng

DEFAULT = NetworkSpec()


def random_networks(spec: NetworkSpec, n: int, seed: int = 0) -> list[Network]:
    rng = make_rng(seed)
    return [Network.unflatten(spec, rng.standard_normal(spec.n_parameters)) for _ in range(n)]


def test_hand_example_each_car_uses_its_own_network():
    spec = NetworkSpec((2, 1), output_activation="linear")
    nets = [
        Network(spec, [[[1.0, 2.0]]], [[0.0]]),
        Network(spec, [[[-1.0, 1.0]]], [[0.5]]),
    ]
    obs = [[1.0, 1.0], [2.0, 3.0]]
    out = Population.from_networks(nets).predict(obs)
    np.testing.assert_allclose(out, [[3.0], [1.5]], atol=1e-9)
    # network 0 alone on the second row would give 8, so the result is not one shared network
    np.testing.assert_allclose(nets[0].weights[0] @ np.array([2.0, 3.0]), [8.0])


def test_equals_loop_of_single_forward_passes_for_n_20():
    nets = random_networks(DEFAULT, 20, seed=1)
    pop = Population.from_networks(nets)
    obs = make_rng(2).uniform(-1, 1, size=(20, 7))
    trace = pop.forward(obs)
    for i, net in enumerate(nets):
        single = forward(net, obs[i])
        for a_p, a_s in zip(trace.activations, single.activations, strict=True):
            np.testing.assert_allclose(a_p[i], a_s, atol=1e-9)
        for z_p, z_s in zip(trace.weighted_sums, single.weighted_sums, strict=True):
            np.testing.assert_allclose(z_p[i], z_s, atol=1e-9)


@pytest.mark.parametrize(
    "spec", [NetworkSpec((3, 5, 4, 2), "relu", "sigmoid"), NetworkSpec((4, 2))]
)
def test_other_specs_match_single_passes(spec):
    nets = random_networks(spec, 6, seed=3)
    obs = make_rng(4).uniform(-1, 1, size=(6, spec.n_inputs))
    out = Population.from_networks(nets).predict(obs)
    for i, net in enumerate(nets):
        np.testing.assert_allclose(out[i], forward(net, obs[i]).output, atol=1e-9)


def test_trace_shapes():
    trace = Population.from_networks(random_networks(DEFAULT, 5)).forward(np.zeros((5, 7)))
    assert [a.shape for a in trace.activations] == [(5, 7), (5, 16), (5, 16), (5, 2)]
    assert [z.shape for z in trace.weighted_sums] == [(5, 16), (5, 16), (5, 2)]


def test_population_of_one_equals_single_network():
    net = random_networks(DEFAULT, 1, seed=5)[0]
    obs = make_rng(6).uniform(-1, 1, size=(1, 7))
    np.testing.assert_allclose(
        Population.from_networks([net]).predict(obs)[0], forward(net, obs[0]).output, atol=1e-9
    )


def test_from_genomes_matches_unflatten_and_round_trips():
    nets = random_networks(DEFAULT, 4, seed=7)
    genomes = np.stack([n.flatten() for n in nets])
    pop = Population.from_genomes(DEFAULT, genomes)
    np.testing.assert_array_equal(pop.genomes(), genomes)
    for i, net in enumerate(nets):
        assert pop.network(i).equals(net)
    assert pop.size == 4


def test_from_networks_and_from_genomes_agree():
    nets = random_networks(DEFAULT, 3, seed=8)
    a = Population.from_networks(nets)
    b = Population.from_genomes(DEFAULT, a.genomes())
    for wa, wb in zip(a.weights + a.biases, b.weights + b.biases, strict=True):
        np.testing.assert_array_equal(wa, wb)


def test_population_owns_its_data():
    nets = random_networks(DEFAULT, 2, seed=9)
    pop = Population.from_networks(nets)
    nets[0].weights[0][:] = 0.0
    assert pop.network(0).weights[0].any()
    pop.network(1).weights[0][:] = 0.0  # a copy, so the population is unchanged
    assert pop.network(1).weights[0].any()


def test_members_are_different_networks():
    pop = Population.from_networks(random_networks(DEFAULT, 3, seed=10))
    out = pop.predict(np.tile(np.full(7, 0.3), (3, 1)))
    assert not np.allclose(out[0], out[1])
    assert not np.allclose(out[1], out[2])


def test_mixed_specs_and_empty_rejected():
    mixed = [Network.zeros(NetworkSpec((2, 1))), Network.zeros(NetworkSpec((3, 1)))]
    with pytest.raises(ValueError, match="share one spec"):
        Population.from_networks(mixed)
    with pytest.raises(ValueError, match="at least one"):
        Population.from_networks([])


@pytest.mark.parametrize("shape", [(4, 7), (5,), (5, 6), (5, 7, 1)])
def test_wrong_observation_shape_rejected(shape):
    pop = Population.from_networks(random_networks(DEFAULT, 5))
    with pytest.raises(ValueError, match="expected observations"):
        pop.forward(np.zeros(shape))


def test_bad_genome_matrix_and_index_rejected():
    with pytest.raises(ValueError, match="genomes"):
        Population.from_genomes(DEFAULT, np.zeros((3, 433)))
    with pytest.raises(ValueError, match="genomes"):
        Population.from_genomes(DEFAULT, np.zeros(434))
    pop = Population.from_networks(random_networks(DEFAULT, 2))
    with pytest.raises(IndexError):
        pop.network(2)
