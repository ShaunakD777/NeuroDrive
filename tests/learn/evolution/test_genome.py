"""Tests for learn/evolution/genome.py: the gene map, one gene = one parameter, genome matrices,
random starting genomes, and permutation symmetry (the Brief's worked example of task 7.1)."""

import numpy as np
import pytest

from neurodrive.core.nn.forward import predict
from neurodrive.core.nn.init import init_network, init_population, weight_limit
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.population import Population
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.learn.evolution.genome import (
    GeneAddress,
    gene_address,
    gene_index,
    genomes_of,
    layer_slices,
    networks_of,
    random_genomes,
)

TINY = NetworkSpec((2, 2, 1))
DEFAULT = NetworkSpec()
SPECS = [
    TINY,
    NetworkSpec((7, 2)),
    NetworkSpec((3, 4, 2)),
    NetworkSpec((2, 3, 3, 3, 1)),
    DEFAULT,
]
SPEC_IDS = ["2-2-1", "7-2", "3-4-2", "2-3-3-3-1", "7-16-16-2"]

# The Brief's worked example: W0 rows (0.5, -1) and (0, 1), b0 = (0, 0), W1 = (1, 2), b1 = 0.
G = np.array([0.5, -1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0])
X = np.array([2.0, 1.0])


def output(genome) -> float:
    return float(predict(Network.unflatten(TINY, genome), X)[0])


def swap_hidden(net: Network, layer: int, i: int, j: int) -> Network:
    """Swap hidden neurons i and j of ``layer``: rows of W and b there, columns of the next W."""
    weights = [w.copy() for w in net.weights]
    biases = [b.copy() for b in net.biases]
    weights[layer][[i, j]] = weights[layer][[j, i]]
    biases[layer][[i, j]] = biases[layer][[j, i]]
    weights[layer + 1][:, [i, j]] = weights[layer + 1][:, [j, i]]
    return Network(net.spec, weights, biases)


# ── the gene map ───────────────────────────────────────────────────────────────────────────


def test_hand_map_of_the_tiny_network():
    names = [gene_address(TINY, i).name for i in range(9)]
    assert names == [
        "W0[0,0]",
        "W0[0,1]",
        "W0[1,0]",
        "W0[1,1]",
        "b0[0]",
        "b0[1]",
        "W1[0,0]",
        "W1[0,1]",
        "b1[0]",
    ]
    assert gene_address(TINY, 3) == GeneAddress(0, "weight", 1, 1)  # k = 3: row 1, column 1
    assert gene_address(TINY, 7) == GeneAddress(1, "weight", 0, 1)  # s_1 = 6, k = 1
    net = Network.unflatten(TINY, G)
    assert net.weights[0][1, 1] == G[3] == 1.0 and net.weights[1][0, 1] == G[7] == 2.0


def test_default_layout_and_the_briefs_named_genes():
    slices = [
        (g.weights.start, g.weights.stop, g.biases.start, g.biases.stop)
        for g in layer_slices(DEFAULT)
    ]
    assert slices == [(0, 112, 112, 128), (128, 384, 384, 400), (400, 432, 432, 434)]
    assert gene_address(DEFAULT, 37).name == "W0[5,2]"  # 37 = 5 * 7 + 2
    assert gene_address(DEFAULT, 400).name == "W2[0,0]"
    assert gene_address(DEFAULT, 433).name == "b2[1]"  # the throttle bias
    assert gene_index(DEFAULT, GeneAddress(1, "weight", 3, 7)) == 183  # 128 + 3 * 16 + 7


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_slices_cover_every_gene_once(spec):
    covered = np.zeros(spec.n_parameters, dtype=int)
    for genes in layer_slices(spec):
        covered[genes.weights] += 1
        covered[genes.biases] += 1
    assert np.all(covered == 1)


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_index_and_address_are_exact_inverses(spec):
    addresses = [gene_address(spec, i) for i in range(spec.n_parameters)]
    assert len(set(addresses)) == spec.n_parameters  # every address appears once
    assert [gene_index(spec, a) for a in addresses] == list(range(spec.n_parameters))


@pytest.mark.parametrize("spec", SPECS, ids=SPEC_IDS)
def test_one_gene_changes_exactly_one_parameter_the_one_at_its_address(spec):
    base = make_rng(0).standard_normal(spec.n_parameters)
    before = Network.unflatten(spec, base)
    for i in range(spec.n_parameters):
        g = base.copy()
        g[i] += 1.0
        after = Network.unflatten(spec, g)
        a = gene_address(spec, i)
        changed = []
        for layer in range(spec.n_layers):
            for kind, old, new in (
                ("weight", before.weights[layer], after.weights[layer]),
                ("bias", before.biases[layer], after.biases[layer]),
            ):
                for pos in zip(*np.nonzero(old != new), strict=True):
                    changed.append((layer, kind, *map(int, pos)))
        want = (a.layer, a.kind, a.row) + ((a.column,) if a.kind == "weight" else ())
        assert changed == [want]


# ── the worked example: one gene, one weight; permutation symmetry ─────────────────────────


def test_the_briefs_worked_example():
    assert output(G) == pytest.approx(0.9093, abs=5e-5)  # tanh(2 * tanh(1))
    mutated = G.copy()
    mutated[3] = 0.5  # W0[1, 1] only
    assert output(mutated) == pytest.approx(0.7279, abs=5e-5)  # tanh(2 * tanh(0.5))
    swapped = swap_hidden(Network.unflatten(TINY, G), 0, 0, 1).flatten()
    np.testing.assert_array_equal(swapped, [0, 1, 0.5, -1, 0, 0, 2, 1, 0])
    assert int((swapped != G).sum()) == 6
    assert output(swapped) == pytest.approx(output(G), abs=1e-12)  # same driver, other genome
    child = np.r_[G[:6], swapped[6:]]  # first six genes from one parent, last three from the other
    assert output(child) == pytest.approx(0.6420, abs=5e-5)  # tanh(tanh(1)): like neither


def test_permutation_symmetry_on_the_default_network():
    net = Network.unflatten(DEFAULT, make_rng(1).standard_normal(DEFAULT.n_parameters))
    swapped = swap_hidden(net, 0, 3, 11)  # two neurons of the first hidden layer
    assert int((swapped.flatten() != net.flatten()).sum()) == 2 * (7 + 1 + 16)  # 48 genes
    obs = make_rng(2).uniform(-1.0, 1.0, (100, 7))
    np.testing.assert_allclose(predict(swapped, obs), predict(net, obs), rtol=0, atol=1e-12)


# ── genome matrices ────────────────────────────────────────────────────────────────────────


def test_matrix_round_trips_are_exact():
    rng = make_rng(4)
    nets = [init_network(DEFAULT, rng) for _ in range(6)]
    g = genomes_of(nets)
    assert g.shape == (6, 434)
    for i, net in enumerate(nets):
        assert g[i].tobytes() == net.flatten().tobytes()
    back = networks_of(DEFAULT, g)
    assert all(a.equals(b) for a, b in zip(back, nets, strict=True))
    via_genomes = Population.from_genomes(DEFAULT, g)
    via_networks = Population.from_networks(networks_of(DEFAULT, g))
    assert via_genomes.genomes().tobytes() == via_networks.genomes().tobytes()


def test_random_genomes_are_seeded_xavier_starts():
    g = random_genomes(DEFAULT, 5, make_rng(9))
    assert g.tobytes() == random_genomes(DEFAULT, 5, make_rng(9)).tobytes()
    assert g.tobytes() == init_population(DEFAULT, 5, make_rng(9)).genomes().tobytes()
    assert not np.array_equal(g, random_genomes(DEFAULT, 5, make_rng(10)))
    for layer, genes in enumerate(layer_slices(DEFAULT)):
        n_out, n_in = DEFAULT.weight_shape(layer)
        assert np.all(g[:, genes.biases] == 0.0)
        assert np.all(np.abs(g[:, genes.weights]) <= weight_limit("xavier", n_in, n_out))
        assert np.any(g[:, genes.weights] != 0.0)


# ── errors ─────────────────────────────────────────────────────────────────────────────────


def test_bad_indices_addresses_and_matrices_are_refused():
    for i in (-1, TINY.n_parameters):
        with pytest.raises(IndexError):
            gene_address(TINY, i)
    with pytest.raises(TypeError):
        gene_address(TINY, 1.0)
    for bad in (
        GeneAddress(2, "bias", 0),
        GeneAddress(0, "weight", 2, 0),
        GeneAddress(0, "weight", 0, 2),
        GeneAddress(1, "bias", 1),
    ):
        with pytest.raises(IndexError):
            gene_index(TINY, bad)
    with pytest.raises(ValueError):
        GeneAddress(0, "weight", 0)  # a weight needs a column
    with pytest.raises(ValueError):
        GeneAddress(0, "bias", 0, 1)  # a bias has none
    with pytest.raises(ValueError):
        networks_of(TINY, np.zeros((3, 8)))
    with pytest.raises(ValueError):
        networks_of(TINY, np.zeros(9))
    with pytest.raises(ValueError):
        genomes_of([Network.zeros(TINY), Network.zeros(DEFAULT)])
    with pytest.raises(ValueError):
        genomes_of([])
