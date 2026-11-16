"""Tests for task 7.4: Gaussian mutation and uniform crossover (``learn/evolution/variation.py``).

The roadmap asks for mutation statistics and for crossover taking each gene from one parent. The
random tests are seeded, so they are deterministic; their tolerances are 5 standard deviations or
the 0.1 % critical value of the test statistic, so a correct operator would fail them less than
once in a thousand seeds, and the seeds used here pass.
"""

import math

import numpy as np
import pytest

from neurodrive.core.nn.forward import predict
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.learn.evolution.genome import networks_of, random_genomes
from neurodrive.learn.evolution.variation import (
    CrossoverResult,
    MutationResult,
    gaussian_mutation,
    uniform_crossover,
)

DEFAULT = NetworkSpec()  # [7, 16, 16, 2], P = 434


class ScriptedRng:
    """A stand-in generator that hands out prepared arrays in order (checking each shape)."""

    def __init__(self, *draws: tuple[str, list]) -> None:
        self.draws = [(kind, np.asarray(values, dtype=np.float64)) for kind, values in draws]

    def _next(self, kind: str, size) -> np.ndarray:
        expected_kind, values = self.draws.pop(0)
        assert kind == expected_kind, f"asked for {kind}, the script has {expected_kind}"
        assert values.shape == tuple(np.atleast_1d(size)), f"shape {size} vs {values.shape}"
        return values

    def random(self, size):
        return self._next("random", size)

    def standard_normal(self, size):
        return self._next("normal", size)


def chi2_critical_999(df: int) -> float:
    """The 0.1 % critical value of chi-squared with df degrees of freedom (Wilson-Hilferty)."""
    z = 3.090232306167813  # the standard normal's 99.9 % point
    return df * (1 - 2 / (9 * df) + z * math.sqrt(2 / (9 * df))) ** 3


def binomial_chi2(counts: np.ndarray, trials: int, p: float) -> float:
    """Sum over independent binomial counts of (count - trials p)^2 / (trials p (1 - p))."""
    return float((((counts - trials * p) ** 2) / (trials * p * (1 - p))).sum())


def permute_hidden(net: Network, rng) -> Network:
    """The renumbered twin: every hidden layer's neurons put in a random new order (their rows
    of W and b, and the matching columns of the next layer's W). The outputs do not change."""
    weights = [w.copy() for w in net.weights]
    biases = [b.copy() for b in net.biases]
    for layer in range(net.spec.n_layers - 1):
        order = rng.permutation(weights[layer].shape[0])
        weights[layer] = weights[layer][order]
        biases[layer] = biases[layer][order]
        weights[layer + 1] = weights[layer + 1][:, order]
    return Network(net.spec, weights, biases)


# ── the Brief's worked example ─────────────────────────────────────────────────────────────

A = [[0.4, -0.2, 0.7, 0.0]]
B = [[-0.6, 0.5, 0.1, 0.3]]
U = [[0.31, 0.77, 0.12, 0.58]]
V = [[0.62, 0.04, 0.91, 0.30]]
Z = [[-0.8, 1.2, 0.5, -1.5]]


def test_the_briefs_worked_example():
    crossed = uniform_crossover(A, B, ScriptedRng(("random", [0.0]), ("random", U)), rate=1.0)
    assert isinstance(crossed, CrossoverResult)
    np.testing.assert_array_equal(crossed.genomes, [[0.4, 0.5, 0.7, 0.3]])
    np.testing.assert_array_equal(crossed.from_a, [[True, False, True, False]])
    np.testing.assert_array_equal(crossed.crossed, [True])

    mutated = gaussian_mutation(
        crossed.genomes, ScriptedRng(("random", V), ("normal", Z)), rate=0.25, strength=0.1
    )
    assert isinstance(mutated, MutationResult)
    np.testing.assert_allclose(mutated.genomes, [[0.4, 0.62, 0.7, 0.3]], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(mutated.mutated, [[False, True, False, False]])

    # the same draws at p = 0.5: gene 3 is mutated as well, gene 1 gets the same nudge
    wider = gaussian_mutation(
        crossed.genomes, ScriptedRng(("random", V), ("normal", Z)), rate=0.5, strength=0.1
    )
    np.testing.assert_allclose(wider.genomes, [[0.4, 0.62, 0.7, 0.15]], rtol=0, atol=1e-15)
    np.testing.assert_array_equal(wider.mutated, [[False, True, False, True]])


# ── mutation statistics ────────────────────────────────────────────────────────────────────

N_ROWS = 2000


@pytest.fixture(scope="module")
def parents() -> np.ndarray:
    return random_genomes(DEFAULT, N_ROWS, make_rng(7))


@pytest.mark.parametrize("rate", [0.01, 0.05, 0.3])
def test_mutated_fraction_matches_the_rate(parents, rate):
    out = gaussian_mutation(parents, make_rng(11), rate=rate, strength=0.1)
    n = out.mutated.size
    count = int(out.mutated.sum())
    sd = math.sqrt(n * rate * (1 - rate))
    assert abs(count - n * rate) < 5 * sd, (count, n * rate, sd)


def test_every_gene_position_is_equally_likely_to_mutate(parents):
    rate = 0.05
    out = gaussian_mutation(parents, make_rng(12), rate=rate)
    per_gene = out.mutated.sum(axis=0)
    stat = binomial_chi2(per_gene, N_ROWS, rate)
    assert stat < chi2_critical_999(DEFAULT.n_parameters), stat


def test_mutation_changes_are_normal_with_mean_0_and_sd_sigma(parents):
    sigma = 0.1
    out = gaussian_mutation(parents, make_rng(13), rate=0.05, strength=sigma)
    changes = (out.genomes - parents)[out.mutated]
    n = changes.size
    assert abs(changes.mean()) < 5 * sigma / math.sqrt(n)
    assert abs(changes.std() - sigma) < 5 * sigma / math.sqrt(2 * n)
    # Kolmogorov-Smirnov: the largest gap between the changes' cumulative share and the normal
    # curve's, against the 0.1 % critical value 1.95 / sqrt(n)
    x = np.sort(changes / sigma)
    cdf = np.array([0.5 * (1 + math.erf(t / math.sqrt(2))) for t in x])
    rank = np.arange(1, n + 1)
    gap = max(float(np.max(rank / n - cdf)), float(np.max(cdf - (rank - 1) / n)))
    assert gap < 1.95 / math.sqrt(n), gap


def test_unmutated_genes_are_exact_copies(parents):
    out = gaussian_mutation(parents, make_rng(14), rate=0.05)
    np.testing.assert_array_equal(out.genomes[~out.mutated], parents[~out.mutated])
    assert (out.genomes[out.mutated] != parents[out.mutated]).all()


def test_rate_0_copies_rate_1_changes_everything_strength_0_changes_nothing(parents):
    g = parents[:20]
    none = gaussian_mutation(g, make_rng(15), rate=0.0)
    np.testing.assert_array_equal(none.genomes, g)
    assert not none.mutated.any()
    every = gaussian_mutation(g, make_rng(15), rate=1.0)
    assert every.mutated.all()
    assert (every.genomes != g).all()
    still = gaussian_mutation(g, make_rng(15), rate=0.5, strength=0.0)
    np.testing.assert_array_equal(still.genomes, g)
    assert still.mutated.any()  # the genes were picked, with a nudge of size 0


def test_a_higher_rate_only_adds_mutated_genes_with_the_same_nudges(parents):
    low = gaussian_mutation(parents, make_rng(16), rate=0.05)
    high = gaussian_mutation(parents, make_rng(16), rate=0.1)
    assert (low.mutated <= high.mutated).all()
    assert high.mutated.sum() > low.mutated.sum()
    np.testing.assert_array_equal(low.genomes[low.mutated], high.genomes[low.mutated])


def test_mutation_is_seeded_and_leaves_its_input_alone(parents):
    g = parents[:50].copy()
    first = gaussian_mutation(g, make_rng(17))
    np.testing.assert_array_equal(g, parents[:50])
    again = gaussian_mutation(g, make_rng(17))
    other = gaussian_mutation(g, make_rng(18))
    np.testing.assert_array_equal(first.genomes, again.genomes)
    np.testing.assert_array_equal(first.mutated, again.mutated)
    assert not np.array_equal(first.genomes, other.genomes)


# ── crossover statistics ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def distinct_parents() -> tuple[np.ndarray, np.ndarray]:
    """Parents with every value different (a > 0, b < 0), so each child gene names its source."""
    shape = (N_ROWS, DEFAULT.n_parameters)
    a = 1.0 + np.arange(np.prod(shape), dtype=np.float64).reshape(shape)
    return a, -a


def test_each_gene_comes_from_one_parent_at_the_same_position(distinct_parents):
    a, b = distinct_parents
    out = uniform_crossover(a, b, make_rng(21))
    assert out.crossed.all()
    np.testing.assert_array_equal(out.genomes[out.from_a], a[out.from_a])
    np.testing.assert_array_equal(out.genomes[~out.from_a], b[~out.from_a])
    assert ((out.genomes == a) | (out.genomes == b)).all()


def test_half_the_genes_come_from_each_parent_at_every_position(distinct_parents):
    a, b = distinct_parents
    out = uniform_crossover(a, b, make_rng(22))
    n = out.from_a.size
    assert abs(int(out.from_a.sum()) - n / 2) < 5 * math.sqrt(n / 4)
    stat = binomial_chi2(out.from_a.sum(axis=0), N_ROWS, 0.5)
    assert stat < chi2_critical_999(DEFAULT.n_parameters), stat


def test_neighbouring_genes_are_chosen_independently(distinct_parents):
    a, b = distinct_parents
    out = uniform_crossover(a, b, make_rng(23))
    same = out.from_a[:, 1:] == out.from_a[:, :-1]
    n = same.size
    # one cut or two would keep neighbours together almost always; fair coins do it half the time
    assert abs(int(same.sum()) - n / 2) < 5 * math.sqrt(n / 4)


def test_the_crossover_rate_and_uncrossed_children_copy_parent_a():
    rows, rate = 20_000, 0.7
    a = make_rng(24).standard_normal((rows, 5))
    b = make_rng(25).standard_normal((rows, 5))
    out = uniform_crossover(a, b, make_rng(26), rate=rate)
    assert abs(int(out.crossed.sum()) - rows * rate) < 5 * math.sqrt(rows * rate * (1 - rate))
    kept = ~out.crossed
    np.testing.assert_array_equal(out.genomes[kept], a[kept])
    assert out.from_a[kept].all()
    assert not out.from_a[out.crossed].all(axis=1).all()  # crossed rows do take genes from b
    never = uniform_crossover(a, b, make_rng(26), rate=0.0)
    np.testing.assert_array_equal(never.genomes, a)
    assert not never.crossed.any()


def test_crossing_a_genome_with_itself_gives_it_back(parents):
    out = uniform_crossover(parents, parents, make_rng(27))
    np.testing.assert_array_equal(out.genomes, parents)


def test_crossover_is_seeded_and_leaves_its_inputs_alone(distinct_parents):
    a, b = (p[:50].copy() for p in distinct_parents)
    first = uniform_crossover(a, b, make_rng(28))
    np.testing.assert_array_equal(a, distinct_parents[0][:50])
    np.testing.assert_array_equal(b, distinct_parents[1][:50])
    again = uniform_crossover(a, b, make_rng(28))
    other = uniform_crossover(a, b, make_rng(29))
    np.testing.assert_array_equal(first.genomes, again.genomes)
    assert not np.array_equal(first.genomes, other.genomes)


# ── why crossover is awkward for networks: permutation symmetry ────────────────────────────


def test_crossing_a_network_with_its_renumbered_twin_breaks_it():
    net = Network.unflatten(DEFAULT, random_genomes(DEFAULT, 1, make_rng(31))[0])
    twin = permute_hidden(net, make_rng(32))
    obs = make_rng(33).uniform(-1.0, 1.0, (200, 7))
    reference = predict(net, obs)
    np.testing.assert_allclose(predict(twin, obs), reference, rtol=0, atol=1e-12)
    # most of the 400 weights moved (382 here; a neuron the shuffle leaves in place keeps its
    # row); the 34 biases start at 0, so renumbering cannot change them
    assert int((twin.flatten() != net.flatten()).sum()) > 350

    copies = np.tile(net.flatten(), (50, 1))
    twins = np.tile(twin.flatten(), (50, 1))
    with_self = uniform_crossover(copies, copies, make_rng(34))
    for child in networks_of(DEFAULT, with_self.genomes):
        np.testing.assert_array_equal(predict(child, obs), reference)

    with_twin = uniform_crossover(copies, twins, make_rng(34))
    gaps = [
        float(np.abs(predict(child, obs) - reference).max())
        for child in networks_of(DEFAULT, with_twin.genomes)
    ]
    # both parents give identical outputs, yet every child's outputs differ from them
    assert min(gaps) > 0.05, min(gaps)


# ── refused inputs ─────────────────────────────────────────────────────────────────────────


def test_bad_inputs_are_refused():
    g = np.zeros((3, 4))
    rng = make_rng(0)
    with pytest.raises(ValueError, match="genome matrix"):
        gaussian_mutation(np.zeros(4), rng)
    with pytest.raises(ValueError, match="finite"):
        gaussian_mutation(np.array([[0.0, np.nan]]), rng)
    for rate in (-0.1, 1.5, True):
        with pytest.raises(ValueError, match="mutation rate"):
            gaussian_mutation(g, rng, rate=rate)
    for strength in (-0.1, np.inf, True):
        with pytest.raises(ValueError, match="mutation strength"):
            gaussian_mutation(g, rng, strength=strength)
    with pytest.raises(ValueError, match="same shape"):
        uniform_crossover(g, np.zeros((3, 5)), rng)
    with pytest.raises(ValueError, match="genome matrix"):
        uniform_crossover(np.zeros(4), np.zeros(4), rng)
    with pytest.raises(ValueError, match="finite"):
        uniform_crossover(g, np.full((3, 4), np.inf), rng)
    for rate in (-0.1, 1.5):
        with pytest.raises(ValueError, match="crossover rate"):
            uniform_crossover(g, g, rng, rate=rate)
