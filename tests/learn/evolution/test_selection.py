"""Tests for learn/evolution/selection.py (task 7.3): tournament and rank selection on fixed fitness
arrays. Exact probabilities are checked against hand values and brute-force enumeration; drawn
counts against n * p; plus order invariance, fair ties, determinism and errors."""

import itertools
import math

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.learn.evolution.selection import (
    rank_probabilities,
    rank_selection,
    stochastic_universal_sampling,
    tournament_probabilities,
    tournament_selection,
)

EXAMPLE = (0.3, 1.2, 0.3, 1.9)  # the Brief's worked example: ranking 3, 1, then 0 and 2 tied
# 10 networks with ties: four at exactly 0 (stalled), two at 0.5, the rest distinct
FIXED = np.array([0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 1.2, 0.0, 1.7, 0.9])


def brute_tournament(f: np.ndarray, k: int) -> np.ndarray:
    """Every one of the N^k equally likely draws, the first-drawn fittest contestant winning."""
    n = len(f)
    wins = np.zeros(n)
    for draw in itertools.product(range(n), repeat=k):
        wins[draw[int(np.argmax(f[list(draw)]))]] += 1
    return wins / n**k


def brute_rank(f: np.ndarray, s: float) -> np.ndarray:
    """Linear ranking with ties broken uniformly at random: average e(place) over every order of
    the networks that sorts the fitness from best to worst."""
    n = len(f)
    total = np.zeros(n)
    orders = 0
    for order in itertools.permutations(range(n)):
        if all(f[order[i]] >= f[order[i + 1]] for i in range(n - 1)):
            orders += 1
            for place, i in enumerate(order):
                total[i] += s - 2 * (s - 1) * place / (n - 1)
    return total / orders / n


# --- exact probabilities ---------------------------------------------------------------------


def test_worked_example_tournament() -> None:
    p = tournament_probabilities(EXAMPLE, size=2)
    np.testing.assert_allclose(p, [2 / 16, 5 / 16, 2 / 16, 7 / 16], rtol=0, atol=1e-15)
    np.testing.assert_allclose(4 * p, [0.5, 1.25, 0.5, 1.75], atol=1e-15)


def test_worked_example_rank() -> None:
    p = rank_probabilities(EXAMPLE, pressure=1.5)
    np.testing.assert_allclose(p, [1 / 6, 7 / 24, 1 / 6, 3 / 8], rtol=0, atol=1e-15)
    np.testing.assert_allclose(4 * p, [2 / 3, 7 / 6, 2 / 3, 1.5], atol=1e-15)


@pytest.mark.parametrize(
    "f",
    [
        np.array([0.4, 2.0, -1.0, 0.7, 1.1]),  # distinct
        np.array([0.0, 0.5, 0.0, 0.5, 0.0]),  # two tied levels
        np.array([1.0, 1.0, 1.0, 1.0, 1.0]),  # all equal
    ],
)
@pytest.mark.parametrize("k", [1, 2, 3])
def test_tournament_matches_brute_force(f: np.ndarray, k: int) -> None:
    np.testing.assert_allclose(tournament_probabilities(f, k), brute_tournament(f, k), atol=1e-14)


@pytest.mark.parametrize(
    "f",
    [
        np.array([0.4, 2.0, -1.0, 0.7, 1.1]),
        np.array([0.0, 0.5, 0.0, 0.5, 0.0]),
        np.array([3.0, 3.0, 1.0, 3.0, 2.0, 1.0]),
    ],
)
@pytest.mark.parametrize("s", [1.0, 1.5, 2.0])
def test_rank_matches_random_tie_breaking(f: np.ndarray, s: float) -> None:
    np.testing.assert_allclose(rank_probabilities(f, s), brute_rank(f, s), atol=1e-14)


def test_tournament_formula_without_ties() -> None:
    n, k = 50, 3
    f = np.linspace(-1.0, 1.0, n)[::-1]  # network i is at place i
    r = np.arange(n)
    want = ((n - r) / n) ** k - ((n - r - 1) / n) ** k
    p = tournament_probabilities(f, k)
    np.testing.assert_allclose(p, want, atol=1e-15)
    assert n * p[0] == pytest.approx(50 * (1 - 0.98**3))  # 2.94 copies of the best
    assert n * p[-1] == pytest.approx(1 / 50**2)  # the worst wins only if drawn 3 times


def test_rank_formula_without_ties() -> None:
    n, s = 50, 1.5
    f = np.arange(n, dtype=float)  # network n-1 is best
    e = n * rank_probabilities(f, s)
    assert e[-1] == pytest.approx(1.5) and e[0] == pytest.approx(0.5)
    np.testing.assert_allclose(np.diff(e), 1.0 / (n - 1), atol=1e-12)  # equal steps
    assert e.mean() == pytest.approx(1.0)


@pytest.mark.parametrize("n", [1, 2, 7, 50])
@pytest.mark.parametrize("seed", [0, 1])
def test_probabilities_sum_to_one(n: int, seed: int) -> None:
    f = np.round(make_rng(seed).normal(size=n), 1)  # rounding makes ties
    for k in (1, 2, 3, 7):
        assert tournament_probabilities(f, k).sum() == pytest.approx(1.0, abs=1e-12)
    for s in (1.0, 1.2, 1.5, 2.0):
        p = rank_probabilities(f, s)
        assert p.sum() == pytest.approx(1.0, abs=1e-12)
        assert (p >= 0).all()


def test_no_pressure_and_equal_fitness_are_uniform() -> None:
    f = np.array([0.1, 1.9, 0.5, 0.5, -2.0])
    np.testing.assert_allclose(tournament_probabilities(f, 1), 0.2, atol=1e-15)
    np.testing.assert_allclose(rank_probabilities(f, 1.0), 0.2, atol=1e-15)
    same = np.zeros(6)
    np.testing.assert_allclose(tournament_probabilities(same, 5), 1 / 6, atol=1e-15)
    np.testing.assert_allclose(rank_probabilities(same, 2.0), 1 / 6, atol=1e-15)
    np.testing.assert_array_equal(rank_probabilities([4.2], 1.5), [1.0])
    np.testing.assert_array_equal(tournament_probabilities([4.2], 3), [1.0])


# --- the index never decides; only the order matters -----------------------------------------


def test_permuting_fitness_permutes_probabilities() -> None:
    perm = make_rng(3).permutation(len(FIXED))
    for prob in (
        lambda f: tournament_probabilities(f, 3),
        lambda f: rank_probabilities(f, 1.5),
    ):
        np.testing.assert_allclose(prob(FIXED[perm]), prob(FIXED)[perm], atol=1e-15)
        p = prob(FIXED)
        zeros = FIXED == 0.0
        assert np.ptp(p[zeros]) == 0.0  # the four stalled networks: one shared chance
        assert p[3] == p[4]


@pytest.mark.parametrize(
    "select",
    [
        lambda f, n, g: tournament_selection(f, n, g, size=3),
        lambda f, n, g: rank_selection(f, n, g, pressure=1.5),
    ],
)
def test_increasing_change_of_fitness_picks_the_same_parents(select) -> None:
    base = select(FIXED, 500, make_rng(11))
    np.testing.assert_array_equal(select(10 * FIXED + 3, 500, make_rng(11)), base)
    np.testing.assert_array_equal(select(np.exp(FIXED), 500, make_rng(11)), base)


# --- drawn counts against expected counts ----------------------------------------------------


def test_tournament_counts_match_expectation() -> None:
    n = 200_000
    picks = tournament_selection(FIXED, n, make_rng(2026), size=3)
    assert picks.shape == (n,) and picks.dtype == np.intp
    counts = np.bincount(picks, minlength=len(FIXED))
    p = tournament_probabilities(FIXED, 3)
    expected = n * p
    sd = np.sqrt(n * p * (1 - p))
    assert (np.abs(counts - expected) < 5 * sd).all(), (counts, expected)
    chi2 = float(((counts - expected) ** 2 / expected).sum())
    assert chi2 < 27.877  # the 0.1 % critical value with 9 degrees of freedom


def test_tied_networks_win_tournaments_equally_often() -> None:
    # three networks tie for best at rows 0, 4 and 8: whoever is drawn first among them wins
    f = np.zeros(10)
    f[[0, 4, 8]] = 1.0
    n = 60_000
    counts = np.bincount(tournament_selection(f, n, make_rng(4), size=3), minlength=10)
    p = tournament_probabilities(f, 3)[0]  # (1 - 0.7^3) / 3 = 0.219
    assert p == pytest.approx((1 - 0.7**3) / 3)
    sd = math.sqrt(n * p * (1 - p))
    for i in (0, 4, 8):
        assert abs(counts[i] - n * p) < 5 * sd, counts
    # all equal: every network is equally likely, whatever its row
    same = np.bincount(tournament_selection(np.zeros(10), n, make_rng(5), 3), minlength=10)
    assert float(((same - n / 10) ** 2 / (n / 10)).sum()) < 27.877


def test_sus_copies_are_floor_or_ceil_and_average_to_expectation() -> None:
    reps = 5_000
    g = make_rng(7)
    for n in (10, 50, 7):
        p = rank_probabilities(FIXED, 1.5)
        e = n * p
        low, high = np.floor(e - 1e-9), np.ceil(e + 1e-9)
        total = np.zeros(len(FIXED))
        for _ in range(reps):
            counts = np.bincount(rank_selection(FIXED, n, g, 1.5), minlength=len(FIXED))
            assert counts.sum() == n
            assert ((counts >= low) & (counts <= high)).all(), (counts, e)
            total += counts
        frac = e - np.floor(e)
        se = np.sqrt(frac * (1 - frac) / reps)
        assert (np.abs(total / reps - e) <= 5 * se + 1e-9).all(), (total / reps, e)


def test_sus_on_the_worked_example() -> None:
    p = rank_probabilities(EXAMPLE, 1.5)  # 0.6667, 1.1667, 0.6667, 1.5 copies of 4
    g = make_rng(5)
    total = np.zeros(4)
    reps = 4_000
    for _ in range(reps):
        picks = stochastic_universal_sampling(p, 4, g)
        assert (np.diff(picks) >= 0).all()  # wheel order
        c = np.bincount(picks, minlength=4)
        assert c[3] in (1, 2) and c[1] in (1, 2) and c[0] in (0, 1) and c[2] in (0, 1)
        total += c
    np.testing.assert_allclose(total / reps, [2 / 3, 7 / 6, 2 / 3, 1.5], atol=0.04)


def test_full_pressure_never_picks_the_worst() -> None:
    g = make_rng(9)
    for f in (np.array([1.0, 2.0, 3.0, 0.5]), np.array([0.5, 1.0, 2.0, 3.0])):
        worst = int(np.argmin(f))
        assert rank_probabilities(f, 2.0)[worst] == 0.0
        for _ in range(2_000):
            assert worst not in rank_selection(f, 4, g, 2.0)


# --- determinism, edge cases, errors ---------------------------------------------------------


def test_same_seed_same_parents_other_seed_different() -> None:
    for select in (tournament_selection, rank_selection):
        a = select(FIXED, 50, make_rng(1))
        np.testing.assert_array_equal(select(FIXED, 50, make_rng(1)), a)
        assert not np.array_equal(select(FIXED, 50, make_rng(2)), a)


def test_zero_parents() -> None:
    for select in (tournament_selection, rank_selection):
        out = select(FIXED, 0, make_rng(0))
        assert out.shape == (0,) and out.dtype == np.intp


@pytest.mark.parametrize("bad", [[], [[0.1, 0.2]], [0.1, math.nan], [0.1, math.inf]])
def test_bad_fitness_is_refused(bad) -> None:
    with pytest.raises(ValueError):
        tournament_probabilities(bad, 2)
    with pytest.raises(ValueError):
        rank_selection(bad, 2, make_rng(0))


def test_bad_settings_are_refused() -> None:
    g = make_rng(0)
    with pytest.raises(ValueError, match="number of parents"):
        tournament_selection(FIXED, -1, g)
    with pytest.raises(ValueError, match="number of parents"):
        rank_selection(FIXED, 2.5, g)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tournament size"):
        tournament_selection(FIXED, 5, g, size=0)
    with pytest.raises(ValueError, match="tournament size"):
        tournament_probabilities(FIXED, True)  # type: ignore[arg-type]
    for s in (0.9, 2.1):
        with pytest.raises(ValueError, match="pressure"):
            rank_probabilities(FIXED, s)
    with pytest.raises(ValueError):
        stochastic_universal_sampling([0.0, 0.0], 3, g)
    with pytest.raises(ValueError):
        stochastic_universal_sampling([0.5, -0.1], 3, g)
