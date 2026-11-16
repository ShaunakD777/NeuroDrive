"""The genetic algorithm, part 2 (task 7.3): selection, turning a generation's fitness into parents.

Selection is the only step of a GA where "better" becomes "more children". Both operators here
use only the *order* of the fitness values (which network is better, never by how much), so any
increasing change of the fitness (doubling it, adding 3, taking exp) picks exactly the same
parents. Networks with equal fitness always get equal chances: the row index never decides
(ADR-055; in early generations many networks stall and tie at exactly 0).

Symbols: N networks with fitness F_i; a network's place r in the ranking (0 = best, N - 1 =
worst); n parents to pick; p_i the probability that one pick chooses network i, so network i's
expected number of copies is n * p_i. *Selection pressure* is the expected copies of the best.

Tournament selection, size k. Each tournament draws k networks uniformly *with replacement*; the
fittest contestant wins, and among equally fit contestants the one drawn first (the draw order is
random, so this is uniform among them). For a fitness level v, with m_le the number of networks
with F <= v and m_lt the number with F < v, every contestant is at most v and at least one is
exactly v with probability

    P(winner has fitness v) = (m_le / N)^k - (m_lt / N)^k,

shared equally by the m_le - m_lt networks at that level. Without ties this is
p(r) = ((N - r) / N)^k - ((N - r - 1) / N)^k; the best gets about k copies when N is large.

Rank selection, pressure s in [1, 2] (linear ranking, Baker 1985). The network at place r gets

    e(r) = s - 2 (s - 1) r / (N - 1)

expected copies per N picks (the best s, the worst 2 - s, equal steps, mean exactly 1), so
p = e / N. Tied networks share the mean e of the places they fill together (e is linear in r, so
that is e at their mean place). The parents are then drawn by stochastic universal sampling.

Worked example: F = (0.3, 1.2, 0.3, 1.9), N = n = 4. Tournament k = 2: network 3 wins with
1 - (3/4)^2 = 7/16, network 1 with (3/4)^2 - (2/4)^2 = 5/16, and networks 0 and 2 share
(2/4)^2 = 4/16, so p = (0.125, 0.3125, 0.125, 0.4375). Rank s = 1.5: e = 1.5, 1.1667 for places
0 and 1, and the tied pair shares (0.8333 + 0.5) / 2 = 0.6667, so p = (0.1667, 0.2917, 0.1667,
0.375).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.rng import Rng

Array = NDArray[np.float64]
Indices = NDArray[np.intp]


def _fitness(fitness: ArrayLike) -> Array:
    """The fitness as a checked 1-D float array (not empty, every value finite)."""
    f = np.asarray(fitness, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError(f"fitness must be 1-D, got shape {f.shape}")
    if f.size == 0:
        raise ValueError("fitness is empty: there is nobody to select")
    if not np.isfinite(f).all():
        raise ValueError("fitness must be finite (no NaN or infinity)")
    return f


def _count(n: int) -> int:
    """The number of parents to pick, checked: a whole number, 0 or more."""
    if isinstance(n, bool) or not isinstance(n, int | np.integer):
        raise ValueError(f"the number of parents must be a whole number, got {n!r}")
    if n < 0:
        raise ValueError(f"the number of parents must be 0 or more, got {n}")
    return int(n)


def _size(size: int) -> int:
    """The tournament size k, checked: a whole number, at least 1."""
    if isinstance(size, bool) or not isinstance(size, int | np.integer) or size < 1:
        raise ValueError(f"the tournament size must be a whole number of at least 1, got {size!r}")
    return int(size)


def _levels(f: Array) -> tuple[Array, Array]:
    """For every network i: m_lt, the number of networks with F < F_i, and m_le, with F <= F_i."""
    s = np.sort(f)
    m_lt = np.searchsorted(s, f, side="left").astype(np.float64)
    m_le = np.searchsorted(s, f, side="right").astype(np.float64)
    return m_lt, m_le


def tournament_probabilities(fitness: ArrayLike, size: int = 3) -> Array:
    """p_i, the probability that one tournament of ``size`` contestants (k) is won by network i.

    Per network: ((m_le / N)^k - (m_lt / N)^k) / (m_le - m_lt), its fitness level's chance of
    providing the winner, shared by everyone at that level. The values sum to 1; k = 1 gives
    every network 1 / N.
    """
    f = _fitness(fitness)
    k = _size(size)
    n_net = float(f.size)
    m_lt, m_le = _levels(f)
    level = (m_le / n_net) ** k - (m_lt / n_net) ** k
    return level / (m_le - m_lt)


def rank_probabilities(fitness: ArrayLike, pressure: float = 1.5) -> Array:
    """p_i for linear ranking with selection pressure s: e(mean place of i's level) / N.

    A fitness level filled by m_le - m_lt tied networks occupies the places N - m_le to
    N - m_lt - 1, so its mean place is (2N - m_le - m_lt - 1) / 2. s = 1 gives every network
    1 / N; s = 2 gives the worst network 0. A single network gets probability 1.
    """
    f = _fitness(fitness)
    s = float(pressure)
    if not 1.0 <= s <= 2.0:
        raise ValueError(f"the rank selection pressure must be between 1 and 2, got {pressure!r}")
    n_net = f.size
    if n_net == 1:
        return np.ones(1)
    m_lt, m_le = _levels(f)
    mean_place = (2 * n_net - m_le - m_lt - 1) / 2
    expected = s - 2 * (s - 1) * mean_place / (n_net - 1)
    return expected / n_net


def stochastic_universal_sampling(probabilities: ArrayLike, n: int, rng: Rng) -> Indices:
    """Pick n indices with one spin of a wheel that has n equally spaced pointers.

    The wheel has length 1 and gives index i a slice of length p_i, in index order. One offset u
    is drawn from [0, 1 / n); the pointers sit at u + j / n for j = 0 .. n - 1, and each picks the
    slice it lands in. So index i is picked either floor(n p_i) or ceil(n p_i) times, and its
    expected count is exactly n p_i. The picks come out in wheel order (sorted by index); shuffle
    them before pairing parents. ``probabilities`` must be non-negative with a positive sum (it is
    divided by its sum, which also makes the last slice end exactly at 1).
    """
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 1 or p.size == 0:
        raise ValueError(f"probabilities must be a non-empty 1-D array, got shape {p.shape}")
    if not np.isfinite(p).all() or (p < 0).any():
        raise ValueError("probabilities must be finite and non-negative")
    count = _count(n)
    edges = np.cumsum(p)
    if edges[-1] <= 0:
        raise ValueError("probabilities must not all be zero")
    if count == 0:
        return np.empty(0, dtype=np.intp)
    edges = edges / edges[-1]
    offset = rng.random() / count
    pointers = offset + np.arange(count) / count
    # side="right": a pointer exactly on a slice's end belongs to the next slice, so a slice of
    # length 0 is never picked; every pointer is below 1 = the last edge, so no index runs past
    return np.searchsorted(edges, pointers, side="right").astype(np.intp)


def tournament_selection(fitness: ArrayLike, n: int, rng: Rng, size: int = 3) -> Indices:
    """Pick n parents by n tournaments of ``size`` contestants drawn with replacement.

    1. Draw an n x k table of network indices, uniformly from 0 .. N - 1.
    2. In each row, the contestant with the largest fitness wins; among equals the first drawn.
    3. The winners, in the order of the tournaments, are the parents.
    """
    f = _fitness(fitness)
    k = _size(size)
    count = _count(n)
    draws = rng.integers(0, f.size, size=(count, k))
    winner = np.argmax(f[draws], axis=1)  # argmax returns the first of equal maxima
    return draws[np.arange(count), winner].astype(np.intp)


def rank_selection(fitness: ArrayLike, n: int, rng: Rng, pressure: float = 1.5) -> Indices:
    """Pick n parents by linear ranking with pressure s and stochastic universal sampling.

    1. p = ``rank_probabilities(fitness, pressure)`` (ties share their places' mean).
    2. One spin of the n-pointer wheel (``stochastic_universal_sampling``).
    3. Shuffle the picks, so that pairing parents later does not follow the row order.
    """
    p = rank_probabilities(fitness, pressure)
    picks = stochastic_universal_sampling(p, _count(n), rng)
    return rng.permutation(picks).astype(np.intp)
