"""The genetic algorithm, part 3 (task 7.4): variation, making new genomes from the parents.

Selection (task 7.3) only decides *who* gets children; it never invents anything. The two
*variation operators* here are where new genomes come from: mutation nudges a few genes of a
child, and crossover builds a child from two parents, each gene from one of them. Both work on a
whole generation at once, as a genome matrix (N, P) with one genome per row (task 7.1), and both
return, next to the children, a record of what happened, so the lineage view (task 7.8) can show
exactly which genes were mutated and which parent each gene came from. Neither changes its inputs.

Symbols: N children of P genes; g a parent's genome and g' the child's; p the mutation rate and
sigma the mutation strength; c the crossover rate; u, v, w uniform random numbers in [0, 1) and z
a standard normal one (mean 0, standard deviation 1).

Gaussian mutation. For every gene i of every child, independently:

    g'_i = g_i + sigma * z_i   if v_i < p,   and   g'_i = g_i   otherwise.

So about P * p genes change per child (21.7 of the default network's 434 at p = 0.05), each by a
normal amount with standard deviation sigma (on average sigma * sqrt(2 / pi) = 0.08 in size at
sigma = 0.1). Every gene draws its v and z whatever p is, so with the same generator a higher
rate only *adds* mutated genes and never changes the nudge a gene gets: the mutated genes at
p = 0.05 are a subset of those at p = 0.1, which lets a rate sweep compare settings on the same
random numbers.

Uniform crossover. For a pair of parents a and b, one w per child and one u per gene:

    the child is a copy of a       if w >= c,
    g'_i = a_i if u_i < 0.5, else b_i   otherwise.

A crossed child takes on average half its genes from each parent, each gene independently of its
neighbours. Crossover never invents a value: every child gene is one parent's gene at the same
position. That is also why it is awkward for networks (permutation symmetry, see ``genome.py``):
two parents that drive identically can number their hidden neurons differently, and a gene-by-gene
mix of them can drive like neither.

Worked example (P = 4, c = 1, p = 0.25, sigma = 0.1): a = (0.4, -0.2, 0.7, 0.0) and
b = (-0.6, 0.5, 0.1, 0.3) with u = (0.31, 0.77, 0.12, 0.58) give the crossed child
(0.4, 0.5, 0.7, 0.3); then v = (0.62, 0.04, 0.91, 0.30) mutates only gene 1, and z_1 = 1.2 makes
it 0.5 + 0.12 = 0.62. At p = 0.5 gene 3 (v = 0.30) is mutated too: with z_3 = -1.5 it becomes
0.3 - 0.15 = 0.15, and gene 1 is still 0.62.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.rng import Rng

Genomes = NDArray[np.float64]
Mask = NDArray[np.bool_]


@dataclass(frozen=True)
class MutationResult:
    """The mutated children (N, P) and the mask (N, P) of the genes that were changed.

    ``mutated[n, i]`` is True when gene i of child n received a nudge sigma * z (with
    sigma = 0 the value is the same, but the gene still counts as mutated).
    """

    genomes: Genomes
    mutated: Mask


@dataclass(frozen=True)
class CrossoverResult:
    """The children (N, P), which parent each gene came from, and which children were crossed.

    ``from_a[n, i]`` is True when gene i of child n is parent a's, False when it is parent b's;
    a child that was not crossed (``crossed[n]`` False) is a copy of a, so its row is all True.
    """

    genomes: Genomes
    from_a: Mask
    crossed: Mask


def _genomes(genomes: ArrayLike, name: str) -> Genomes:
    """A genome matrix as a checked 2-D float array of finite numbers."""
    g = np.asarray(genomes, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"{name} must be a genome matrix (N, P), got shape {g.shape}")
    if not np.isfinite(g).all():
        raise ValueError(f"{name} must be finite (no NaN or infinity)")
    return g


def _probability(value: float, name: str) -> float:
    """A rate, checked: a number between 0 and 1."""
    if isinstance(value, bool):
        raise ValueError(f"the {name} must be a number between 0 and 1, got {value!r}")
    v = float(value)
    if not 0.0 <= v <= 1.0:
        raise ValueError(f"the {name} must be between 0 and 1, got {value!r}")
    return v


def gaussian_mutation(
    genomes: ArrayLike, rng: Rng, rate: float = 0.05, strength: float = 0.1
) -> MutationResult:
    """Mutate every row of a genome matrix: each gene, with probability p (``rate``), moves by
    sigma * z, a normal amount with standard deviation sigma (``strength``).

    1. Draw an (N, P) table of uniform numbers v, then an (N, P) table of standard normal z.
    2. The mask is v < p.
    3. The children are the genomes plus sigma * z where the mask is True; every other gene is
       copied exactly.

    Every gene draws its v and z at any p, so the same generator gives nested masks and equal
    nudges across rates. Raises ``ValueError`` for a non-2-D or non-finite matrix, p outside
    [0, 1], or a negative or non-finite sigma.
    """
    g = _genomes(genomes, "genomes")
    p = _probability(rate, "mutation rate")
    if isinstance(strength, bool) or not np.isfinite(sigma := float(strength)) or sigma < 0:
        raise ValueError(f"the mutation strength must be a finite number >= 0, got {strength!r}")
    v = rng.random(g.shape)
    z = rng.standard_normal(g.shape)
    mutated = v < p
    children = np.where(mutated, g + sigma * z, g)
    return MutationResult(genomes=children, mutated=mutated)


def uniform_crossover(
    parents_a: ArrayLike, parents_b: ArrayLike, rng: Rng, rate: float = 1.0
) -> CrossoverResult:
    """One child per row pair (a, b): crossed with probability c (``rate``), each gene from a
    or b by a fair coin; otherwise a copy of a.

    1. Draw N uniform numbers w (one per child), then an (N, P) table of uniform numbers u.
    2. A child is crossed when w < c.
    3. ``from_a`` is u < 0.5 in crossed rows and all True in the others.
    4. Each child gene is a's where ``from_a`` is True and b's elsewhere.

    Raises ``ValueError`` when the parent matrices are not 2-D, not finite, or of different
    shapes, or c is outside [0, 1].
    """
    a = _genomes(parents_a, "parents_a")
    b = _genomes(parents_b, "parents_b")
    if a.shape != b.shape:
        raise ValueError(
            f"the parent matrices must have the same shape, got {a.shape} and {b.shape}"
        )
    c = _probability(rate, "crossover rate")
    w = rng.random(a.shape[0])
    u = rng.random(a.shape)
    crossed = w < c
    from_a = (u < 0.5) | ~crossed[:, None]
    children = np.where(from_a, a, b)
    return CrossoverResult(genomes=children, from_a=from_a, crossed=crossed)
