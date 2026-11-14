"""A GA run's family history (task 7.8, ADR-060): the line of descent and the mutation diff.

Every line g of a GA run's ``extra/lineage.jsonl`` says how each row of generation g came about
from generation g - 1 (``learn/evolution/ga.py: lineage_record``): ``random`` (generation 1),
``elite`` (an exact copy of row r) or ``child`` (parents a and b, crossed or not, B the genes from
b, M the mutated genes). This module turns those records into the two views of task 7.8.

Symbols: P genes per genome; x the child's genome; for gene k its *source parent* s_k (b when k is
in B, else a) and its inherited value y_k (gene k of the source parent's genome, generation
g - 1). The *main parent* m gave more genes (a when P - |B| >= |B|), o is the other.

**Mutation diff.** For every gene: delta_k = x_k - y_k when k is in M, else 0 (and x_k = y_k
exactly, because crossover copies values; a folder where that fails is damaged). D = the largest
|delta_k|, the drawn change c_k = delta_k / D (all 0 when nothing was mutated). Genes from o are
marked, so the diagram can draw them grey. Comparing with the source parent, not with m, keeps
crossover out of the colours. Worked example (the 7.8 Brief): generation 1 rows (0, 0),
(0.5, -0.5), (1, 1), (0.2, 0.4); child (0.6, 0.4) of a = 3, b = 1 with B = M = [0]: m = a (a tie),
y = (0.5, 0.4), delta = (+0.1, 0), c = (+1, 0), gene 0 from o. Child (0.0, 0.5) of (3, 3) with
M = [0, 1]: delta = (-0.2, +0.1), c = (-1, +0.5).

**Line of descent.** From (g, r): note the step, then go to (g - 1, main parent) (an elite's copied
row) until a random genome of generation 1, or a generation whose origins are unknown.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.learn.evolution.ga import main_parent, ranking

ORIGINS = ("random", "elite", "child", "unknown")


@dataclass(frozen=True)
class AncestryStep:
    """One network on a line of descent: generation ``unit``, ``row``, its fitness F and rank
    (0 = best), ``origin``, ``parents`` (rows of generation unit - 1: one for an elite, a and b for
    a child, none otherwise), ``crossed`` (children only), ``main_parent`` (the next step's row),
    the genes it took from the main and the other parent, the number of mutated genes, and
    ``distance`` = the Euclidean distance of its genome from the main parent's (None when the
    genomes are not given or there is no parent)."""

    unit: int
    row: int
    fitness: float
    rank: int
    origin: str
    parents: list[int]
    crossed: bool | None
    main_parent: int | None
    from_main: int | None
    from_other: int | None
    mutated: int | None
    distance: float | None


def _lines_by_unit(lines: Sequence[Mapping[str, Any]], name: str) -> dict[int, Mapping[str, Any]]:
    out: dict[int, Mapping[str, Any]] = {}
    for line in lines:
        unit = line.get("unit")
        if not isinstance(unit, int):
            raise ValueError(f"a line of {name} has no unit number")
        out[unit] = line
    return out


def _rows(lineage: Mapping[int, Mapping[str, Any]], unit: int) -> list[dict[str, Any]] | None:
    """Generation ``unit``'s lineage records, None when they are unknown."""
    if unit not in lineage:
        raise ValueError(
            f"generation {unit} has no lineage line (a run trained before task 7.8 has none): "
            "train the run again to see its ancestry"
        )
    return lineage[unit].get("rows")


def ancestry(
    lineage_lines: Sequence[Mapping[str, Any]],
    population_lines: Sequence[Mapping[str, Any]],
    unit: int,
    row: int | None,
    n_genes: int,
    genomes: Mapping[int, ArrayLike] | None = None,
) -> list[AncestryStep]:
    """The line of descent of row ``row`` of generation ``unit`` (default: its champion).

    1. Start at (unit, row); look up its fitness (population line) and lineage record.
    2. Note the step: origin, parents, the genes from each parent, the mutated genes and, with
       ``genomes`` ({unit: matrix (N, P)}), the distance to the main parent.
    3. Stop at a random genome or an unknown origin; else go to (unit - 1, main parent).

    The steps run from generation ``unit`` back to the first. ``ValueError`` in plain words for a
    generation without its lines, a row that does not exist or records that do not fit.
    """
    lineage = _lines_by_unit(lineage_lines, "extra/lineage.jsonl")
    population = _lines_by_unit(population_lines, "extra/population.jsonl")
    if unit not in population:
        raise ValueError(f"there is no generation {unit}")
    if row is None:
        row = int(population[unit]["champion"])
    steps: list[AncestryStep] = []
    u, r = unit, row
    while True:
        if u not in population:
            raise ValueError(f"generation {u} has no population line: the run folder is damaged")
        fitness = population[u]["fitness"]
        if not 0 <= r < len(fitness):
            raise ValueError(f"generation {u} has {len(fitness)} networks; there is no row {r}")
        rows = _rows(lineage, u)
        if rows is not None and len(rows) != len(fitness):
            raise ValueError(
                f"generation {u}'s lineage has {len(rows)} rows for {len(fitness)} networks"
            )
        _, ranks = ranking(fitness)
        record = rows[r] if rows is not None else {"origin": "unknown"}
        origin = record["origin"]
        if origin not in ORIGINS:
            raise ValueError(f"unknown origin {origin!r} in generation {u}, row {r}")
        if origin == "random" and u != 1:
            raise ValueError(f"generation {u}, row {r} says random, but only generation 1 is")
        if origin in ("elite", "child") and u == 1:
            raise ValueError(f"generation 1, row {r} cannot have parents")
        parent = main_parent(record, n_genes) if origin in ("elite", "child") else None
        from_main = from_other = mutated = None
        crossed = None
        if origin == "child":
            n_b = len(record["from_b"])
            main_is_a = parent == record["parents"][0]
            from_main, from_other = (n_genes - n_b, n_b) if main_is_a else (n_b, n_genes - n_b)
            if record["parents"][0] == record["parents"][1]:  # one parent twice
                from_main, from_other = n_genes, 0
            mutated, crossed = len(record["mutated"]), bool(record["crossed"])
        elif origin == "elite":
            from_main, from_other, mutated = n_genes, 0, 0
        distance = None
        if parent is not None and genomes is not None and u in genomes and u - 1 in genomes:
            child, older = np.asarray(genomes[u]), np.asarray(genomes[u - 1])
            distance = float(np.sqrt(((child[r] - older[parent]) ** 2).sum()))
        steps.append(
            AncestryStep(
                unit=u,
                row=r,
                fitness=float(fitness[r]),
                rank=int(ranks[r]),
                origin=origin,
                parents=[int(p) for p in record.get("parents", [])],
                crossed=crossed,
                main_parent=parent,
                from_main=from_main,
                from_other=from_other,
                mutated=mutated,
                distance=distance,
            )
        )
        if parent is None:
            return steps
        u, r = u - 1, parent


@dataclass(frozen=True)
class MutationChange:
    """What made one network differ from its parents, gene by gene (each array has P entries).

    ``delta`` = x - y for mutated genes (0 elsewhere), ``change`` = delta / D with D =
    ``max_abs_delta``, ``mutated`` and ``from_other`` the two masks; ``main_parent`` and
    ``other_parent`` rows of the previous generation (other None for an elite or one parent
    used twice)."""

    origin: str
    parents: list[int]
    crossed: bool | None
    main_parent: int
    other_parent: int | None
    delta: NDArray[np.float64]
    change: NDArray[np.float64]
    mutated: NDArray[np.bool_]
    from_other: NDArray[np.bool_]
    max_abs_delta: float

    @property
    def n_mutated(self) -> int:
        return int(self.mutated.sum())

    @property
    def n_from_other(self) -> int:
        return int(self.from_other.sum())


def mutation_change(
    record: Mapping[str, Any], child: ArrayLike, previous: ArrayLike
) -> MutationChange:
    """The mutation diff of one network from its lineage ``record``, its genome ``child`` (P,)
    and the previous generation's genome matrix ``previous`` (N, P).

    1. Refuse a random genome (it has no parent) or an unknown origin.
    2. Find the main parent m and the other parent o (``main_parent``).
    3. For every gene, its source parent (b for the genes in B, else a; an elite's copied row) and
       inherited value y_k; delta_k = x_k - y_k for the mutated genes, 0 for the others.
    4. Check every unmutated gene equals y_k exactly (else the folder is damaged).
    5. D = the largest |delta_k|; change = delta / D (0 everywhere when D = 0).
    """
    x = np.asarray(child, dtype=np.float64)
    old = np.asarray(previous, dtype=np.float64)
    if x.ndim != 1 or old.ndim != 2 or old.shape[1] != x.shape[0]:
        raise ValueError(f"need a genome (P,) and a matrix (N, P), got {x.shape} and {old.shape}")
    p = x.shape[0]
    origin = record.get("origin")
    if origin == "random":
        raise ValueError("a network of generation 1 is random: it has no parent to compare with")
    if origin not in ("elite", "child"):
        raise ValueError("this generation's origins are unknown (a run resumed from before 7.8)")
    parents = [int(v) for v in record["parents"]]
    if any(not 0 <= v < old.shape[0] for v in parents):
        raise ValueError(f"parents {parents} are not rows of the previous generation")
    m = main_parent(record, p)
    assert m is not None
    mutated = np.zeros(p, dtype=bool)
    from_b = np.zeros(p, dtype=bool)
    if origin == "child":
        mutated[np.asarray(record["mutated"], dtype=np.intp)] = True
        from_b[np.asarray(record["from_b"], dtype=np.intp)] = True
        a, b = parents
        source = np.where(from_b, old[b], old[a])
        other = None if a == b else (b if m == a else a)
        from_other = (from_b if m == a else ~from_b) if other is not None else np.zeros(p, bool)
        crossed: bool | None = bool(record["crossed"])
    else:
        source = old[m]
        other, from_other, crossed = None, np.zeros(p, dtype=bool), None
    if not np.array_equal(x[~mutated], source[~mutated]):
        raise ValueError(
            "a gene that was not mutated differs from the parent it came from: "
            "the run folder is damaged"
        )
    delta = np.where(mutated, x - source, 0.0)
    biggest = float(np.max(np.abs(delta), initial=0.0))
    change = delta / biggest if biggest > 0 else np.zeros(p)
    return MutationChange(
        origin=str(origin),
        parents=parents,
        crossed=crossed,
        main_parent=m,
        other_parent=other,
        delta=delta,
        change=change,
        mutated=mutated,
        from_other=from_other,
        max_abs_delta=biggest,
    )


def as_layers(
    spec: NetworkSpec, genes: ArrayLike
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """A per-gene vector (P,) in the network's shapes: the weight matrices W_l[j, i] and bias
    vectors b_l, by the gene map of task 7.1 (``Network.unflatten``)."""
    net = Network.unflatten(spec, np.asarray(genes, dtype=np.float64))
    return list(net.weights), list(net.biases)
