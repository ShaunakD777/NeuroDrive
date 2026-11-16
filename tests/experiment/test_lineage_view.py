"""Tests for the lineage views (task 7.8, ADR-060): the Brief's worked example, the main parent,
the line of descent on a real small GA run, and the refusals."""

import numpy as np
import pytest

from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.experiment.config import parse_config
from neurodrive.experiment.lineage_view import (
    ancestry,
    as_layers,
    mutation_change,
)
from neurodrive.experiment.runner import train_ga_run

# ── the Brief's worked example (P = 2, N = 4, E = 1) ──────────────────────────────────────────

GEN1 = np.array([[0.0, 0.0], [0.5, -0.5], [1.0, 1.0], [0.2, 0.4]])
GEN2 = np.array([[0.2, 0.4], [0.6, 0.4], [0.5, -0.5], [0.0, 0.5]])
ROWS2 = [
    {"origin": "elite", "parents": [3]},
    {"origin": "child", "parents": [3, 1], "crossed": True, "from_b": [0], "mutated": [0]},
    {"origin": "child", "parents": [1, 0], "crossed": False, "from_b": [], "mutated": []},
    {"origin": "child", "parents": [3, 3], "crossed": False, "from_b": [], "mutated": [0, 1]},
]


def test_the_crossed_and_mutated_child():
    d = mutation_change(ROWS2[1], GEN2[1], GEN1)
    assert d.main_parent == 3 and d.other_parent == 1  # 1 gene each: a tie goes to a
    assert d.delta == pytest.approx([0.1, 0.0], abs=1e-15)  # 0.6 - 0.5, not 0.6 - 0.2
    assert d.change == pytest.approx([1.0, 0.0], abs=1e-12)
    assert d.max_abs_delta == pytest.approx(0.1, abs=1e-15)
    assert d.mutated.tolist() == [True, False] and d.from_other.tolist() == [True, False]
    assert (d.n_mutated, d.n_from_other, d.crossed) == (1, 1, True)


def test_the_child_of_one_parent_twice():
    d = mutation_change(ROWS2[3], GEN2[3], GEN1)
    assert d.delta == pytest.approx([-0.2, 0.1], abs=1e-15)
    assert d.change == pytest.approx([-1.0, 0.5], abs=1e-12)
    assert d.other_parent is None and d.n_from_other == 0


def test_an_unmutated_copy_and_an_elite_change_nothing():
    copy = mutation_change(ROWS2[2], GEN2[2], GEN1)
    assert copy.main_parent == 1 and copy.other_parent == 0 and copy.n_from_other == 0
    elite = mutation_change(ROWS2[0], GEN2[0], GEN1)
    assert (elite.main_parent, elite.other_parent, elite.crossed) == (3, None, None)
    for d in (copy, elite):
        assert d.n_mutated == 0 and d.max_abs_delta == 0.0
        assert d.change.tolist() == [0.0, 0.0]


def test_the_main_parent_can_be_b():
    # b gave genes 0 and 1 of 3: b is the main parent, a's gene 2 is "from the other parent"
    record = {"origin": "child", "parents": [0, 2], "crossed": True, "from_b": [0, 1]}
    record["mutated"] = []
    old = np.array([[0.0, 0.0, 0.0], [9.0, 9.0, 9.0], [1.0, 2.0, 3.0]])
    d = mutation_change(record, np.array([1.0, 2.0, 0.0]), old)
    assert (d.main_parent, d.other_parent) == (2, 0)
    assert d.from_other.tolist() == [False, False, True]


def test_refusals():
    with pytest.raises(ValueError, match="no parent"):
        mutation_change({"origin": "random"}, GEN1[0], GEN1)
    with pytest.raises(ValueError, match="unknown"):
        mutation_change({"origin": "unknown"}, GEN1[0], GEN1)
    with pytest.raises(ValueError, match="damaged"):  # gene 1 is not mutated but is not a's
        mutation_change(ROWS2[1], np.array([0.6, 0.45]), GEN1)
    with pytest.raises(ValueError, match="rows of the previous"):
        mutation_change({"origin": "elite", "parents": [7]}, GEN1[0], GEN1)


def lines(*generations):
    """Population and lineage lines for the worked example's generations (fitness made up)."""
    fitness = {1: [0.3, 1.2, 0.3, 1.9], 2: [1.9, 2.0, 1.2, 0.1], 3: [2.0, 0.5, 0.4, 0.3]}
    champion = {1: 3, 2: 1, 3: 0}
    rows = {
        1: [{"origin": "random"}] * 4,
        2: ROWS2,
        3: [{"origin": "elite", "parents": [1]}]
        + [{"origin": "child", "parents": [0, 2], "crossed": False, "from_b": [], "mutated": [1]}]
        * 3,
    }
    population = [{"unit": u, "fitness": fitness[u], "champion": champion[u]} for u in generations]
    lineage = [{"unit": u, "rows": rows[u]} for u in generations]
    return lineage, population


def test_the_line_of_descent_of_the_worked_example():
    lineage, population = lines(1, 2, 3)
    steps = ancestry(lineage, population, 3, None, 2, genomes={1: GEN1, 2: GEN2})
    assert [(s.unit, s.row, s.origin) for s in steps] == [
        (3, 0, "elite"),
        (2, 1, "child"),
        (1, 3, "random"),
    ]
    elite, child, first = steps
    assert (elite.fitness, elite.rank, elite.mutated, elite.distance) == (2.0, 0, 0, None)
    assert child.parents == [3, 1] and child.crossed and child.main_parent == 3
    assert (child.from_main, child.from_other, child.mutated) == (1, 1, 1)
    assert child.distance == pytest.approx(0.4)  # (0.6, 0.4) - (0.2, 0.4)
    assert (first.fitness, first.rank, first.main_parent, first.parents) == (1.9, 0, None, [])
    # any row, not only the champion
    assert [s.row for s in ancestry(lineage, population, 2, 3, 2)] == [3, 3]


def test_the_line_stops_at_unknown_origins_and_refuses_missing_lines():
    lineage, population = lines(1, 2, 3)
    lineage[1] = {"unit": 2, "rows": None}  # resumed from a state saved before task 7.8
    steps = ancestry(lineage, population, 3, None, 2)
    assert [s.origin for s in steps] == ["elite", "unknown"]
    with pytest.raises(ValueError, match="no lineage line"):
        ancestry(lineage[2:], population, 3, None, 2)
    with pytest.raises(ValueError, match="no row 9"):
        ancestry(lineage, population, 3, 9, 2)
    with pytest.raises(ValueError, match="no generation 4"):
        ancestry(lineage, population, 4, None, 2)


def test_as_layers_follows_the_gene_map():
    spec = NetworkSpec(layer_sizes=(2, 2, 1))  # W0 (2 x 2), b0 (2), W1 (1 x 2), b1 (1): 9 genes
    weights, biases = as_layers(spec, np.arange(9.0))
    assert weights[0].tolist() == [[0.0, 1.0], [2.0, 3.0]] and biases[0].tolist() == [4.0, 5.0]
    assert weights[1].tolist() == [[6.0, 7.0]] and biases[1].tolist() == [8.0]


# ── a real small GA run ───────────────────────────────────────────────────────────────────────


def test_every_network_of_a_real_run_is_explained_by_its_record(tmp_path):
    config = parse_config(
        {
            "method": "ga",
            "tag": "lineage",
            "population": {"size": 6, "elite": 1},
            "variation": {"mutation_rate": 0.1, "mutation_strength": 0.2, "crossover_rate": 1.0},
            "fitness": {"start_fractions": [0.0], "lateral_offsets": [0.0], "target_laps": 1},
            "budget": {"units": 4},
            "exams": False,
        }
    )
    store = train_ga_run(config, tmp_path).store
    lineage, population = store.read_extra_lines("lineage"), store.read_extra_lines("population")
    genomes = {u: store.load_unit_arrays("generations", u)["genomes"] for u in range(1, 5)}
    p = genomes[1].shape[1]
    for line in lineage[1:]:
        g = line["unit"]
        for i, record in enumerate(line["rows"]):
            d = mutation_change(record, genomes[g][i], genomes[g - 1])
            assert d.n_mutated == len(record.get("mutated", []))
            # the inherited genes plus the nudges give the child back exactly
            if record["origin"] == "child":
                a, b = record["parents"]
                source = np.where(
                    np.isin(np.arange(p), record["from_b"]), genomes[g - 1][b], genomes[g - 1][a]
                )
                assert np.array_equal(np.where(d.mutated, genomes[g][i], source), genomes[g][i])
                assert np.allclose(source + d.delta, genomes[g][i], rtol=0, atol=1e-15)
    steps = ancestry(lineage, population, 4, None, p, genomes)
    assert [s.unit for s in steps] == [4, 3, 2, 1] and steps[-1].origin == "random"
    assert steps[0].row == population[3]["champion"]
    for s in steps:
        assert s.fitness == population[s.unit - 1]["fitness"][s.row]
