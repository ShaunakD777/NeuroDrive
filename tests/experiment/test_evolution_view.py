"""Tests for the evolution charts' numbers (task 7.7, ADR-058): the statistics of a generation,
the fitness bands (a value on an edge goes into the band above), and agreement with what the GA
itself stored for a real (small) run."""

from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.experiment.config import parse_config
from neurodrive.experiment.evolution_view import (
    BAND_WIDTH,
    band_counts,
    band_edges,
    evolution_view,
    generation_stats,
)
from neurodrive.experiment.runner import train_ga_run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.evolution.ga import diversity

EXAMPLE = [0.0, 0.0, 0.12, 0.31, 1.27]  # the Brief's worked example


def test_worked_example_statistics():
    s = generation_stats(EXAMPLE)
    assert s.best == 1.27
    assert s.q3 == 0.31  # position 0.75 x 4 = 3
    assert s.median == 0.12
    assert s.mean == pytest.approx(0.34, abs=1e-15)
    assert s.q1 == 0.0  # position 0.25 x 4 = 1
    assert s.worst == 0.0
    assert s.q1 <= s.median <= s.q3


def test_quartiles_interpolate_between_neighbours():
    s = generation_stats([1.0, 2.0, 3.0, 4.0])  # positions 0.75 and 2.25
    assert s.q1 == pytest.approx(1.75)
    assert s.median == pytest.approx(2.5)
    assert s.q3 == pytest.approx(3.25)


def test_worked_example_bands():
    edges = band_edges([EXAMPLE])
    assert edges.size == 41 and edges[0] == 0.0 and edges[-1] == 2.0  # 40 bands of 0.05
    assert edges[3] == 0.15  # stored rounded, not 0.15000000000000002
    counts = band_counts(EXAMPLE, edges)
    assert counts.sum() == 5
    nonzero = np.nonzero(counts)[0]
    assert nonzero.tolist() == [0, 2, 6, 25]
    assert counts[nonzero].tolist() == [2, 1, 1, 1]


def test_a_value_on_an_edge_goes_into_the_band_above():
    edges = band_edges([[0.0]])
    assert 0.15 / BAND_WIDTH < 3  # why rounding F / w down would be wrong
    for value, band in [(0.0, 0), (0.05, 1), (0.15, 3), (0.1499999, 2), (1.9999, 39), (0.3, 6)]:
        counts = band_counts([value], edges)
        assert counts.argmax() == band and counts.sum() == 1, value


def test_bands_grow_to_hold_every_fitness():
    up = band_edges([[0.1], [2.3]])
    assert up[0] == 0.0 and up[-1] == 2.35  # 2.3 lies in [2.30, 2.35)
    assert band_counts([2.3], up).argmax() == 46
    exactly_two = band_edges([[2.0]])
    assert exactly_two[-1] == 2.05  # 2.0 needs the band [2.00, 2.05)
    down = band_edges([[-0.07, 0.4]])
    assert down[0] == -0.1 and down[-1] == 2.0
    assert band_counts([-0.07, -0.1, 0.0], down).tolist()[:3] == [2, 0, 1]
    assert band_edges([[0.0, 1.0]]).size == 41  # never fewer bands than 0 to 2


def test_counts_refuse_values_outside_the_bands():
    edges = band_edges([[0.5]])
    with pytest.raises(ValueError, match="outside"):
        band_counts([2.0], edges)
    with pytest.raises(ValueError, match="outside"):
        band_counts([-0.01], edges)


def test_counts_add_up_for_random_generations():
    rng = make_rng(0)
    rows = [rng.uniform(0.0, 1.99, 50) for _ in range(5)] + [np.zeros(50), np.full(50, 1.95)]
    edges = band_edges(rows)
    for row in rows:
        counts = band_counts(row, edges)
        assert counts.sum() == 50
        # each value is in the band whose edges hold it
        band = np.repeat(np.arange(counts.size), counts)
        assert np.array_equal(np.sort(row) >= edges[band], np.ones(50, bool))
        assert np.array_equal(np.sort(row) < edges[band + 1], np.ones(50, bool))


def _line(unit: int, fitness: list[float]) -> dict:
    return {"unit": unit, "fitness": fitness, "champion": int(np.argmax(fitness))}


def _record(unit: int, best: float) -> dict:
    return {"unit": unit, "best_fitness": best, "diversity": 1.0 / unit}


def test_view_of_hand_made_lines():
    metrics = [_record(1, 1.27), _record(2, 1.27)]
    lines = [_line(1, EXAMPLE), _line(2, [1.27, 0.2, 0.2, 0.0, 0.05])]
    view = evolution_view(metrics, lines, elite=1)
    assert view.population == 5 and view.elite == 1 and view.unit == [1, 2]
    assert view.zero == [2, 1]
    assert view.champion == [4, 0]
    assert view.diversity == [1.0, 0.5]
    assert view.finishers == [None, None]  # lines without them stay empty
    assert view.counts.shape == (2, 40) and view.counts.sum(axis=1).tolist() == [5, 5]
    assert view.counts[1, 4] == 2 and view.counts[1, 1] == 1


def test_view_refuses_damaged_runs():
    with pytest.raises(ValueError, match="no population line"):
        evolution_view([_record(1, 1.0), _record(2, 1.0)], [_line(1, EXAMPLE)])
    with pytest.raises(ValueError, match="networks"):
        evolution_view([_record(1, 1.0), _record(2, 1.0)], [_line(1, EXAMPLE), _line(2, [0.1])])


@pytest.fixture(scope="module")
def ga_run(tmp_path_factory) -> RunStore:
    """A small GA run: 6 networks, 1 elite, 2 starts, 1 lap, 3 generations."""
    config = parse_config(
        {
            "method": "ga",
            "seed": 2,
            "tag": "evo",
            "population": {"size": 6, "elite": 1},
            "variation": {"mutation_rate": 0.2, "mutation_strength": 0.3},
            "fitness": {"start_fractions": [0.0, 0.5], "lateral_offsets": [0.0], "target_laps": 1},
            "budget": {"units": 3},
            "exams": False,
        }
    )
    return train_ga_run(config, tmp_path_factory.mktemp("runs")).store


def test_view_agrees_with_what_the_ga_stored(ga_run: RunStore):
    metrics = ga_run.read_metrics()
    view = evolution_view(metrics, ga_run.read_extra_lines("population"), elite=1)
    assert view.unit == [1, 2, 3] and view.population == 6
    for g, record in enumerate(metrics):
        s = view.stats[g]
        assert s.mean == pytest.approx(record["mean_fitness"], abs=1e-12)
        assert s.median == pytest.approx(record["median_fitness"], abs=1e-12)
        assert s.worst == record["worst_fitness"]
        assert s.best == record["generation_best_fitness"]
        assert view.best_so_far[g] == record["best_fitness"]
        assert view.zero[g] == record["zero_fitness"]
        assert view.finishers[g] == record["finishers"]
        assert view.finished_share[g] == record["finished_share"]
        # the diversity of the generation that was scored, as saved by task 7.6
        genomes = ga_run.load_unit_arrays("generations", record["unit"])["genomes"]
        assert view.diversity[g] == pytest.approx(diversity(genomes), rel=1e-12)
        assert view.counts[g].sum() == 6
    assert all(b >= a for a, b in zip(view.best_so_far, view.best_so_far[1:], strict=False))


def test_the_view_does_not_change_the_run(ga_run: RunStore):
    before = {p: p.read_bytes() for p in Path(ga_run.path).rglob("*") if p.is_file()}
    evolution_view(ga_run.read_metrics(), ga_run.read_extra_lines("population"))
    after = {p: p.read_bytes() for p in Path(ga_run.path).rglob("*") if p.is_file()}
    assert before == after
