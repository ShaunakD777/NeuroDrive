"""Tests for ``GET /api/runs/{id}/evolution``, a GA run's evolution charts (api v1.23, task 7.7,
ADR-058), and since api v1.24 (ADR-059) ``?start=k`` and
``GET /api/runs/{id}/generations/{g}/starts``."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodrive.experiment.config import parse_config
from neurodrive.experiment.evolution_view import evolution_view
from neurodrive.experiment.runner import train_ga_run
from neurodrive.experiment.store import RunStore
from neurodrive.server.app import create_app
from neurodrive.server.schemas import API_VERSION, EvolutionView, GenerationStarts

GENERATIONS, N = 3, 6


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    """A runs folder with a small GA run and a run of another method."""
    folder = tmp_path_factory.mktemp("runs")
    config = parse_config(
        {
            "method": "ga",
            "seed": 4,
            "tag": "evo",
            "population": {"size": N, "elite": 1},
            "variation": {"mutation_rate": 0.2, "mutation_strength": 0.3},
            "fitness": {"start_fractions": [0.0, 0.5], "lateral_offsets": [0.0], "target_laps": 1},
            "budget": {"units": GENERATIONS},
            "exams": False,
        }
    )
    store = train_ga_run(config, folder).store
    old = folder / store.path.name.replace("-evo", "-old")  # as trained before ADR-059
    shutil.copytree(store.path, old)
    for path in (old / "extra" / "generations").glob("unit_*.npz"):
        with np.load(path) as data:
            genomes = data["genomes"]
        np.savez(path, genomes=genomes)
    RunStore.create(folder, "random_search", "other", {"method": "random_search"}).finish(
        "completed", "units"
    )
    return folder


@pytest.fixture(scope="module")
def client(runs: Path) -> TestClient:
    return TestClient(create_app(runs_dir=runs, datasets_dir=runs / "none"))


def run_id(runs: Path, tag: str) -> str:
    (path,) = [p for p in runs.iterdir() if p.name.endswith(f"-{tag}")]
    return path.name


def test_a_ga_run_has_its_evolution_charts(client, runs):
    run = run_id(runs, "evo")
    answer = client.get(f"/api/runs/{run}/evolution")
    assert answer.status_code == 200
    view = EvolutionView.model_validate(answer.json())
    assert view.run_id == run and view.population == N and view.elite == 1
    assert view.unit == list(range(1, GENERATIONS + 1))
    columns = [view.best_so_far, view.best, view.q3, view.median, view.mean, view.q1, view.worst]
    columns += [view.zero, view.diversity, view.finished_share, view.finishers, view.champion]
    assert all(len(c) == GENERATIONS for c in columns)
    assert len(view.counts) == GENERATIONS
    assert all(len(row) == len(view.edges) - 1 and sum(row) == N for row in view.counts)
    assert view.band_width == 0.05 and view.edges[0] == 0.0 and view.edges[-1] == 2.0
    for g in range(GENERATIONS):
        assert view.worst[g] <= view.q1[g] <= view.median[g] <= view.q3[g] <= view.best[g]
        assert view.worst[g] <= view.mean[g] <= view.best[g]
        assert view.best[g] <= view.best_so_far[g]


def test_the_answer_is_the_view_rounded_to_6_digits(client, runs):
    run = run_id(runs, "evo")
    store = RunStore.open(runs / run)
    view = evolution_view(store.read_metrics(), store.read_extra_lines("population"), elite=1)
    answer = client.get(f"/api/runs/{run}/evolution").json()
    exact = [s.mean for s in view.stats]
    assert answer["mean"] == [float(f"{v:.6g}") for v in exact]
    assert answer["counts"] == view.counts.tolist()
    assert np.allclose(answer["diversity"], view.diversity, rtol=1e-5)


def test_refusals(client, runs):
    assert client.get("/api/runs/20990101-000000-ga-none/evolution").status_code == 404
    assert client.get("/api/runs/not-a-run/evolution").status_code == 404
    other = client.get(f"/api/runs/{run_id(runs, 'other')}/evolution")
    assert other.status_code == 400
    assert "need a GA run" in other.json()["detail"]


def test_api_version():
    assert API_VERSION == "v1.25"


# ── one start at a time (task 7.7 follow-up, ADR-059, api v1.24) ─────────────────────────────


def test_every_start_has_its_finish_share(client, runs):
    view = EvolutionView.model_validate(
        client.get(f"/api/runs/{run_id(runs, 'evo')}/evolution").json()
    )
    assert view.start is None and view.n_starts == 2
    assert len(view.start_finish_share) == len(view.start_mean) == GENERATIONS
    assert all(len(row) == 2 and all(0 <= v <= 1 for v in row) for row in view.start_finish_share)


@pytest.mark.parametrize("start", [0, 1])
def test_one_start_is_the_view_of_that_column(client, runs, start):
    run = run_id(runs, "evo")
    store = RunStore.open(runs / run)
    starts = {u: store.load_unit_arrays("generations", u) for u in range(1, GENERATIONS + 1)}
    exact = evolution_view(
        store.read_metrics(), store.read_extra_lines("population"), 1, starts=starts, start=start
    )
    answer = client.get(f"/api/runs/{run}/evolution", params={"start": start})
    assert answer.status_code == 200
    view = EvolutionView.model_validate(answer.json())
    assert view.start == start
    assert view.median == [float(f"{s.median:.6g}") for s in exact.stats]
    assert view.counts == exact.counts.tolist()
    assert view.best_so_far == [float(f"{v:.6g}") for v in exact.best_so_far]


def test_a_generation_s_outcome_grid(client, runs):
    run = run_id(runs, "evo")
    store = RunStore.open(runs / run)
    lines = {r["unit"]: r for r in store.read_extra_lines("population")}
    for unit in range(1, GENERATIONS + 1):
        answer = client.get(f"/api/runs/{run}/generations/{unit}/starts")
        assert answer.status_code == 200
        grid = GenerationStarts.model_validate(answer.json())
        arrays = store.load_unit_arrays("generations", unit)
        assert (grid.unit, grid.population, grid.n_starts) == (unit, N, 2)
        assert grid.start_fraction == [0.0, 0.5] and grid.start_lateral_m == [0.0, 0.0]
        assert grid.order[0] == lines[unit]["champion"] == grid.champion
        assert sorted(grid.order) == list(range(N))
        assert [row.count("finished") for row in grid.status] == lines[unit]["finished"]
        assert np.allclose(grid.start_fitness, arrays["start_fitness"], rtol=1e-5)
        for i in range(N):
            for k in range(2):
                done = grid.status[i][k] == "finished"
                assert (grid.finish_time[i][k] is not None) == done


def test_per_start_refusals(client, runs):
    run, old = run_id(runs, "evo"), run_id(runs, "old")
    assert client.get(f"/api/runs/{run}/evolution", params={"start": 2}).status_code == 400
    assert client.get(f"/api/runs/{run}/evolution", params={"start": -1}).status_code == 422
    # a run trained before ADR-059: every start still works, one start and the grid do not
    whole = client.get(f"/api/runs/{old}/evolution").json()
    assert whole["n_starts"] is None and whole["start_finish_share"] is None
    one = client.get(f"/api/runs/{old}/evolution", params={"start": 0})
    assert one.status_code == 400 and "no per-start results" in one.json()["detail"]
    grid = client.get(f"/api/runs/{old}/generations/1/starts")
    assert grid.status_code == 400 and "no per-start results" in grid.json()["detail"]
    assert client.get(f"/api/runs/{run}/generations/9/starts").status_code == 400
    assert client.get("/api/runs/20990101-000000-ga-none/generations/1/starts").status_code == 404
    other = client.get(f"/api/runs/{run_id(runs, 'other')}/generations/1/starts")
    assert other.status_code == 400
