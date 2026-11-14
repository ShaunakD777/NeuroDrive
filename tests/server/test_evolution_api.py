"""Tests for ``GET /api/runs/{id}/evolution``, a GA run's evolution charts (api v1.23, task 7.7,
ADR-058)."""

from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodrive.experiment.config import parse_config
from neurodrive.experiment.evolution_view import evolution_view
from neurodrive.experiment.runner import train_ga_run
from neurodrive.experiment.store import RunStore
from neurodrive.server.app import create_app
from neurodrive.server.schemas import API_VERSION, EvolutionView

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
    train_ga_run(config, folder)
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
    assert API_VERSION == "v1.23"
