"""Tests for a GA run's lineage endpoints (api v1.25, task 7.8, ADR-060):
``GET /api/runs/{id}/generations/{g}/ancestry`` and ``.../generations/{g}/rows/{r}/mutation``."""

import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodrive.experiment.config import parse_config
from neurodrive.experiment.runner import train_ga_run
from neurodrive.experiment.store import RunStore
from neurodrive.server.app import create_app
from neurodrive.server.schemas import API_VERSION, Ancestry, MutationView

GENERATIONS, N = 3, 6


@pytest.fixture(scope="module")
def runs(tmp_path_factory) -> Path:
    """A runs folder with a small GA run, a copy without lineage (as trained before task 7.8)
    and a run of another method."""
    folder = tmp_path_factory.mktemp("runs")
    config = parse_config(
        {
            "method": "ga",
            "seed": 2,
            "tag": "lin",
            "population": {"size": N, "elite": 1},
            "variation": {"mutation_rate": 0.1, "mutation_strength": 0.3, "crossover_rate": 1.0},
            "fitness": {"start_fractions": [0.0], "lateral_offsets": [0.0], "target_laps": 1},
            "budget": {"units": GENERATIONS},
            "exams": False,
        }
    )
    store = train_ga_run(config, folder).store
    old = folder / store.path.name.replace("-lin", "-old")
    shutil.copytree(store.path, old)
    (old / "extra" / "lineage.jsonl").unlink()
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


def test_the_champions_line_of_descent(client, runs):
    rid = run_id(runs, "lin")
    store = RunStore.open(runs / rid)
    population = store.read_extra_lines("population")
    answer = client.get(f"/api/runs/{rid}/generations/{GENERATIONS}/ancestry")
    assert answer.status_code == 200, answer.text
    view = Ancestry.model_validate(answer.json())
    assert view.row == population[-1]["champion"] and view.n_genes == 434
    assert [s.unit for s in view.steps] == [3, 2, 1]
    assert view.steps[-1].origin == "random" and view.steps[-1].main_parent is None
    for step, older in zip(view.steps, view.steps[1:], strict=False):
        assert step.main_parent == older.row and step.origin in ("elite", "child")
        assert step.distance is not None and step.distance >= 0
    for s in view.steps:
        assert s.fitness == pytest.approx(population[s.unit - 1]["fitness"][s.row], rel=1e-5)
    other = client.get(f"/api/runs/{rid}/generations/2/ancestry", params={"row": 4}).json()
    assert other["row"] == 4 and other["steps"][0]["row"] == 4


def test_a_childs_mutation_in_network_shapes(client, runs):
    rid = run_id(runs, "lin")
    store = RunStore.open(runs / rid)
    record = store.read_extra_lines("lineage")[2]["rows"][3]  # generation 3, row 3: a child
    answer = client.get(f"/api/runs/{rid}/generations/3/rows/3/mutation")
    assert answer.status_code == 200, answer.text
    view = MutationView.model_validate(answer.json())
    assert view.origin == "child" and view.parents == record["parents"]
    assert view.layer_sizes == [7, 16, 16, 2]
    flags = [x for layer in view.weight_mutated for r in layer for x in r]
    flags += [x for layer in view.bias_mutated for x in layer]
    assert sum(flags) == view.n_mutated == len(record["mutated"])
    assert [len(w) for w in view.weight_delta] == [16, 16, 2]
    changes = [abs(c) for layer in view.weight_change for r in layer for c in r]
    changes += [abs(c) for layer in view.bias_change for c in layer]
    if view.n_mutated:
        assert max(changes) == pytest.approx(1.0) and view.max_abs_delta > 0
    # a gene from the other parent that was not mutated has no change
    for layer in range(3):
        other = np.array(view.weight_from_other[layer]) & ~np.array(view.weight_mutated[layer])
        assert np.all(np.array(view.weight_change[layer])[other] == 0)
    elite = client.get(f"/api/runs/{rid}/generations/3/rows/0/mutation").json()
    assert elite["origin"] == "elite" and elite["n_mutated"] == 0 and elite["max_abs_delta"] == 0


def test_refusals(client, runs):
    rid = run_id(runs, "lin")
    assert client.get("/api/runs/nope/generations/2/ancestry").status_code == 404
    assert client.get("/api/runs/nope/generations/2/rows/0/mutation").status_code == 404
    for url, words in [
        (f"/api/runs/{run_id(runs, 'other')}/generations/1/ancestry", "GA run"),
        (f"/api/runs/{run_id(runs, 'old')}/generations/2/ancestry", "before task 7.8"),
        (f"/api/runs/{run_id(runs, 'old')}/generations/2/rows/1/mutation", "before task 7.8"),
        (f"/api/runs/{rid}/generations/1/rows/1/mutation", "random"),
        (f"/api/runs/{rid}/generations/2/rows/{N}/mutation", "no row"),
        (f"/api/runs/{rid}/generations/9/ancestry", "no saved generation"),
    ]:
        answer = client.get(url)
        assert answer.status_code == 400, url
        assert words in answer.json()["detail"], answer.json()["detail"]
    assert client.get(f"/api/runs/{rid}/generations/2/ancestry?row=-1").status_code == 422


def test_the_api_version():
    assert API_VERSION == "v1.25"
