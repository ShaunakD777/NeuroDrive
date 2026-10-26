"""Tests for the run-folder REST endpoints (api v1.10, ADR-031)."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodrive.core.nn.gradcheck import relative_error
from neurodrive.core.rng import make_rng
from neurodrive.experiment.runner import run
from neurodrive.experiment.store import RunStore
from neurodrive.experiment.toy_view import toy_checkpoint_view
from neurodrive.learn.supervised.train_toy import ToyConfig, ToyLearner
from neurodrive.server.app import create_app
from neurodrive.server.schemas import (
    CheckpointList,
    CheckpointNetwork,
    RunDetail,
    RunListResponse,
    RunMetrics,
    ToyCheckpointResponse,
)


def make_run(runs_dir, task: str = "sine", epochs: int = 20, tag: str = "t", **over) -> RunStore:
    config = ToyConfig.for_task(task, epochs=epochs, tag=tag, **over)
    learner = ToyLearner(config, make_rng(config.seed))
    store = RunStore.create(runs_dir, learner.name, config.tag, config.to_dict())
    store.save_extra("dataset", x=learner.dataset.x, y=learner.dataset.y)
    run(learner, store, epochs)
    return store


@pytest.fixture
def runs_dir(tmp_path):
    return tmp_path / "runs"


@pytest.fixture
def client(runs_dir):
    return TestClient(create_app(runs_dir=runs_dir))


def test_empty_or_missing_runs_dir_lists_nothing(client):
    response = client.get("/api/runs")
    assert response.status_code == 200 and response.json() == {"runs": []}


def test_list_newest_first_with_status(client, runs_dir):
    a = make_run(runs_dir, "xor", 5, tag="a")
    b = make_run(runs_dir, "sine", 3000, tag="b", learning_rate=50.0)
    (runs_dir / "not-a-run").mkdir()  # ignored: wrong name, no meta.json
    runs = RunListResponse.model_validate(client.get("/api/runs").json()).runs
    assert [r.run_id for r in runs] == sorted([a.path.name, b.path.name], reverse=True)
    by_tag = {r.tag: r for r in runs}
    assert by_tag["a"].status == "completed" and by_tag["a"].last_unit == 5
    assert by_tag["a"].method == "toy_xor" and by_tag["a"].checkpoints == 4  # 0, 1, 2, 5
    assert by_tag["b"].status == "diverged"


def test_detail_and_checkpoints(client, runs_dir):
    store = make_run(runs_dir, "xor", 10)
    rid = store.path.name
    detail = RunDetail.model_validate(client.get(f"/api/runs/{rid}").json())
    assert detail.config == store.read_config() and detail.meta["status"] == "completed"
    units = CheckpointList.model_validate(client.get(f"/api/runs/{rid}/checkpoints").json())
    assert units.units == [0, 1, 2, 5, 10]


def test_metrics_as_columns(client, runs_dir):
    store = make_run(runs_dir, "xor", 30)
    m = RunMetrics.model_validate(client.get(f"/api/runs/{store.path.name}/metrics").json())
    records = store.read_metrics()
    assert m.unit == list(range(1, 31))
    np.testing.assert_allclose(m.loss, [r["loss"] for r in records], rtol=1e-5)
    np.testing.assert_allclose(m.loss_before, [r["loss_before"] for r in records], rtol=1e-5)
    assert len(m.grad_norm) == 2 and all(len(col) == 30 for col in m.grad_norm)
    np.testing.assert_allclose(m.grad_norm[1], [r["grad_norm"][1] for r in records], rtol=1e-5)
    assert m.diverged == [False] * 30
    assert len(json.dumps(m.model_dump())) < 10_000  # columns, 6 significant digits


def test_metrics_of_a_diverged_run_keep_nulls(client, runs_dir):
    store = make_run(runs_dir, "sine", 3000, learning_rate=50.0)
    m = RunMetrics.model_validate(client.get(f"/api/runs/{store.path.name}/metrics").json())
    assert m.diverged[-1] is True and m.loss[-1] is None and m.grad_norm[0][-1] is None


def test_checkpoint_network(client, runs_dir):
    store = make_run(runs_dir, "xor", 5)
    net = CheckpointNetwork.model_validate(
        client.get(f"/api/runs/{store.path.name}/checkpoints/5").json()
    )
    saved = store.load_checkpoint(5)
    assert net.layer_sizes == [2, 4, 1] and net.activations == ["tanh", "sigmoid"]
    np.testing.assert_allclose(net.weights[0], saved.weights[0], atol=5e-5)
    np.testing.assert_allclose(net.biases[1], saved.biases[1], atol=5e-5)


@pytest.mark.parametrize("task", ["sine", "xor"])
def test_toy_checkpoint_matches_the_toy_view(client, runs_dir, task):
    store = make_run(runs_dir, task, 20)
    url = f"/api/runs/{store.path.name}/checkpoints/20/toy"
    toy = ToyCheckpointResponse.model_validate(client.get(url).json())
    view = toy_checkpoint_view(store, 20)
    assert toy.unit == 20 and toy.loss == pytest.approx(view.loss, rel=1e-5)
    assert toy.max_gradient == pytest.approx(view.max_gradient, rel=1e-5)
    for got, want in zip(toy.weight_gradients, view.weight_gradients, strict=True):
        assert relative_error(np.array(got), want) < 1e-5  # 6 significant digits: <= 5e-6 each
    np.testing.assert_allclose(toy.glow[-1], view.glow[-1], atol=5e-4)
    if task == "sine":
        assert len(toy.prediction_curve) == 201 and toy.prediction_map is None
        np.testing.assert_allclose(toy.prediction_curve, view.prediction_curve, atol=5e-5)
    else:
        assert toy.prediction_curve is None and len(toy.prediction_map) == 41
        np.testing.assert_allclose(toy.prediction_map, view.prediction_map, atol=5e-5)
    assert toy.network.layer_sizes == list(view.network.spec.layer_sizes)


def test_unknown_run_and_checkpoint_are_404(client, runs_dir):
    store = make_run(runs_dir, "xor", 2)
    rid = store.path.name
    for url in (
        "/api/runs/20990101-000000-toy_xor-none",
        "/api/runs/20990101-000000-toy_xor-none/metrics",
        "/api/runs/20990101-000000-toy_xor-none/checkpoints",
        f"/api/runs/{rid}/checkpoints/3",
        f"/api/runs/{rid}/checkpoints/3/toy",
    ):
        assert client.get(url).status_code == 404, url


def test_run_ids_cannot_leave_the_runs_folder(client, runs_dir, tmp_path):
    make_run(runs_dir, "xor", 2)
    secret = tmp_path / "20260101-000000-x-y"  # a run-shaped folder outside runs_dir
    secret.mkdir()
    (secret / "meta.json").write_text("{}")
    for rid in ("..", "..%2F20260101-000000-x-y", "20260101-000000-x-y", "%2E%2E"):
        assert client.get(f"/api/runs/{rid}").status_code == 404, rid


def test_toy_view_of_a_non_toy_run_is_refused(client, runs_dir):
    store = RunStore.create(runs_dir, "random_search", "t", {})
    store.finish("completed")
    response = client.get(f"/api/runs/{store.path.name}/checkpoints/0/toy")
    assert response.status_code == 400 and "not a toy run" in response.json()["detail"]
