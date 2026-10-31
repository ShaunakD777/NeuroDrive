"""Tests for GET /api/runs/compare and the `sweep` field of GET /api/runs (api v1.11, ADR-033)."""

import math
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from neurodrive.experiment.config import load_config, parse_config, resolved
from neurodrive.experiment.runner import sweep_seeds, train_toy_run
from neurodrive.experiment.stats import curve_bands, summarise
from neurodrive.server.app import create_app
from neurodrive.server.schemas import CompareResponse, RunListResponse

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    """A 3-seed SGD sweep, a 2-seed sweep with a validation set, a single run, a diverged run."""
    runs_dir = tmp_path_factory.mktemp("runs")
    base = load_config(CONFIGS / "toy_sine.yaml", epochs=15)
    sgd_id, sgd = sweep_seeds(
        load_config(CONFIGS / "toy_sine.yaml", tag="sgd", epochs=15), [0, 1, 2], runs_dir
    )
    val_config = parse_config(
        {**resolved(base), "tag": "val", "training": {"epochs": 15, "val_fraction": 0.25}}
    )
    val_id, val = sweep_seeds(val_config, [0, 1], runs_dir)
    single = train_toy_run(load_config(CONFIGS / "toy_sine.yaml", tag="one", epochs=10), runs_dir)
    hot = parse_config(
        {
            **resolved(base),
            "tag": "hot",
            "training": {"epochs": 200},
            "optimizer": {"name": "sgd", "learning_rate": 50.0},
        }
    )
    diverged = train_toy_run(hot, runs_dir)
    assert diverged.status == "diverged"
    return {
        "client": TestClient(create_app(runs_dir=runs_dir)),
        "sgd": (sgd_id, [r.store.path.name for r in sgd]),
        "val": (val_id, [r.store.path.name for r in val]),
        "single": single.store.path.name,
        "diverged": diverged.store.path.name,
        "stores": {r.store.path.name: r.store for r in [*sgd, *val, single, diverged]},
    }


def compare(runs, ids, metric="loss"):
    response = runs["client"].get(
        "/api/runs/compare", params={"ids": ",".join(ids), "metric": metric}
    )
    assert response.status_code == 200, response.text
    return CompareResponse.model_validate(response.json())


def test_run_list_carries_the_sweep_id(runs):
    listing = RunListResponse.model_validate(runs["client"].get("/api/runs").json())
    by_id = {r.run_id: r.sweep for r in listing.runs}
    sgd_id, sgd_runs = runs["sgd"]
    assert all(by_id[r] == sgd_id for r in sgd_runs)
    assert by_id[runs["single"]] is None


def test_groups_follow_sweeps_and_request_order(runs):
    sgd_id, sgd_runs = runs["sgd"]
    result = compare(runs, [runs["single"], *sgd_runs])
    assert [g.group_id for g in result.groups] == [runs["single"], sgd_id]
    assert result.groups[1].run_ids == sgd_runs
    assert [r.seed for r in result.runs] == [0, 0, 1, 2]
    assert result.skipped == []


def test_bands_equal_the_statistics_of_the_raw_metrics(runs):
    sgd_id, sgd_runs = runs["sgd"]
    group = compare(runs, sgd_runs).groups[0]
    curves = []
    for run_id in sgd_runs:
        records = runs["stores"][run_id].read_metrics()
        curves.append([records[0]["loss_before"], *[r["loss"] for r in records]])
    bands = curve_bands(curves, [False] * 3)
    assert group.unit == list(range(16))  # epoch 0 (before training) to 15
    assert group.median == pytest.approx([float(f"{v:.6g}") for v in bands.median])
    assert group.q1 == pytest.approx([float(f"{v:.6g}") for v in bands.q1], rel=1e-5)
    finals = summarise([c[-1] for c in curves])
    assert group.final_median == pytest.approx(finals.median, rel=1e-5)
    assert (group.final_q1, group.final_q3) == pytest.approx((finals.q1, finals.q3), rel=1e-5)
    assert group.n_present == [3] * 16 and group.n_diverged == 0


def test_a_diverged_run_is_infinity_sent_as_null(runs):
    result = compare(runs, [runs["diverged"]])
    run, group = result.runs[0], result.groups[0]
    assert run.diverged and run.values[-1] is None and run.final is not None
    assert group.median[-1] is None  # +inf has no JSON form
    assert group.n_finite[-1] == 0 and group.n_diverged == 1
    assert group.final_median is None
    assert all(v is None or math.isfinite(v) for v in group.median)


def test_val_loss_starts_at_epoch_1_and_runs_without_it_are_skipped(runs):
    val_id, val_runs = runs["val"]
    result = compare(runs, [*val_runs, runs["single"]], metric="val_loss")
    assert result.metric == "val_loss"
    assert result.skipped == [runs["single"]]
    assert [g.group_id for g in result.groups] == [val_id]
    assert result.groups[0].unit[0] == 1 and result.runs[0].unit[0] == 1
    records = runs["stores"][val_runs[0]].read_metrics()
    assert result.runs[0].values[-1] == pytest.approx(records[-1]["val_loss"], rel=1e-5)


def test_bad_requests(runs):
    client = runs["client"]
    sgd_runs = runs["sgd"][1]
    assert client.get("/api/runs/compare", params={"ids": "nope"}).status_code == 404
    assert (
        client.get("/api/runs/compare", params={"ids": "20261031-120000-toy_x-y"}).status_code
        == 404
    )
    assert client.get("/api/runs/compare", params={"ids": ""}).status_code == 422
    assert client.get("/api/runs/compare").status_code == 422
    bad_metric = {"ids": sgd_runs[0], "metric": "accuracy"}
    assert client.get("/api/runs/compare", params=bad_metric).status_code == 422
    too_many = {"ids": ",".join([sgd_runs[0]] * 65)}
    assert client.get("/api/runs/compare", params=too_many).status_code == 422
    # repeats are counted once
    assert len(compare(runs, [sgd_runs[0], sgd_runs[0]]).runs) == 1
