"""Tests for the demonstration-dataset REST endpoints (api v1.15, task 6.2, ADR-045; dense
coverage api v1.16, ADR-046)."""

import json
from datetime import UTC, datetime

import numpy as np
import pytest
from fastapi.testclient import TestClient

from neurodrive.core.rng import make_rng
from neurodrive.experiment.dataset_view import DENSE_SHARE, MAP_POINTS, dense_coverage
from neurodrive.learn.supervised.demos import (
    DATASET_FILE,
    DATASET_VERSION,
    META_FILE,
    DemoDataset,
    save_dataset,
    straight_share,
)
from neurodrive.server.app import create_app
from neurodrive.server.schemas import API_VERSION, DatasetDetail, DatasetListResponse


def make_dataset(frames: tuple[int, ...] = (10, 15), seed: int = 0) -> DemoDataset:
    """A small hand-made dataset: episodes of the given lengths with in-range random values."""
    rng = make_rng(seed)
    f = sum(frames)
    observation = rng.uniform(0.0, 1.0, (f, 7))
    observation[:, 6] = rng.uniform(-0.2, 0.2, f)
    label = np.column_stack([rng.uniform(-0.5, 0.5, f), np.ones(f)])  # full throttle throughout
    meta = {
        "dataset_version": DATASET_VERSION,
        "teacher": {"kind": "expert", "params": {}},
        "track": "train_loop_01",
        "track_split": "train",
        "episodes": [
            {"index": i, "status": "finished" if i == 0 else "crashed", "frames": n}
            for i, n in enumerate(frames)
        ],
        "demo_frames": f,
        "env_steps": f,
        "straight_share": straight_share(label),
    }
    return DemoDataset(
        meta=meta,
        observation=observation,
        label=label,
        episode=np.repeat(np.arange(len(frames)), frames).astype(np.int64),
        step=np.concatenate([np.arange(n) for n in frames]).astype(np.int64),
        x=np.linspace(0.0, 100.0, f),
        y=np.linspace(-50.0, 50.0, f),
        psi=np.zeros(f),
        v=np.full(f, 10.0),
    )


@pytest.fixture
def datasets_dir(tmp_path):
    return tmp_path / "demos"


@pytest.fixture
def client(datasets_dir, tmp_path):
    return TestClient(create_app(runs_dir=tmp_path / "runs", datasets_dir=datasets_dir))


def test_missing_folder_lists_nothing(client):
    response = client.get("/api/datasets")
    assert response.status_code == 200 and response.json() == {"datasets": []}


def test_list_newest_first_and_skips_non_datasets(client, datasets_dir):
    old = save_dataset(make_dataset(), datasets_dir, "old", when=datetime(2026, 1, 1, tzinfo=UTC))
    new = save_dataset(make_dataset(), datasets_dir, "new", when=datetime(2026, 2, 1, tzinfo=UTC))
    (datasets_dir / "not-a-dataset").mkdir()  # wrong name
    (datasets_dir / "20260301-000000-expert-empty").mkdir()  # no meta.json
    broken = datasets_dir / "20260302-000000-expert-broken"
    broken.mkdir()
    (broken / META_FILE).write_text("{not json", encoding="utf-8")  # unreadable meta.json
    listed = DatasetListResponse.model_validate(client.get("/api/datasets").json()).datasets
    assert [d.dataset_id for d in listed] == [new.name, old.name]
    first = listed[0]
    assert (first.teacher, first.tag, first.track) == ("expert", "new", "train_loop_01")
    assert first.demo_frames == 25 and first.episodes == 2 and first.finished == 1
    assert first.created_at == "2026-02-01T00:00:00+00:00"


def test_detail_histograms_and_map(client, datasets_dir):
    dataset = make_dataset()
    path = save_dataset(dataset, datasets_dir, "t")
    response = client.get(f"/api/datasets/{path.name}")
    assert response.status_code == 200
    detail = DatasetDetail.model_validate(response.json())
    assert detail.dataset_id == path.name and detail.meta["demo_frames"] == 25
    assert [h.name for h in detail.labels] == ["steer", "throttle"]
    assert [h.name for h in detail.observations] == [
        "ray_far_left",
        "ray_left",
        "ray_centre",
        "ray_right",
        "ray_far_right",
        "speed",
        "heading_error",
    ]
    for h in detail.labels + detail.observations:
        assert len(h.counts) == 20 and len(h.edges) == 21 and sum(h.counts) == 25
        assert h.edges[0] == h.lo and h.edges[-1] == h.hi
    throttle = detail.labels[1]
    assert throttle.counts[-1] == 25 and throttle.min == throttle.max == 1.0  # all at hi
    assert throttle.covered == 0.0
    steer = detail.labels[0]
    assert steer.min == pytest.approx(dataset.label[:, 0].min())
    assert steer.covered == pytest.approx(np.ptp(dataset.label[:, 0]) / 2, abs=1e-6)
    # 25 frames fit within MAP_POINTS, so every frame is on the map, in row order.
    m = detail.map
    assert m.stride == 1 and len(m.x) == len(m.y) == len(m.steer) == len(m.episode) == 25
    assert m.x == np.round(dataset.x, 2).tolist() and m.episode == dataset.episode.tolist()
    assert m.throttle == [1.0] * 25


def test_a_large_dataset_is_thinned(client, datasets_dir):
    f = MAP_POINTS + 1  # one frame too many: every second frame
    path = save_dataset(make_dataset(frames=(f,)), datasets_dir, "big")
    m = DatasetDetail.model_validate(client.get(f"/api/datasets/{path.name}").json()).map
    assert m.stride == 2 and len(m.x) == (f + 1) // 2


def test_unknown_malformed_and_broken_ids_are_refused(client, datasets_dir):
    assert client.get("/api/datasets/20260101-000000-expert-nope").status_code == 404
    for bad in ["..", "not-an-id", "20260101-000000-expert-a.b"]:
        assert client.get(f"/api/datasets/{bad}").status_code == 404
    assert client.get("/api/datasets/..%2F..%2Fsecret").status_code == 404
    # A folder that has meta.json but does not load as dataset v1: 400 with the reason.
    path = save_dataset(make_dataset(), datasets_dir, "broken")
    meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
    meta["demo_frames"] = 99
    (path / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
    response = client.get(f"/api/datasets/{path.name}")
    assert response.status_code == 400 and "99 frames" in response.json()["detail"]
    path2 = save_dataset(make_dataset(), datasets_dir, "noarrays")
    (path2 / DATASET_FILE).write_bytes(b"not a zip file")
    assert client.get(f"/api/datasets/{path2.name}").status_code == 400


def test_detail_dense_coverage(client, datasets_dir):
    # 6.2 follow-up (api v1.16, ADR-046): every histogram carries d and tau·F; d never exceeds c.
    dataset = make_dataset()
    path = save_dataset(dataset, datasets_dir, "dense")
    detail = DatasetDetail.model_validate(client.get(f"/api/datasets/{path.name}").json())
    for h in detail.labels + detail.observations:
        assert h.dense_threshold == pytest.approx(DENSE_SHARE * 25)
        assert 0.0 <= h.dense <= h.covered + 1e-9
        expected = dense_coverage(np.array(h.edges), np.array(h.counts), h.min, h.max, h.lo, h.hi)
        assert h.dense == pytest.approx(expected, abs=1e-6)
    # 25 frames: tau·F = 0.25, so every non-empty bin is dense; an empty bin between min and max
    # is the only thing that makes d smaller than c.
    for h in detail.labels + detail.observations:
        inside = [b for b in range(20) if h.edges[b + 1] > h.min and h.edges[b] < h.max]
        if all(h.counts[b] > 0 for b in inside):
            assert h.dense == pytest.approx(h.covered, abs=1e-6)
        else:
            assert h.dense < h.covered
    throttle = detail.labels[1]  # every label exactly 1.0: nothing spanned, nothing dense
    assert throttle.dense == throttle.covered == 0.0


def test_hello_says_v1_17(client):
    assert API_VERSION == "v1.17"
    with client.websocket_connect("/ws/live") as ws:
        assert json.loads(ws.receive_text())["api"] == "v1.17"
