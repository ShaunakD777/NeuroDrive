"""Tests for experiment/store.py: run folder v0 is written and read back faithfully."""

import json
import re
from datetime import UTC, datetime

import numpy as np
import pytest

from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.experiment.store import RUN_VERSION, RunStore

WHEN = datetime(2026, 10, 24, 9, 5, 7, tzinfo=UTC)
CONFIG = {"task": "xor", "layer_sizes": [2, 4, 1], "learning_rate": 1.0, "seed": 3}


def new_store(tmp_path, **kwargs) -> RunStore:
    return RunStore.create(tmp_path, kwargs.get("method", "toy_xor"), "t1", CONFIG, when=WHEN)


def test_folder_name_and_layout(tmp_path):
    store = new_store(tmp_path)
    assert store.path.name == "20261024-090507-toy_xor-t1"
    for item in ("config.yaml", "meta.json", "metrics.jsonl", "checkpoints/index.json"):
        assert (store.path / item).is_file()
    assert (store.path / "extra").is_dir()


def test_same_second_gets_a_suffix(tmp_path):
    a, b, c = new_store(tmp_path), new_store(tmp_path), new_store(tmp_path)
    assert [s.path.name[-5:] for s in (a, b, c)] == ["or-t1", "-t1-2", "-t1-3"]


def test_config_round_trips(tmp_path):
    assert RunStore.open(new_store(tmp_path).path).read_config() == CONFIG


def test_meta_contents_and_finish(tmp_path):
    store = new_store(tmp_path)
    meta = store.read_meta()
    assert meta["run_version"] == RUN_VERSION == 0
    assert meta["status"] == "running" and meta["ended_at"] is None
    assert meta["run_id"] == store.path.name and meta["method"] == "toy_xor"
    assert {"git_commit", "git_dirty", "python_version", "numpy_version"} <= set(meta)
    store.finish("completed")
    meta = store.read_meta()
    assert meta["status"] == "completed" and meta["ended_at"] is not None
    with pytest.raises(ValueError, match="final status"):
        store.finish("running")
    with pytest.raises(ValueError, match="final status"):
        store.finish("great")


def test_no_absolute_paths_in_written_files(tmp_path):
    store = new_store(tmp_path)
    store.finish("completed")
    for name in ("meta.json", "config.yaml"):
        text = (store.path / name).read_text(encoding="utf-8")
        assert str(tmp_path) not in text and not re.search(r"[A-Za-z]:\\|/home/|/Users/", text)


def test_metrics_lines_round_trip(tmp_path):
    store = new_store(tmp_path)
    store.append_metrics({"unit": 1, "loss": 0.5, "grad_norm": [1.0, 2.0]})
    store.append_metrics({"unit": 2, "loss": None})
    assert store.read_metrics() == [
        {"unit": 1, "loss": 0.5, "grad_norm": [1.0, 2.0]},
        {"unit": 2, "loss": None},
    ]
    lines = (store.path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["unit"] for line in lines] == [1, 2]
    with pytest.raises(ValueError, match="unit"):
        store.append_metrics({"loss": 1.0})
    with pytest.raises(ValueError):
        store.append_metrics({"unit": 3, "loss": float("nan")})  # NaN is not valid JSON


def test_checkpoints_and_index(tmp_path):
    store = new_store(tmp_path)
    net = init_network(NetworkSpec((2, 4, 1)), make_rng(0))
    for unit in (5, 0, 5, 1):
        store.save_checkpoint(unit, net)
    assert store.checkpoint_units() == [0, 1, 5]
    assert sorted(p.name for p in (store.path / "checkpoints").glob("*.npz")) == [
        "unit_0.npz", "unit_1.npz", "unit_5.npz"
    ]  # fmt: skip
    assert store.load_checkpoint(5).equals(net)
    with pytest.raises(KeyError):
        store.load_checkpoint(2)


def test_extra_arrays(tmp_path):
    store = new_store(tmp_path)
    x, y = np.arange(6.0).reshape(3, 2), np.array([[0.0], [1.0], [1.0]])
    store.save_extra("dataset", x=x, y=y)
    data = store.load_extra("dataset")
    np.testing.assert_array_equal(data["x"], x)
    np.testing.assert_array_equal(data["y"], y)


def test_bad_names_and_missing_folder(tmp_path):
    with pytest.raises(ValueError, match="tag"):
        RunStore.create(tmp_path, "toy", "../escape", CONFIG)
    with pytest.raises(ValueError, match="method"):
        RunStore.create(tmp_path, "toy xor", "t", CONFIG)
    with pytest.raises(FileNotFoundError):
        RunStore.open(tmp_path / "nothing")
