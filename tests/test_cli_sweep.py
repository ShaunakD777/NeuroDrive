"""Tests for `nd sweep` (cli.py) and parse_seeds / sweep_seeds (experiment/runner.py), task 4.7a;
`--study` (task 5.10)."""

import json
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

import pytest
from typer.testing import CliRunner

from neurodrive.cli import app
from neurodrive.experiment.config import load_config
from neurodrive.experiment.db import DB_NAME, connect, runs_of_study
from neurodrive.experiment.runner import parse_seeds, sweep_seeds, train_toy_run
from neurodrive.experiment.store import RunStore

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
runner = CliRunner()


@pytest.mark.parametrize(
    "spec,expected",
    [
        ("0-4", [0, 1, 2, 3, 4]),
        ("0,2,7", [0, 2, 7]),
        ("0-2,5", [0, 1, 2, 5]),
        ("3", [3]),
        (" 1 , 4 ", [1, 4]),
    ],
)
def test_parse_seeds(spec, expected):
    assert parse_seeds(spec) == expected


@pytest.mark.parametrize("spec", ["", "a", "1-", "-1", "3-1", "0,0", "0-2,1", "1.5"])
def test_parse_seeds_refuses_bad_lists(spec):
    with pytest.raises(ValueError):
        parse_seeds(spec)


def test_sweep_runs_equal_single_runs(tmp_path):
    config = load_config(CONFIGS / "toy_xor.yaml", tag="sw", epochs=30)
    when = datetime(2026, 10, 31, 12, 0, 0, tzinfo=UTC)
    sweep_id, results = sweep_seeds(config, [0, 3], tmp_path / "sweep", when=when)
    assert sweep_id == "20261031-120000-toy_xor-sw"
    for seed, result in zip([0, 3], results, strict=True):
        meta = result.store.read_meta()
        assert meta["sweep"] == sweep_id and meta["status"] == "completed"
        assert result.store.path.name.endswith(f"-toy_xor-sw-s{seed}")
        stored = result.store.read_config()
        assert (stored["seed"], stored["tag"]) == (seed, f"sw-s{seed}")
        single = train_toy_run(
            load_config(CONFIGS / "toy_xor.yaml", seed=seed, tag=f"sw-s{seed}", epochs=30),
            tmp_path / "single",
        )
        assert "sweep" not in single.store.read_meta()
        assert [m["loss"] for m in single.store.read_metrics()] == [
            m["loss"] for m in result.store.read_metrics()
        ]


def test_cli_sweep_prints_runs_and_summary(tmp_path):
    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(CONFIGS / "toy_sine.yaml"),
            "--seeds",
            "0-4",
            "--epochs",
            "20",
            "--tag",
            "cli",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "(5 runs)" in result.stdout
    assert "final loss: median" in result.stdout and "over 5 seeds" in result.stdout
    folders = sorted(p.name for p in tmp_path.iterdir() if p.is_dir())  # not neurodrive.db
    assert [f.rsplit("-", 1)[1] for f in folders] == ["s0", "s1", "s2", "s3", "s4"]
    sweeps = {RunStore(tmp_path / f).read_meta()["sweep"] for f in folders}
    assert len(sweeps) == 1
    assert all("study" not in RunStore(tmp_path / f).read_meta() for f in folders)


def test_cli_sweep_with_a_study_tags_every_run_and_fills_the_studies_table(tmp_path):
    args = ["sweep", "--config", str(CONFIGS / "toy_sine.yaml"), "--seeds", "0-2", "--epochs", "3"]
    result = runner.invoke(app, [*args, "--study", "S4", "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    metas = [RunStore(p).read_meta() for p in tmp_path.iterdir() if p.is_dir()]
    assert [m["study"] for m in metas] == ["S4"] * 3
    with closing(connect(tmp_path / DB_NAME)) as conn:
        assert runs_of_study(conn, "S4") == sorted(m["run_id"] for m in metas)
        row = conn.execute("SELECT n_runs, sweeps FROM studies WHERE study_id = 'S4'").fetchone()
    assert row == (3, json.dumps([metas[0]["sweep"]]))
    bad = tmp_path / "bad"
    result = runner.invoke(app, [*args, "--study", "S4/x", "--runs-dir", str(bad)])
    assert result.exit_code == 2 and "study must be letters" in result.output
    assert not bad.exists()


def test_cli_sweep_refuses_bad_seeds_and_configs(tmp_path):
    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(CONFIGS / "toy_sine.yaml"),
            "--seeds",
            "4-0",
            "--runs-dir",
            str(tmp_path / "r"),
        ],
    )
    assert result.exit_code == 2 and "end is before the start" in result.output
    result = runner.invoke(
        app,
        [
            "sweep",
            "--config",
            str(CONFIGS / "toy_sine.yaml"),
            "--seeds",
            "0-1",
            "--epochs",
            "0",
            "--runs-dir",
            str(tmp_path / "r"),
        ],
    )
    assert result.exit_code == 2 and "training.epochs" in result.output
    assert not (tmp_path / "r").exists()
