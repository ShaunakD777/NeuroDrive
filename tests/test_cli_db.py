"""Tests for `nd db rebuild` and `nd db check` (cli.py), task 5.5b."""

import sqlite3
from contextlib import closing
from pathlib import Path

from typer.testing import CliRunner

from neurodrive.cli import app
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.experiment.db import DB_NAME

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

runner = CliRunner()


def nd(*args: str):
    return runner.invoke(app, list(args))


def test_rebuild_then_check_then_a_tampered_database(tmp_path):
    runs, config = str(tmp_path), str(CONFIGS / "toy_xor.yaml")
    trained = nd("train", "--config", config, "--epochs", "10", "--runs-dir", runs)
    assert trained.exit_code == 0, trained.output
    assert (tmp_path / DB_NAME).is_file()  # nd train mirrored the run as it went

    clean = nd("db", "check", "--runs-dir", runs)
    assert clean.exit_code == 0 and "the database matches the folders (1 runs)" in clean.stdout

    with closing(sqlite3.connect(tmp_path / DB_NAME)) as conn, conn:
        conn.execute("UPDATE metrics SET value = 0 WHERE unit = 3 AND name = 'loss'")
    dirty = nd("db", "check", "--runs-dir", runs)
    assert dirty.exit_code == 1
    assert "metrics:" in dirty.stdout and "0 rows missing, 0 extra, 1 different" in dirty.stdout
    assert "nd db rebuild" in dirty.stdout

    rebuilt = nd("db", "rebuild", "--runs-dir", runs)
    assert rebuilt.exit_code == 0, rebuilt.output
    # 10 epochs × 9 values: 3 totals, wall clock, loss, loss_before, grad_norm[0], [1], diverged
    assert "from 1 run folders" in rebuilt.stdout and "metrics: 90 rows" in rebuilt.stdout
    assert "checkpoints: 5 rows" in rebuilt.stdout  # units 0, 1, 2, 5, 10
    assert nd("db", "check", "--runs-dir", runs).exit_code == 0


def test_nd_exam_updates_the_mirror(tmp_path):
    from neurodrive.experiment.store import RunStore

    store = RunStore.create(tmp_path, "random", "exam", {"seed": 3})
    store.save_checkpoint(0, init_network(NetworkSpec(), make_rng(3)))
    assert nd("db", "rebuild", "--runs-dir", str(tmp_path)).exit_code == 0
    assert nd("exam", "--run", str(store.path), "--unit", "0").exit_code == 0
    check = nd("db", "check", "--runs-dir", str(tmp_path))
    assert check.exit_code == 0, check.output
    with closing(sqlite3.connect(tmp_path / DB_NAME)) as conn:
        assert conn.execute("SELECT count(*) FROM exam_episodes").fetchone()[0] == 50


def test_rebuild_refuses_a_missing_folder_and_check_needs_a_database(tmp_path):
    missing = nd("db", "rebuild", "--runs-dir", str(tmp_path / "nope"))
    assert missing.exit_code == 2 and "no such folder" in missing.output
    none = nd("db", "check", "--runs-dir", str(tmp_path))
    assert none.exit_code == 1 and "no database at neurodrive.db" in none.stdout
