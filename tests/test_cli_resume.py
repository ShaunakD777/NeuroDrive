"""Tests for `nd resume` and random search through `nd train` / `nd sweep` (cli.py, task 5.7).

A small random-search run is killed in a separate process at two moments and finished with
`nd resume`; the result must equal a run that was never interrupted."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from neurodrive.cli import app
from neurodrive.experiment.config import load_config
from neurodrive.experiment.db import check
from neurodrive.experiment.runner import train_random_search_run
from neurodrive.experiment.store import RunStore
from random_search_kill import KILL_UNIT, KILLED, KILLS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
KILLER = Path(__file__).resolve().parent / "random_search_kill.py"
SCHEDULE = [0, 1, 2, 5, 6]  # 0, 1, 2, 5 from the schedule, and the last unit, 6
runner = CliRunner()


@pytest.fixture(scope="module")
def small_config(tmp_path_factory) -> Path:
    """random_search.yaml made small: 4 candidates, 2 starts, 6 units, exams on."""
    data = yaml.safe_load((CONFIGS / "random_search.yaml").read_text(encoding="utf-8"))
    data["tag"] = "small"
    data["search"]["candidates_per_unit"] = 4
    data["fitness"]["start_fractions"], data["fitness"]["lateral_offsets"] = [0.0, 0.5], [0.0]
    data["budget"]["units"] = 6
    path = tmp_path_factory.mktemp("config") / "small.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


@pytest.fixture(scope="module")
def reference(small_config, tmp_path_factory) -> RunStore:
    """The uninterrupted run every resumed run must equal."""
    runs_dir = tmp_path_factory.mktemp("reference")
    return train_random_search_run(load_config(small_config), runs_dir).store


def same_run(a: RunStore, b: RunStore) -> None:
    """Equal in everything except wall-clock times and the run's own name."""

    def metrics(store):
        return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]

    assert metrics(a) == metrics(b)
    assert a.checkpoint_units() == b.checkpoint_units() == SCHEDULE
    for unit in SCHEDULE:
        assert a.load_checkpoint(unit).equals(b.load_checkpoint(unit))
        card_a, card_b = a.load_exam(unit).to_dict(), b.load_exam(unit).to_dict()
        assert card_a.pop("policy")["unit"] == card_b.pop("policy")["unit"] == unit
        assert card_a == card_b
    (state_a, runner_a), (state_b, runner_b) = a.load_state(), b.load_state()
    assert runner_a["unit"] == runner_b["unit"] == 6 and runner_a["totals"] == runner_b["totals"]
    assert np.array_equal(state_a.arrays["best_genome"], state_b.arrays["best_genome"])
    assert state_a.info == state_b.info  # best fitness, counters and the generator


@pytest.mark.parametrize("kill", KILLS)
def test_a_killed_random_search_run_finished_by_nd_resume_equals_the_uninterrupted_one(
    kill, small_config, reference, tmp_path
):
    killed = subprocess.run(
        [sys.executable, str(KILLER), str(small_config), str(tmp_path), kill],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert killed.returncode == KILLED, killed.stderr
    (folder,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    store = RunStore.open(folder)
    assert store.read_meta()["status"] == "running"  # killed: nobody marked it finished
    lines = len(store.read_metrics())
    assert lines == (KILL_UNIT - 1 if kill == "during_unit" else KILL_UNIT)

    result = runner.invoke(app, ["resume", "--run", str(folder)])
    assert result.exit_code == 0, result.output
    assert f"status: completed after 6 units ({lines} were kept)" in result.stdout
    assert "best fitness: " in result.stdout
    meta = store.read_meta()
    assert meta["status"] == "completed" and len(meta["resumed_at"]) == 1
    same_run(store, reference)
    assert check(tmp_path) == []  # the database mirror followed the resume


def test_resume_refuses_toy_finished_and_missing_runs(reference, tmp_path):
    trained = runner.invoke(
        app,
        ["train", "--config", str(CONFIGS / "toy_xor.yaml"), "--epochs", "5"]
        + ["--runs-dir", str(tmp_path)],
    )
    assert trained.exit_code == 0, trained.output
    (toy,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    cases = [
        (toy, "toy_xor runs cannot be resumed"),
        (reference.path, "already completed"),
        (tmp_path / "nope", "not a run folder"),
    ]
    for folder, message in cases:
        result = runner.invoke(app, ["resume", "--run", str(folder)])
        assert result.exit_code == 2 and message in result.output, result.output


def test_nd_train_and_nd_sweep_run_random_search(small_config, tmp_path):
    trained = runner.invoke(
        app,
        ["train", "--config", str(small_config), "--units", "2", "--runs-dir", str(tmp_path / "t")],
    )
    assert trained.exit_code == 0, trained.output
    assert "status: completed after 2 units" in trained.stdout
    assert "best fitness: " in trained.stdout
    swept = runner.invoke(
        app,
        ["sweep", "--config", str(small_config), "--seeds", "0-1", "--units", "1"]
        + ["--runs-dir", str(tmp_path / "s")],
    )
    assert swept.exit_code == 0, swept.output
    assert "-random_search-small (2 runs)" in swept.stdout
    assert "best fitness: median" in swept.stdout and "over 2 seeds" in swept.stdout
    bad = runner.invoke(
        app, ["train", "--config", str(small_config), "--epochs", "3", "--runs-dir", str(tmp_path)]
    )
    assert bad.exit_code == 2 and "--epochs is for toy and imitation configs" in bad.output
