"""Tests for `nd train` (cli.py) and train_toy_run (experiment/runner.py), task 4.6b."""

from pathlib import Path

import yaml
from typer.testing import CliRunner

from neurodrive.cli import app
from neurodrive.experiment.config import load_config, parse_config, toy_config_from_run
from neurodrive.experiment.runner import train_toy_run
from neurodrive.experiment.store import RunStore
from neurodrive.experiment.toy_view import toy_checkpoint_view

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
runner = CliRunner()


def train(*args: str):
    return runner.invoke(app, ["train", *args])


def only_run(runs_dir: Path) -> RunStore:
    (folder,) = [p for p in runs_dir.iterdir() if p.is_dir()]
    return RunStore(folder)


def test_train_creates_a_completed_run_with_a_resolved_config(tmp_path):
    result = train(
        "--config",
        str(CONFIGS / "toy_xor.yaml"),
        "--epochs",
        "40",
        "--tag",
        "cli",
        "--runs-dir",
        str(tmp_path),
    )
    assert result.exit_code == 0, result.output
    assert "status: completed after 40 epochs" in result.stdout
    store = only_run(tmp_path)
    assert store.path.name.endswith("-toy_xor-cli")
    stored = store.read_config()
    assert stored["method"] == "toy" and stored["tag"] == "cli"
    assert stored["training"] == {
        "epochs": 40,
        "loss": "mse",
        "batch_size": None,
        "val_fraction": 0.0,
    }
    assert stored["optimizer"] == {"name": "sgd", "learning_rate": 1.0}
    assert store.read_meta()["status"] == "completed"
    assert len(store.read_metrics()) == 40
    # the stored config is itself a valid config file
    again = load_config(store.path / "config.yaml")
    assert again == load_config(CONFIGS / "toy_xor.yaml", tag="cli", epochs=40)


def test_same_config_and_seed_give_the_same_losses(tmp_path):
    config = load_config(CONFIGS / "toy_sine.yaml", epochs=30)
    adam = parse_config(
        {
            **yaml.safe_load((CONFIGS / "toy_sine.yaml").read_text("utf-8")),
            "optimizer": {"name": "adam", "learning_rate": 0.01},
            "training": {"epochs": 30, "batch_size": 32, "val_fraction": 0.25},
        }
    )
    a = train_toy_run(config, tmp_path / "a")
    b = train_toy_run(config, tmp_path / "b")
    c = train_toy_run(adam, tmp_path / "c")
    losses = [[m["loss"] for m in r.store.read_metrics()] for r in (a, b, c)]
    assert losses[0] == losses[1]
    assert losses[2] != losses[0]
    assert all("val_loss" in m for m in c.store.read_metrics())
    assert c.end_loss < a.end_loss  # Adam with minibatches beats SGD full batch after 30 epochs


def test_toy_lab_opens_a_run_made_by_nd_train(tmp_path):
    result = train_toy_run(load_config(CONFIGS / "toy_sine.yaml", epochs=20), tmp_path)
    view = toy_checkpoint_view(result.store, 20)
    assert view.loss == result.end_loss
    assert toy_config_from_run(result.store.read_config()).optimizer == "sgd"


def test_invalid_config_exits_with_status_2_and_creates_no_folder(tmp_path):
    bad = tmp_path / "bad.yaml"
    text = (CONFIGS / "toy_xor.yaml").read_text("utf-8").replace("name: sgd", "name: adma")
    bad.write_text(text, encoding="utf-8")
    runs = tmp_path / "runs"
    result = train("--config", str(bad), "--runs-dir", str(runs))
    assert result.exit_code == 2
    assert "optimizer.name: unknown optimizer 'adma'" in result.output
    assert not runs.exists()


def test_missing_file_and_bad_override(tmp_path):
    result = train("--config", str(tmp_path / "nope.yaml"), "--runs-dir", str(tmp_path / "r"))
    assert result.exit_code == 2 and "cannot read" in result.output
    result = train(
        "--config", str(CONFIGS / "toy_xor.yaml"), "--seed", "-3", "--runs-dir", str(tmp_path / "r")
    )
    assert result.exit_code == 2 and "seed" in result.output
    assert not (tmp_path / "r").exists()
