"""Tests for imitation runs (task 6.3, ADR-047): the config, the runner's ``checkpoint_every``,
the pinned dataset, a short run with an exam card after every epoch, and an exact resume."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner
from walk_learner import SEED, UNITS, WalkLearner

from neurodrive.cli import app
from neurodrive.env.track import Track
from neurodrive.experiment.config import (
    ConfigError,
    ImitationRunConfig,
    load_config,
    parse_config,
    to_imitation_settings,
)
from neurodrive.experiment.runner import (
    Budget,
    checkpoint_due,
    learner_from_config,
    pin_dataset,
    resume,
    run,
    sweep_seeds,
    train_imitation_run,
)
from neurodrive.experiment.store import RunStore
from neurodrive.learn.supervised.demos import DATASET_FILE, record_expert, save_dataset
from neurodrive.learn.supervised.imitation import ImitationLearner

REPO = Path(__file__).resolve().parents[2]
WHEN = datetime(2026, 11, 4, 12, 0, 0, tzinfo=UTC)
runner = CliRunner()


def config_data(**overrides) -> dict:
    """A small imitation config: 3 epochs, batches of 512, exams off, as plain data."""
    data = {
        "method": "imitation",
        "tag": "small",
        "data": {"dataset": "expert-t"},
        "training": {"epochs": 3, "batch_size": 512},
        "optimizer": {"name": "adam", "learning_rate": 0.003},
        "exams": False,
    }
    return {**data, **overrides}


@pytest.fixture(scope="module")
def datasets(tmp_path_factory) -> Path:
    """A folder with one small expert dataset: 2 exam-like starts and 2 random starts."""
    folder = tmp_path_factory.mktemp("demos")
    track = Track.from_json(REPO / "tracks" / "train_loop_01.json")
    demo = record_expert(
        track, start_fractions=(0.0, 0.5), lateral_offsets=(0.0,), random_starts=2, seed=0
    )
    save_dataset(demo, folder, tag="t", when=WHEN)
    return folder


def metrics_of(store: RunStore) -> list[dict]:
    return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]


# ── the config ──────────────────────────────────────────────────────────────────────────────


def test_an_imitation_config_gets_its_defaults():
    config = parse_config(config_data())
    assert isinstance(config, ImitationRunConfig)
    assert config.network.layer_sizes == [7, 16, 16, 2]
    assert config.network.output_activation == "tanh"
    assert (config.data.sha256, config.data.val_fraction) == (None, 0.2)
    assert (config.checkpoint_every, config.training.loss) == (None, "mse")
    settings = to_imitation_settings(config)
    assert (settings.optimizer, settings.learning_rate, settings.batch_size) == ("adam", 0.003, 512)
    assert settings.optimizer_settings == {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
    assert parse_config(config_data(), epochs=7).training.epochs == 7  # --epochs works


def test_the_shipped_config_is_valid():
    config = load_config(REPO / "configs" / "imitation.yaml")
    assert isinstance(config, ImitationRunConfig)
    assert config.data.dataset == "expert-default" and config.data.sha256 is None
    assert (config.training.epochs, config.checkpoint_every, config.exams) == (100, 1, True)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"lr": 1}, "lr: unknown field"),
        ({"network": {"layer_sizes": [2, 4, 1]}}, "7 inputs first"),
        ({"data": {"dataset": "../elsewhere"}}, "data.dataset"),
        ({"data": {"dataset": "expert-t", "sha256": "abc"}}, "data.sha256"),
        ({"data": {"dataset": "expert-t", "val_fraction": 1.0}}, "data.val_fraction"),
        ({"checkpoint_every": 0}, "checkpoint_every"),
        ({"training": {"epochs": 0}}, "training.epochs"),
        ({"data": {}}, "data.dataset: Field required"),
    ],
)
def test_bad_imitation_configs_are_refused(change, message):
    with pytest.raises(ConfigError, match=message.replace(".", r"\.")):
        parse_config(config_data(**change))


def test_units_are_not_for_imitation():
    with pytest.raises(ConfigError, match="--units is for random_search"):
        parse_config(config_data(), units=5)


# ── the runner: checkpoint_every (decision D3) ──────────────────────────────────────────────


def test_checkpoint_due_adds_every_kth_unit():
    assert [u for u in range(21) if checkpoint_due(u, 5)] == [0, 1, 2, 5, 10, 15, 20]
    assert [u for u in range(21) if checkpoint_due(u)] == [0, 1, 2, 5, 10, 20]
    assert all(checkpoint_due(u, 1) for u in range(21))


@pytest.mark.parametrize(
    ("every", "expected"),
    [
        (None, [0, 1, 2, 5, 10, 12]),
        (3, [0, 1, 2, 3, 5, 6, 9, 10, 12]),
        (1, list(range(13))),
    ],
)
def test_the_runner_checkpoints_every_kth_unit(tmp_path, every, expected):
    store = RunStore.create(tmp_path, "walk", "t", {"seed": SEED})
    run(WalkLearner(SEED), store, Budget(units=UNITS), exams=False, checkpoint_every=every)
    assert store.checkpoint_units() == expected
    assert store.read_meta()["checkpoint_every"] == every
    with pytest.raises(ValueError, match="checkpoint_every"):
        run(WalkLearner(SEED), store, Budget(units=1), exams=False, checkpoint_every=0)


# ── the dataset is pinned (decision D1) ─────────────────────────────────────────────────────


def test_pinning_names_the_exact_dataset_and_its_checksum(datasets):
    pinned, dataset = pin_dataset(parse_config(config_data()), datasets)
    assert pinned.data.dataset == "20261104-120000-expert-t"
    assert pinned.data.sha256 is not None and len(pinned.data.sha256) == 64
    assert dataset.n_frames == dataset.meta["demo_frames"] and len(dataset.meta["episodes"]) == 4
    again, _ = pin_dataset(pinned, datasets)  # pinning a pinned config changes nothing
    assert again == pinned
    with pytest.raises(FileNotFoundError, match="no dataset 'expert-other'"):
        pin_dataset(parse_config(config_data(data={"dataset": "expert-other"})), datasets)


# ── the task's verification: an exam card after every epoch ────────────────────────────────


def test_a_short_run_has_an_exam_card_after_every_epoch(datasets, tmp_path):
    config = parse_config(config_data(training={"epochs": 4}, checkpoint_every=1, exams=True))
    result = train_imitation_run(config, tmp_path, datasets_dir=datasets)
    store = result.store
    assert result.status == "completed" and result.epochs_done == 4
    # 3 is on no schedule: only checkpoint_every puts it there.
    assert store.checkpoint_units() == store.exam_units() == [0, 1, 2, 3, 4]
    card = store.load_exam(3)
    assert card.policy == {"kind": "checkpoint", "run": store.path.name, "unit": 3}
    meta, saved = store.read_meta(), store.read_config()
    frames = meta["dataset"]["demo_frames"]
    assert meta["method"] == "imitation" and meta["checkpoint_every"] == 1
    assert saved["data"]["dataset"] == meta["dataset"]["id"] == "20261104-120000-expert-t"
    assert saved["data"]["sha256"] == meta["dataset"]["sha256"]
    assert len(meta["dataset"]["val_episodes"]) == 1  # round(0.2 · 4) = 1 of the 4 drives
    metrics = store.read_metrics()
    assert [m["demo_frames_total"] for m in metrics] == [frames] * 4  # charged once
    assert all(m["env_steps_total"] == 0 and m["teacher_queries_total"] == 0 for m in metrics)
    assert result.end_loss == metrics[-1]["loss"] and result.end_val_loss == metrics[-1]["val_loss"]
    assert metrics[-1]["loss"] < metrics[0]["loss_before"]


# ── resuming, and refusing changed data ─────────────────────────────────────────────────────


def test_a_failed_run_resumes_exactly(datasets, tmp_path, monkeypatch):
    config = parse_config(
        config_data(training={"epochs": 6, "batch_size": 512}, checkpoint_every=2)
    )
    reference = train_imitation_run(config, tmp_path / "a", datasets_dir=datasets).store
    original = ImitationLearner.train_unit

    def fails_in_unit_4(self):
        if self.epochs == 3:
            raise RuntimeError("the process died")
        return original(self)

    monkeypatch.setattr(ImitationLearner, "train_unit", fails_in_unit_4)
    with pytest.raises(RuntimeError):
        train_imitation_run(config, tmp_path / "b", datasets_dir=datasets)
    monkeypatch.undo()
    (folder,) = [p for p in (tmp_path / "b").iterdir() if p.is_dir()]  # besides neurodrive.db
    store = RunStore.open(folder)
    assert store.read_meta()["status"] == "failed" and len(store.read_metrics()) == 3
    answer = runner.invoke(app, ["resume", "--run", str(folder), "--datasets-dir", str(datasets)])
    assert answer.exit_code == 0, answer.output
    assert "status: completed after 6 units (3 were kept)" in answer.output
    assert metrics_of(store) == metrics_of(reference)
    # 4 and 6 come from checkpoint_every 2, which the resumed run read back from meta.json.
    assert store.checkpoint_units() == reference.checkpoint_units() == [0, 1, 2, 4, 5, 6]
    for unit in store.checkpoint_units():
        assert store.load_checkpoint(unit).equals(reference.load_checkpoint(unit))
    (state_a, _), (state_b, _) = store.load_state(), reference.load_state()
    assert state_a.info == state_b.info
    assert all(np.array_equal(state_a.arrays[k], state_b.arrays[k]) for k in state_b.arrays)


def test_a_changed_dataset_is_refused(datasets, tmp_path):
    copy = tmp_path / "demos"
    shutil.copytree(datasets, copy)
    store = train_imitation_run(
        parse_config(config_data(training={"epochs": 1})), tmp_path / "runs", datasets_dir=copy
    ).store
    (folder,) = copy.iterdir()
    raw = bytearray((folder / DATASET_FILE).read_bytes())
    raw[-1] ^= 1
    (folder / DATASET_FILE).write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="has changed"):
        learner_from_config(store.read_config(), datasets_dir=copy)
    config_file = tmp_path / "again.yaml"
    config_file.write_text(yaml.safe_dump(store.read_config()), encoding="utf-8")
    answer = runner.invoke(
        app,
        [
            "train",
            "--config",
            str(config_file),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--datasets-dir",
            str(copy),
        ],
    )
    assert answer.exit_code == 2 and "has changed" in answer.output


def test_a_run_killed_before_any_state_resumes_from_unit_1(datasets, tmp_path):
    store = RunStore.create(tmp_path, "imitation", "t", {})
    store.update_meta(budget={"units": 2, "env_steps": None, "wall_clock_s": None})
    pinned, _ = pin_dataset(parse_config(config_data()), datasets)
    learner = learner_from_config(pinned.model_dump(mode="json"), datasets_dir=datasets)
    store.update_meta(exams=False, resumed_at=[], exam_wall_clock_s=0.0)
    assert resume(learner, store) == "completed"  # killed before any state: starts at unit 1
    assert [m["unit"] for m in store.read_metrics()] == [1, 2]


# ── the command line ────────────────────────────────────────────────────────────────────────


def test_nd_train_and_sweep_with_a_datasets_dir(datasets, tmp_path):
    config_file = tmp_path / "small.yaml"
    config_file.write_text(yaml.safe_dump(config_data(training={"epochs": 2})), encoding="utf-8")
    common = ["--config", str(config_file), "--datasets-dir", str(datasets)]
    answer = runner.invoke(app, ["train", *common, "--runs-dir", str(tmp_path / "runs")])
    assert answer.exit_code == 0, answer.output
    assert "dataset: 20261104-120000-expert-t" in answer.output
    assert "status: completed after 2 epochs" in answer.output
    assert "validation" in answer.output
    missing = runner.invoke(app, ["train", "--config", str(config_file), "--datasets-dir", "none"])
    assert missing.exit_code == 2 and "no dataset 'expert-t'" in missing.output
    sweep = runner.invoke(
        app, ["sweep", *common, "--seeds", "0-1", "--runs-dir", str(tmp_path / "sweep")]
    )
    assert sweep.exit_code == 0, sweep.output
    assert "final validation loss: median" in sweep.output


def test_a_sweep_pins_the_dataset_once(datasets, tmp_path):
    config = parse_config(config_data(training={"epochs": 1}))
    _, results = sweep_seeds(config, [0, 1], tmp_path, datasets_dir=datasets)
    configs = [r.store.read_config() for r in results]
    assert configs[0]["data"] == configs[1]["data"]  # the same id and checksum
    assert [c["seed"] for c in configs] == [0, 1]
    assert results[0].end_loss != results[1].end_loss
