"""Tests for DAgger runs (task 6.6, ADR-049): the config, a short run with an exam card per round
and the budget in every currency, an exact resume, the dataset of collected frames, and the
command line."""

import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import yaml
from typer.testing import CliRunner

from neurodrive.cli import app
from neurodrive.env.track import Track
from neurodrive.experiment.config import (
    ConfigError,
    DaggerRunConfig,
    load_config,
    parse_config,
    to_dagger_settings,
)
from neurodrive.experiment.runner import learner_from_config, sweep_seeds, train_dagger_run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.supervised.dagger import DaggerLearner
from neurodrive.learn.supervised.demos import DemoDataset, record_expert, save_dataset

REPO = Path(__file__).resolve().parents[2]
WHEN = datetime(2026, 11, 10, 12, 0, 0, tzinfo=UTC)
runner = CliRunner()


def config_data(**overrides) -> dict:
    """A small DAgger config: 3 rounds of 2 epochs, 2 cars per collection, exams off."""
    data = {
        "method": "dagger",
        "tag": "small",
        "data": {"dataset": "expert-t"},
        "training": {"epochs_per_round": 2, "batch_size": 512},
        "dagger": {"rounds": 3, "random_starts": 2},
        "optimizer": {"name": "adam", "learning_rate": 0.003},
        "exams": False,
    }
    return {**data, **overrides}


@pytest.fixture(scope="module")
def recorded(tmp_path_factory) -> Path:
    """A folder with one small expert dataset: 2 exam-like starts and 2 random starts."""
    folder = tmp_path_factory.mktemp("demos")
    track = Track.from_json(REPO / "tracks" / "train_loop_01.json")
    demo = record_expert(
        track, start_fractions=(0.0, 0.5), lateral_offsets=(0.0,), random_starts=2, seed=0
    )
    save_dataset(demo, folder, tag="t", when=WHEN)
    return folder


@pytest.fixture
def datasets(recorded, tmp_path) -> Path:
    """A fresh copy per test, because DAgger runs save their collected frames into it."""
    copy = tmp_path / "demos"
    shutil.copytree(recorded, copy)
    return copy


def metrics_of(store: RunStore) -> list[dict]:
    return [{k: v for k, v in m.items() if k != "wall_clock_s"} for m in store.read_metrics()]


def dagger_folders(datasets: Path) -> list[Path]:
    return sorted(p for p in datasets.iterdir() if "-dagger-" in p.name)


# ── the config ──────────────────────────────────────────────────────────────────────────────


def test_a_dagger_config_gets_its_defaults():
    config = parse_config(config_data())
    assert isinstance(config, DaggerRunConfig)
    assert config.network.layer_sizes == [7, 16, 16, 2]
    assert (config.dagger.track, config.dagger.target_laps) == ("train_loop_01", 2)
    assert (config.training.loss, config.data.val_fraction) == ("mse", 0.2)
    settings = to_dagger_settings(config)
    assert (settings.epochs_per_round, settings.random_starts) == (2, 2)
    assert (settings.imitation.optimizer, settings.imitation.batch_size) == ("adam", 512)
    assert parse_config(config_data(), units=7).dagger.rounds == 7  # --units sets the rounds


def test_the_shipped_config_is_valid():
    config = load_config(REPO / "configs" / "dagger.yaml")
    assert isinstance(config, DaggerRunConfig)
    assert config.data.dataset == "expert-default" and config.data.sha256 is None
    assert (config.dagger.rounds, config.training.epochs_per_round) == (10, 30)
    assert (config.dagger.random_starts, config.checkpoint_every, config.exams) == (10, 1, True)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"dagger": {"rounds": 3, "track": "hairpins_01"}}, "held-out track"),
        ({"dagger": {"rounds": 0}}, "dagger.rounds"),
        ({"dagger": {"rounds": 2, "random_starts": 0}}, "dagger.random_starts"),
        ({"dagger": {}}, "dagger.rounds: Field required"),
        ({"training": {"epochs": 5}}, "training.epochs: unknown field"),
        ({"network": {"layer_sizes": [2, 4, 1]}}, "7 inputs first"),
    ],
)
def test_bad_dagger_configs_are_refused(change, message):
    with pytest.raises(ConfigError, match=message.replace(".", r"\.")):
        parse_config(config_data(**change))


def test_epochs_are_not_for_dagger():
    with pytest.raises(ConfigError, match="--epochs is for toy and imitation"):
        parse_config(config_data(), epochs=5)


# ── the task's verification: the budget, an exam per round ─────────────────────────────────


def test_a_short_run_counts_every_teacher_query(datasets, tmp_path):
    config = parse_config(config_data(checkpoint_every=1, exams=True))
    result = train_dagger_run(config, tmp_path / "runs", datasets_dir=datasets)
    store = result.store
    assert result.status == "completed" and result.rounds_done == 3
    assert store.checkpoint_units() == store.exam_units() == [0, 1, 2, 3]
    meta, metrics = store.read_meta(), store.read_metrics()
    frames = meta["dataset"]["demo_frames"]
    assert meta["method"] == "dagger"
    collected = [m["collect_frames"] for m in metrics]
    assert collected[0] is None and all(c > 0 for c in collected[1:])
    # Round 1 is the demonstrations; every later frame is one step and one teacher query.
    assert [m["demo_frames_total"] for m in metrics] == [frames] * 3
    assert [m["teacher_queries_total"] for m in metrics] == [0, collected[1], sum(collected[1:])]
    assert all(m["env_steps_total"] == m["teacher_queries_total"] for m in metrics)
    assert result.teacher_queries == sum(collected[1:])
    assert result.end_onpolicy_loss == metrics[-1]["onpolicy_loss"]
    # The collected frames were saved as a dataset (decision D4A) and named in meta.json.
    (folder,) = dagger_folders(datasets)
    assert meta["dagger_dataset"] == result.dagger_dataset == folder.name
    saved = DemoDataset.load(folder)
    assert saved.n_frames == result.teacher_queries
    assert saved.meta["dagger_run"] == store.path.name
    assert saved.meta["tag"] == "small"


# ── resuming ────────────────────────────────────────────────────────────────────────────────


def test_a_failed_run_resumes_exactly(datasets, tmp_path, monkeypatch):
    config = parse_config(config_data(dagger={"rounds": 4, "random_starts": 2}))
    reference = train_dagger_run(config, tmp_path / "a", datasets_dir=datasets)
    original = DaggerLearner.train_unit

    def fails_in_round_3(self):
        if self.rounds == 2:
            raise RuntimeError("the process died")
        return original(self)

    monkeypatch.setattr(DaggerLearner, "train_unit", fails_in_round_3)
    with pytest.raises(RuntimeError):
        train_dagger_run(config, tmp_path / "b", datasets_dir=datasets)
    monkeypatch.undo()
    (folder,) = [p for p in (tmp_path / "b").iterdir() if p.is_dir()]
    store = RunStore.open(folder)
    assert store.read_meta()["status"] == "failed" and len(store.read_metrics()) == 2
    assert len(dagger_folders(datasets)) == 1  # a failed run saves no dataset
    answer = runner.invoke(app, ["resume", "--run", str(folder), "--datasets-dir", str(datasets)])
    assert answer.exit_code == 0, answer.output
    assert "status: completed after 4 units (2 were kept)" in answer.output
    assert "teacher queries:" in answer.output and "saved as dataset" in answer.output
    assert metrics_of(store) == metrics_of(reference.store)
    for unit in store.checkpoint_units():
        assert store.load_checkpoint(unit).equals(reference.store.load_checkpoint(unit))
    (state_a, _), (state_b, _) = store.load_state(), reference.store.load_state()
    assert state_a.info == state_b.info
    assert all(np.array_equal(state_a.arrays[k], state_b.arrays[k]) for k in state_b.arrays)
    first, second = (DemoDataset.load(p) for p in dagger_folders(datasets))
    assert all(
        np.array_equal(getattr(first, k), getattr(second, k))
        for k in ("observation", "label", "episode", "step")
    )
    assert store.read_meta()["dagger_dataset"] == dagger_folders(datasets)[1].name


def test_a_run_folders_config_rebuilds_the_learner(datasets, tmp_path):
    store = train_dagger_run(
        parse_config(config_data(dagger={"rounds": 1, "random_starts": 2})),
        tmp_path,
        datasets_dir=datasets,
    ).store
    learner = learner_from_config(store.read_config(), datasets_dir=datasets)
    assert isinstance(learner, DaggerLearner) and learner.track.name == "train_loop_01"
    assert dagger_folders(datasets) == []  # one round collects nothing, so nothing is saved
    assert "dagger_dataset" not in store.read_meta()


# ── the command line ────────────────────────────────────────────────────────────────────────


def test_nd_train_and_sweep(datasets, tmp_path):
    config_file = tmp_path / "small.yaml"
    config_file.write_text(yaml.safe_dump(config_data()), encoding="utf-8")
    common = ["--config", str(config_file), "--datasets-dir", str(datasets)]
    answer = runner.invoke(
        app, ["train", *common, "--units", "2", "--runs-dir", str(tmp_path / "r")]
    )
    assert answer.exit_code == 0, answer.output
    assert "dataset: 20261110-120000-expert-t" in answer.output
    assert "status: completed after 2 rounds" in answer.output
    assert "teacher queries:" in answer.output and "on-policy error" in answer.output
    refused = runner.invoke(app, ["train", *common, "--epochs", "2"])
    assert refused.exit_code == 2 and "--epochs is for toy and imitation" in refused.output
    sweep = runner.invoke(
        app, ["sweep", *common, "--seeds", "0-1", "--units", "2", "--runs-dir", str(tmp_path / "s")]
    )
    assert sweep.exit_code == 0, sweep.output
    assert "final validation loss: median" in sweep.output
    assert "final on-policy error: median" in sweep.output


def test_a_sweep_pins_the_dataset_once(datasets, tmp_path):
    config = parse_config(config_data(dagger={"rounds": 2, "random_starts": 2}))
    _, results = sweep_seeds(config, [0, 1], tmp_path, datasets_dir=datasets)
    configs = [r.store.read_config() for r in results]
    assert configs[0]["data"] == configs[1]["data"]
    assert [c["tag"] for c in configs] == ["small-s0", "small-s1"]
    assert [p.name.split("-dagger-")[1] for p in dagger_folders(datasets)] == [
        "small-s0",
        "small-s1",
    ]


def test_the_wide_config_and_its_ranges():
    # Task 6.6 follow-up (ADR-050): the collection starts' ranges, defaults as before.
    wide = load_config(REPO / "configs" / "dagger_wide.yaml")
    assert (wide.dagger.start_lateral_m, wide.dagger.start_heading_rad) == (3.0, 0.35)
    assert wide.tag == "wide" and wide.dagger.rounds == 10
    plain = parse_config(config_data())
    assert (plain.dagger.start_lateral_m, plain.dagger.start_heading_rad) == (1.0, 0.1)
    for change, field in (
        ({"start_lateral_m": 5.0}, "start_lateral_m"),
        ({"start_heading_rad": 1.0}, "start_heading_rad"),
    ):
        with pytest.raises(ConfigError, match=rf"dagger\.{field}"):
            parse_config(config_data(dagger={"rounds": 2, **change}))
