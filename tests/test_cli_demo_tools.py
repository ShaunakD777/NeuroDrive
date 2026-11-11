"""Tests for `nd demo-merge` and `nd demo-record --starts-from` (cli.py), task 6.7."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from neurodrive.cli import app
from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig
from neurodrive.learn.supervised.demos import DemoDataset, DemoRecorder, save_dataset

runner = CliRunner()
TRACKS = Path(__file__).resolve().parents[1] / "tracks"


def save_keyboard(out_dir, tag: str, frames: list[int]) -> Path:
    track = Track.from_json(TRACKS / "train_loop_01.json")
    world = World(track, WorldConfig.for_track(track))
    rec = DemoRecorder(track, world.config, world.dt, world.params, seed=None)
    for i, n in enumerate(frames):
        rec.begin_episode(
            {
                "s0": np.array([50.0 * (i + 1)]),
                "lateral": np.array([0.0]),
                "heading_offset": np.array([0.0]),
                "speed": np.array([0.0]),
            }
        )
        for t in range(n):
            rec.add(np.zeros(7), np.array([0.0, 1.0]), t, 0, 0, 0)
        rec.end_episode("stopped")
    return save_dataset(rec.to_dataset(), out_dir, tag)


def test_merge_joins_datasets_by_name_and_leaves_the_sources(tmp_path):
    a = save_keyboard(tmp_path, "a", [3, 5])
    b = save_keyboard(tmp_path, "b", [4])
    result = runner.invoke(
        app, ["demo-merge", "keyboard-a", b.name, "--tag", "both", "--datasets-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert "frames: 12 from 3 episodes of 2 datasets" in result.stdout
    folders = sorted(p.name for p in tmp_path.iterdir())
    assert len(folders) == 3 and a.name in folders and b.name in folders
    merged = DemoDataset.load(tmp_path / next(f for f in folders if f.endswith("-both")))
    assert merged.n_frames == 12 and merged.meta["merged_from"] == [a.name, b.name]


def test_merge_refuses_what_it_cannot_merge(tmp_path):
    save_keyboard(tmp_path, "a", [3])
    for args, message in (
        (["keyboard-a", "keyboard-nowhere"], "no dataset"),
        (["keyboard-a", "--tag", "bad tag"], "tag must be"),
    ):
        result = runner.invoke(app, ["demo-merge", *args, "--datasets-dir", str(tmp_path)])
        assert result.exit_code == 2, args
        assert message in result.output
    assert len(list(tmp_path.iterdir())) == 1


def test_record_from_the_starts_of_a_dataset(tmp_path):
    source = save_keyboard(tmp_path, "src", [3, 5])
    args = ["demo-record", "--starts-from", source.name, "--match-frames", "--tag", "m"]
    result = runner.invoke(app, [*args, "--out-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    folder = next(p for p in tmp_path.iterdir() if p.name.endswith("-expert-m"))
    dataset = DemoDataset.load(folder)
    assert [e["frames"] for e in dataset.meta["episodes"]] == [3, 5]
    assert dataset.meta["starts"]["source_dataset"] == source.name
    assert "frames: 8 from 2 episodes" in result.stdout


def test_record_from_starts_refusals(tmp_path):
    source = save_keyboard(tmp_path, "src", [3])
    for args, message in (
        (["--match-frames"], "--match-frames needs --starts-from"),
        (["--starts-from", "keyboard-missing"], "no dataset"),
        (["--starts-from", source.name, "--track", "hairpins_01"], "held-out"),
    ):
        result = runner.invoke(app, ["demo-record", *args, "--out-dir", str(tmp_path)])
        assert result.exit_code == 2, args
        assert message in result.output
    assert len(list(tmp_path.iterdir())) == 1
