"""Tests for ``DemoRecorder`` (keyboard demonstrations, task 6.1b, ADR-051)."""

from pathlib import Path

import numpy as np
import pytest

from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig
from neurodrive.learn.supervised.demos import (
    DemoDataset,
    DemoRecorder,
    save_dataset,
    straight_share,
)

TRACKS = Path(__file__).resolve().parents[3] / "tracks"
START = {
    "s0": np.array([10.0]),
    "lateral": np.array([0.5]),
    "heading_offset": np.array([0.02]),
    "speed": np.array([0.0]),
}


def recorder(name: str = "train_loop_01") -> DemoRecorder:
    track = Track.from_json(TRACKS / f"{name}.json")
    world = World(track, WorldConfig.for_track(track))
    return DemoRecorder(track, world.config, world.dt, world.params, seed=None)


def feed(rec: DemoRecorder, steers: list[float], status: str) -> None:
    rec.begin_episode(START)
    for i, steer in enumerate(steers):
        rec.add(np.full(7, 0.1 * i), np.array([steer, 1.0]), float(i), 0.0, 0.0, 1.0)
    rec.end_episode(status)


def test_worked_example_three_plus_five_frames():
    rec = recorder()
    feed(rec, [0.0, 1.0, 0.0], "crashed")
    feed(rec, [0.0, 0.0, 1.0, 0.0, -1.0], "stopped")
    dataset = rec.to_dataset()
    assert dataset.n_frames == 8 == dataset.meta["demo_frames"] == dataset.meta["env_steps"]
    assert [e["frames"] for e in dataset.meta["episodes"]] == [3, 5]
    assert list(dataset.episode) == [0, 0, 0, 1, 1, 1, 1, 1]
    assert list(dataset.step) == [0, 1, 2, 0, 1, 2, 3, 4]
    # q: steers [0, 0, +1, 0, -1] give 3 / 5 for the second drive
    assert straight_share(dataset.label[3:]) == pytest.approx(0.6)


def test_actions_are_clipped_and_the_start_is_stored():
    rec = recorder()
    rec.begin_episode(START)
    rec.add(np.zeros(7), np.array([2.5, -3.0]), 0.0, 0.0, 0.0, 0.0)
    rec.end_episode("stopped")
    dataset = rec.to_dataset()
    assert list(dataset.label[0]) == [1.0, -1.0]
    episode = dataset.meta["episodes"][0]
    assert episode["start_s0_m"] == 10.0 and episode["start_lateral_m"] == 0.5
    assert episode["start_heading_offset_rad"] == 0.02 and episode["start_kind"] == "live"


def test_counts_include_the_open_drive_and_empty_drives_vanish():
    rec = recorder()
    rec.begin_episode(START)
    assert (rec.n_episodes, rec.n_frames) == (0, 0)
    rec.add(np.zeros(7), np.zeros(2), 0, 0, 0, 0)
    assert (rec.n_episodes, rec.n_frames) == (1, 1)
    rec.end_episode("stopped")
    rec.begin_episode(START)
    rec.end_episode("stopped")  # no frames: dropped
    assert (rec.n_episodes, rec.n_frames) == (1, 1)


def test_misuse_is_refused():
    rec = recorder()
    with pytest.raises(RuntimeError):
        rec.add(np.zeros(7), np.zeros(2), 0, 0, 0, 0)  # no open drive
    with pytest.raises(ValueError, match="nothing recorded"):
        rec.to_dataset()
    rec.begin_episode(START)
    with pytest.raises(RuntimeError):
        rec.begin_episode(START)
    with pytest.raises(RuntimeError, match="close"):
        rec.to_dataset()
    with pytest.raises(ValueError, match="held-out"):
        recorder("hairpins_01")


def test_save_and_load_round_trip(tmp_path):
    rec = recorder()
    feed(rec, [0.0, 1.0, -1.0], "stopped")
    dataset = rec.to_dataset()
    folder = save_dataset(dataset, tmp_path, "me")
    assert folder.name.endswith("-keyboard-me")
    assert DemoDataset.load(folder).identical_to(dataset)
