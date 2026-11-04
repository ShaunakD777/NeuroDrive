"""Tests for learn/supervised/demos.py (task 6.1): which frames count, the labels are the
teacher's, the files round-trip, and recording is deterministic and refuses held-out tracks."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from neurodrive.env.episode import EpisodeRecording, run_episode
from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.sensors import edge_grid, observe
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig
from neurodrive.experiment.exam import EXAM_V1, exam_starts
from neurodrive.experiment.metrics import episode_metrics
from neurodrive.learn.supervised.demos import (
    DATASET_VERSION,
    DemoDataset,
    driven_steps,
    exam_like_starts,
    frames_from_recording,
    record_expert,
    save_dataset,
    straight_share,
)

TRACKS_DIR = Path(__file__).resolve().parents[3] / "tracks"
EXAM = {"start_fractions": EXAM_V1.start_fractions, "lateral_offsets": EXAM_V1.lateral_offsets}


def load(name: str) -> Track:
    return Track.from_json(TRACKS_DIR / f"{name}.json")


def circle(radius: float = 40.0) -> Track:
    angles = 2 * np.pi * np.arange(24) / 24
    return Track("circle", "train", [[radius * np.cos(a), radius * np.sin(a)] for a in angles])


def hand_recording() -> EpisodeRecording:
    """Two cars, 5 steps: car A stops after 3 steps (alive in frames 0-2), car B drives all 5.
    Every value encodes its frame and car (10 * t + car) so rows can be traced back."""
    t_frames, n = 6, 2
    code = 10.0 * np.arange(t_frames)[:, None] + np.arange(n)[None, :]
    alive = np.ones((t_frames, n), dtype=bool)
    alive[3:, 0] = False
    alive[5:, 1] = False
    return EpisodeRecording(
        meta={},
        start_s0=np.zeros(n),
        start_lateral=np.zeros(n),
        start_heading_offset=np.zeros(n),
        start_speed=np.zeros(n),
        x=code,
        y=code + 0.5,
        psi=code * 0.01,
        v=code + 0.25,
        alive=alive,
        progress=code,
        rays=np.repeat(code[:, :, None], 5, axis=2),
        observation=np.repeat(code[:, :, None], 7, axis=2),
        action=np.repeat(code[:5, :, None], 2, axis=2) * -1.0,
        status=np.array([Status.CRASHED, Status.FINISHED], dtype=np.int8),
    )


@pytest.fixture(scope="module")
def loop_demo() -> DemoDataset:
    """The expert on the training loop: the exam's 10 starts plus 2 random starts."""
    return record_expert(load("train_loop_01"), **EXAM, random_starts=2, seed=0)


# ── which frames count ──────────────────────────────────────────────────────────────────────


def test_worked_example_three_plus_five_frames():
    recording = hand_recording()
    assert recording.n_steps == 5 and recording.action.shape == (5, 2, 2)  # 10 actions logged
    assert driven_steps(recording).tolist() == [3, 5]
    rows = frames_from_recording(recording, first_episode=7)
    assert len(rows["observation"]) == 8  # F = 3 + 5
    assert rows["episode"].tolist() == [7, 7, 7, 8, 8, 8, 8, 8]
    assert rows["step"].tolist() == [0, 1, 2, 0, 1, 2, 3, 4]
    # car A's rows are frames 0, 10, 20 (car 0); car B's are 1, 11, ..., 41: nothing frozen
    assert rows["observation"][:, 0].tolist() == [0, 10, 20, 1, 11, 21, 31, 41]
    assert rows["label"][:, 1].tolist() == [0, -10, -20, -1, -11, -21, -31, -41]
    assert rows["x"].tolist() == rows["observation"][:, 0].tolist()


def test_frames_equal_the_steps_of_an_independent_expert_drive(loop_demo):
    track = load("train_loop_01")
    world = World(track, WorldConfig.for_track(track))
    s0, lateral = exam_starts(track.length)
    world.reset(s0=s0, lateral=lateral)
    recording = run_episode(world, ScriptedExpert(world))
    steps = [m.steps for m in episode_metrics(recording, track.length)]
    episodes = loop_demo.meta["episodes"]
    assert [e["frames"] for e in episodes[:10]] == steps
    assert [e["status"] for e in episodes[:10]] == ["finished"] * 10
    exam_rows = loop_demo.episode < 10
    assert int(exam_rows.sum()) == sum(steps)
    # the rows are exactly the recording's running frames, labelled with what the expert did
    for car in range(10):
        mine = loop_demo.episode == car
        n = steps[car]
        assert np.array_equal(loop_demo.observation[mine], recording.observation[:n, car])
        assert np.array_equal(loop_demo.label[mine], recording.action[:n, car])
        assert recording.alive[:n, car].all() and not recording.alive[n, car]


def test_budget_is_the_number_of_rows(loop_demo):
    meta = loop_demo.meta
    assert meta["demo_frames"] == loop_demo.n_frames == sum(e["frames"] for e in meta["episodes"])
    assert meta["env_steps"] == loop_demo.n_frames
    assert np.bincount(loop_demo.episode).tolist() == [e["frames"] for e in meta["episodes"]]


# ── shapes, ranges and labels ───────────────────────────────────────────────────────────────


def test_shapes_and_ranges(loop_demo):
    f = loop_demo.n_frames
    assert loop_demo.observation.shape == (f, 7) and loop_demo.label.shape == (f, 2)
    for name in ("episode", "step", "x", "y", "psi", "v"):
        assert getattr(loop_demo, name).shape == (f,)
    obs = loop_demo.observation
    assert obs[:, :6].min() >= 0.0 and obs[:, :6].max() <= 1.0
    assert obs[:, 6].min() > -1.0 and obs[:, 6].max() <= 1.0
    assert np.abs(loop_demo.label).max() <= 1.0
    assert loop_demo.meta["episodes"][10]["start_kind"] == "random"
    assert len(loop_demo.meta["episodes"]) == 12


def test_replaying_the_labels_from_each_start_reproduces_the_drive(loop_demo):
    """One episode at a time, alone in a fresh world: the stored labels drive the car through
    the stored observations (bit for bit), a fresh expert asked at each state gives the stored
    label, and the episode ends exactly after its frames with its status."""
    track = load("train_loop_01")
    segments = edge_grid(track)
    for episode in (0, 7, 11):
        e = loop_demo.meta["episodes"][episode]
        rows = loop_demo.episode == episode
        world = World(track, WorldConfig.for_track(track))
        world.reset(
            s0=[e["start_s0_m"]],
            lateral=[e["start_lateral_m"]],
            heading_offset=[e["start_heading_offset_rad"]],
            speed=[e["start_speed_mps"]],
        )
        expert = ScriptedExpert(world)
        for obs, label in zip(loop_demo.observation[rows], loop_demo.label[rows], strict=True):
            assert world.running[0]
            assert np.array_equal(observe(world, segments)[0], obs)
            assert np.array_equal(expert.label()[0], label)
            world.step(label[:1], label[1:])
        assert not world.running[0]
        assert Status(int(world.status[0])).name.lower() == e["status"]


def test_straight_share_worked_example():
    steers = np.array([0.02, -0.05, 0.4, 0.01, -0.6])
    assert straight_share(np.stack([steers, np.zeros(5)], axis=1)) == pytest.approx(0.6)


def test_exam_like_starts_are_the_exams_starts():
    for length in (730.7, 1540.0):
        mine = exam_like_starts(length, **EXAM)
        theirs = exam_starts(length)
        assert all(np.array_equal(a, b) for a, b in zip(mine, theirs, strict=True))


# ── files ───────────────────────────────────────────────────────────────────────────────────


def test_save_and_load_give_the_same_dataset(tmp_path):
    dataset = record_expert(
        circle(), start_fractions=(0.0, 0.5), lateral_offsets=(0.0,), random_starts=1, seed=3
    )
    folder = save_dataset(dataset, tmp_path, "t1", provenance={"git_commit": "abc"})
    assert folder.name.endswith("-expert-t1") and folder.parent == tmp_path
    assert sorted(p.name for p in folder.iterdir()) == ["dataset.npz", "meta.json"]
    again = DemoDataset.load(folder)
    assert again.identical_to(dataset)
    assert again.meta["dataset_id"] == folder.name and again.meta["git_commit"] == "abc"
    assert again.meta["dataset_version"] == DATASET_VERSION
    assert again.episode.dtype == np.int64 and again.observation.dtype == np.float64
    when = datetime.strptime(folder.name[:15], "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    second = save_dataset(dataset, tmp_path, "t1", when=when)  # the same second: a new name
    assert second.name == f"{folder.name}-2"


def test_broken_files_are_refused(tmp_path):
    dataset = record_expert(
        circle(), start_fractions=(0.0,), lateral_offsets=(0.0,), random_starts=0, seed=0
    )
    folder = save_dataset(dataset, tmp_path / "a")
    meta_path = folder / "meta.json"
    text = meta_path.read_text(encoding="utf-8")
    meta_path.write_text(text.replace('"dataset_version": 1', '"dataset_version": 9'), "utf-8")
    with pytest.raises(ValueError, match="unsupported dataset version 9"):
        DemoDataset.load(folder)
    meta_path.write_text(text.replace('"demo_frames": ', '"demo_frames": 1'), "utf-8")
    with pytest.raises(ValueError, match="frames but the arrays hold"):
        DemoDataset.load(folder)
    with pytest.raises(FileNotFoundError, match="not a dataset folder"):
        DemoDataset.load(tmp_path)
    with pytest.raises(ValueError, match="label must be"):
        DemoDataset(
            meta={},
            observation=np.zeros((3, 7)),
            label=np.zeros((2, 2)),
            **{k: np.zeros(3) for k in ("episode", "step", "x", "y", "psi", "v")},
        )
    with pytest.raises(ValueError, match="tag must be"):
        save_dataset(dataset, tmp_path, "a b")


# ── rules ───────────────────────────────────────────────────────────────────────────────────


def test_held_out_tracks_are_refused():
    with pytest.raises(ValueError, match="held-out track"):
        record_expert(load("hairpins_01"), **EXAM, random_starts=0, seed=0)
    with pytest.raises(ValueError, match="no starts"):
        record_expert(circle(), start_fractions=(), lateral_offsets=(0.0,), random_starts=0, seed=0)


def test_recording_is_deterministic_and_the_seed_picks_the_random_starts():
    args = {"start_fractions": (0.0,), "lateral_offsets": (0.0,), "random_starts": 3}
    a = record_expert(circle(), **args, seed=5)
    b = record_expert(circle(), **args, seed=5)
    c = record_expert(circle(), **args, seed=6)
    assert a.identical_to(b)
    assert not a.identical_to(c)
    starts = [[e["start_s0_m"] for e in d.meta["episodes"]] for d in (a, c)]
    assert starts[0][0] == starts[1][0]  # the exam-like start does not depend on the seed
    assert starts[0][1:] != starts[1][1:]
    for e in a.meta["episodes"][1:]:
        assert abs(e["start_lateral_m"]) <= 1.0 and abs(e["start_heading_offset_rad"]) <= 0.1
