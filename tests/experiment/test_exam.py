"""Tests for experiment/exam.py: the fixed spec, the starts, determinism, and that a card is
exactly what the recorded drives say."""

import json
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.env.episode import EpisodeRecording, replay, run_episode
from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.policies import MLPPolicy, NeuronPolicy
from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig
from neurodrive.experiment.exam import (
    EXAM_V1,
    ExamResult,
    exam_starts,
    exam_track,
    format_card,
    run_exam,
)
from neurodrive.experiment.metrics import episode_metrics

TRACKS = Path(__file__).resolve().parents[2] / "tracks"
START_FIELDS = ("start_index", "start_s0_m", "start_lateral_m")


def load(name: str) -> Track:
    return Track.from_json(TRACKS / f"{name}.json")


def s1_neuron(world: World) -> NeuronPolicy:
    return NeuronPolicy((-12.0, 12.0), 0.0, "tanh", 0.24)  # Study S1's hand-wired baseline


@pytest.fixture(scope="module")
def expert_card() -> ExamResult:
    return run_exam(ScriptedExpert, {"kind": "expert"})


# ── the spec ────────────────────────────────────────────────────────────────────────────────


def test_v1_is_the_training_track_plus_every_heldout_track_on_disk():
    splits = {
        p.stem: json.loads(p.read_text(encoding="utf-8"))["split"] for p in TRACKS.glob("*.json")
    }
    heldout = sorted(name for name, split in splits.items() if split == "heldout")
    # Adding a held-out track file must not change exam v1 silently: this fails instead.
    assert EXAM_V1.tracks == ("train_loop_01", *heldout)
    assert EXAM_V1.version == 1 and EXAM_V1.target_laps == 2


def test_starts_match_the_worked_example():
    length = load("train_loop_01").length  # 730.7 m
    s0, lateral = exam_starts(length)
    expected = [0.0, 146.1, 292.3, 438.4, 584.6]
    assert s0 == pytest.approx(expected * 2, abs=0.05)
    assert lateral.tolist() == [0.0] * 5 + [1.5] * 5


# ── a card is what the recorded drives say ──────────────────────────────────────────────────


def record_by_hand(track: Track, policy_for, tmp_path: Path) -> EpisodeRecording:
    """The exam's 10 starts written out independently of exam_starts, one batch, saved and
    reloaded from disk."""
    fractions = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    world = World(track, WorldConfig.for_track(track))
    world.reset(
        s0=np.concatenate([fractions, fractions]) * track.length, lateral=[0.0] * 5 + [1.5] * 5
    )
    recording = run_episode(world, policy_for(world))
    return EpisodeRecording.load(recording.save(tmp_path / f"{track.name}.npz"))


def card_metrics(part) -> list[dict]:
    return [{k: v for k, v in e.items() if k not in START_FIELDS} for e in part.episodes]


@pytest.mark.parametrize(
    ("name", "policy_for"),
    [
        ("hairpins_01", ScriptedExpert),
        ("train_loop_01", lambda world: MLPPolicy(init_network(NetworkSpec(), make_rng(3)))),
    ],
    ids=["expert-hairpins", "random-network-loop"],
)
def test_the_exam_of_a_recorded_policy_matches_its_recording(name, policy_for, tmp_path):
    track = load(name)
    recording = record_by_hand(track, policy_for, tmp_path)
    assert replay(recording, track).identical_to(recording)  # the recording is the drive
    part, exam_recording = exam_track(track, policy_for)
    from_recording = [m.to_dict() for m in episode_metrics(recording, track.length)]
    assert card_metrics(part) == from_recording  # exactly, not approximately
    assert np.array_equal(exam_recording.x, recording.x)


# ── determinism and the first cards ─────────────────────────────────────────────────────────


def test_the_same_policy_gets_the_same_card_twice():
    first = run_exam(s1_neuron, {"kind": "s1_neuron"})
    second = run_exam(s1_neuron, {"kind": "s1_neuron"})
    assert first.to_dict() == second.to_dict()
    per_track = {p.track: p.aggregate["completion_rate"] for p in first.tracks}
    # It laps the loop and its mirror (it steers symmetrically) and crashes in the tight bends.
    assert per_track["train_loop_01"] == 1.0 and per_track["train_loop_01_mirror"] == 1.0
    assert per_track["hairpins_01"] == 0.0 and per_track["sweeper_01"] == 0.0


def test_the_expert_completes_every_episode_of_the_exam(expert_card):
    assert [p.track for p in expert_card.tracks] == list(EXAM_V1.tracks)
    for part in expert_card.tracks:
        assert len(part.episodes) == 10
        assert part.aggregate["completion_rate"] == 1.0 and part.aggregate["crash_rate"] == 0.0
        assert part.time_limit_s == WorldConfig.for_track(load(part.track)).time_limit
    assert expert_card.training["completion_rate"] == 1.0 and expert_card.training["episodes"] == 10
    assert expert_card.heldout["completion_rate"] == 1.0 and expert_card.heldout["episodes"] == 40
    assert expert_card.env_steps > 0


def test_heldout_lap_times_are_given_per_track_not_pooled(expert_card):
    # Lap times compare only within a track (ADR-022, ADR-036): no pooled median best lap.
    assert "median_best_lap_s" not in expert_card.heldout
    by_track = expert_card.heldout["median_best_lap_s_by_track"]
    heldout_parts = [p for p in expert_card.tracks if p.split == "heldout"]
    assert by_track == {p.track: p.aggregate["median_best_lap_s"] for p in heldout_parts}
    assert list(by_track) == ["hairpins_01", "monza_ccw", "sweeper_01", "train_loop_01_mirror"]
    assert by_track["monza_ccw"] > by_track["hairpins_01"] > by_track["sweeper_01"]
    # The rates are still pooled: 40 episodes, each track weighing the same.
    assert expert_card.heldout["episodes"] == 40
    assert "per track" in format_card(expert_card)


def test_a_card_survives_json_and_prints(expert_card, tmp_path):
    path = expert_card.save(tmp_path / "exams" / "unit_0.json")
    assert ExamResult.load(path).to_dict() == expert_card.to_dict()
    text = format_card(expert_card)
    assert "HELD-OUT (pooled)" in text and "monza_ccw (heldout)" in text
    with pytest.raises(ValueError):
        ExamResult.from_dict({**expert_card.to_dict(), "exam_version": 2})
