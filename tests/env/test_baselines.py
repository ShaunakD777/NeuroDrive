"""Guards for the committed baselines: configs/baselines/hand_neuron.yaml must keep working."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from neurodrive.env.episode import EpisodeRecording, replay, run_episode
from neurodrive.env.policies import NeuronPolicy
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "configs" / "baselines" / "hand_neuron.yaml"
TRACK_FILE = ROOT / "tracks" / "train_loop_01.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return yaml.safe_load(BASELINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def track() -> Track:
    return Track.from_json(TRACK_FILE)


def policy_from(spec: dict) -> NeuronPolicy:
    n = spec["neuron"]
    return NeuronPolicy((n["w_left"], n["w_right"]), n["bias"], n["activation"], n["throttle"])


def test_the_baseline_file_describes_a_tanh_neuron_with_opposite_sign_weights(spec):
    assert spec["name"] == "hand_neuron" and spec["policy"] == "neuron"
    n = spec["neuron"]
    assert n["w_left"] < 0 < n["w_right"]  # steer away from the nearer wall
    assert n["activation"] == "tanh"  # sigmoid could never steer left (study S1)
    assert 0.0 < n["throttle"] <= 1.0
    assert policy_from(spec).neuron.weights.tolist() == [n["w_left"], n["w_right"]]


def test_the_baseline_completes_two_laps_from_the_start_line(spec, track):
    world = World(track, WorldConfig())  # the exam's limits: 2 laps, 90 s
    world.reset()
    rec = run_episode(world, policy_from(spec))
    assert rec.status[0] == Status.FINISHED and world.laps[0] == 2
    assert 2 * track.length <= rec.progress[-1, 0] < 2 * track.length + 5.0
    assert world.time < 90.0


def test_the_baseline_finishes_from_all_ten_exam_starts(spec, track):
    fractions = [0.0, 0.2, 0.4, 0.6, 0.8]
    world = World(track, WorldConfig())
    world.reset(
        s0=[f * track.length for _ in (0, 1) for f in fractions],
        lateral=[lat for lat in (0.0, 1.5) for _ in fractions],
    )
    rec = run_episode(world, policy_from(spec))
    assert np.all(rec.status == Status.FINISHED)


def test_a_baseline_lap_is_recorded_and_replays_identically(spec, track, tmp_path):
    world = World(track, WorldConfig())
    world.reset()
    rec = run_episode(
        world, policy_from(spec), policy_ref={"kind": "NeuronPolicy", "config": "hand_neuron"}
    )
    loaded = EpisodeRecording.load(rec.save(tmp_path / "lap.npz"))
    assert loaded.identical_to(rec)
    assert replay(loaded, track).identical_to(rec)


def test_the_study_s1_lesson_flipped_signs_and_sigmoid_do_not_finish(spec, track):
    """The two failures S1 explains: flipped weights steer into the wall, sigmoid cannot go left."""
    n = spec["neuron"]
    flipped = NeuronPolicy((-n["w_left"], -n["w_right"]), n["bias"], "tanh", n["throttle"])
    sigmoid = NeuronPolicy((n["w_left"], n["w_right"]), n["bias"], "sigmoid", n["throttle"])
    for policy in (flipped, sigmoid):
        world = World(track, WorldConfig())
        world.reset()
        rec = run_episode(world, policy)
        assert rec.status[0] == Status.CRASHED and rec.progress[-1, 0] < 100.0
    assert rec.action[:, 0, 0].min() >= 0.0  # the sigmoid run never steered left, not once
