"""Tests for env/experts.py: the worked examples by hand, the speed profile's promises, and the
acceptance test of task 5.3: the expert laps every track from every exam start."""

from pathlib import Path

import numpy as np
import pytest

from neurodrive.env.car import DEFAULT_PARAMS
from neurodrive.env.episode import run_episode
from neurodrive.env.experts import ExpertParams, ScriptedExpert, pursuit_steer, speed_profile
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig

TRACKS_DIR = Path(__file__).resolve().parents[2] / "tracks"
ALL_TRACKS = sorted(p.stem for p in TRACKS_DIR.glob("*.json"))
EXAM_FRACTIONS = np.array([0.0, 0.2, 0.4, 0.6, 0.8])


def load(name: str) -> Track:
    return Track.from_json(TRACKS_DIR / f"{name}.json")


def circle(radius: float = 50.0) -> Track:
    angles = 2 * np.pi * np.arange(24) / 24
    return Track("circle", "train", [[radius * np.cos(a), radius * np.sin(a)] for a in angles])


# ── the worked examples ─────────────────────────────────────────────────────────────────────


def test_pure_pursuit_worked_example():
    # Target (10, 5) from (0, 0) heading east: kappa = 2 sin(26.57 deg) / 11.18 = 0.08 /m,
    # wheel angle atan(0.08 * 2.5) = 0.1974 rad, steer = -0.1974 / 0.5 = -0.395 (left).
    assert pursuit_steer(0.0, 0.0, 0.0, 10.0, 5.0) == pytest.approx(-np.arctan(0.2) / 0.5)
    assert pursuit_steer(0.0, 0.0, 0.0, 10.0, 5.0) == pytest.approx(-0.3948, abs=1e-4)
    assert pursuit_steer(0.0, 0.0, 0.0, 10.0, -5.0) == pytest.approx(0.3948, abs=1e-4)  # right
    assert pursuit_steer(0.0, 0.0, 0.0, 10.0, 0.0) == 0.0  # straight ahead
    # The same geometry turned by 90 degrees (heading north, target up and to the left).
    assert pursuit_steer(0.0, 0.0, np.pi / 2, -5.0, 10.0) == pytest.approx(-0.3948, abs=1e-4)
    # A target far to the side asks for more than full lock: clipped to -1.
    assert pursuit_steer(0.0, 0.0, 0.0, 1.0, 5.0) == -1.0


def test_speed_profile_and_speed_control_on_a_circle():
    track = circle(50.0)
    v_ref = speed_profile(track)
    # A circle has no braking zones: sqrt(0.8 * 9 * 50) = 18.97 m/s all the way round (the
    # resampled circle bends a fraction of a percent unevenly).
    assert v_ref == pytest.approx(np.sqrt(0.8 * 9 * 50), rel=0.01)
    world = World(track)
    world.reset(s0=[0.0, 0.0], speed=v_ref[0] + np.array([-2.0, 1.0]))
    throttle = ScriptedExpert(world).label()[:, 1]
    assert throttle == pytest.approx([1.0, -0.5])  # 2 m/s slow: full; 1 m/s fast: half brake


def test_hairpin_worked_example_speed_and_braking_distance():
    track = load("hairpins_01")
    v_ref = speed_profile(track)
    # The 13 m hairpins allow sqrt(0.8 * 9 * 13) = 9.67 m/s; the 9 m average of the curvature
    # makes the measured radius a little smaller or larger.
    assert 9.0 < v_ref.min() < 10.5
    # Braking at 0.7 * 16 = 11.2 m/s^2 from 30 m/s to 9.67 m/s takes (900 - 93.6) / 22.4 = 36 m.
    # Inside the braking zone v^2 falls by exactly 2 * 11.2 per metre. The zone ends a few metres
    # before the slowest point, because the averaged curvature tightens gradually into the
    # hairpin, so braking starts 39 m before the slowest point, not 36 m.
    slow = int(np.argmin(v_ref[:400]))  # the first hairpin, about 300 m in
    start = slow - int(np.argmax(v_ref[slow::-1] >= DEFAULT_PARAMS.v_max - 1e-9))
    # From the first sample below top speed (the braking curve meets the 30 m/s cap between two
    # samples), 30 m of pure braking.
    first = start + 1
    ds = np.diff(track.s[first : first + 31])
    drop = v_ref[first : first + 30] ** 2 - v_ref[first + 1 : first + 31] ** 2
    assert drop == pytest.approx(2 * 0.7 * 16 * ds, rel=1e-9)
    assert 36.0 <= track.s[slow] - track.s[start] <= 42.0


@pytest.mark.parametrize("name", ALL_TRACKS)
def test_speed_profile_brakes_no_harder_than_planned(name):
    track = load(name)
    v = speed_profile(track)
    ds = np.diff(np.append(track.s, track.length))
    assert np.all((v > 0) & (v <= DEFAULT_PARAMS.v_max))
    # Going from each sample to the next never needs more than 11.2 m/s^2 of braking.
    assert np.all(v**2 <= np.roll(v, -1) ** 2 + 2 * 0.7 * 16 * ds + 1e-9)


def test_settings_are_checked():
    for bad in ({"grip_share": 1.5}, {"brake_share": 0.0}, {"curvature_window": 4}):
        with pytest.raises(ValueError):
            ExpertParams(**bad)


# ── acceptance: every track, every exam start ───────────────────────────────────────────────


def exam_drive(track: Track) -> tuple[World, np.ndarray, float]:
    """The 10 exam starts at once, driven by the expert. Returns the world, each car's finish
    time (seconds; inf if it did not finish) and the largest distance any car strayed from the
    centreline while running."""
    world = World(track, WorldConfig.for_track(track))
    s0 = np.concatenate([EXAM_FRACTIONS, EXAM_FRACTIONS]) * track.length
    world.reset(s0=s0, lateral=np.repeat([0.0, 1.5], 5))
    expert = ScriptedExpert(world)
    finish = np.full(world.n, np.inf)
    widest = 0.0
    while not world.done:
        running = world.cars.alive
        action = expert.label()
        world.step(action[:, 0], action[:, 1])
        widest = max(widest, float(np.abs(world.lateral_offset[running]).max()))
        finish[running & (world.status == Status.FINISHED)] = world.time
    return world, finish, widest


@pytest.mark.parametrize("name", ALL_TRACKS)
def test_the_expert_laps_every_track_from_every_exam_start(name):
    track = load(name)
    world, finish, widest = exam_drive(track)
    assert np.all(world.status == Status.FINISHED), [Status(s).name for s in world.status]
    assert widest < 3.5  # a crash is at 5 m; the prototype stayed within 2.3 m
    assert finish.max() <= 0.8 * world.config.time_limit


# ── as a policy ─────────────────────────────────────────────────────────────────────────────


def test_as_a_policy_it_is_deterministic_and_ignores_the_observations():
    track = load("train_loop_01")
    recordings = []
    for _ in range(2):
        world = World(track, WorldConfig.for_track(track))
        world.reset()
        recordings.append(run_episode(world, ScriptedExpert(world)))
    assert recordings[0].identical_to(recordings[1])
    assert recordings[0].status[0] == Status.FINISHED

    world = World(track)
    world.reset(s0=[0.0, 100.0])
    expert = ScriptedExpert(world)
    real = recordings[0].observation[0, :1].repeat(2, axis=0)
    assert np.array_equal(expert.act(np.zeros((2, 7))), expert.act(real))
    assert np.array_equal(expert.act(np.zeros((2, 7))), expert.label())
    with pytest.raises(ValueError):
        expert.act(np.zeros((3, 7)))
