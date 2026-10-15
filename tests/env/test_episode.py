"""Tests for the observation builder, policies, episode runner and recordings (task 1.6)."""

import math
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.env.episode import EpisodeRecording, replay, run_episode
from neurodrive.env.policies import ConstantPolicy, KeyboardPolicy, Policy
from neurodrive.env.sensors import (
    OBSERVATION_SIZE,
    build_observation,
    edge_segments,
    ray_readings,
    wrap_angle,
)
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig

TRACK_FILE = Path(__file__).resolve().parents[2] / "tracks" / "train_loop_01.json"


@pytest.fixture(scope="module")
def loop() -> Track:
    return Track.from_json(TRACK_FILE)


class RayDriver:
    """A tiny hand-made driver that sees only the observation: steer away from the nearer wall
    and back towards the road direction; hold a gentle throttle."""

    def act(self, observations):
        left, right, heading_error = observations[:, 0], observations[:, 4], observations[:, 6]
        steer = np.clip(3.0 * (right - left) + 2.0 * heading_error, -1.0, 1.0)
        return np.column_stack([steer, np.full(len(observations), 0.35)])


def wall(x0, y0, x1, y1):
    return np.array([[[x0, y0], [x1, y1]]], dtype=float)


# ── observation builder ─────────────────────────────────────────────────────────────────────


def test_wrap_angle_maps_into_minus_pi_exclusive_to_pi_inclusive():
    angles = [0.0, math.pi, -math.pi, 1.5 * math.pi, -1.5 * math.pi, 7.0]
    expected = [0.0, math.pi, math.pi, -0.5 * math.pi, 0.5 * math.pi, 7.0 - 2 * math.pi]
    assert wrap_angle(angles) == pytest.approx(expected)


def test_observation_hand_example():
    seg = wall(10, -20, 10, 20)
    # Car at the origin facing +x at 15 m/s; the road points 90 degrees to the right of the car,
    # so the car points 90 degrees left of the road: o7 = (pi/2) / pi = 0.5.
    obs = build_observation(seg, 0.0, 0.0, 0.0, 15.0, -math.pi / 2, 30.0)
    assert obs.shape == (1, OBSERVATION_SIZE)
    assert obs[0, :5] == pytest.approx(ray_readings(seg, 0.0, 0.0, 0.0)[0], abs=1e-12)
    assert obs[0, 2] == pytest.approx(0.2)  # centre ray: wall 10 m ahead
    assert obs[0, 5] == pytest.approx(0.5)  # 15 / 30
    assert obs[0, 6] == pytest.approx(0.5)


def test_heading_error_sign_and_wrap():
    seg = wall(10, -20, 10, 20)

    def o7(psi, road):
        return build_observation(seg, 0.0, 0.0, psi, 0.0, road, 30.0)[0, 6]

    assert o7(0.3, 0.0) > 0.0  # pointing left of the road
    assert o7(-0.3, 0.0) < 0.0
    assert o7(1.5 * math.pi, 0.0) == pytest.approx(-0.5)  # 270 deg left = 90 deg right
    assert o7(math.pi, 0.0) == pytest.approx(1.0)  # exactly backwards reads +1, never -1
    assert o7(0.1 + 2 * math.pi, 2 * math.pi) == pytest.approx(0.1 / math.pi)


def test_observation_values_stay_in_the_documented_ranges(loop):
    world = World(loop)
    world.reset(n=60, rng=make_rng(2), random_s0=True, lateral_jitter=4.0, heading_jitter=3.0)
    cars = world.cars
    obs = build_observation(
        edge_segments(loop), cars.x, cars.y, cars.psi, cars.v, world.road_heading, 30.0
    )
    assert obs.shape == (60, 7)
    assert np.all((obs[:, :6] >= 0.0) & (obs[:, :6] <= 1.0))
    assert np.all((obs[:, 6] > -1.0) & (obs[:, 6] <= 1.0))


# ── policies ────────────────────────────────────────────────────────────────────────────────


def test_policies_follow_the_protocol_and_return_one_action_per_car():
    obs = np.zeros((4, 7))
    for policy in (ConstantPolicy(0.2, 0.7), KeyboardPolicy(), RayDriver()):
        assert isinstance(policy, Policy)
        assert policy.act(obs).shape == (4, 2)
    assert ConstantPolicy(0.2, 0.7).act(obs) == pytest.approx(np.tile([0.2, 0.7], (4, 1)))


def test_constant_policy_clips_and_validates():
    assert ConstantPolicy(5.0, -9.0).act(np.zeros((2, 7))).tolist() == [[1.0, -1.0], [1.0, -1.0]]
    with pytest.raises(ValueError, match="finite"):
        ConstantPolicy(float("nan"), 0.0)
    with pytest.raises(ValueError, match="shape"):
        ConstantPolicy().act(np.zeros((3, 5)))


def test_keyboard_policy_reports_the_last_input_given():
    keys = KeyboardPolicy()
    assert keys.act(np.zeros((2, 7))).tolist() == [[0.0, 0.0], [0.0, 0.0]]
    keys.set_input(-1.0, 1.0)  # left + accelerate
    assert keys.act(np.zeros((2, 7))).tolist() == [[-1.0, 1.0], [-1.0, 1.0]]
    keys.set_input(3.0, 0.5)
    assert keys.act(np.zeros((1, 7))).tolist() == [[1.0, 0.5]]


# ── episode runner ──────────────────────────────────────────────────────────────────────────


def test_doing_nothing_ends_in_a_stall_with_a_consistent_recording(loop):
    world = World(loop)
    world.reset()
    rec = run_episode(world, ConstantPolicy(0.0, 0.0), seed=5)
    assert rec.n_steps == 60 and rec.n_cars == 1  # stalls after the 2 s grace period
    assert rec.x.shape == (61, 1) and rec.rays.shape == (61, 1, 5)
    assert rec.observation.shape == (61, 1, 7) and rec.action.shape == (60, 1, 2)
    assert rec.status.tolist() == [Status.STALLED]
    assert rec.alive[0, 0] and not rec.alive[-1, 0]
    assert np.all(rec.action == 0.0)
    assert rec.x[0, 0] == pytest.approx(0.0) and rec.progress[0, 0] == 0.0
    assert rec.meta["track"] == "train_loop_01" and rec.meta["seed"] == 5
    assert rec.meta["dt"] == pytest.approx(1 / 30)
    assert rec.meta["policy"] == {"kind": "ConstantPolicy"}
    assert (rec.meta["physics_version"], rec.meta["observation_version"]) == (1, 1)


def test_recorded_rays_and_observation_agree_with_the_recorded_pose(loop):
    world = World(loop, WorldConfig(time_limit=10.0))
    world.reset(speed=5.0)
    rec = run_episode(world, RayDriver())
    segments = edge_segments(loop)
    for t in (0, 17, rec.n_steps):
        rays = ray_readings(segments, rec.x[t], rec.y[t], rec.psi[t])
        assert rec.rays[t] == pytest.approx(rays, abs=1e-12)
        assert rec.observation[t, :, :5] == pytest.approx(rays, abs=1e-12)
        assert rec.observation[t, :, 5] == pytest.approx(rec.v[t] / 30.0, abs=1e-12)
        road = loop.project(np.stack([rec.x[t], rec.y[t]], axis=1)).heading
        heading_error = wrap_angle(rec.psi[t] - road) / np.pi
        assert rec.observation[t, :, 6] == pytest.approx(heading_error, abs=1e-6)


def test_a_hand_driver_that_sees_only_rays_makes_progress(loop):
    world = World(loop, WorldConfig(time_limit=20.0))
    world.reset(speed=4.0)
    rec = run_episode(world, RayDriver())
    assert rec.progress[-1, 0] > 30.0
    assert np.all((rec.observation[:, :, :6] >= 0.0) & (rec.observation[:, :, :6] <= 1.0))
    assert np.all(np.abs(rec.observation[:, :, 6]) <= 1.0)
    assert np.all(np.abs(rec.action) <= 1.0)


def test_max_steps_stops_the_run_early(loop):
    world = World(loop)
    world.reset(speed=5.0)
    rec = run_episode(world, RayDriver(), max_steps=12)
    assert rec.n_steps == 12 and rec.x.shape[0] == 13 and rec.alive[-1, 0]


def test_runner_rejects_a_world_that_is_not_freshly_reset(loop):
    world = World(loop)
    with pytest.raises(ValueError, match="just been reset"):
        run_episode(world, ConstantPolicy())
    world.reset(speed=5.0)
    world.step(0.0, 0.0)
    with pytest.raises(ValueError, match="just been reset"):
        run_episode(world, ConstantPolicy())


def test_runner_rejects_actions_of_the_wrong_shape(loop):
    class Broken:
        def act(self, observations):
            return np.zeros((len(observations) + 1, 2))

    world = World(loop)
    world.reset(speed=5.0)
    with pytest.raises(ValueError, match="actions of shape"):
        run_episode(world, Broken())


def test_same_start_and_policy_give_identical_recordings(loop):
    def run():
        world = World(loop, WorldConfig(time_limit=8.0))
        world.reset(n=3, rng=make_rng(4), random_s0=True, lateral_jitter=1.0, speed=5.0)
        return run_episode(world, RayDriver(), seed=4)

    assert run().identical_to(run())


# ── saving, loading, replaying ──────────────────────────────────────────────────────────────


def multi_car_recording(loop: Track) -> EpisodeRecording:
    world = World(loop, WorldConfig(time_limit=15.0))
    world.reset(
        n=4, rng=make_rng(8), random_s0=True, lateral_jitter=1.5, heading_jitter=0.1, speed=6.0
    )
    return run_episode(world, RayDriver(), seed=8, policy_ref={"kind": "RayDriver", "gain": 3.0})


def test_a_saved_recording_reloads_identically(loop, tmp_path):
    rec = multi_car_recording(loop)
    path = rec.save(tmp_path / "episode")
    assert path.suffix == ".npz" and path.exists()
    loaded = EpisodeRecording.load(path)
    assert loaded.identical_to(rec)
    assert loaded.meta["policy"] == {"kind": "RayDriver", "gain": 3.0}
    assert loaded.alive.dtype == bool and loaded.status.dtype == np.int8
    assert loaded.x.dtype == np.float64


def test_replaying_recorded_actions_reproduces_the_trajectory_exactly(loop):
    rec = multi_car_recording(loop)
    assert rec.n_steps > 30
    again = replay(rec, loop)
    assert again.identical_to(rec)  # every position, ray, observation and status, bit for bit


def test_replay_works_after_a_save_and_load_round_trip(loop, tmp_path):
    rec = multi_car_recording(loop)
    loaded = EpisodeRecording.load(rec.save(tmp_path / "round_trip.npz"))
    assert replay(loaded, loop).identical_to(rec)


def test_replay_notices_a_changed_action(loop):
    rec = multi_car_recording(loop)
    original_x = rec.x.copy()
    rec.action[5, 0, 0] = -rec.action[5, 0, 0] + 0.5  # tamper with one recorded steer value
    assert not np.array_equal(replay(rec, loop).x, original_x)


def test_replay_rejects_the_wrong_track(loop):
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    other = Track.from_dict(
        {
            "name": "other",
            "version": 1,
            "split": "train",
            "width": 12,
            "start": {"s0": 0.0, "direction": 1},
            "centerline": [[50 * math.cos(a), 50 * math.sin(a)] for a in angles],
        }
    )
    with pytest.raises(ValueError, match="made on track"):
        replay(multi_car_recording(loop), other)


def test_loading_rejects_an_unknown_recording_version(loop, tmp_path):
    rec = multi_car_recording(loop)
    rec.meta["recording_version"] = 99
    path = rec.save(tmp_path / "future.npz")
    with pytest.raises(ValueError, match="unsupported recording version"):
        EpisodeRecording.load(path)


def test_the_start_is_recorded_so_random_starts_can_be_recreated(loop):
    rec = multi_car_recording(loop)
    world = World(loop)
    world.reset(
        s0=rec.start_s0,
        lateral=rec.start_lateral,
        heading_offset=rec.start_heading_offset,
        speed=rec.start_speed,
    )
    assert np.array_equal(world.cars.x, rec.x[0]) and np.array_equal(world.cars.psi, rec.psi[0])
