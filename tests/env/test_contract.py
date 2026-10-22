"""Contract test (task 2.5): observation v1 and action v1 (ARCHITECTURE §6.4-6.5, ADR-005).

Every learning method sees these 7 numbers and acts through these 2, so a change to their count,
order, meaning, range or sign needs an ADR and a new version. This file fails when one drifts.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.env.car import (
    ACTION_NAMES,
    ACTION_SIZE,
    ACTION_VERSION,
    DEFAULT_PARAMS,
    CarState,
    step_cars,
)
from neurodrive.env.sensors import (
    OBSERVATION_NAMES,
    OBSERVATION_SIZE,
    OBSERVATION_VERSION,
    RAY_ANGLES,
    RAY_RANGE,
    build_observation,
    edge_grid,
    observe,
)
from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig

TRACK_FILE = Path(__file__).resolve().parents[2] / "tracks" / "train_loop_01.json"
V_MAX = DEFAULT_PARAMS.v_max


@pytest.fixture(scope="module")
def loop() -> Track:
    return Track.from_json(TRACK_FILE)


def wall(x0, y0, x1, y1):
    return np.array([[[x0, y0], [x1, y1]]], dtype=float)


def obs_at(segments, x=0.0, y=0.0, psi=0.0, v=0.0, road_heading=0.0):
    return build_observation(segments, x, y, psi, v, road_heading, V_MAX)


# ── observation v1: what the numbers are ────────────────────────────────────────────────────


def test_version_names_and_size():
    assert OBSERVATION_VERSION == "v1"
    assert OBSERVATION_SIZE == 7
    assert OBSERVATION_NAMES == (
        "ray_far_left",
        "ray_left",
        "ray_centre",
        "ray_right",
        "ray_far_right",
        "speed",
        "heading_error",
    )


def test_ray_angles_and_range():
    assert np.degrees(RAY_ANGLES) == pytest.approx([60, 30, 0, -30, -60], abs=1e-12)
    assert RAY_RANGE == 50.0


def test_shape_is_n_by_7_for_one_and_many_cars():
    far = wall(1000, 1000, 1001, 1000)
    assert obs_at(far).shape == (1, 7)
    many = build_observation(far, np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4), 0.0, V_MAX)
    assert many.shape == (4, 7)


def test_ray_order_is_far_left_to_far_right():
    # A wall along the left side of a car that faces +x (left is +y), 5 m away.
    left_wall = wall(-100, 5, 100, 5)
    o = obs_at(left_wall)[0]
    assert o[0] < o[1] < o[2]  # the +60 and +30 degree rays reach the wall; o3 (ahead) does not
    assert o[2] == o[3] == o[4] == 1.0  # the rays that point right or ahead see nothing
    assert o[0] == pytest.approx(5 / math.sin(math.radians(60)) / RAY_RANGE)
    assert o[1] == pytest.approx(5 / math.sin(math.radians(30)) / RAY_RANGE)
    # the mirror image: a wall on the right fills the right-hand end instead
    o_right = obs_at(wall(-100, -5, 100, -5))[0]
    assert o_right[4] == pytest.approx(o[0]) and o_right[3] == pytest.approx(o[1])
    assert o_right[0] == o_right[1] == o_right[2] == 1.0


def test_ray_hand_values_wall_ahead_and_nothing_in_range():
    assert obs_at(wall(10, -20, 10, 20))[0, 2] == pytest.approx(0.2)  # 10 m / 50 m
    assert obs_at(wall(80, -20, 80, 20))[0, :5] == pytest.approx(np.ones(5))  # out of range -> 1


def test_speed_is_v_over_v_max():
    far = wall(1000, 1000, 1001, 1000)
    speeds = [0.0, 7.5, 15.0, V_MAX]
    got = [obs_at(far, v=v)[0, 5] for v in speeds]
    assert got == pytest.approx([0.0, 0.25, 0.5, 1.0])


def test_heading_error_sign_scale_and_wrap():
    far = wall(1000, 1000, 1001, 1000)

    def o7(psi, road):
        return obs_at(far, psi=psi, road_heading=road)[0, 6]

    assert o7(0.0, 0.0) == 0.0
    assert o7(math.pi / 2, 0.0) == pytest.approx(0.5)  # points left of the road: positive
    assert o7(-math.pi / 2, 0.0) == pytest.approx(-0.5)  # points right of the road: negative
    assert o7(0.1, -0.2) == pytest.approx(0.3 / math.pi)  # difference, not the raw heading
    # across the +-pi seam: 3.0 rad vs -3.0 rad is a gap of 0.283 rad the short way round
    assert o7(3.0, -3.0) == pytest.approx((6.0 - 2 * math.pi) / math.pi)
    # facing exactly backwards is +1 from either side: the range is (-1, 1], never -1
    assert o7(math.pi, 0.0) == pytest.approx(1.0)
    assert o7(-math.pi, 0.0) == pytest.approx(1.0)


def test_each_row_equals_the_single_car_observation():
    segs = wall(10, -20, 10, 20)
    xs, ys = np.array([0.0, 2.0, -1.0]), np.array([0.0, 3.0, 1.0])
    psis, vs, roads = np.array([0.0, 0.4, -2.0]), np.array([1.0, 12.0, 30.0]), [0.1, -0.3, 2.0]
    batch = build_observation(segs, xs, ys, psis, vs, np.array(roads), V_MAX)
    for i in range(3):
        one = build_observation(segs, xs[i], ys[i], psis[i], vs[i], roads[i], V_MAX)
        assert batch[i] == pytest.approx(one[0], abs=1e-12)


def test_ranges_hold_while_driving_with_random_actions(loop):
    # 12 cars at random spots on a real track, steered at random for 600 steps: every
    # observation must stay inside its stated range, including after crashes (frozen cars).
    rng = make_rng(0)
    world = World(loop, WorldConfig.for_track(loop))
    world.reset(n=12, rng=rng, random_s0=True, lateral_jitter=1.5, heading_jitter=0.3, speed=10.0)
    grid = edge_grid(loop)
    seen = []
    for _ in range(600):
        cars = world.cars
        seen.append(
            build_observation(
                grid, cars.x, cars.y, cars.psi, cars.v, world.road_heading, world.params.v_max
            )
        )
        world.step(rng.uniform(-1, 1, 12), rng.uniform(-1, 1, 12))
    obs = np.concatenate(seen)
    assert obs.shape[1] == 7 and np.isfinite(obs).all()
    assert obs[:, :6].min() >= 0.0 and obs[:, :6].max() <= 1.0
    assert obs[:, 6].min() > -1.0 and obs[:, 6].max() <= 1.0
    assert obs[:, :5].min() < 1.0  # the cars did see walls, so the check was not vacuous


# ── action v1: what the numbers do ──────────────────────────────────────────────────────────


def drive(steer, throttle, v0=10.0, steps=15):
    state = CarState.from_pose(0.0, 0.0, 0.0, v0)
    for _ in range(steps):
        state = step_cars(state, steer, throttle)
    return state


def test_action_is_two_numbers_per_car_steer_then_throttle():
    state = CarState.from_pose([0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [10.0, 10.0])
    after = step_cars(state, [1.0, -1.0], [0.5, 0.5])  # steer, throttle: one entry per car
    assert after.psi[0] < 0.0 < after.psi[1]
    with pytest.raises(ValueError):
        step_cars(state, [1.0, 0.0, 0.0], [0.0, 0.0])


def test_positive_steer_turns_right_negative_turns_left():
    assert drive(+1.0, 0.0).psi[0] < 0.0  # heading angle falls: clockwise, to the right
    assert drive(-1.0, 0.0).psi[0] > 0.0
    assert drive(+1.0, 0.0).y[0] < 0.0 < drive(-1.0, 0.0).y[0]  # facing +x, left is +y
    assert drive(0.0, 0.0).psi[0] == 0.0


def test_throttle_plus_one_accelerates_minus_one_brakes():
    assert drive(0.0, +1.0).v[0] > 10.0
    assert drive(0.0, -1.0).v[0] < 10.0
    assert drive(0.0, -1.0, steps=300).v[0] == 0.0  # braking stops the car; it never reverses


def test_out_of_range_actions_are_clipped_to_the_contract_range():
    for steer, throttle in [(5.0, 0.0), (-5.0, 0.0), (0.0, 9.0), (0.0, -9.0), (3.0, -3.0)]:
        clipped = drive(np.clip(steer, -1, 1), np.clip(throttle, -1, 1))
        wild = drive(steer, throttle)
        assert (wild.x, wild.y, wild.psi, wild.v) == (clipped.x, clipped.y, clipped.psi, clipped.v)


def test_action_version_names_and_size():
    assert ACTION_VERSION == "v1"
    assert ACTION_NAMES == ("steer", "throttle")
    assert ACTION_SIZE == 2


def test_observe_is_build_observation_applied_to_the_world_state(loop):
    world = World(loop, WorldConfig.for_track(loop))
    world.reset(n=5, rng=make_rng(1), random_s0=True, lateral_jitter=1.0, speed=12.0)
    grid = edge_grid(loop)
    cars = world.cars
    direct = build_observation(
        grid, cars.x, cars.y, cars.psi, cars.v, world.road_heading, world.params.v_max
    )
    np.testing.assert_array_equal(observe(world, grid), direct)
    assert observe(world, grid).shape == (5, OBSERVATION_SIZE)
