"""Tests for env/car.py: hand-computed steps, limits, the grip limit, dead cars and batching."""

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.env.car import DT, CarParams, CarState, step_cars

P = CarParams()


def run(state: CarState, steer, throttle, steps: int) -> CarState:
    for _ in range(steps):
        state = step_cars(state, steer, throttle)
    return state


# ── hand-computed single steps ──────────────────────────────────────────────────────────────


def test_one_step_with_grip_limited_turn():
    # v = 10, full throttle, full right steer. v -> 10 + (8 - 1)/30 = 10.23333.
    # kappa_req = tan(0.5)/2.5 = 0.2185, kappa_max = 9/10.2333^2 = 0.0860 -> grip limits it.
    # psi = -10.2333 * 0.0860 / 30 = -0.029316.
    s = step_cars(CarState.from_pose(0.0, 0.0, 0.0, 10.0), 1.0, 1.0)
    assert s.v[0] == pytest.approx(10.0 + 7.0 / 30.0, abs=1e-9)
    assert s.psi[0] == pytest.approx(-9.0 / s.v[0] / 30.0, abs=1e-9)
    assert s.psi[0] == pytest.approx(-0.029316, abs=1e-6)
    assert s.x[0] == pytest.approx(s.v[0] * np.cos(s.psi[0]) / 30.0, abs=1e-12)
    assert s.y[0] == pytest.approx(s.v[0] * np.sin(s.psi[0]) / 30.0, abs=1e-12)


def test_one_step_without_grip_limit_uses_the_steering_curvature():
    # v = 4: kappa_max = 9/16 -> larger than kappa_req = tan(0.5)/2.5, so steering decides.
    s = step_cars(CarState.from_pose(0.0, 0.0, 0.0, 4.0), 1.0, 0.0)
    v = 4.0 * (1.0 - 0.1 * DT)
    assert s.v[0] == pytest.approx(v, abs=1e-12)
    assert s.psi[0] == pytest.approx(-v * np.tan(0.5) / 2.5 * DT, abs=1e-12)


def test_straight_driving_moves_along_the_heading():
    s = step_cars(CarState.from_pose(1.0, 2.0, np.pi / 2, 6.0), 0.0, 0.0)
    assert s.x[0] == pytest.approx(1.0, abs=1e-12)
    assert s.y[0] == pytest.approx(2.0 + s.v[0] * DT, abs=1e-12)
    assert s.psi[0] == pytest.approx(np.pi / 2)


# ── speed behaviour ─────────────────────────────────────────────────────────────────────────


def test_full_throttle_reaches_but_never_exceeds_v_max():
    s = CarState.from_pose(0.0, 0.0, 0.0, 0.0)
    speeds = []
    for _ in range(600):
        s = step_cars(s, 0.0, 1.0)
        speeds.append(s.v[0])
    assert max(speeds) == P.v_max
    assert speeds[-1] == P.v_max
    assert np.all(np.diff(speeds) >= 0)
    assert next(i for i, v in enumerate(speeds) if v >= P.v_max) * DT == pytest.approx(4.7, abs=0.1)


def test_zero_throttle_slows_by_the_drag_factor_each_step():
    s = run(CarState.from_pose(0.0, 0.0, 0.0, 20.0), 0.0, 0.0, 90)  # three seconds
    assert s.v[0] == pytest.approx(20.0 * (1.0 - P.drag * DT) ** 90, abs=1e-9)
    assert 14.0 < s.v[0] < 15.5  # close to 20 * exp(-0.3) = 14.8


def test_braking_stops_the_car_and_never_reverses_it():
    s = CarState.from_pose(0.0, 0.0, 0.0, 10.0)
    for _ in range(60):
        s = step_cars(s, 0.0, -1.0)
        assert s.v[0] >= 0.0
    assert s.v[0] == 0.0
    assert step_cars(s, 0.0, -1.0).x[0] == s.x[0]  # stopped: no creeping backwards


# ── steering ────────────────────────────────────────────────────────────────────────────────


def test_positive_steer_turns_right_and_negative_turns_left():
    start = CarState.from_pose(0.0, 0.0, 0.0, 5.0)
    assert step_cars(start, 1.0, 0.0).psi[0] < 0.0
    assert step_cars(start, -1.0, 0.0).psi[0] > 0.0
    assert step_cars(start, 0.0, 0.0).psi[0] == 0.0


def test_a_stopped_car_does_not_turn():
    assert step_cars(CarState.from_pose(0.0, 0.0, 0.3, 0.0), 1.0, 0.0).psi[0] == pytest.approx(0.3)


def test_heading_stays_wrapped():
    s = run(CarState.from_pose(0.0, 0.0, 3.1, 5.0), -1.0, 0.0, 120)
    assert np.all((s.psi > -np.pi) & (s.psi <= np.pi))


def test_grip_limit_bounds_the_turn_radius_at_every_speed():
    # Radius of the path travelled in one step is v*dt/|dpsi|; grip says it is >= v^2/a_lat_max.
    speeds = np.linspace(0.5, P.v_max, 60)
    start = CarState.from_pose(0.0, 0.0, 0.0, speeds)
    after = step_cars(start, 1.0, 0.0)
    dpsi = np.abs(after.psi - start.psi)
    radius = after.v * DT / dpsi
    assert np.all(radius >= after.v**2 / P.a_lat_max - 1e-9)
    assert np.all(radius >= P.wheelbase / np.tan(P.delta_max) - 1e-9)  # steering limit too


def test_hard_steer_at_speed_runs_wide_but_slow_steer_does_not():
    # A quarter circle at 30 m/s needs radius >= 100 m, much more than the ~4.6 m the wheel
    # alone could give; at 5 m/s the steering limit decides (radius 4.6 m, no grip limit).
    fast = run(CarState.from_pose(0.0, 0.0, 0.0, 30.0), 1.0, 0.0, 30)
    slow = run(CarState.from_pose(0.0, 0.0, 0.0, 5.0), 1.0, 0.0, 30)
    assert abs(fast.psi[0]) < 0.35  # about 0.30 rad after one second
    assert abs(slow.psi[0]) > 1.0  # more than 1 rad after one second


# ── dead cars, batching, actions, determinism ───────────────────────────────────────────────


def test_dead_cars_do_not_move():
    start = CarState.from_pose([0.0, 5.0], [0.0, 5.0], [0.0, 1.0], [10.0, 10.0]).with_alive(
        [True, False]
    )
    after = run(start, 1.0, 1.0, 20)
    assert after.x[1] == 5.0 and after.y[1] == 5.0 and after.psi[1] == 1.0 and after.v[1] == 10.0
    assert not after.alive[1] and after.alive[0]
    assert after.x[0] != 0.0  # the live car did move


def test_batch_of_three_equals_three_separate_runs():
    rng = make_rng(3)
    steps = 200
    steer = rng.uniform(-1.0, 1.0, size=(steps, 3))
    throttle = rng.uniform(-1.0, 1.0, size=(steps, 3))
    x0, y0, psi0 = rng.normal(size=3), rng.normal(size=3), rng.uniform(-3.0, 3.0, size=3)
    v0 = rng.uniform(0.0, 30.0, size=3)

    batch = CarState.from_pose(x0, y0, psi0, v0)
    singles = [CarState.from_pose(x0[i], y0[i], psi0[i], v0[i]) for i in range(3)]
    for t in range(steps):
        batch = step_cars(batch, steer[t], throttle[t])
        singles = [step_cars(c, steer[t, i], throttle[t, i]) for i, c in enumerate(singles)]
    for i, c in enumerate(singles):
        for name in ("x", "y", "psi", "v"):
            assert getattr(batch, name)[i] == getattr(c, name)[0]  # bit-for-bit equal


def test_actions_outside_the_range_are_clipped():
    start = CarState.from_pose(0.0, 0.0, 0.0, 10.0)
    a = step_cars(start, 5.0, 9.0)
    b = step_cars(start, 1.0, 1.0)
    for name in ("x", "y", "psi", "v"):
        assert getattr(a, name)[0] == getattr(b, name)[0]


def test_step_does_not_change_the_input_state():
    start = CarState.from_pose(1.0, 2.0, 0.5, 8.0)
    step_cars(start, 1.0, 1.0)
    assert (start.x[0], start.y[0], start.psi[0], start.v[0]) == (1.0, 2.0, 0.5, 8.0)


def test_same_inputs_give_identical_trajectories():
    def trajectory():
        rng = make_rng(11)
        s = CarState.from_pose(0.0, 0.0, 0.0, 5.0)
        for _ in range(100):
            s = step_cars(s, rng.uniform(-1, 1), rng.uniform(-1, 1))
        return s

    a, b = trajectory(), trajectory()
    assert np.array_equal(a.x, b.x) and np.array_equal(a.psi, b.psi) and np.array_equal(a.v, b.v)


def test_bad_inputs_are_rejected():
    start = CarState.from_pose([0.0, 1.0], 0.0, 0.0, 1.0)
    with pytest.raises(ValueError, match="steer must have shape"):
        step_cars(start, [0.0, 0.0, 0.0], 0.0)
    with pytest.raises(ValueError, match="throttle must have shape"):
        step_cars(start, 0.0, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="speed must not be negative"):
        CarState.from_pose(0.0, 0.0, 0.0, -1.0)
    with pytest.raises(ValueError, match="wheelbase must be positive"):
        CarParams(wheelbase=0.0)
