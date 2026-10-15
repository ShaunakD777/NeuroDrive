"""Tests for env/world.py (progress, crash, termination) and the windowed projection it uses."""

import math
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig

TRACK_FILE = Path(__file__).resolve().parents[2] / "tracks" / "train_loop_01.json"


@pytest.fixture(scope="module")
def loop() -> Track:
    return Track.from_json(TRACK_FILE)


def circle_track(radius: float = 50.0, width: float = 12.0) -> Track:
    angles = 2 * np.pi * np.arange(24) / 24
    return Track.from_dict(
        {
            "name": "circle",
            "version": 1,
            "split": "train",
            "width": width,
            "start": {"s0": 0.0, "direction": 1},
            "centerline": [[radius * np.cos(a), radius * np.sin(a)] for a in angles],
        }
    )


def wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def drive(world: World, speed: float = 8.0):
    """Steer back towards the centreline with simple proportional rules; hold a steady speed.

    Heading error h > 0 means the car points left of the road, and lateral offset e > 0 means it
    is left of the centreline; both call for a right turn, which is a positive steer.
    """
    h = wrap(world.cars.psi - world.road_heading)
    steer = np.clip(1.5 * h + 0.15 * world.lateral_offset, -1.0, 1.0)
    throttle = np.clip((speed - world.cars.v) * 0.5, -1.0, 1.0)
    return world.step(steer, throttle)


# ── windowed projection and point_at (Track additions used by the world) ────────────────────


def test_window_search_equals_global_search_near_the_previous_segment(loop):
    rng = make_rng(5)
    # Points within 4 m of the centreline, then "previous index" within 3 segments of the truth.
    s = rng.uniform(0.0, loop.length, size=200)
    x, y, h = loop.point_at(s)
    lateral = rng.uniform(-5.0, 5.0, size=200)
    q = np.stack([x - lateral * np.sin(h), y + lateral * np.cos(h)], axis=1)
    truth = loop.project(q)
    near = (truth.index + rng.integers(-3, 4, size=200)) % len(loop.points)
    windowed = loop.project_window(q, near)
    gap = np.abs(windowed.index - truth.index)  # a tie at a segment joint may pick the neighbour
    assert np.all(np.minimum(gap, len(loop.points) - gap) <= 1)
    assert windowed.s == pytest.approx(truth.s, abs=1e-12)
    assert windowed.e == pytest.approx(truth.e, abs=1e-12)


def test_window_search_wraps_round_the_start_of_the_loop(loop):
    n = len(loop.points)
    q = np.array([[loop.points[2, 0], loop.points[2, 1]]])
    proj = loop.project_window(q, near=n - 3)  # previous segment near the end of the list
    assert proj.index[0] in (1, 2) and proj.s[0] == pytest.approx(loop.s[2], abs=1e-9)


def test_window_search_cannot_jump_to_a_far_stretch_of_road(loop):
    # Global search picks the true nearest point; a window around a far segment cannot.
    q = loop.points[[400]]
    far = loop.project_window(q, near=100)
    assert abs(int(far.index[0]) - 400) > 100


def test_point_at_matches_the_samples_and_wraps(loop):
    x, y, h = loop.point_at(loop.s[[0, 100, 500]])
    assert np.stack([x, y], axis=1) == pytest.approx(loop.points[[0, 100, 500]], abs=1e-9)
    assert h == pytest.approx(loop.heading[[0, 100, 500]], abs=1e-9)
    x2, y2, _ = loop.point_at([loop.length + 10.0, -loop.length + 10.0])
    x3, y3, _ = loop.point_at([10.0, 10.0])
    assert np.allclose(x2, x3) and np.allclose(y2, y3)


def test_point_at_then_project_round_trips(loop):
    s = np.linspace(0.3, loop.length - 0.3, 50)
    x, y, _ = loop.point_at(s)
    proj = loop.project(np.stack([x, y], axis=1))
    assert proj.s == pytest.approx(s, abs=1e-6)
    assert proj.e == pytest.approx(0.0, abs=1e-6)


# ── reset ───────────────────────────────────────────────────────────────────────────────────


def test_reset_puts_cars_on_the_road_facing_along_it(loop):
    world = World(loop)
    world.reset(s0=[0.0, 100.0, 300.0], lateral=[0.0, 1.5, -2.0], speed=3.0)
    assert world.n == 3 and world.cars.alive.all() and world.time == 0.0
    assert world.s == pytest.approx([0.0, 100.0, 300.0], abs=0.05)
    assert world.lateral_offset == pytest.approx([0.0, 1.5, -2.0], abs=0.05)
    assert wrap(world.cars.psi - world.road_heading) == pytest.approx(0.0, abs=1e-3)
    assert world.cars.v == pytest.approx(3.0)
    assert world.progress == pytest.approx(0.0) and (world.status == Status.RUNNING).all()


def test_reset_defaults_to_the_track_start_and_broadcasts(loop):
    world = World(loop)
    world.reset(n=4)
    assert world.n == 4 and np.all(world.s == world.s[0]) and world.s[0] == pytest.approx(0.0)


# ── progress ────────────────────────────────────────────────────────────────────────────────


def test_driving_a_lap_adds_about_one_track_length_of_progress(loop):
    world = World(loop, WorldConfig(target_laps=1, time_limit=120.0))
    world.reset()
    steps = 0
    while not world.done and steps < 4000:
        drive(world, speed=10.0)
        steps += 1
    assert world.status[0] == Status.FINISHED
    assert world.laps[0] == 1
    assert loop.length <= world.progress[0] < loop.length + 1.0  # ends within one step of the line
    assert 70 < world.time < 90  # roughly 731 m at 10 m/s


def test_two_laps_with_wrap_round_the_start_line(loop):
    world = World(loop, WorldConfig(target_laps=2, time_limit=300.0))
    world.reset()
    while not world.done:
        drive(world)
    assert world.status[0] == Status.FINISHED and world.laps[0] == 2
    assert 2 * loop.length <= world.progress[0] < 2 * loop.length + 1.0


def test_progress_per_step_never_jumps_by_a_lap_when_crossing_the_start_line():
    world = World(circle_track(), WorldConfig(target_laps=3, time_limit=300.0))
    world.reset(s0=300.0, speed=8.0)  # 14 m before the start line of a 314 m loop
    biggest = 0.0
    for _ in range(300):
        biggest = max(biggest, float(np.abs(drive(world)).max()))
    assert biggest < 1.0  # a car at 8-10 m/s moves about 0.3 m per step, never ~314 m


def test_driving_backwards_reduces_progress_and_never_counts_negative_laps(loop):
    world = World(loop)
    world.reset(heading_offset=np.pi, speed=5.0, s0=200.0)  # facing against the traffic
    for _ in range(25):
        world.step(0.0, 0.0)
    assert world.progress[0] < -1.0
    assert world.laps[0] == 0


def test_forward_progress_equals_distance_along_a_straight_stretch(loop):
    world = World(loop)
    world.reset(s0=20.0, speed=10.0)
    before = world.s[0]
    for _ in range(15):  # half a second, drive steadily
        drive(world, speed=10.0)
    assert world.progress[0] == pytest.approx(world.s[0] - before, abs=1e-9)
    assert world.progress[0] == pytest.approx(5.0, abs=0.7)


# ── crash ───────────────────────────────────────────────────────────────────────────────────


def test_crash_threshold_is_half_width_minus_car_half_width(loop):
    # road half-width 6 m, car half-width 1 m -> crash beyond |e| = 5 m
    world = World(loop)
    world.reset(lateral=[4.9, 5.1, -4.9, -5.1])
    world.step(0.0, 0.0)
    assert list(world.status) == [Status.RUNNING, Status.CRASHED, Status.RUNNING, Status.CRASHED]
    assert list(world.cars.alive) == [True, False, True, False]


def test_driving_off_the_road_ends_the_episode_and_freezes_the_car(loop):
    world = World(loop)
    world.reset(speed=8.0)
    for _ in range(200):
        world.step(1.0, 0.3)  # keeps turning right on a left-curving road
        if world.done:
            break
    assert world.status[0] == Status.CRASHED
    frozen = (world.cars.x[0], world.cars.y[0], world.progress[0], world.step_count)
    world.step(0.0, 1.0)
    assert (world.cars.x[0], world.cars.y[0], world.progress[0]) == frozen[:3]
    assert world.status[0] == Status.CRASHED and world.last_ds[0] == 0.0


def test_one_crash_does_not_stop_the_other_cars(loop):
    world = World(loop)
    world.reset(lateral=[0.0, 5.5], speed=5.0)
    drive(world)
    assert list(world.running) == [True, False]
    before = world.cars.x[0]
    drive(world)
    assert world.cars.x[0] != before


# ── stall, timeout ──────────────────────────────────────────────────────────────────────────


def test_a_stopped_car_stalls_after_the_two_second_grace_period(loop):
    world = World(loop)
    world.reset()
    for k in range(1, 61):
        world.step(0.0, 0.0)
        assert (world.status[0] == Status.STALLED) == (k == 60), k  # 60 steps = 2 s
    assert not world.cars.alive[0]


def test_a_car_that_keeps_making_progress_never_stalls(loop):
    world = World(loop, WorldConfig(time_limit=20.0))
    world.reset()
    for _ in range(500):
        drive(world)
        assert world.status[0] != Status.STALLED
        if world.done:
            break


def test_a_car_that_slows_to_a_crawl_stalls_once_the_window_shows_under_two_metres(loop):
    world = World(loop, WorldConfig(time_limit=60.0))
    world.reset(speed=0.4)
    while not world.done:
        world.step(0.0, 0.0)
    assert world.status[0] == Status.STALLED and world.step_count == 60


def test_time_limit_ends_the_episode_with_timeout(loop):
    world = World(loop, WorldConfig(time_limit=5.0, target_laps=5))
    world.reset()
    while not world.done:
        drive(world)
    assert world.status[0] == Status.TIMEOUT
    assert world.step_count == 150  # 5 s at 30 steps per second


def test_status_priority_crash_beats_stall(loop):
    # A stopped car outside the crash line: crash is reported in the first step, not stall.
    world = World(loop)
    world.reset(lateral=5.5)
    world.step(0.0, 0.0)
    assert world.status[0] == Status.CRASHED


# ── determinism, batching, errors ───────────────────────────────────────────────────────────


def run_random(seed: int, track: Track, n: int = 3, steps: int = 120) -> World:
    rng = make_rng(seed)
    world = World(track, WorldConfig(time_limit=10.0))
    world.reset(
        s0=rng.uniform(0, track.length, size=n), lateral=rng.uniform(-2, 2, size=n), speed=6.0
    )
    for _ in range(steps):
        world.step(rng.uniform(-0.6, 0.6, size=n), rng.uniform(-0.2, 1.0, size=n))
    return world


def test_same_seed_gives_identical_results(loop):
    a, b = run_random(7, loop), run_random(7, loop)
    assert np.array_equal(a.cars.x, b.cars.x) and np.array_equal(a.progress, b.progress)
    assert np.array_equal(a.status, b.status)
    c = run_random(8, loop)
    assert not np.array_equal(a.cars.x, c.cars.x)  # a different seed really differs


def test_batched_world_equals_separate_single_car_worlds(loop):
    rng = make_rng(21)
    n, steps = 3, 150
    s0 = rng.uniform(0, loop.length, size=n)
    lateral = rng.uniform(-2, 2, size=n)
    steer = rng.uniform(-0.5, 0.5, size=(steps, n))
    throttle = rng.uniform(-0.2, 1.0, size=(steps, n))

    batch = World(loop)
    batch.reset(s0=s0, lateral=lateral, speed=5.0)
    singles = []
    for i in range(n):
        w = World(loop)
        w.reset(s0=s0[i], lateral=lateral[i], speed=5.0)
        singles.append(w)
    for t in range(steps):
        batch.step(steer[t], throttle[t])
        for i, w in enumerate(singles):
            w.step(steer[t, i], throttle[t, i])
    for i, w in enumerate(singles):
        assert batch.cars.x[i] == w.cars.x[0] and batch.cars.y[i] == w.cars.y[0]
        assert batch.progress[i] == w.progress[0] and batch.status[i] == w.status[0]


def test_step_before_reset_and_bad_config_are_rejected(loop):
    with pytest.raises(RuntimeError, match="reset"):
        World(loop).step(0.0, 0.0)
    with pytest.raises(ValueError, match="target_laps"):
        WorldConfig(target_laps=0)
    assert math.isclose(World(loop).crash_offset, 5.0)
