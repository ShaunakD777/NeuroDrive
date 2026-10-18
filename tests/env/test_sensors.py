"""Tests for env/sensors.py: hand cases, edge cases, and a slow scalar reference (Hypothesis)."""

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from neurodrive.core.rng import make_rng
from neurodrive.env.sensors import (
    RAY_ANGLES,
    RAY_RANGE,
    edge_grid,
    edge_segments,
    ray_distances,
    ray_readings,
    wall_grid,
)
from neurodrive.env.track import Track

TRACK_FILE = Path(__file__).resolve().parents[2] / "tracks" / "train_loop_01.json"


def wall(x0, y0, x1, y1) -> np.ndarray:
    """One wall segment as an array of shape (1, 2, 2)."""
    return np.array([[[x0, y0], [x1, y1]]], dtype=float)


# ── slow, simple reference: one ray, one segment at a time, plain Python ────────────────────


def reference_distance(segments, x, y, psi, rel_angle, max_range=RAY_RANGE) -> float:
    """Scalar version of the ray-segment maths, written independently for comparison."""
    angle = psi + rel_angle
    dx, dy = math.cos(angle), math.sin(angle)
    best = math.inf
    for (ax, ay), (bx, by) in segments:
        ex, ey = bx - ax, by - ay
        denom = dx * ey - dy * ex
        if abs(denom) < 1e-12:
            continue  # parallel
        wx, wy = ax - x, ay - y
        t = (wx * ey - wy * ex) / denom
        u = (wx * dy - wy * dx) / denom
        if t >= 0.0 and -1e-9 <= u <= 1.0 + 1e-9:
            best = min(best, t)
    return min(best, max_range)


def reference_readings(segments, x, y, psi) -> list[float]:
    return [reference_distance(segments, x, y, psi, a) / RAY_RANGE for a in RAY_ANGLES]


# ── hand cases ──────────────────────────────────────────────────────────────────────────────


def test_wall_ten_metres_ahead_reads_point_two_on_the_centre_ray():
    r = ray_readings(wall(10, -20, 10, 20), 0.0, 0.0, 0.0)
    assert r.shape == (1, 5)
    assert r[0, 2] == pytest.approx(10.0 / 50.0, abs=1e-12)  # centre ray: 10/50 = 0.2


def test_side_rays_travel_farther_to_the_same_wall():
    # Wall x = 10, y in [-20, 20]. Ray at angle a hits at distance 10 / cos(a).
    r = ray_readings(wall(10, -20, 10, 20), 0.0, 0.0, 0.0)[0]
    expected = [20.0 / 50.0, 10 / math.cos(math.radians(30)) / 50.0, 0.2]
    assert r[:3] == pytest.approx(expected, abs=1e-12)  # +60: 20 m -> 0.4; +30: 11.547 m
    assert r == pytest.approx(r[::-1], abs=1e-12)  # a symmetric wall gives symmetric readings


def test_no_hit_reads_one():
    assert np.all(ray_readings(wall(-10, -20, -10, 20), 0.0, 0.0, 0.0) == 1.0)  # wall behind


def test_hits_beyond_the_range_read_one_and_exactly_at_range_reads_one():
    assert ray_readings(wall(80, -90, 80, 90), 0.0, 0.0, 0.0)[0, 2] == 1.0
    assert ray_readings(wall(50, -90, 50, 90), 0.0, 0.0, 0.0)[0, 2] == 1.0
    assert ray_readings(wall(49, -90, 49, 90), 0.0, 0.0, 0.0)[0, 2] == pytest.approx(0.98)


def test_heading_rotates_the_rays():
    # Wall along y = 10; a car facing +y (psi = 90 degrees) sees it 10 m ahead.
    r = ray_readings(wall(-30, 10, 30, 10), 0.0, 0.0, math.pi / 2)[0]
    assert r[2] == pytest.approx(0.2, abs=1e-12)
    # Far-left ray (+60 degrees from +y) points towards -x and still hits: 10 / cos(60) = 20 m.
    assert r[0] == pytest.approx(0.4, abs=1e-12)


def test_position_matters_not_just_heading():
    r = ray_readings(wall(10, -20, 10, 20), 4.0, 0.0, 0.0)[0]
    assert r[2] == pytest.approx(6.0 / 50.0, abs=1e-12)


def test_ray_parallel_to_a_wall_does_not_hit_it():
    # Wall along y = 5 from x = 0 to 20; the centre ray travels along y = 0, parallel to it.
    r = ray_readings(wall(0, 5, 20, 5), 0.0, 0.0, 0.0)[0]
    assert r[2] == 1.0
    # Collinear: the ray runs along the wall itself. Still treated as a miss, never a crash.
    assert ray_readings(wall(5, 0, 20, 0), 0.0, 0.0, 0.0)[0, 2] == 1.0


def test_ray_through_the_shared_corner_of_two_segments_is_not_lost():
    # Two walls meeting at (10, 0), forming a "<" shape; the centre ray points at the corner.
    walls = np.array([[[10, 0], [20, 10]], [[10, 0], [20, -10]]], dtype=float)
    assert ray_readings(walls, 0.0, 0.0, 0.0)[0, 2] == pytest.approx(0.2, abs=1e-9)
    chain = np.array([[[10, -10], [10, 0]], [[10, 0], [10, 10]]], dtype=float)  # ends at (10, 0)
    assert ray_readings(chain, 0.0, 0.0, 0.0)[0, 2] == pytest.approx(0.2, abs=1e-9)


def test_nearest_of_several_walls_wins_and_a_car_on_a_wall_reads_zero():
    walls = np.concatenate([wall(30, -9, 30, 9), wall(12, -9, 12, 9), wall(20, -9, 20, 9)])
    assert ray_readings(walls, 0.0, 0.0, 0.0)[0, 2] == pytest.approx(12.0 / 50.0)
    assert ray_readings(wall(0, -5, 0, 5), 0.0, 0.0, 0.0)[0, 2] == 0.0


def test_empty_wall_list_reads_one():
    assert np.all(ray_readings(np.zeros((0, 2, 2)), [0.0, 1.0], 0.0, 0.0) == 1.0)


def test_batch_shapes_and_scalar_broadcast():
    r = ray_readings(wall(10, -20, 10, 20), [0.0, 1.0, 2.0], 0.0, 0.0)
    assert r.shape == (3, 5)
    assert r[:, 2] == pytest.approx([0.2, 0.18, 0.16])
    with pytest.raises(ValueError, match="segments must have shape"):
        ray_readings(np.zeros((3, 2)), 0.0, 0.0, 0.0)


def test_distances_are_metres_and_readings_are_distances_over_range():
    segs = wall(10, -20, 10, 20)
    assert ray_distances(segs, 0.0, 0.0, 0.0)[0, 2] == pytest.approx(10.0)
    assert ray_readings(segs, 0.0, 0.0, 0.0) == pytest.approx(
        ray_distances(segs, 0.0, 0.0, 0.0) / 50.0
    )


# ── on a real road ──────────────────────────────────────────────────────────────────────────


def circle_track() -> Track:
    angles = 2 * np.pi * np.arange(24) / 24
    corners = [[50 * np.cos(a), 50 * np.sin(a)] for a in angles]
    return Track.from_dict(
        {
            "name": "circle",
            "version": 1,
            "split": "train",
            "width": 12,
            "start": {"s0": 0.0, "direction": 1},
            "centerline": corners,
        }
    )


def test_centre_ray_on_a_circular_road_matches_the_geometry():
    # Car on the centreline at (50, 0) heading +y. The outer wall has radius 56 m, so the centre
    # ray hits it where 50^2 + t^2 = 56^2, i.e. t = sqrt(636) = 25.219 m. The inner wall (radius
    # 44 m) is never reached by the line x = 50.
    r = ray_readings(edge_segments(circle_track()), 50.0, 0.0, math.pi / 2)[0]
    assert r[2] == pytest.approx(math.sqrt(636.0) / 50.0, abs=0.005)
    assert np.all((r >= 0.0) & (r <= 1.0))


def test_readings_shrink_as_the_car_drives_towards_a_wall():
    segs = edge_segments(circle_track())
    # Drive outwards from the centreline towards the outer wall (heading 0 = +x, wall at x = 56).
    centre = [ray_readings(segs, x, 0.0, 0.0)[0, 2] for x in (50.0, 52.0, 54.0, 55.5)]
    assert centre == sorted(centre, reverse=True)
    assert centre[-1] == pytest.approx(0.5 / 50.0, abs=0.003)


def test_all_readings_on_the_real_track_are_in_range():
    track = Track.from_json(TRACK_FILE)
    segs = edge_segments(track)
    r = ray_readings(segs, track.points[::20, 0], track.points[::20, 1], track.heading[::20])
    assert r.shape == (len(track.points[::20]), 5)
    assert np.all((r >= 0.0) & (r <= 1.0))
    # On the centreline of a 12 m road the +-60 degree rays reach a wall in about 6/sin(60) m.
    assert r[:, [0, 4]].max() < 0.5


# ── batched result equals the scalar reference (Hypothesis) ─────────────────────────────────

TRACK = Track.from_json(TRACK_FILE)
TRACK_SEGMENTS = edge_segments(TRACK)
TRACK_SEGMENT_LIST = TRACK_SEGMENTS.tolist()

poses = st.lists(
    st.tuples(
        st.floats(-130.0, 230.0),
        st.floats(-40.0, 190.0),
        st.floats(-math.pi, math.pi),
    ),
    min_size=1,
    max_size=5,
)


@settings(max_examples=60, deadline=None)
@given(poses)
def test_batched_readings_match_the_scalar_reference_on_the_track(cars):
    x, y, psi = (np.array(column) for column in zip(*cars, strict=True))
    batched = ray_readings(TRACK_SEGMENTS, x, y, psi)
    for i, (xi, yi, pi_) in enumerate(cars):
        assert batched[i] == pytest.approx(
            reference_readings(TRACK_SEGMENT_LIST, xi, yi, pi_), abs=1e-9
        )


coordinate = st.floats(-30.0, 30.0)
random_walls = st.lists(st.tuples(coordinate, coordinate, coordinate, coordinate), max_size=8)


@settings(max_examples=150, deadline=None)
@given(random_walls, poses)
def test_batched_readings_match_the_scalar_reference_on_random_walls(walls, cars):
    segments = np.array([[[a, b], [c, d]] for a, b, c, d in walls]).reshape(-1, 2, 2)
    x, y, psi = (np.array(column) for column in zip(*cars, strict=True))
    batched = ray_readings(segments, x / 5.0, y / 5.0, psi)  # keep cars near the random walls
    for i, (xi, yi, pi_) in enumerate(cars):
        assert batched[i] == pytest.approx(
            reference_readings(segments.tolist(), xi / 5.0, yi / 5.0, pi_), abs=1e-9
        )


def test_same_poses_give_identical_readings():
    a = ray_readings(TRACK_SEGMENTS, [0.0, 50.0], [0.0, 3.0], [0.1, 2.0])
    b = ray_readings(TRACK_SEGMENTS, [0.0, 50.0], [0.0, 3.0], [0.1, 2.0])
    assert np.array_equal(a, b)


# ── the wall grid (only nearby walls) gives exactly the full search's readings ──────────────

MONZA = Track.from_json(TRACK_FILE.parent / "monza_cw.json")


@pytest.mark.parametrize("track", [TRACK, MONZA], ids=["train_loop_01", "monza_cw"])
def test_wall_grid_readings_are_bit_identical_to_the_full_search(track):
    segments = edge_segments(track)
    grid = edge_grid(track)
    rng = make_rng(7)
    lo, hi = segments.reshape(-1, 2).min(axis=0) - 80.0, segments.reshape(-1, 2).max(axis=0) + 80.0
    # Cars anywhere round the track (many far off the road or off the grid) and cars on the road.
    on_road = rng.integers(0, len(track.points), 400)
    x = np.r_[rng.uniform(lo[0], hi[0], 600), track.points[on_road, 0] + rng.uniform(-6, 6, 400)]
    y = np.r_[rng.uniform(lo[1], hi[1], 600), track.points[on_road, 1] + rng.uniform(-6, 6, 400)]
    psi = rng.uniform(-math.pi, math.pi, 1000)
    for k in range(0, 1000, 250):
        part = slice(k, k + 250)
        assert np.array_equal(
            ray_distances(grid, x[part], y[part], psi[part]),
            ray_distances(segments, x[part], y[part], psi[part]),
        )
    assert grid.n_walls == len(segments)
    assert grid.table.shape[1] < len(segments) / 3  # each car tests well under a third of the walls


def test_wall_grid_handles_cars_on_cell_edges_and_outside_the_grid():
    segments = wall(10, -20, 10, 20)
    grid = wall_grid(segments, cell=5.0)
    # A car exactly on a cell corner, cars on the grid's outer edge, and cars far away.
    x0, y0 = grid.origin
    xs = [x0 + 5.0 * 3, x0, x0 + grid.cell * grid.nx, -1000.0, 1000.0, 0.0]
    ys = [y0 + 5.0 * 4, y0, y0 + grid.cell * grid.ny, 0.0, 0.0, 0.0]
    for psi in (0.0, 1.0, math.pi):
        assert np.array_equal(
            ray_distances(grid, xs, ys, psi), ray_distances(segments, xs, ys, psi)
        )
    assert ray_readings(grid, 0.0, 0.0, 0.0)[0, 2] == pytest.approx(0.2)
    assert np.all(ray_readings(grid, 1000.0, 0.0, 0.0) == 1.0)


def test_wall_grid_rejects_longer_rays_and_works_without_walls():
    grid = wall_grid(wall(10, -20, 10, 20), reach=20.0)
    with pytest.raises(ValueError, match="built for rays up to"):
        ray_distances(grid, 0.0, 0.0, 0.0)  # the default range is 50 m
    assert ray_distances(grid, 0.0, 0.0, 0.0, max_range=20.0)[0, 2] == pytest.approx(10.0)
    empty = wall_grid(np.zeros((0, 2, 2)))
    assert np.all(ray_readings(empty, [0.0, 3.0], 0.0, 0.0) == 1.0)
    with pytest.raises(ValueError, match="shape"):
        wall_grid(np.zeros((3, 2)))
