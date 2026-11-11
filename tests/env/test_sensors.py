"""Tests for env/sensors.py: hand cases, edge cases, and a slow scalar reference (Hypothesis)."""

import math
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from neurodrive.core.rng import make_rng
from neurodrive.env.sensors import (
    GRID_CELL,
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
    assert grid.longest_list < len(segments) / 10  # each ray tests well under a tenth of the walls


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


# ── per-direction lists (task 7.0b): conservative, so still exactly the full search ─────────

ALL_TRACKS = sorted((TRACK_FILE.parent).glob("*.json"))


def assert_grid_equals_full_search(grid, segments, x, y, psi, angles=RAY_ANGLES, chunk=500):
    for k in range(0, len(x), chunk):
        part = slice(k, k + chunk)
        got = ray_distances(grid, x[part], y[part], psi[part], angles)
        want = ray_distances(segments, x[part], y[part], psi[part], angles)
        assert got.tobytes() == want.tobytes()


@pytest.mark.parametrize("path", ALL_TRACKS, ids=[p.stem for p in ALL_TRACKS])
def test_every_track_grid_is_bit_identical_to_the_full_search(path):
    track = Track.from_json(path)
    segments, grid = edge_segments(track), edge_grid(track)
    assert (grid.cell, grid.bins) == (10.0, 16)
    rng = make_rng(11)
    lo, hi = segments.reshape(-1, 2).min(axis=0) - 70.0, segments.reshape(-1, 2).max(axis=0) + 70.0
    on_road = rng.integers(0, len(track.points), 2000)
    x = np.r_[rng.uniform(lo[0], hi[0], 2000), track.points[on_road, 0] + rng.uniform(-6, 6, 2000)]
    y = np.r_[rng.uniform(lo[1], hi[1], 2000), track.points[on_road, 1] + rng.uniform(-6, 6, 2000)]
    psi = rng.uniform(-10.0, 10.0, 4000)  # headings far outside (-pi, pi] too
    assert_grid_equals_full_search(grid, segments, x, y, psi)


def test_rays_exactly_on_bin_edges_and_just_beside_them():
    segments, grid = edge_segments(TRACK), edge_grid(TRACK)
    width = 2 * math.pi / grid.bins
    rng = make_rng(3)
    spots = rng.integers(0, len(TRACK.points), 40)
    edges = np.array(
        [h * width + turn for h in range(grid.bins) for turn in (0.0, 2 * math.pi, -4 * math.pi)]
    )
    phi = np.concatenate([edges, edges - 1e-12, edges + 1e-12, np.nextafter(edges, np.inf)])
    x = np.repeat(TRACK.points[spots, 0], len(phi))
    y = np.repeat(TRACK.points[spots, 1], len(phi))
    # one ray straight ahead, so every ray's direction is exactly the chosen angle
    assert_grid_equals_full_search(grid, segments, x, y, np.tile(phi, len(spots)), angles=[0.0])
    # the five real rays with the car turned so that the centre ray sits on an edge
    assert_grid_equals_full_search(grid, segments, x, y, np.tile(phi, len(spots)))


def test_cars_exactly_on_cell_edges_and_corners():
    segments, grid = edge_segments(TRACK), edge_grid(TRACK)
    (x0, y0), c = grid.origin, grid.cell
    px, py = TRACK.points[::7, 0], TRACK.points[::7, 1]
    snap_x = x0 + np.round((px - x0) / c) * c
    snap_y = y0 + np.round((py - y0) / c) * c
    x = np.r_[snap_x, snap_x, px, np.nextafter(snap_x, -np.inf)]
    y = np.r_[snap_y, py, snap_y, np.nextafter(snap_y, -np.inf)]
    psi = make_rng(5).uniform(-math.pi, math.pi, len(x))
    assert_grid_equals_full_search(grid, segments, x, y, psi)


@settings(max_examples=120, deadline=None)
@given(
    random_walls,
    poses,
    st.floats(1.0, 30.0),
    st.sampled_from([1, 4, 7, 16, 33]),
    st.floats(5.0, 60.0),
)
def test_any_cell_size_and_bin_count_gives_the_full_search(walls, cars, cell, bins, reach):
    segments = np.array([[[a, b], [c, d]] for a, b, c, d in walls]).reshape(-1, 2, 2)
    x, y, psi = (np.array(column) for column in zip(*cars, strict=True))
    grid = wall_grid(segments, reach=reach, cell=cell, bins=bins)
    got = ray_distances(grid, x / 5.0, y / 5.0, psi, max_range=reach)
    want = ray_distances(segments, x / 5.0, y / 5.0, psi, max_range=reach)
    assert got.tobytes() == want.tobytes()


def test_the_briefs_worked_example_one_cell_one_wedge_two_walls():
    # W1 behind the cell, W2 ahead of it. The grid's origin is (-70, -48), so one cell is the
    # square x 0..10, y 2..12, and a ray from (6, 2.5) at angle 0 lies in bin 0 (0 to 22.5 deg).
    w1, w2 = wall(-20, 5, -20, 6), wall(55, 2, 55, 3)
    segments = np.concatenate([w1, w2])
    grid = wall_grid(segments, cell=10.0, bins=16)
    assert np.allclose(grid.origin, [-70.0, -48.0])
    key = grid.lists(np.array([6.0]), np.array([2.5]), np.array([[0.0]]))[0, 0]
    on_list = set(grid.walls[grid.offsets[key] : grid.offsets[key + 1]].tolist())
    assert on_list == {1}  # W2 kept, W1 (index 0) left off: no ray of bin 0 can reach it
    assert ray_readings(grid, 6.0, 2.5, 0.0, angles=[0.0])[0, 0] == pytest.approx(49.0 / 50.0)
    behind = wall_grid(segments, cell=10.0, bins=1)  # without directions W1 stays on the list
    key1 = behind.lists(np.array([6.0]), np.array([2.5]), np.array([[0.0]]))[0, 0]
    assert set(behind.walls[behind.offsets[key1] : behind.offsets[key1 + 1]].tolist()) == {0, 1}


def test_directions_shorten_the_lists_and_bins_one_is_still_exact():
    segments = edge_segments(TRACK)
    plain, binned = wall_grid(segments, bins=1), edge_grid(TRACK)
    mean = lambda g: g.list_lengths[g.list_lengths > 0].mean()  # noqa: E731
    assert mean(binned) < 0.3 * mean(plain)
    rng = make_rng(9)
    spots = rng.integers(0, len(TRACK.points), 500)
    x = TRACK.points[spots, 0] + rng.uniform(-6, 6, 500)
    y = TRACK.points[spots, 1] + rng.uniform(-6, 6, 500)
    assert_grid_equals_full_search(plain, segments, x, y, rng.uniform(-math.pi, math.pi, 500))


@pytest.mark.parametrize("bins", [0, 2, 3, -1, 2.0, True])
def test_wall_grid_rejects_bin_counts_it_cannot_enclose(bins):
    with pytest.raises(ValueError, match="bins"):
        wall_grid(wall(0, 0, 1, 1), bins=bins)


def test_edge_grid_is_built_once_per_track_and_cannot_be_changed():
    grid = edge_grid(TRACK)
    assert edge_grid(Track.from_json(TRACK_FILE)) is grid  # same edges, same grid object
    assert edge_grid(MONZA) is not grid
    for array in (grid.starts, grid.vectors, grid.origin, grid.offsets, grid.walls):
        with pytest.raises(ValueError):
            array[0] = 0


def test_walls_just_inside_the_range_seen_from_cell_corners():
    # The hardest place for the lists: a wall almost R away, hit by a ray from the corner of its
    # cell that points the ray's way, at any angle inside a bin. Two anchor walls fix the grid's
    # origin at (-250, -250), so cell corners lie on multiples of 10 m. Each car sits just inside
    # one of the four cells around a corner, and its own short wall stands across its ray at a
    # distance between 49 and 50 m.
    rng = make_rng(21)
    n = 3000
    corner = rng.integers(-5, 6, (n, 2)) * GRID_CELL
    nudge = rng.choice([-1e-7, 0.0, 1e-7], (n, 2))
    car = corner + nudge
    theta = rng.uniform(0.0, 2 * math.pi, n)
    dist = rng.uniform(49.0, 50.0, n)
    direction = np.stack([np.cos(theta), np.sin(theta)], axis=1)
    across = np.stack([-direction[:, 1], direction[:, 0]], axis=1) * 0.05
    hit = car + dist[:, None] * direction
    walls = np.stack([hit - across, hit + across], axis=1)
    anchors = np.array([[[-200.0, -200.0], [-200.0, -199.0]], [[200.0, 200.0], [200.0, 199.0]]])
    segments = np.concatenate([walls, anchors])
    grid = wall_grid(segments)
    assert np.allclose(grid.origin, [-250.0, -250.0])
    got = np.concatenate(
        [
            ray_distances(grid, car[i : i + 1, 0], car[i : i + 1, 1], theta[i : i + 1], [0.0])
            for i in range(n)
        ]
    )
    want = np.concatenate(
        [
            ray_distances(segments, car[i : i + 1, 0], car[i : i + 1, 1], theta[i : i + 1], [0.0])
            for i in range(n)
        ]
    )
    assert got.tobytes() == want.tobytes()
    assert np.mean(want < RAY_RANGE) > 0.9  # nearly every ray really meets its wall
