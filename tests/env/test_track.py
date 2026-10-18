"""Tests for env/track.py: geometry on a circle with known answers, and the real training track."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.env.track import Track

ROOT = Path(__file__).resolve().parents[2]
TRACK_FILE = ROOT / "tracks" / "train_loop_01.json"
RADIUS = 50.0


def circle_data(radius: float = RADIUS, width: float = 12.0, n_corners: int = 24) -> dict:
    """A counter-clockwise circular road whose first corner is (radius, 0)."""
    angles = 2 * np.pi * np.arange(n_corners) / n_corners
    corners = [[radius * np.cos(a), radius * np.sin(a)] for a in angles]
    return {
        "name": "circle",
        "version": 1,
        "split": "train",
        "width": width,
        "start": {"s0": 0.0, "direction": 1},
        "centerline": corners,
    }


@pytest.fixture(scope="module")
def circle() -> Track:
    return Track.from_dict(circle_data())


@pytest.fixture(scope="module")
def loop() -> Track:
    return Track.from_json(TRACK_FILE)


# ── hand cases on the circle (radius 50 m, width 12 m) ──────────────────────────────────────


def test_circle_length_is_two_pi_r(circle):
    assert circle.length == pytest.approx(2 * np.pi * RADIUS, rel=0.002)  # about 314.16 m


def test_circle_samples_are_about_one_metre_apart(circle):
    spacing = np.diff(np.append(circle.s, circle.length))
    assert spacing.min() > 0.99 and spacing.max() < 1.01
    assert circle.s[0] == 0.0 and np.all(np.diff(circle.s) > 0)


def test_circle_tangent_at_start_points_up(circle):
    assert circle.points[0] == pytest.approx([RADIUS, 0.0], abs=1e-9)
    assert circle.tangent[0] == pytest.approx([0.0, 1.0], abs=1e-3)
    assert circle.heading[0] == pytest.approx(np.pi / 2, abs=1e-3)
    assert circle.normal[0] == pytest.approx([-1.0, 0.0], abs=1e-3)  # left = towards the centre


def test_circle_boundaries_sit_half_a_width_either_side(circle):
    assert np.linalg.norm(circle.left, axis=1) == pytest.approx(RADIUS - 6.0, abs=0.02)
    assert np.linalg.norm(circle.right, axis=1) == pytest.approx(RADIUS + 6.0, abs=0.02)
    assert circle.left[0] == pytest.approx([44.0, 0.0], abs=0.02)
    assert circle.right[0] == pytest.approx([56.0, 0.0], abs=0.02)


def test_edge_segments_join_consecutive_edge_points_and_wrap(circle):
    assert circle.left_segments.shape == (len(circle.points), 2, 2)
    assert circle.left_segments[3, 0] == pytest.approx(circle.left[3])
    assert circle.left_segments[3, 1] == pytest.approx(circle.left[4])
    assert circle.right_segments[-1, 1] == pytest.approx(circle.right[0])  # loop is closed


def test_tangents_and_normals_are_unit_and_perpendicular(circle):
    assert np.linalg.norm(circle.tangent, axis=1) == pytest.approx(1.0, abs=1e-12)
    assert np.sum(circle.tangent * circle.normal, axis=1) == pytest.approx(0.0, abs=1e-12)


# ── projection: arc length s and lateral offset e ───────────────────────────────────────────


def test_point_on_centreline_has_zero_offset(circle):
    p = circle.project(circle.points[[0, 100, 250]])
    assert p.e == pytest.approx(0.0, abs=1e-9)
    assert p.s == pytest.approx(circle.s[[0, 100, 250]], abs=1e-9)


def test_point_three_metres_left_has_offset_plus_three(circle):
    p = circle.project([RADIUS - 3.0, 0.0])
    assert p.e == pytest.approx([3.0], abs=0.02)
    assert p.s == pytest.approx([0.0], abs=0.05)


def test_point_three_metres_right_has_offset_minus_three(circle):
    assert circle.project([RADIUS + 3.0, 0.0]).e == pytest.approx([-3.0], abs=0.02)


def test_offsets_all_round_the_circle_and_between_samples():
    track = Track.from_dict(circle_data())
    rng = make_rng(7)
    theta = rng.uniform(0.0, 2 * np.pi, size=200)  # arbitrary angles, mostly between samples
    r = RADIUS - rng.uniform(-5.0, 5.0, size=200)  # 5 m either side of the centreline
    p = track.project(np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1))
    assert p.e == pytest.approx(RADIUS - r, abs=0.02)
    expected_s = (RADIUS * theta) % track.length  # CCW: arc length grows with the angle
    wrap = np.abs(p.s - expected_s)
    assert np.minimum(wrap, track.length - wrap).max() < 0.1


def test_projection_heading_follows_the_road(circle):
    p = circle.project([0.0, RADIUS - 2.0])  # top of the circle, 2 m inside; road points to -x
    assert abs(np.cos(p.heading[0]) + 1.0) < 1e-3 and abs(np.sin(p.heading[0])) < 0.05


def test_project_accepts_single_point_and_batch(circle):
    one = circle.project([RADIUS, 1.0])
    many = circle.project(np.array([[RADIUS, 1.0], [0.0, RADIUS]]))
    assert one.s.shape == (1,) and many.s.shape == (2,)
    assert one.e[0] == pytest.approx(many.e[0])
    with pytest.raises(ValueError, match="query must have shape"):
        circle.project(np.zeros((3, 3)))


# ── the real training track ─────────────────────────────────────────────────────────────────


def test_train_loop_metadata_and_plausible_length(loop):
    assert loop.name == "train_loop_01" and loop.split == "train" and loop.width == 12.0
    assert loop.start_s0 == 0.0 and loop.start_direction == 1
    assert 500.0 < loop.length < 1200.0


def test_train_loop_samples_are_evenly_spaced_with_increasing_arc_length(loop):
    spacing = np.diff(np.append(loop.s, loop.length))
    assert spacing.min() > 0.99 and spacing.max() < 1.01
    assert np.all(np.diff(loop.s) > 0)


def test_train_loop_runs_counter_clockwise_and_bends_smoothly(loop):
    turn = np.diff(np.unwrap(np.append(loop.heading, loop.heading[0])))
    assert turn.sum() == pytest.approx(2 * np.pi, abs=1e-6)  # one full anticlockwise turn
    assert np.abs(turn).max() < np.radians(5.0)  # no sudden kinks between neighbours
    min_radius = 1.0 / (np.abs(turn) / 1.0).max()
    assert min_radius > 15.0  # far tighter than half the width (6 m) would fold the edges


def test_train_loop_edges_do_not_cross(loop):
    # Loading already checks this; a track that failed would have raised ValueError.
    assert loop.left.shape == loop.right.shape == loop.points.shape
    gap = np.linalg.norm(loop.left - loop.right, axis=1)
    assert gap == pytest.approx(12.0, abs=1e-9)


def test_loading_twice_gives_identical_arrays():
    a, b = Track.from_json(TRACK_FILE), Track.from_json(TRACK_FILE)
    for name in ("points", "s", "tangent", "left", "right"):
        assert np.array_equal(getattr(a, name), getattr(b, name))


def test_committed_track_files_match_the_generator_script():
    spec = importlib.util.spec_from_file_location(
        "make_tracks", ROOT / "scripts" / "make_tracks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for file_name, text in module.build_tracks().items():
        assert (ROOT / "tracks" / file_name).read_text(encoding="utf-8") == text


# ── loader checks ───────────────────────────────────────────────────────────────────────────


def test_bad_split_is_rejected():
    data = circle_data() | {"split": "validation"}
    with pytest.raises(ValueError, match="split must be"):
        Track.from_dict(data)


def test_missing_field_is_rejected():
    data = circle_data()
    del data["centerline"]
    with pytest.raises(ValueError, match="missing field.*centerline"):
        Track.from_dict(data)


def test_too_few_points_is_rejected():
    with pytest.raises(ValueError, match="at least 4"):
        Track.from_dict(circle_data(n_corners=3))


def test_repeated_points_are_rejected():
    data = circle_data()
    data["centerline"][5] = list(data["centerline"][4])
    with pytest.raises(ValueError, match="repeated"):
        Track.from_dict(data)


def test_unsupported_version_and_direction_are_rejected():
    with pytest.raises(ValueError, match="unsupported track version"):
        Track.from_dict(circle_data() | {"version": 2})
    with pytest.raises(ValueError, match="direction"):
        Track.from_dict(circle_data() | {"start": {"s0": 0.0, "direction": -1}})


def test_start_beyond_the_loop_is_rejected():
    with pytest.raises(ValueError, match="start s0"):
        Track.from_dict(circle_data() | {"start": {"s0": 9999.0, "direction": 1}})


def test_road_wider_than_the_bend_allows_is_rejected():
    # Half the width (30 m) exceeds the 20 m radius of the bend, so the inside edge folds over.
    with pytest.raises(ValueError, match="crosses itself"):
        Track.from_dict(circle_data(radius=20.0, width=60.0))


def test_road_whose_edges_cross_is_rejected_but_a_thin_one_is_accepted():
    # A dumbbell: two wide discs joined by a neck. The same centreline is fine as a thin road,
    # but a 12 m road is too wide for the neck, so its edges cross (this shows the edge check
    # can both pass and fail).
    corners = [[-60, -40], [-100, -20], [-100, 20], [-60, 40], [-20, 4], [20, 4]]
    corners += [[60, 40], [100, 20], [100, -20], [60, -40], [20, -4], [-20, -4]]
    data = circle_data() | {"centerline": corners}
    assert Track.from_dict(data | {"width": 0.5}).length > 0
    with pytest.raises(ValueError, match="crosses itself"):
        Track.from_dict(data | {"width": 12.0})


# ── half-size Monza, both directions ────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def monza() -> dict[str, Track]:
    return {d: Track.from_json(ROOT / "tracks" / f"monza_{d}.json") for d in ("cw", "ccw")}


def turn_per_sample(track: Track) -> np.ndarray:
    """Signed heading change from each sample to the next, radians (positive = turning left)."""
    return np.angle(np.exp(1j * np.diff(np.append(track.heading, track.heading[0]))))


def test_monza_is_half_the_real_length_in_both_directions(monza):
    for track in monza.values():
        assert track.length == pytest.approx(5793.0 / 2, rel=0.01)  # about 2.9 km
        assert track.width == 12.0
    assert monza["cw"].length == pytest.approx(monza["ccw"].length, abs=0.5)


def test_monza_cw_is_clockwise_for_training_and_ccw_is_counter_clockwise_and_held_out(monza):
    assert turn_per_sample(monza["cw"]).sum() == pytest.approx(-2 * np.pi, abs=1e-6)
    assert turn_per_sample(monza["ccw"]).sum() == pytest.approx(2 * np.pi, abs=1e-6)
    assert monza["cw"].split == "train" and monza["ccw"].split == "heldout"


def test_both_directions_share_the_start_line_and_face_opposite_ways(monza):
    cw, ccw = monza["cw"], monza["ccw"]
    assert cw.points[0] == pytest.approx(ccw.points[0], abs=1e-9)
    assert np.cos(cw.heading[0] - ccw.heading[0]) == pytest.approx(-1.0, abs=1e-3)


def test_both_directions_are_the_same_road(monza):
    # Every point of one centreline lies on the other (within a few centimetres of resampling).
    proj = monza["ccw"].project(monza["cw"].points[::25])
    assert np.abs(proj.e).max() < 0.1


def test_monza_has_every_kind_of_corner(monza):
    track = monza["cw"]
    curvature = turn_per_sample(track) / track._seg_len.mean()  # 1 / radius, signed
    radius = 1.0 / np.maximum(np.abs(curvature), 1e-12)
    # Long straights: the longest stretch with no bend tighter than 500 m is over 400 m.
    straight, longest = 0, 0
    for r in radius:
        straight = straight + 1 if r > 500 else 0
        longest = max(longest, straight)
    assert longest * track._seg_len.mean() > 400
    # Tight chicanes (radius under 20 m) and both left and right bends.
    assert radius.min() > 6.0  # still drivable: the edges never fold
    assert radius.min() < 20.0
    assert (curvature > 1 / 60).any() and (curvature < -1 / 60).any()
    # The Parabolica: one long bend that turns the car round by close to 180 degrees.
    turning = np.convolve(turn_per_sample(track), np.ones(250), mode="valid")
    assert np.degrees(np.abs(turning).max()) > 150


def test_a_car_can_be_placed_and_projected_anywhere_on_monza(monza):
    track = monza["ccw"]
    s = np.linspace(0, track.length, 40, endpoint=False)
    x, y, _ = track.point_at(s)
    assert track.project(np.stack([x, y], axis=1)).e == pytest.approx(0.0, abs=1e-6)
