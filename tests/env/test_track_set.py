"""Tests for the track set of task 5.1: the held-out tracks have the character they are named
for, the mirror really is the training loop reflected, and every track is drivable in time.
"""

from pathlib import Path

import numpy as np
import pytest

from neurodrive.env.car import DEFAULT_PARAMS, CarParams
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig, time_limit_for

TRACKS_DIR = Path(__file__).resolve().parents[2] / "tracks"
TRAINING = {"train_loop_01", "monza_cw"}
HELDOUT = {"hairpins_01", "sweeper_01", "train_loop_01_mirror", "monza_ccw"}
EXAM_TRACKS = ["train_loop_01", *sorted(HELDOUT)]  # Driving Exam v1 (EXPERIMENT_PROTOCOL §3)
MAX_IDEAL_RATIO = 0.75  # ideal 2-lap time / time limit: room for a driver 25 % off perfect


@pytest.fixture(scope="module")
def tracks() -> dict[str, Track]:
    return {path.stem: Track.from_json(path) for path in sorted(TRACKS_DIR.glob("*.json"))}


def turn_per_sample(track: Track) -> np.ndarray:
    """Signed heading change from each sample to the next, radians (positive = turning left)."""
    return np.angle(np.exp(1j * np.diff(np.append(track.heading, track.heading[0]))))


def bends(track: Track, straight_radius: float = 100.0) -> list[dict]:
    """Every stretch of road whose radius stays under ``straight_radius`` metres, in order.

    Each bend gives its first sample, its length (metres), its total signed turn (degrees) and
    its median radius (metres; the median ignores the noisy single samples where the 1 m
    sampling makes the radius look a little tighter than it is).
    """
    turn = turn_per_sample(track)
    radius = track._seg_len / np.maximum(np.abs(turn), 1e-12)
    curved = radius < straight_radius
    if curved.all():
        raise ValueError("no straight anywhere; cannot split the loop into bends")
    first = int(np.argmin(curved))  # start the walk on a straight, so no bend wraps round
    order = np.roll(np.arange(len(turn)), -first)
    found, run = [], []
    for i in [*order, order[0]]:
        if curved[i]:
            run.append(i)
        elif run:
            found.append(
                {
                    "start": run[0],
                    "length": float(track._seg_len[run].sum()),
                    "turn": float(np.degrees(turn[run].sum())),
                    "radius": float(np.median(radius[run])),
                    "samples": np.array(run),
                }
            )
            run = []
    return found


def ideal_lap_time(track: Track, params: CarParams = DEFAULT_PARAMS) -> float:
    """Fastest possible lap, in seconds, for a car that keeps to the centreline.

    At each 1 m sample the speed is capped by grip, v <= sqrt(a_lat / kappa) with kappa the
    curvature (1 / radius), and by the top speed. A forward pass lowers each cap to what full
    throttle can reach from the sample before, v_next^2 <= v^2 + 2 (a_acc - k v) ds; a backward
    pass lowers it to what full braking can shed before the sample after,
    v^2 <= v_next^2 + 2 (a_brk + k v_next) ds (k is drag; ds the step length). Three rounds
    let the start line's wrap-round settle. The lap time adds ds / (mean speed of the step).
    """
    ds = track._seg_len
    kappa = np.abs(turn_per_sample(track)) / ds
    v = np.minimum(params.v_max, np.sqrt(params.a_lat_max / np.maximum(kappa, 1e-12)))
    n = len(v)
    for _ in range(3):
        for i in range(n):
            j = (i + 1) % n
            gain = max(params.a_acc - params.drag * v[i], 0.0)
            v[j] = min(v[j], np.sqrt(v[i] ** 2 + 2.0 * gain * ds[i]))
        for i in range(n - 1, -1, -1):
            j = (i + 1) % n
            v[i] = min(v[i], np.sqrt(v[j] ** 2 + 2.0 * (params.a_brk + params.drag * v[j]) * ds[i]))
    return float(np.sum(ds / ((v + np.roll(v, -1)) / 2.0)))


# ── the set ─────────────────────────────────────────────────────────────────────────────────


def test_every_track_file_loads_with_the_expected_split(tracks):
    assert set(tracks) == TRAINING | HELDOUT
    assert {name for name, t in tracks.items() if t.split == "train"} == TRAINING
    assert {name for name, t in tracks.items() if t.split == "heldout"} == HELDOUT
    for name, track in tracks.items():
        assert track.name == name and track.width == 12.0 and track.start_s0 == 0.0


# ── the hand check of the helpers ───────────────────────────────────────────────────────────


def test_ideal_lap_on_a_circle_matches_the_grip_speed():
    # A 50 m circle: grip speed sqrt(9 * 50) = 21.21 m/s all the way round, so a lap of
    # 2 pi 50 = 314.16 m takes 314.16 / 21.21 = 14.81 s.
    angles = 2 * np.pi * np.arange(24) / 24
    corners = [[50 * np.cos(a), 50 * np.sin(a)] for a in angles]
    circle = Track("circle", "train", corners)
    assert ideal_lap_time(circle) == pytest.approx(2 * np.pi * 50 / np.sqrt(9 * 50), rel=0.01)


# ── each held-out track has its character ───────────────────────────────────────────────────


def test_hairpins_are_tighter_than_any_bend_of_the_training_loop(tracks):
    hairpins = [b for b in bends(tracks["hairpins_01"]) if abs(b["turn"]) >= 140.0]
    assert len(hairpins) >= 2
    assert all(11.0 <= b["radius"] <= 15.0 for b in hairpins)
    assert {np.sign(b["turn"]) for b in hairpins} == {-1.0, 1.0}  # one left, one right
    # Even the single tightest metre of the training loop (22.5 m) is wider than the hairpins.
    loop = tracks["train_loop_01"]
    loop_tightest = (loop._seg_len / np.abs(turn_per_sample(loop))).min()
    assert max(b["radius"] for b in hairpins) < loop_tightest


def test_the_sweeper_holds_one_steady_radius_for_over_200_m(tracks):
    track = tracks["sweeper_01"]
    sweeper = max(bends(track), key=lambda b: b["length"])
    assert sweeper["length"] >= 200.0
    assert 45.0 <= sweeper["radius"] <= 80.0
    # Along its middle 200 m the radius (averaged over 9 m to smooth the sampling) varies by
    # less than 10 %, so the steering a driver needs there is nearly constant.
    turn = np.convolve(np.abs(turn_per_sample(track)), np.ones(9) / 9, mode="same")
    mid = sweeper["samples"][len(sweeper["samples"]) // 2]
    radius = 1.0 / turn[mid - 100 : mid + 100]
    assert radius.max() / radius.min() < 1.10


def test_the_mirror_is_the_training_loop_reflected_left_to_right(tracks):
    loop, mirror = tracks["train_loop_01"], tracks["train_loop_01_mirror"]
    assert mirror.length == pytest.approx(loop.length, abs=1e-9)
    assert mirror.points == pytest.approx(loop.points * [-1.0, 1.0], abs=1e-9)
    # A heading h becomes pi - h: compare as unit vectors so the angle wrap does not matter.
    assert np.cos(mirror.heading) == pytest.approx(-np.cos(loop.heading), abs=1e-9)
    assert np.sin(mirror.heading) == pytest.approx(np.sin(loop.heading), abs=1e-9)
    # The same bends in the same order, each turning the other way: one full clockwise turn.
    assert turn_per_sample(mirror) == pytest.approx(-turn_per_sample(loop), abs=1e-9)
    assert turn_per_sample(mirror).sum() == pytest.approx(-2 * np.pi, abs=1e-6)


def test_monza_ccw_is_the_long_held_out_track(tracks):
    lengths = {name: tracks[name].length for name in EXAM_TRACKS}
    assert max(lengths, key=lengths.get) == "monza_ccw"
    assert lengths["monza_ccw"] > 3 * lengths["train_loop_01"]


# ── every track can be driven, and every exam start is on the road ──────────────────────────


@pytest.mark.parametrize("name", sorted(TRAINING | HELDOUT))
def test_ideal_two_laps_fit_well_inside_the_time_limit(tracks, name):
    track = tracks[name]
    ratio = 2.0 * ideal_lap_time(track) / time_limit_for(track.length)
    assert 0.3 < ratio <= MAX_IDEAL_RATIO


@pytest.mark.parametrize("name", EXAM_TRACKS)
def test_exam_starts_are_on_the_road_and_face_along_it(tracks, name):
    track = tracks[name]
    fractions = np.array([0.0, 0.2, 0.4, 0.6, 0.8])
    s0 = np.concatenate([fractions, fractions]) * track.length
    lateral = np.repeat([0.0, 1.5], 5)  # centred, then 1.5 m to the left
    world = World(track, WorldConfig.for_track(track))
    world.reset(s0=s0, lateral=lateral)
    # Within a centimetre and 0.01 rad (0.6 degrees): in a bend the car is pushed sideways along
    # the blended normal, then projected back onto 1 m straight segments. Inside a 13 m hairpin
    # (the road turns 4.4 degrees per metre) that reads the road heading a little further on:
    # 0.0044 rad for the 20 % start on hairpins_01, which begins at the first hairpin.
    assert world.lateral_offset == pytest.approx(lateral, abs=0.01)
    heading_error = np.angle(np.exp(1j * (world.cars.psi - world.road_heading)))
    assert heading_error == pytest.approx(0.0, abs=0.01)
    world.step(np.zeros(10), np.zeros(10))
    assert np.all(world.status == Status.RUNNING)
