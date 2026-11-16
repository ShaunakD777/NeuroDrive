"""Tests for the racing-line measure (task 7.9, Study S6, ADR-062): bends found from the signed
curvature, the inside offset at the apex and the outside offsets at entry and exit of every pass,
on hand-made arrays (the Brief's worked example) and on the training loop with the expert."""

import numpy as np
import pytest

from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.track import Track
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, exam_track
from neurodrive.experiment.racing_line import (
    Bend,
    bend_offsets,
    bends,
    bends_from_curvature,
    drive_offsets,
    racing_line_summary,
    signed_curvature,
)

TRAIN = Track.from_json(DEFAULT_TRACKS_DIR / "train_loop_01.json")
LAP = 100.0
LEFT = Bend(index=0, start_s=25.0, end_s=35.0, apex_s=30.0, sign=1, radius_m=30.0)
RIGHT = Bend(index=1, start_s=65.0, end_s=75.0, apex_s=70.0, sign=-1, radius_m=40.0)


# ── finding bends ──────────────────────────────────────────────────────────────────────────────


def test_bends_from_hand_made_curvature():
    s = np.arange(100.0)
    k = np.zeros(100)
    k[10:15] = [0.03, 0.05, 0.06, 0.05, 0.03]  # a left bend, apex at 12 (radius 16.7 m)
    k[40:45] = -0.025  # a right bend of radius 40 m: its apex is the first of equals
    k[60:62] = 0.01  # radius 100 m: not a bend at 50 m
    found = bends_from_curvature(s, k, radius=50.0)
    assert [(b.start_s, b.end_s, b.apex_s, b.sign) for b in found] == [
        (10.0, 14.0, 12.0, 1),
        (40.0, 44.0, 40.0, -1),
    ]
    assert found[0].radius_m == pytest.approx(1 / 0.06) and found[1].radius_m == pytest.approx(40)
    assert [b.index for b in found] == [0, 1]


def test_a_bend_across_the_start_line_and_an_s_bend():
    s = np.arange(100.0)
    k = np.zeros(100)
    k[[97, 98, 99, 0, 1]] = [0.03, 0.04, 0.05, 0.04, 0.03]  # one bend through sample 0
    k[50:53] = 0.03
    k[53:56] = -0.03  # left then right at once: two bends
    found = bends_from_curvature(s, k)
    assert [(b.start_s, b.end_s, b.apex_s, b.sign) for b in found] == [
        (50.0, 52.0, 50.0, 1),
        (53.0, 55.0, 53.0, -1),
        (97.0, 1.0, 99.0, 1),
    ]


def test_no_bend_and_one_endless_bend():
    assert bends_from_curvature(np.arange(10.0), np.zeros(10)) == []
    with pytest.raises(ValueError, match="one bend"):
        bends_from_curvature(np.arange(10.0), np.full(10, 0.1))
    with pytest.raises(ValueError):
        bends_from_curvature(np.arange(10.0), np.zeros(9))


def test_the_training_loops_curvature_and_bends():
    k = signed_curvature(TRAIN)
    # the loop turns once round, anticlockwise: the curvature integrates to +2 pi
    ds = np.diff(np.append(TRAIN.s, TRAIN.length))
    assert float(np.sum(k * ds)) == pytest.approx(2 * np.pi, abs=1e-2)
    found = bends(TRAIN)
    assert len(found) == 7
    assert [b.sign for b in found] == [1, 1, -1, -1, 1, 1, 1]
    assert [round(b.radius_m) for b in found] == [40, 28, 33, 45, 26, 45, 36]
    assert all(b.start_s <= b.apex_s <= b.end_s for b in found)


# ── offsets through a bend ─────────────────────────────────────────────────────────────────────


def briefs_drive() -> tuple[np.ndarray, np.ndarray]:
    """One lap, a frame a metre: e = -1.0 m 20 m before the left apex, +1.2 m at it, and
    +0.8 m at the right apex (linear in between)."""
    d = np.arange(0.0, 100.0)
    e = np.interp(d, [0, 10, 30, 50, 70, 90, 99], [0, -1.0, 1.2, 0.4, 0.8, 0.0, 0.0])
    return d, e


def test_the_briefs_worked_example():
    d, e = briefs_drive()
    passes = bend_offsets(d, e, LAP, [LEFT, RIGHT])
    assert [p["bend"] for p in passes] == [0, 1]
    left, right = passes
    assert left["e_in"] == pytest.approx(1.2)  # cuts the left apex
    assert left["e_wide"] == pytest.approx(1.0)  # came in from the outside (the right)
    assert left["e_exit"] == pytest.approx(-0.4)  # 20 m after: 0.4 m left = inside
    assert right["e_in"] == pytest.approx(-0.8)  # on the outside (left) of a right bend
    summary = racing_line_summary(passes, 2)
    assert np.mean([p["e_in"] for p in passes]) == pytest.approx(0.2)
    assert summary["passes"] == 2 and summary["per_bend"][1]["e_in"] == pytest.approx(-0.8)


@pytest.mark.parametrize("c", [-2.0, 0.5, 3.0])
def test_a_constant_offset_is_inside_on_left_bends_and_outside_on_right_bends(c):
    d = np.arange(0.0, 300.0, 0.5)
    passes = bend_offsets(d, np.full_like(d, c), LAP, [LEFT, RIGHT])
    assert len(passes) == 6  # apexes at 30, 70, 130, 170, 230, 270 (its exit, 290, is reached)
    for p in passes:
        sign = 1 if p["bend"] == 0 else -1
        assert p["e_in"] == pytest.approx(sign * c)
        assert p["e_wide"] == pytest.approx(-sign * c) and p["e_exit"] == pytest.approx(-sign * c)


def test_half_measured_passes_are_skipped():
    d, e = briefs_drive()
    # starting at 15 m: the left bend's entry (10 m) is behind the car, so only the right counts
    assert [p["bend"] for p in bend_offsets(d[15:], e[15:], LAP, [LEFT, RIGHT])] == [1]
    # ending at 80 m: the right bend's exit (90 m) is never reached
    assert [p["bend"] for p in bend_offsets(d[:81], e[:81], LAP, [LEFT, RIGHT])] == [0]
    assert bend_offsets(np.zeros(0), np.zeros(0), LAP, [LEFT]) == []


def test_values_between_frames_are_interpolated_and_the_first_arrival_counts():
    d = np.array([0.0, 4.0, 8.0, 12.0, 28.0, 32.0, 29.0, 52.0])  # backs up from 32 to 29
    e = np.array([0.0, 0.0, 0.0, -1.0, 1.0, 2.0, 9.0, 0.0])
    (p,) = bend_offsets(d, e, LAP, [LEFT])
    assert p["e_wide"] == pytest.approx(0.5)  # 10 m: halfway from 8 m (0) to 12 m (-1)
    assert p["e_in"] == pytest.approx(1.5)  # 30 m: halfway from 28 (1) to 32 (2), not 9
    assert p["e_exit"] == pytest.approx(-9.0 * 2 / 23)  # 50 m: between 29 m (9) and 52 m (0)


def test_drive_offsets_projects_the_frames():
    idx = np.arange(100, 400)
    xy = TRAIN.points[idx] + 1.5 * TRAIN.normal[idx]  # 1.5 m left of the centreline
    d, e = drive_offsets(TRAIN, xy[:, 0], xy[:, 1], TRAIN.s[idx] - TRAIN.s[100], TRAIN.s[100])
    assert np.allclose(e, 1.5, atol=0.02)
    assert np.allclose(d, TRAIN.s[idx])


# ── the expert on the training loop (measured, not assumed) ──────────────────────────────────


@pytest.mark.slow
def test_the_expert_cuts_every_bend_of_the_training_loop():
    # pure pursuit aims at a point on the centreline 8 m + 0.5 v ahead and so drives the chord:
    # inside the centreline at every apex, by about 0.6 m in median (S6's reference line)
    found = bends(TRAIN)
    _, rec = exam_track(TRAIN, ScriptedExpert)
    passes = []
    for car in range(rec.n_cars):
        n = int(rec.alive[:, car].sum()) + 1
        d, e = drive_offsets(
            TRAIN, rec.x[:n, car], rec.y[:n, car], rec.progress[:n, car], rec.start_s0[car]
        )
        passes += bend_offsets(d, e, TRAIN.length, found)
    summary = racing_line_summary(passes, len(found))
    assert summary["passes"] >= 130  # 10 drives x 2 laps x 7 bends, less those cut by a start
    assert all(b["e_in"] > 0.2 for b in summary["per_bend"])
    assert 0.4 < summary["e_in"] < 0.9
    assert max(abs(p["e_in"]) for p in passes) < 2.0  # far from the 5 m that would crash
