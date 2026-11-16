"""Where a car drives through the bends: the racing line, measured (task 7.9, Study S6, ADR-062).

The scripted expert follows the centreline on purpose (task 5.3). A fast driver does not: it
comes into a bend from the *outside*, touches the *inside* at the *apex* (the tightest point)
and drifts back out, which makes the arc wider, so the same grip allows more speed. This module
measures how far a drive departs from the centreline at those three places of every bend.

Symbols:

* e: sideways offset from the centreline, metres, positive = left (ARCHITECTURE §6.6);
* kappa: signed curvature of the centreline, 1/m, positive for a left bend: the turn angle from
  one sample to the next divided by the spacing, averaged over ``window`` samples (9, about 9 m,
  as the expert's speed profile does, ``env/experts.py: speed_profile``);
* a *bend*: a stretch of samples where |kappa| > 1 / radius (radius 50 m by default); its apex is
  its sample of largest |kappa|, its sign sigma = +1 for a left bend and -1 for a right bend;
* d: the distance a car has come along the centreline, its start's arc length plus its progress,
  so the k-th time it reaches an apex at arc length a is the moment d = a + k L (L the lap).

At every pass of a bend three numbers are read, each by linear interpolation in d between the
two frames around the point (the first time the car reaches it):

* inside offset at the apex, e_in = sigma * e(apex): positive = toward the inside (cutting);
* outside offset at entry, e_wide = -sigma * e(apex - entry_m): positive = coming in wide;
* outside offset at exit, e_exit = -sigma * e(apex + entry_m): positive = drifting out.

Example: at a left apex the car is at e = +1.2 m, so e_in = +1.2; at a right apex at e = +0.8 m,
so e_in = -0.8; 20 m before the left apex at e = -1.0 m, so e_wide = +1.0. The mean e_in over the
two bends is 0.2 m. A pass counts only when the car reached the entry point after its own start
and also reached the exit point (so a drive starting inside a bend, or ending in one, gives no
half-measured pass there).
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.env.track import Track

Array = NDArray[np.float64]

BEND_RADIUS_M = 50.0  # a bend: centreline radius under 50 m
ENTRY_M = 20.0  # entry and exit points: 20 m before and after the apex
CURVATURE_WINDOW = 9  # samples (about metres), as the expert's speed profile


@dataclass(frozen=True)
class Bend:
    """One bend of a track: where it starts and ends, its apex, its direction and radius."""

    index: int
    start_s: float
    end_s: float
    apex_s: float
    sign: int  # +1 left bend, -1 right bend
    radius_m: float  # 1 / |kappa| at the apex

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start_s_m": round(self.start_s, 3),
            "end_s_m": round(self.end_s, 3),
            "apex_s_m": round(self.apex_s, 3),
            "direction": "left" if self.sign > 0 else "right",
            "radius_m": round(self.radius_m, 3),
        }


def signed_curvature(track: Track, window: int = CURVATURE_WINDOW) -> Array:
    """kappa at every centreline sample, positive for a left bend, averaged over ``window``.

    The turn angle from sample i to i + 1 (wrapped into (-pi, pi]) divided by their spacing,
    then a centred moving average round the closed loop: the expert's curvature with its sign.
    """
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd number of samples")
    turn = np.angle(np.exp(1j * np.diff(np.append(track.heading, track.heading[0]))))
    ds = np.diff(np.append(track.s, track.length))
    kappa = turn / ds
    half = window // 2
    padded = np.concatenate([kappa[len(kappa) - half :], kappa, kappa[:half]])
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def bends_from_curvature(
    s: ArrayLike, kappa: ArrayLike, radius: float = BEND_RADIUS_M
) -> list[Bend]:
    """The bends of a closed loop sampled at arc lengths ``s`` with signed curvature ``kappa``.

    1. Mark the samples with |kappa| > 1 / radius.
    2. Join neighbouring marked samples into stretches (round the loop: a stretch may wrap past
       sample 0); a change of sign starts a new stretch (an S-bend is two bends).
    3. Each stretch is a bend: its apex is its sample of largest |kappa|.
    Bends are numbered in order of their apex along the lap.
    """
    s_arr, k = np.asarray(s, dtype=np.float64), np.asarray(kappa, dtype=np.float64)
    if s_arr.shape != k.shape or s_arr.ndim != 1:
        raise ValueError("s and kappa must be 1-D arrays of the same length")
    if not radius > 0:
        raise ValueError(f"radius must be positive, got {radius}")
    n = len(k)
    sign = np.where(np.abs(k) > 1.0 / radius, np.sign(k), 0.0).astype(int)
    if not sign.any():
        return []
    if sign.all() and len(set(sign.tolist())) == 1:
        raise ValueError("the whole loop is one bend; it has no start or end")
    # start the walk on a sample where the sign changes, so no stretch is cut in two
    first = int(np.flatnonzero(sign != np.roll(sign, 1))[0])
    order = (np.arange(n) + first) % n
    stretches: list[list[int]] = []
    for i in order:
        if sign[i] == 0:
            continue
        if stretches and sign[stretches[-1][-1]] == sign[i] and stretches[-1][-1] == (i - 1) % n:
            stretches[-1].append(int(i))
        else:
            stretches.append([int(i)])
    found = []
    for idx in stretches:
        members = np.asarray(idx)
        apex = int(members[np.argmax(np.abs(k[members]))])
        found.append((s_arr[apex], members, apex))
    found.sort(key=lambda item: item[0])
    return [
        Bend(
            index=j,
            start_s=float(s_arr[members[0]]),
            end_s=float(s_arr[members[-1]]),
            apex_s=float(s_arr[apex]),
            sign=int(sign[apex]),
            radius_m=float(1.0 / abs(k[apex])),
        )
        for j, (_, members, apex) in enumerate(found)
    ]


def bends(
    track: Track, radius: float = BEND_RADIUS_M, window: int = CURVATURE_WINDOW
) -> list[Bend]:
    """The bends of ``track`` (radius under ``radius`` metres); see :func:`bends_from_curvature`."""
    return bends_from_curvature(track.s, signed_curvature(track, window), radius)


def _value_at(d: Array, e: Array, target: float) -> float | None:
    """e where d first reaches ``target`` (linear between the two frames around it), or None if
    d never reaches it or already starts past it."""
    if len(d) == 0 or d[0] > target:
        return None
    reached = np.flatnonzero(d >= target)
    if len(reached) == 0:
        return None
    i = int(reached[0])
    if i == 0:
        return float(e[0])
    span = d[i] - d[i - 1]
    w = 0.0 if span <= 0 else (target - d[i - 1]) / span
    return float(e[i - 1] + w * (e[i] - e[i - 1]))


def bend_offsets(
    distance: ArrayLike,
    offset: ArrayLike,
    lap_length: float,
    track_bends: list[Bend],
    entry_m: float = ENTRY_M,
) -> list[dict[str, Any]]:
    """Every complete pass of every bend in one drive: e_in, e_wide and e_exit (module docstring).

    ``distance`` is d per frame (start arc length + progress), ``offset`` e per frame, both of
    the frames the car was alive in. Returns one record per pass: bend index, lap pass k and the
    three offsets, in order of d.
    """
    d, e = np.asarray(distance, dtype=np.float64), np.asarray(offset, dtype=np.float64)
    if d.shape != e.shape or d.ndim != 1:
        raise ValueError("distance and offset must be 1-D arrays of the same length")
    if len(d) == 0:
        return []
    passes = []
    for bend in track_bends:
        k_low = int(np.floor((d[0] - bend.apex_s) / lap_length)) - 1
        k_high = int(np.ceil((d.max() - bend.apex_s) / lap_length)) + 1
        for k in range(k_low, k_high + 1):
            apex = bend.apex_s + k * lap_length
            before = _value_at(d, e, apex - entry_m)
            at = _value_at(d, e, apex)
            after = _value_at(d, e, apex + entry_m)
            if before is None or at is None or after is None:
                continue
            passes.append(
                {
                    "bend": bend.index,
                    "apex_d_m": apex,
                    "e_in": bend.sign * at,
                    "e_wide": -bend.sign * before,
                    "e_exit": -bend.sign * after,
                }
            )
    passes.sort(key=lambda p: p["apex_d_m"])
    return passes


def drive_offsets(
    track: Track, x: ArrayLike, y: ArrayLike, progress: ArrayLike, s0: float
) -> tuple[Array, Array]:
    """d and e of one car's frames: d = s0 + progress, e from projecting (x, y) onto the road.

    The projection walks along the frames with the windowed search (as the world does), starting
    from a global search at the first frame, so a road that passes close to itself cannot steal
    a frame.
    """
    xs, ys = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    e = np.empty(len(xs))
    proj = track.project(np.array([xs[0], ys[0]]))
    e[0], near = proj.e[0], proj.index
    for t in range(1, len(xs)):
        proj = track.project_window(np.array([xs[t], ys[t]]), near)
        e[t], near = proj.e[0], proj.index
    return float(s0) + np.asarray(progress, dtype=np.float64), e


def racing_line_summary(passes: list[dict[str, Any]], n_bends: int) -> dict[str, Any]:
    """Medians of e_in, e_wide and e_exit: per bend, and over every pass of every bend."""

    def med(values: list[float]) -> float | None:
        return float(np.median(values)) if values else None

    per_bend = []
    for b in range(n_bends):
        mine = [p for p in passes if p["bend"] == b]
        per_bend.append(
            {
                "bend": b,
                "passes": len(mine),
                "e_in": med([p["e_in"] for p in mine]),
                "e_wide": med([p["e_wide"] for p in mine]),
                "e_exit": med([p["e_exit"] for p in mine]),
            }
        )
    return {
        "passes": len(passes),
        "e_in": med([p["e_in"] for p in passes]),
        "e_wide": med([p["e_wide"] for p in passes]),
        "e_exit": med([p["e_exit"] for p in passes]),
        "per_bend": per_bend,
    }
