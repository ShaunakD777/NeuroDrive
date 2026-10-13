"""Track model: a closed road described by a centreline and a width (track v1, ARCHITECTURE §6.2).

A track file lists a few *corner points* of the road's centreline. Loading it does this:

* A smooth closed curve is drawn through the corner points (centripetal Catmull-Rom).
* The curve is resampled into points exactly (about) 1 m apart along the road.
* For each sample we compute the *arc length* ``s`` (distance travelled along the road from the
  first sample), the unit *tangent* (direction of travel) and its angle, and the left-pointing
  unit *normal* (the tangent turned 90 degrees counter-clockwise).
* The road edges are the centreline pushed sideways by half the width ``w`` along the normal:
  left edge = p + (w/2)*n, right edge = p - (w/2)*n. They are stored as line segments, ready for
  the raycast sensors of task 1.4.

World coordinates are metres, x to the right and y up; angles are radians, 0 along +x and
counter-clockwise positive. The *lateral offset* ``e`` of a point is its signed sideways
distance from the centreline: positive on the left of the direction of travel.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

TRACK_VERSION = 1
SPACING = 1.0  # target distance between neighbouring centreline samples, metres
DEFAULT_WIDTH = 12.0  # metres
_DENSE_PER_CORNER = 200  # fine samples per corner-to-corner piece before resampling
_ALPHA = 0.5  # centripetal Catmull-Rom (knot spacing = distance ** 0.5)


def _catmull_rom_closed(corners: Array, per_piece: int) -> Array:
    """Return a dense closed curve through ``corners`` (shape (K, 2)).

    Each piece between corner i and corner i+1 is evaluated from the four corners around it
    (Barry-Goldman form: repeated linear blends). The *centripetal* knot spacing, the square
    root of the distance between corners, stops the curve from looping or bulging when corners
    are unevenly spaced.
    """
    k = len(corners)
    pieces = []
    for i in range(k):
        p0, p1, p2, p3 = (corners[(i + j) % k] for j in (-1, 0, 1, 2))
        t0 = 0.0
        t1 = t0 + np.linalg.norm(p1 - p0) ** _ALPHA
        t2 = t1 + np.linalg.norm(p2 - p1) ** _ALPHA
        t3 = t2 + np.linalg.norm(p3 - p2) ** _ALPHA
        t = np.linspace(t1, t2, per_piece, endpoint=False)[:, None]
        a1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
        a2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
        a3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
        b1 = (t2 - t) / (t2 - t0) * a1 + (t - t0) / (t2 - t0) * a2
        b2 = (t3 - t) / (t3 - t1) * a2 + (t - t1) / (t3 - t1) * a3
        pieces.append((t2 - t) / (t2 - t1) * b1 + (t - t1) / (t2 - t1) * b2)
    return np.vstack(pieces)


def _resample_closed(curve: Array, spacing: float) -> Array:
    """Resample a dense closed polyline at equal distances along it.

    The number of samples is the loop length divided by ``spacing``, rounded, so the spacing is
    adjusted very slightly (never by more than half a sample's worth over the loop) to make the
    loop close exactly. The first sample is the first point of ``curve``.
    """
    closed = np.vstack([curve, curve[:1]])
    chord = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(chord)])
    n = max(int(round(cum[-1] / spacing)), 4)
    targets = np.arange(n) * (cum[-1] / n)
    x = np.interp(targets, cum, closed[:, 0])
    y = np.interp(targets, cum, closed[:, 1])
    return np.stack([x, y], axis=1)


def _segments_cross(a: Array, b: Array, c: Array, d: Array) -> Array:
    """True where segment a-b properly crosses segment c-d (arrays broadcast together).

    Two segments cross when c and d lie on opposite sides of the line through a-b *and* a and b
    lie on opposite sides of the line through c-d. "Side" is the sign of a 2-D cross product.
    """

    def side(p: Array, q: Array, r: Array) -> Array:
        return (q[..., 0] - p[..., 0]) * (r[..., 1] - p[..., 1]) - (q[..., 1] - p[..., 1]) * (
            r[..., 0] - p[..., 0]
        )

    return (side(a, b, c) * side(a, b, d) < 0) & (side(c, d, a) * side(c, d, b) < 0)


def _crossing_pair(first: Array, second: Array, same_loop: bool) -> tuple[int, int] | None:
    """Find a pair (i, j) of crossing segments between two closed polylines, or None.

    ``first`` and ``second`` are vertex arrays (n, 2). Segment i joins vertex i to vertex i+1
    (wrapping round). When both are the same loop, a segment is not compared with itself or its
    two neighbours, which touch it at an end point without crossing it.
    """
    n = len(first)
    a, b = first, np.roll(first, -1, axis=0)
    c, d = second, np.roll(second, -1, axis=0)
    hits = _segments_cross(a[:, None], b[:, None], c[None, :], d[None, :])
    if same_loop:
        idx = np.arange(n)
        gap = np.abs(idx[:, None] - idx[None, :])
        hits &= np.minimum(gap, n - gap) > 1
    found = np.argwhere(hits)
    return (int(found[0, 0]), int(found[0, 1])) if len(found) else None


@dataclass(frozen=True)
class Projection:
    """Where a batch of points sits relative to the road (arrays of length M)."""

    s: Array  # arc length of the nearest centreline point, in [0, L_track)
    e: Array  # lateral offset, metres; positive = left of the direction of travel
    heading: Array  # road direction (tangent angle, radians) at the nearest point
    index: NDArray[np.intp]  # index of the centreline segment that contained the nearest point


class Track:
    """A closed road. Build one with :meth:`from_dict` or :meth:`from_json`.

    Attributes (all NumPy arrays with N centreline samples, about 1 m apart):

    * ``points`` (N, 2): centreline samples; ``s`` (N,): arc length of each, starting at 0.
    * ``length``: total loop length ``L_track`` in metres.
    * ``tangent`` (N, 2) and ``heading`` (N,): unit direction of travel and its angle.
    * ``normal`` (N, 2): left-pointing unit normal, ``(-t_y, t_x)``.
    * ``left`` and ``right`` (N, 2): road edge points; ``left_segments`` and ``right_segments``
      (N, 2, 2): the edges as segments (start point, end point), wrapping round at the end.
    """

    def __init__(
        self,
        name: str,
        split: str,
        corners: ArrayLike,
        width: float = DEFAULT_WIDTH,
        start_s0: float = 0.0,
        start_direction: int = 1,
        version: int = TRACK_VERSION,
    ) -> None:
        if version != TRACK_VERSION:
            raise ValueError(f"unsupported track version {version}; expected {TRACK_VERSION}")
        if split not in ("train", "heldout"):
            raise ValueError(f"split must be 'train' or 'heldout', got {split!r}")
        corners_arr = np.array(corners, dtype=np.float64)
        if corners_arr.ndim != 2 or corners_arr.shape[1] != 2 or len(corners_arr) < 4:
            raise ValueError("centerline needs at least 4 corner points, each [x, y]")
        gaps = np.linalg.norm(corners_arr - np.roll(corners_arr, -1, axis=0), axis=1)
        if np.any(gaps < 1e-9):
            raise ValueError("centerline has repeated consecutive points (or repeats the first)")
        if not width > 0:
            raise ValueError(f"width must be positive, got {width}")
        if start_direction != 1:
            raise ValueError("start direction must be +1 (along the centreline order)")

        self.name = name
        self.split = split
        self.version = version
        self.width = float(width)
        self.corners = corners_arr

        dense = _catmull_rom_closed(corners_arr, _DENSE_PER_CORNER)
        self.points: Array = _resample_closed(dense, SPACING)
        n = len(self.points)
        nxt = np.roll(self.points, -1, axis=0)
        self._seg = nxt - self.points  # vector from sample i to sample i+1
        self._seg_len = np.linalg.norm(self._seg, axis=1)
        self.s: Array = np.concatenate([[0.0], np.cumsum(self._seg_len)[:-1]])
        self.length = float(self._seg_len.sum())

        # Tangent at sample i: from the previous sample to the next one, scaled to length 1.
        chord = nxt - np.roll(self.points, 1, axis=0)
        self.tangent: Array = chord / np.linalg.norm(chord, axis=1, keepdims=True)
        self.heading: Array = np.arctan2(self.tangent[:, 1], self.tangent[:, 0])
        self.normal: Array = np.stack([-self.tangent[:, 1], self.tangent[:, 0]], axis=1)

        half = self.width / 2.0
        self.left: Array = self.points + half * self.normal
        self.right: Array = self.points - half * self.normal
        self.left_segments: Array = np.stack([self.left, np.roll(self.left, -1, axis=0)], axis=1)
        self.right_segments: Array = np.stack([self.right, np.roll(self.right, -1, axis=0)], axis=1)

        if not 0.0 <= start_s0 < self.length:
            raise ValueError(f"start s0 must be in [0, {self.length:.1f}), got {start_s0}")
        self.start_s0 = float(start_s0)
        self.start_direction = int(start_direction)
        self._check_edges(n)

    def _check_edges(self, n: int) -> None:
        """Reject a road whose edges fold over or cross (a bend tighter than half the width).

        First test: the local bend radius ``r = step length / turn angle`` must exceed half the
        width, otherwise the inside edge is pushed past the bend's centre and flips direction.
        Second test: no edge segment may cross another edge segment (or the opposite edge).
        """
        turn = np.angle(np.exp(1j * np.diff(np.append(self.heading, self.heading[0]))))
        tightest = float(np.abs(turn).max() / self._seg_len.mean())  # 1 / radius, per metre
        if tightest * self.width / 2.0 >= 1.0:
            raise ValueError(
                f"track {self.name!r}: a bend of radius {1.0 / tightest:.1f} m is too tight for"
                f" a road {self.width:g} m wide (edge crosses itself)"
            )
        for label, edge, other in (
            ("left", self.left, None),
            ("right", self.right, None),
            ("left/right", self.left, self.right),
        ):
            pair = _crossing_pair(edge, edge if other is None else other, other is None)
            if pair is not None:
                raise ValueError(
                    f"track {self.name!r}: {label} edge crosses itself near samples {pair} of {n};"
                    " a bend is tighter than half the road width allows"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        """Build a track from the parsed JSON dictionary of a track v1 file."""
        required = ("name", "version", "split", "centerline", "start")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"track file is missing field(s): {', '.join(missing)}")
        start = data["start"]
        if not isinstance(start, dict) or "s0" not in start or "direction" not in start:
            raise ValueError("track 'start' must contain 's0' and 'direction'")
        return cls(
            name=data["name"],
            split=data["split"],
            corners=data["centerline"],
            width=data.get("width", DEFAULT_WIDTH),
            start_s0=start["s0"],
            start_direction=start["direction"],
            version=data["version"],
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "Track":
        """Load a track v1 JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def project(self, query: ArrayLike) -> Projection:
        """Find the nearest centreline point to each query point (global search).

        ``query`` has shape (2,) or (M, 2); the result always holds M values. For every query
        point q it tries each centreline segment (from sample ``a`` to the next sample ``b``):

        1. How far along the segment is q? ``u = dot(q - a, b - a) / |b - a|^2``, clamped to
           [0, 1] so the nearest point stays on the segment.
        2. The nearest point is ``a + u (b - a)``; its distance to q is measured.

        The segment with the smallest distance wins. The result gives arc length
        ``s = s_a + u * |b - a|``; the lateral offset ``e`` (that distance, positive when q is
        left of the direction of travel, decided by the sign of the 2-D cross product of the
        segment direction with ``q - a``); and the road heading, a blend of the tangents at a
        and b.

        Task 1.5 adds a faster search restricted to a window of segments near the previous one.
        """
        q = np.atleast_2d(np.asarray(query, dtype=np.float64))
        if q.ndim != 2 or q.shape[1] != 2:
            raise ValueError(f"query must have shape (2,) or (M, 2), got {np.shape(query)}")

        rel = q[:, None, :] - self.points[None, :, :]  # (M, N, 2): q - a for every segment
        u = np.einsum("mnk,nk->mn", rel, self._seg) / (self._seg_len**2)[None, :]
        u = np.clip(u, 0.0, 1.0)
        off = rel - u[..., None] * self._seg[None, :, :]  # q minus its nearest point
        dist = np.linalg.norm(off, axis=2)
        idx = np.argmin(dist, axis=1)
        rows = np.arange(len(q))
        u_best = u[rows, idx]

        seg = self._seg[idx]
        cross = seg[:, 0] * rel[rows, idx, 1] - seg[:, 1] * rel[rows, idx, 0]
        e = np.where(cross >= 0.0, dist[rows, idx], -dist[rows, idx])
        s = self.s[idx] + u_best * self._seg_len[idx]
        t = (1.0 - u_best)[:, None] * self.tangent[idx] + u_best[:, None] * self.tangent[
            (idx + 1) % len(self.points)
        ]
        heading = np.arctan2(t[:, 1], t[:, 0])

        return Projection(s=s % self.length, e=e, heading=heading, index=idx)
