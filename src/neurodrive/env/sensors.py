"""Raycast sensors: how far is the nearest road edge in each of five directions?

Each car fires 5 rays from its centre, at +60, +30, 0, -30 and -60 degrees from its heading
(order: far-left to far-right; positive angles are to the left, counter-clockwise). A ray's
*reading* is the distance to the first road edge it hits, capped at ``RAY_RANGE`` = 50 m and
divided by 50, so every reading lies in [0, 1]: 0 means a wall right at the car, 1 means nothing
within range. Dividing by the range (*normalising*) keeps every network input on the same small
scale, whatever the units of the world.

Ray against segment, in words. A ray starts at point o and goes in unit direction d, so its points
are o + t*d for distance t >= 0. A wall segment runs from point a to point b; its points are
a + u*e for e = b - a and u between 0 and 1. They meet where o + t*d = a + u*e. With the 2-D cross
product cross(p, q) = p_x*q_y - p_y*q_x, solving gives

    t = cross(a - o, e) / cross(d, e)        u = cross(a - o, d) / cross(d, e)

and the ray hits the segment when t >= 0 and 0 <= u <= 1. If cross(d, e) is (nearly) zero the ray
is parallel to the wall and never hits it. The reading uses the smallest t over all segments.

Two numerical safeguards: u is allowed a tiny slack (``U_SLACK``) so a ray that passes exactly
through the shared corner of two neighbouring segments cannot slip between them, and "parallel"
means ``|cross(d, e)| < PARALLEL_EPS``.

All N cars are processed together as NumPy arrays (a single car is N = 1).

Only nearby walls (a speed-up that changes no reading). Testing every ray against every wall costs
N x 5 x S checks per step, and S is about 5,800 on a 2.9 km track. A :class:`WallGrid` cuts that
down. It lays square cells of side c (``GRID_CELL``, 25 m) over the track and, for each cell, lists
every wall whose bounding box comes within ``RAY_RANGE`` of the cell. A ray of length 50 m from a
car in that cell can only reach walls on the list, so each car is tested against its cell's list
alone. A car outside the grid is more than 50 m from every wall and sees nothing. Every check
that remains is the same arithmetic as before, so the readings are bit-for-bit those of the full
search (a test compares the two).
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.env.track import Track

Array = NDArray[np.float64]

RAY_ANGLES: Array = np.radians([60.0, 30.0, 0.0, -30.0, -60.0])  # relative to heading, radians
RAY_RANGE = 50.0  # metres; farther hits read as "nothing"
U_SLACK = 1e-9
PARALLEL_EPS = 1e-12
GRID_CELL = 25.0  # metres: side of a WallGrid cell
_GRID_MARGIN = 1e-6  # metres added to the reach, against rounding when a car sits on a cell edge


@dataclass(frozen=True)
class WallGrid:
    """Walls sorted into grid cells, so each car only tests the walls that can be in range.

    ``segments`` holds the S walls plus one zero-length dummy wall at index S, which no ray can hit
    (it is "parallel" to every ray). ``table`` has one row per cell, plus a last row for "outside
    the grid"; each row lists wall indices, padded with the dummy. A car at (x, y) is in column
    floor((x - x0) / c) and row floor((y - y0) / c) of a grid of ``nx`` by ``ny`` cells whose corner
    is ``origin`` = (x0, y0). ``reach`` is the longest ray the grid was built for.
    """

    segments: Array
    origin: Array
    cell: float
    nx: int
    ny: int
    table: NDArray[np.intp]
    reach: float

    @property
    def n_walls(self) -> int:
        """Number of real walls (the dummy not counted)."""
        return len(self.segments) - 1

    def candidates(self, x: Array, y: Array) -> NDArray[np.intp]:
        """Wall indices each car must test, shape (N, K): its cell's row of the table."""
        col = np.floor((x - self.origin[0]) / self.cell)
        row = np.floor((y - self.origin[1]) / self.cell)
        inside = (col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny)
        cell = np.where(inside, col * self.ny + row, self.nx * self.ny).astype(np.intp)
        return self.table[cell]


def wall_grid(segments: ArrayLike, reach: float = RAY_RANGE, cell: float = GRID_CELL) -> WallGrid:
    """Build a :class:`WallGrid` for walls of shape (S, 2, 2) and rays up to ``reach`` metres.

    1. Take the bounding box of every wall (its smallest and largest x and y).
    2. Cover all the boxes, widened by ``reach`` on every side, with square cells of side ``cell``.
    3. For each cell, keep the walls whose box overlaps the cell widened by ``reach``: any wall a
       ray from inside the cell can reach has a point within ``reach`` of it in both x and y.
    4. Pad every list to the longest one with the dummy wall, and add an all-dummy row for cars
       outside the grid.
    """
    seg = np.asarray(segments, dtype=np.float64)
    if seg.ndim != 3 or seg.shape[1:] != (2, 2):
        raise ValueError(f"segments must have shape (S, 2, 2), got {seg.shape}")
    if not (reach > 0 and cell > 0):
        raise ValueError("reach and cell must be positive")
    n_seg = len(seg)
    lo = seg.min(axis=1) if n_seg else np.zeros((0, 2))  # (S, 2) box corners
    hi = seg.max(axis=1) if n_seg else np.zeros((0, 2))
    origin = (lo.min(axis=0) if n_seg else np.zeros(2)) - reach
    top = (hi.max(axis=0) if n_seg else np.zeros(2)) + reach
    nx, ny = (max(1, int(np.ceil(span / cell))) for span in top - origin)

    widen = reach + _GRID_MARGIN
    rows: list[NDArray[np.intp]] = []
    for i in range(nx):
        x_lo, x_hi = origin[0] + i * cell - widen, origin[0] + (i + 1) * cell + widen
        in_column = (hi[:, 0] >= x_lo) & (lo[:, 0] <= x_hi)
        for j in range(ny):
            y_lo, y_hi = origin[1] + j * cell - widen, origin[1] + (j + 1) * cell + widen
            rows.append(np.flatnonzero(in_column & (hi[:, 1] >= y_lo) & (lo[:, 1] <= y_hi)))
    rows.append(np.zeros(0, dtype=np.intp))  # outside the grid: no wall in range

    width = max(1, max(len(r) for r in rows))
    table = np.full((len(rows), width), n_seg, dtype=np.intp)
    for k, r in enumerate(rows):
        table[k, : len(r)] = r
    dummy = np.zeros((1, 2, 2))
    return WallGrid(np.concatenate([seg, dummy]), origin, float(cell), nx, ny, table, float(reach))


def edge_segments(track: Track) -> Array:
    """All road-edge segments of ``track`` as one array of shape (S, 2, 2): (start, end) points."""
    return np.concatenate([track.left_segments, track.right_segments], axis=0)


def edge_grid(track: Track) -> WallGrid:
    """The road edges of ``track`` as a :class:`WallGrid`, the fast form for ray casting."""
    return wall_grid(edge_segments(track))


def ray_distances(
    segments: ArrayLike | WallGrid,
    x: ArrayLike,
    y: ArrayLike,
    psi: ArrayLike,
    angles: ArrayLike = RAY_ANGLES,
    max_range: float = RAY_RANGE,
) -> Array:
    """Distance in metres from each car along each ray to the nearest wall, capped at ``max_range``.

    ``segments`` has shape (S, 2, 2) (start and end point of each wall), or is a :class:`WallGrid`
    of them, which gives the same numbers faster. ``x``, ``y`` and ``psi`` (heading, radians) hold
    the N cars' poses (scalars allowed). ``angles`` are the ray angles relative to the heading.
    The result has shape (N, number of rays); a ray that hits nothing within range returns
    ``max_range``.
    """
    grid = segments if isinstance(segments, WallGrid) else None
    seg = grid.segments if grid is not None else np.asarray(segments, dtype=np.float64)
    if seg.ndim != 3 or seg.shape[1:] != (2, 2):
        raise ValueError(f"segments must have shape (S, 2, 2), got {seg.shape}")
    if grid is not None and max_range > grid.reach:
        raise ValueError(f"the WallGrid was built for rays up to {grid.reach} m, not {max_range}")
    x_, y_, psi_ = np.broadcast_arrays(*(np.atleast_1d(np.asarray(a, float)) for a in (x, y, psi)))
    if x_.ndim != 1:
        raise ValueError("x, y and psi must be scalars or 1-D")
    rays = np.asarray(angles, dtype=np.float64)

    heading = psi_[:, None] + rays[None, :]  # (N, R) absolute ray angles
    d_x, d_y = np.cos(heading)[..., None], np.sin(heading)[..., None]  # (N, R, 1)

    if grid is None:  # every car tests every wall: walls have shape (1, 1, S)
        a = seg[:, 0, :]  # segment start points (S, 2)
        e = seg[:, 1, :] - a  # segment vectors (S, 2)
        a_x, a_y, e_x, e_y = (v[None, None, :] for v in (a[:, 0], a[:, 1], e[:, 0], e[:, 1]))
    else:  # each car tests its own cell's walls: walls have shape (N, 1, K)
        near = seg[grid.candidates(x_, y_)]  # (N, K, 2, 2)
        a = near[:, :, 0, :]
        e = near[:, :, 1, :] - a
        a_x, a_y, e_x, e_y = (v[:, None, :] for v in (a[..., 0], a[..., 1], e[..., 0], e[..., 1]))
    w_x = a_x - x_[:, None, None]  # a - o, shape (N, 1, S or K)
    w_y = a_y - y_[:, None, None]

    denom = d_x * e_y - d_y * e_x  # cross(d, e), shape (N, R, S or K)
    parallel = np.abs(denom) < PARALLEL_EPS
    safe = np.where(parallel, 1.0, denom)
    t = (w_x * e_y - w_y * e_x) / safe  # cross(a - o, e) / cross(d, e)
    u = (w_x * d_y - w_y * d_x) / safe  # cross(a - o, d) / cross(d, e)

    hit = ~parallel & (t >= 0.0) & (u >= -U_SLACK) & (u <= 1.0 + U_SLACK)
    nearest = np.where(hit, t, np.inf).min(axis=2) if t.shape[2] else np.full(heading.shape, np.inf)
    return np.minimum(nearest, max_range)


def ray_readings(
    segments: ArrayLike | WallGrid,
    x: ArrayLike,
    y: ArrayLike,
    psi: ArrayLike,
    angles: ArrayLike = RAY_ANGLES,
    max_range: float = RAY_RANGE,
) -> Array:
    """Normalised sensor readings in [0, 1]: :func:`ray_distances` divided by ``max_range``."""
    return ray_distances(segments, x, y, psi, angles, max_range) / max_range


# The observation contract, frozen at task 2.5 (ADR-005, ARCHITECTURE §6.4). Changing the number,
# order, meaning or scaling of these inputs needs a new ADR and a new version.
OBSERVATION_VERSION = "v1"
OBSERVATION_NAMES = (
    "ray_far_left",  # o1: +60 degrees
    "ray_left",  # o2: +30 degrees
    "ray_centre",  # o3: straight ahead
    "ray_right",  # o4: -30 degrees
    "ray_far_right",  # o5: -60 degrees
    "speed",  # o6: v / v_max
    "heading_error",  # o7: wrap(psi - road heading) / pi
)
OBSERVATION_SIZE = len(OBSERVATION_NAMES)  # 5 ray readings + normalised speed and heading error


def wrap_angle(angle: ArrayLike) -> Array:
    """Wrap angles (radians) into (-pi, pi]: add or subtract whole turns of 2*pi as needed."""
    a = np.asarray(angle, dtype=np.float64)
    return np.pi - np.mod(np.pi - a, 2.0 * np.pi)


def build_observation(
    segments: ArrayLike | WallGrid,
    x: ArrayLike,
    y: ArrayLike,
    psi: ArrayLike,
    v: ArrayLike,
    road_heading: ArrayLike,
    v_max: float,
) -> Array:
    """The 7-number observation of every car (observation v1, ARCHITECTURE §6.4), shape (N, 7).

    Symbols: ``psi`` is the car's heading, ``road_heading`` the road direction at the car's
    nearest centreline point, ``v`` the speed and ``v_max`` the top speed.

    * o1 .. o5: the five ray readings in [0, 1], ordered far-left to far-right.
    * o6 = v / v_max, in [0, 1].
    * o7 = wrap(psi - road_heading) / pi, in (-1, 1]; positive means the car points to the left
      of the road direction.
    """
    readings = ray_readings(segments, x, y, psi)
    n = readings.shape[0]
    speed = np.broadcast_to(np.asarray(v, dtype=np.float64), (n,)) / v_max
    heading_error = wrap_angle(np.asarray(psi) - np.asarray(road_heading)) / np.pi
    return np.column_stack([readings, speed, np.broadcast_to(heading_error, (n,))])
