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

Only nearby walls, in the ray's direction (a speed-up that changes no reading). Testing every ray
against every wall costs N x 5 x S checks per step, and S is about 5,800 on a 2.9 km track. A
:class:`WallGrid` cuts that down in two ways, as a *broad phase* that throws away walls no ray can
reach before the exact *narrow phase* above runs on the rest:

* **Cells.** Square cells of side c (``GRID_CELL``, 10 m) cover the track. A ray starts at the
  car, so a car in a cell can only reach walls within ``RAY_RANGE`` (R = 50 m) of that cell.
* **Directions** (task 7.0b). The full turn is cut into H bins (``GRID_BINS``, 16, so
  delta = 22.5 degrees each); a ray with direction angle phi (the car's heading plus the ray's own
  angle, taken between 0 and 2 pi) belongs to bin floor(phi / delta). The *region* of a cell and
  a bin is every point that a ray of that bin, starting anywhere in the cell, can reach: the
  square smeared along a wedge of radius R between the bin's two edge angles (their Minkowski
  sum: every "square point + wedge point"). The wedge's arc is enclosed by the two lines that
  touch it at the edges, which meet at distance R / cos(delta / 2) on the middle angle, so the
  region is enclosed by a convex polygon.

Each (cell, bin) pair keeps its own list: every wall the polygon may touch. A wall is left off
only when a *separating axis* proves a gap: along one of the polygon's edge directions or the
wall's own direction, the "shadows" (projections) of wall and polygon are more than
``_GRID_MARGIN`` apart. Each bin is also widened by ``_BIN_SLACK`` on both sides. The two margins
keep a car exactly on a cell edge and a ray exactly on a bin edge safe from rounding. Every wall a
ray can hit is on its list (*conservative culling*), and every check that remains is the same
arithmetic as the full search, in the same order, so the readings are bit-for-bit those of the
full search (tests compare the two on every track). ``bins=1`` gives one list per cell with no
direction, the grid before task 7.0b (slightly tighter). A car outside the grid is more than 50 m
from every wall and sees nothing. Each ray then tests exactly the walls on its own list: all the
(ray, wall) tests of a step are laid out as one flat array (no padding to the longest list), and
each ray keeps its smallest t.
"""

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.env.track import Track

if TYPE_CHECKING:  # only for the type hint of observe(); world.py is not imported at run time
    from neurodrive.env.world import World

Array = NDArray[np.float64]

RAY_ANGLES: Array = np.radians([60.0, 30.0, 0.0, -30.0, -60.0])  # relative to heading, radians
RAY_RANGE = 50.0  # metres; farther hits read as "nothing"
U_SLACK = 1e-9
PARALLEL_EPS = 1e-12
GRID_CELL = 10.0  # metres: side of a WallGrid cell (25 m before task 7.0b)
GRID_BINS = 16  # ray-direction bins of a WallGrid, 22.5 degrees each (task 7.0b)
_GRID_MARGIN = 1e-6  # metres: the smallest gap that lets a wall be left off a list
_BIN_SLACK = 1e-9  # radians added to both edges of every bin, against rounding of the angle
_EDGE_GRID_CACHE_SIZE = 32  # tracks whose edge grids edge_grid() keeps per process


def _read_only(a: NDArray) -> NDArray:
    a.setflags(write=False)
    return a


@dataclass(frozen=True)
class WallGrid:
    """Walls sorted into (cell, ray-direction bin) lists, so each ray only tests reachable walls.

    ``starts`` and ``vectors`` hold every wall's start point a and vector e = b - a, plus one
    zero-length dummy wall at index S that no ray can hit (it is "parallel" to every ray). The
    cell of a car at (x, y) is column floor((x - x0) / c) and row floor((y - y0) / c) of a grid of
    ``nx`` by ``ny`` cells whose corner is ``origin`` = (x0, y0); list number
    (column * ny + row) * bins + bin belongs to that cell and a ray direction bin, and the last
    list (always empty) to cars outside the grid. The lists are stored one after another in
    ``walls``; list k is ``walls[offsets[k]:offsets[k + 1]]`` (a ragged list: no padding).
    ``reach`` is the longest ray the grid was built for. All arrays are read-only.
    """

    starts: Array
    vectors: Array
    origin: Array
    cell: float
    nx: int
    ny: int
    bins: int
    offsets: NDArray[np.intp]
    walls: NDArray[np.int32]
    reach: float

    @property
    def n_walls(self) -> int:
        """Number of real walls (the dummy not counted)."""
        return len(self.starts) - 1

    @property
    def list_lengths(self) -> NDArray[np.intp]:
        """How many walls each (cell, bin) list holds, the outside list last."""
        return np.diff(self.offsets)

    @property
    def longest_list(self) -> int:
        """The most walls any single ray can have to test."""
        return int(self.list_lengths.max(initial=0))

    def lists(self, x: Array, y: Array, heading: Array) -> NDArray[np.intp]:
        """List number of every ray, shape (N, R): its car's cell and its direction's bin.

        ``x`` and ``y`` have shape (N,), ``heading`` (N, R) holds absolute ray angles in radians.
        """
        col = np.floor((x - self.origin[0]) / self.cell)
        row = np.floor((y - self.origin[1]) / self.cell)
        inside = (col >= 0) & (col < self.nx) & (row >= 0) & (row < self.ny)
        cell = np.where(inside, col * self.ny + row, 0).astype(np.intp)
        width = 2.0 * np.pi / self.bins
        # mod gives [0, 2 pi]; a value rounded up to 2 pi wraps to bin 0, whose edges are widened
        b = np.floor(np.mod(heading, 2.0 * np.pi) / width).astype(np.intp) % self.bins
        outside = self.nx * self.ny * self.bins
        return np.where(inside[:, None], cell[:, None] * self.bins + b, outside)

    def pairs(
        self, x: Array, y: Array, heading: Array
    ) -> tuple[NDArray[np.intp], NDArray[np.int32], NDArray[np.intp], NDArray[np.intp]]:
        """Every (ray, wall) test the rays need, as flat arrays: no ray tests a wall twice or pads.

        Rays are numbered row by row over (N, R): ray i * R + j is car i's ray j. Returns
        ``ray`` and ``wall`` (one entry per test, grouped by ray in ray order), and per ray the
        position ``first`` of its first test and its number of tests ``count``.
        """
        key = self.lists(x, y, heading).ravel()
        begin = self.offsets[key]
        count = self.offsets[key + 1] - begin
        first = np.cumsum(count) - count
        ray = np.repeat(np.arange(len(key)), count)
        wall = self.walls[np.repeat(begin - first, count) + np.arange(int(count.sum()))]
        return ray, wall, first, count


def _cross(p: Array, q: Array) -> float:
    return float(p[0] * q[1] - p[1] * q[0])


def _hull(points: Array) -> Array:
    """The convex hull of 2-D points, corners counter-clockwise (Andrew's monotone chain)."""
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]

    def chain(seq: Array) -> list[Array]:
        out: list[Array] = []
        for q in seq:
            while len(out) >= 2 and _cross(out[-1] - out[-2], q - out[-2]) <= 0:
                out.pop()
            out.append(q)
        return out

    lower, upper = chain(pts), chain(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _region(cell: float, reach: float, bins: int, h: int) -> Array:
    """Corners of a convex polygon holding every point a ray of bin ``h`` reaches from the cell.

    The cell is the square from (0, 0) to (c, c). With ``bins=1`` the rays go anywhere, and the
    polygon is the square widened by ``reach`` on every side. Otherwise the wedge of bin ``h``
    (edge angles lo and hi, each pushed out by ``_BIN_SLACK``) is enclosed by the quadrilateral
    0, R(cos lo, sin lo), R / cos((hi - lo) / 2) (cos mid, sin mid), R(cos hi, sin hi), and the
    polygon is the hull of every square corner plus every quadrilateral corner.
    """
    square = np.array([[0.0, 0.0], [cell, 0.0], [cell, cell], [0.0, cell]])
    if bins == 1:
        r = reach
        fan = np.array([[-r, -r], [r, -r], [r, r], [-r, r]])
    else:
        width = 2.0 * np.pi / bins
        lo, hi = h * width - _BIN_SLACK, (h + 1) * width + _BIN_SLACK
        mid, far = (lo + hi) / 2.0, reach / np.cos((hi - lo) / 2.0)
        fan = np.array(
            [
                [0.0, 0.0],
                [reach * np.cos(lo), reach * np.sin(lo)],
                [far * np.cos(mid), far * np.sin(mid)],
                [reach * np.cos(hi), reach * np.sin(hi)],
            ]
        )
    return _hull((square[:, None, :] + fan[None, :, :]).reshape(-1, 2))


def wall_grid(
    segments: ArrayLike,
    reach: float = RAY_RANGE,
    cell: float = GRID_CELL,
    bins: int = GRID_BINS,
) -> WallGrid:
    """Build a :class:`WallGrid` for walls of shape (S, 2, 2) and rays up to ``reach`` metres.

    1. Cover the walls' bounding box, widened by ``reach`` on every side, with square cells of
       side ``cell``.
    2. Pair every wall with each cell whose square, widened by ``reach``, overlaps the wall's
       bounding box (only those cells can reach it).
    3. For every bin, take the region polygon of a cell at (0, 0) (:func:`_region`); the region
       of any other cell is the same polygon moved by that cell's corner.
    4. Keep a (wall, cell) pair in the bin's list unless a separating axis shows a gap of more
       than ``_GRID_MARGIN``: the polygon's edge directions, then the wall's own direction.
    5. Sort the kept pairs by list number and wall number into one ragged list.

    ``bins`` is 1 (no directions) or at least 4 (a wedge of 90 degrees or less).
    """
    seg = np.asarray(segments, dtype=np.float64)
    if seg.ndim != 3 or seg.shape[1:] != (2, 2):
        raise ValueError(f"segments must have shape (S, 2, 2), got {seg.shape}")
    if not (reach > 0 and cell > 0):
        raise ValueError("reach and cell must be positive")
    if isinstance(bins, bool) or not isinstance(bins, int) or not (bins == 1 or bins >= 4):
        raise ValueError(f"bins must be 1 or a whole number >= 4, got {bins!r}")
    n_seg = len(seg)
    lo = seg.min(axis=1) if n_seg else np.zeros((0, 2))  # (S, 2) box corners
    hi = seg.max(axis=1) if n_seg else np.zeros((0, 2))
    origin = (lo.min(axis=0) if n_seg else np.zeros(2)) - reach
    top = (hi.max(axis=0) if n_seg else np.zeros(2)) + reach
    nx, ny = (max(1, int(np.ceil(span / cell))) for span in top - origin)
    n_lists = nx * ny * bins + 1  # the last list: outside the grid

    # 2. (wall, cell) pairs: for each wall the block of cells its widened box touches.
    widen = reach + _GRID_MARGIN

    def cell_index(v: Array, axis: int, size: int) -> NDArray[np.intp]:
        return np.clip(np.floor((v - origin[axis]) / cell), 0, size - 1).astype(np.intp)

    first_col, last_col = cell_index(lo[:, 0] - widen, 0, nx), cell_index(hi[:, 0] + widen, 0, nx)
    first_row, last_row = cell_index(lo[:, 1] - widen, 1, ny), cell_index(hi[:, 1] + widen, 1, ny)
    n_rows = last_row - first_row + 1
    per_wall = (last_col - first_col + 1) * n_rows
    wall = np.repeat(np.arange(n_seg), per_wall)
    local = np.arange(len(wall)) - np.repeat(np.cumsum(per_wall) - per_wall, per_wall)
    col = first_col[wall] + local // n_rows[wall]
    row = first_row[wall] + local % n_rows[wall]
    corner_x, corner_y = origin[0] + col * cell, origin[1] + row * cell
    ax, ay = seg[wall, 0, 0] - corner_x, seg[wall, 0, 1] - corner_y  # ends, relative to the cell
    bx, by = seg[wall, 1, 0] - corner_x, seg[wall, 1, 1] - corner_y
    ex, ey = seg[:, 1, 0] - seg[:, 0, 0], seg[:, 1, 1] - seg[:, 0, 1]
    length = np.hypot(ex, ey)
    safe = np.where(length > 0, length, 1.0)
    wall_nx, wall_ny = (-ey / safe)[wall], (ex / safe)[wall]  # the wall's unit normal
    wall_proj = ax * wall_nx + ay * wall_ny
    has_axis = (length > 0)[wall]  # a zero-length wall has no direction of its own

    keys, kept = [], []
    for h in range(bins):
        poly = _region(cell, reach, bins, h)
        separated = np.zeros(len(wall), dtype=bool)
        for e_x, e_y in np.roll(poly, -1, axis=0) - poly:  # 4a. the polygon's edge normals
            norm = np.hypot(e_x, e_y)
            n_x, n_y = e_y / norm, -e_x / norm
            p = poly @ np.array([n_x, n_y])
            pa, pb = ax * n_x + ay * n_y, bx * n_x + by * n_y
            separated |= np.minimum(pa, pb) > p.max() + _GRID_MARGIN
            separated |= np.maximum(pa, pb) < p.min() - _GRID_MARGIN
        p = np.outer(wall_nx, poly[:, 0]) + np.outer(wall_ny, poly[:, 1])  # 4b. the wall's normal
        separated |= has_axis & (p.min(axis=1) > wall_proj + _GRID_MARGIN)
        separated |= has_axis & (p.max(axis=1) < wall_proj - _GRID_MARGIN)
        keep = ~separated
        keys.append((col[keep] * ny + row[keep]) * bins + h)
        kept.append(wall[keep])
    key = np.concatenate(keys).astype(np.intp)
    walls = np.concatenate(kept).astype(np.intp)
    order = np.lexsort((walls, key))  # 5. by list, then by wall number
    counts = np.bincount(key[order], minlength=n_lists)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.intp)

    dummy = np.zeros((1, 2))
    return WallGrid(
        starts=_read_only(np.concatenate([seg[:, 0, :], dummy])),
        vectors=_read_only(np.concatenate([seg[:, 1, :] - seg[:, 0, :], dummy])),
        origin=_read_only(origin),
        cell=float(cell),
        nx=nx,
        ny=ny,
        bins=bins,
        offsets=_read_only(offsets),
        walls=_read_only(walls[order].astype(np.int32)),
        reach=float(reach),
    )


def edge_segments(track: Track) -> Array:
    """All road-edge segments of ``track`` as one array of shape (S, 2, 2): (start, end) points."""
    return np.concatenate([track.left_segments, track.right_segments], axis=0)


_EDGE_GRIDS: dict[bytes, WallGrid] = {}


def edge_grid(track: Track) -> WallGrid:
    """The road edges of ``track`` as a :class:`WallGrid`, the fast form for ray casting.

    Building one takes about a second (several for a 2.9 km track), and the episode runner,
    random search and the live session ask for one each time they start, so the grid of each
    track is built once per process and kept: the key is a SHA-256 checksum of the edge
    coordinates, so a changed track gets a new grid. At most ``_EDGE_GRID_CACHE_SIZE`` grids are
    kept (the oldest is dropped first). The grid's arrays are read-only, so sharing it is safe.
    """
    segments = np.ascontiguousarray(edge_segments(track))
    key = hashlib.sha256(segments.tobytes() + str(segments.shape).encode()).digest()
    grid = _EDGE_GRIDS.get(key)
    if grid is None:
        grid = wall_grid(segments)
        if len(_EDGE_GRIDS) >= _EDGE_GRID_CACHE_SIZE:
            _EDGE_GRIDS.pop(next(iter(_EDGE_GRIDS)))
        _EDGE_GRIDS[key] = grid
    return grid


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
    if grid is None:
        seg = np.asarray(segments, dtype=np.float64)
        if seg.ndim != 3 or seg.shape[1:] != (2, 2):
            raise ValueError(f"segments must have shape (S, 2, 2), got {seg.shape}")
    elif max_range > grid.reach:
        raise ValueError(f"the WallGrid was built for rays up to {grid.reach} m, not {max_range}")
    x_, y_, psi_ = np.broadcast_arrays(*(np.atleast_1d(np.asarray(a, float)) for a in (x, y, psi)))
    if x_.ndim != 1:
        raise ValueError("x, y and psi must be scalars or 1-D")
    rays = np.asarray(angles, dtype=np.float64)

    heading = psi_[:, None] + rays[None, :]  # (N, R) absolute ray angles

    if grid is None:  # every car tests every wall: (N, R, 1) rays against (1, 1, S) walls
        a = seg[:, 0, :]  # segment start points (S, 2)
        e = seg[:, 1, :] - a  # segment vectors (S, 2)
        a_x, a_y, e_x, e_y = (v[None, None, :] for v in (a[:, 0], a[:, 1], e[:, 0], e[:, 1]))
        d_x, d_y = np.cos(heading)[..., None], np.sin(heading)[..., None]
        w_x, w_y = a_x - x_[:, None, None], a_y - y_[:, None, None]  # a - o, shape (N, 1, S)
        t = _hit_distance(d_x, d_y, w_x, w_y, e_x, e_y)  # (N, R, S)
        nearest = t.min(axis=2) if t.shape[2] else np.full(heading.shape, np.inf)
    else:  # each ray tests its own list only: one flat entry per (ray, wall) test
        ray, wall, first, count = grid.pairs(x_, y_, heading)
        nearest = np.full(heading.size, np.inf)
        if len(wall):
            car = ray // heading.shape[1]
            d_x, d_y = np.cos(heading).ravel()[ray], np.sin(heading).ravel()[ray]
            a, e = grid.starts[wall], grid.vectors[wall]
            t = _hit_distance(d_x, d_y, a[:, 0] - x_[car], a[:, 1] - y_[car], e[:, 0], e[:, 1])
            tested = count > 0
            nearest[tested] = np.minimum.reduceat(t, first[tested])  # each ray's smallest t
        nearest = nearest.reshape(heading.shape)
    return np.minimum(nearest, max_range)


def _hit_distance(d_x: Array, d_y: Array, w_x: Array, w_y: Array, e_x: Array, e_y: Array) -> Array:
    """t for every ray-wall pair that hits, infinity for every miss (any matching shapes).

    d is the ray's unit direction, w = a - o the wall's start seen from the ray's origin, e the
    wall's vector. Both the full search and the grid call this one function, so their numbers
    come from the same operations in the same order.
    """
    denom = d_x * e_y - d_y * e_x  # cross(d, e)
    parallel = np.abs(denom) < PARALLEL_EPS
    safe = np.where(parallel, 1.0, denom)
    t = (w_x * e_y - w_y * e_x) / safe  # cross(a - o, e) / cross(d, e)
    u = (w_x * d_y - w_y * d_x) / safe  # cross(a - o, d) / cross(d, e)
    hit = ~parallel & (t >= 0.0) & (u >= -U_SLACK) & (u <= 1.0 + U_SLACK)
    return np.where(hit, t, np.inf)


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


def observe(world: "World", segments: ArrayLike | WallGrid) -> Array:
    """The (N, 7) observation of every car of ``world`` right now.

    The one place that reads a world's car state into :func:`build_observation`, so the episode
    runner and the live session cannot drift apart. ``segments`` is the track's wall grid
    (:func:`edge_grid`), built once and reused every step.
    """
    cars = world.cars
    return build_observation(
        segments, cars.x, cars.y, cars.psi, cars.v, world.road_heading, world.params.v_max
    )
