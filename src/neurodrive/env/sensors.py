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
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.env.track import Track

Array = NDArray[np.float64]

RAY_ANGLES: Array = np.radians([60.0, 30.0, 0.0, -30.0, -60.0])  # relative to heading, radians
RAY_RANGE = 50.0  # metres; farther hits read as "nothing"
U_SLACK = 1e-9
PARALLEL_EPS = 1e-12


def edge_segments(track: Track) -> Array:
    """All road-edge segments of ``track`` as one array of shape (S, 2, 2): (start, end) points."""
    return np.concatenate([track.left_segments, track.right_segments], axis=0)


def ray_distances(
    segments: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    psi: ArrayLike,
    angles: ArrayLike = RAY_ANGLES,
    max_range: float = RAY_RANGE,
) -> Array:
    """Distance in metres from each car along each ray to the nearest wall, capped at ``max_range``.

    ``segments`` has shape (S, 2, 2) (start and end point of each wall). ``x``, ``y`` and ``psi``
    (heading, radians) hold the N cars' poses (scalars allowed). ``angles`` are the ray angles
    relative to the heading. The result has shape (N, number of rays); a ray that hits nothing
    within range returns ``max_range``.
    """
    seg = np.asarray(segments, dtype=np.float64)
    if seg.ndim != 3 or seg.shape[1:] != (2, 2):
        raise ValueError(f"segments must have shape (S, 2, 2), got {seg.shape}")
    x_, y_, psi_ = np.broadcast_arrays(*(np.atleast_1d(np.asarray(a, float)) for a in (x, y, psi)))
    if x_.ndim != 1:
        raise ValueError("x, y and psi must be scalars or 1-D")
    rays = np.asarray(angles, dtype=np.float64)

    heading = psi_[:, None] + rays[None, :]  # (N, R) absolute ray angles
    d_x, d_y = np.cos(heading)[..., None], np.sin(heading)[..., None]  # (N, R, 1)

    a = seg[:, 0, :]  # segment start points (S, 2)
    e = seg[:, 1, :] - a  # segment vectors (S, 2)
    w_x = a[:, 0][None, None, :] - x_[:, None, None]  # a - o, shape (N, 1, S)
    w_y = a[:, 1][None, None, :] - y_[:, None, None]

    denom = d_x * e[:, 1] - d_y * e[:, 0]  # cross(d, e), shape (N, R, S)
    parallel = np.abs(denom) < PARALLEL_EPS
    safe = np.where(parallel, 1.0, denom)
    t = (w_x * e[:, 1] - w_y * e[:, 0]) / safe  # cross(a - o, e) / cross(d, e)
    u = (w_x * d_y - w_y * d_x) / safe  # cross(a - o, d) / cross(d, e)

    hit = ~parallel & (t >= 0.0) & (u >= -U_SLACK) & (u <= 1.0 + U_SLACK)
    nearest = (
        np.where(hit, t, np.inf).min(axis=2) if seg.shape[0] else np.full(heading.shape, np.inf)
    )
    return np.minimum(nearest, max_range)


def ray_readings(
    segments: ArrayLike,
    x: ArrayLike,
    y: ArrayLike,
    psi: ArrayLike,
    angles: ArrayLike = RAY_ANGLES,
    max_range: float = RAY_RANGE,
) -> Array:
    """Normalised sensor readings in [0, 1]: :func:`ray_distances` divided by ``max_range``."""
    return ray_distances(segments, x, y, psi, angles, max_range) / max_range
