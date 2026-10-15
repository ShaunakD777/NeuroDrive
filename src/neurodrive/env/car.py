"""Batched car physics: a kinematic bicycle model with a grip limit (physics v1, ARCHITECTURE §6.3).

The car is treated like a bicycle: the wheel angle sets how fast the heading turns, with no tyre
forces. Time advances in fixed steps of ``DT`` seconds. All N cars are updated together as NumPy
arrays; a single car is N = 1. Cars never interact with each other.

Each step uses *semi-implicit Euler*: speed is updated first, then the heading using the new
speed, then the position using the new heading. Progress along the track and laps are not stored
here; they need the track and are kept by the World (task 1.5).

Units: metres, seconds, radians. Heading is 0 along +x and counter-clockwise positive, so a
positive steer command (turn right) makes the heading *decrease*.
"""

from dataclasses import dataclass, fields

import numpy as np
from numpy.typing import ArrayLike, NDArray

Array = NDArray[np.float64]

DT = 1.0 / 30.0  # physics step, seconds


@dataclass(frozen=True)
class CarParams:
    """Physical constants of every car (ARCHITECTURE §6.3 initial values)."""

    wheelbase: float = 2.5  # Lw, metres between the axles
    v_max: float = 30.0  # top speed, m/s
    a_acc: float = 8.0  # full-throttle acceleration, m/s^2
    a_brk: float = 16.0  # full-brake deceleration, m/s^2
    drag: float = 0.1  # k, speed lost per second as a fraction of speed
    delta_max: float = 0.5  # largest wheel angle, radians
    half_width: float = 1.0  # half the car's width, metres (used by crash detection)
    a_lat_max: float = 9.0  # grip limit: largest sideways acceleration, m/s^2

    def __post_init__(self) -> None:
        for field in fields(self):
            if not getattr(self, field.name) > 0:
                raise ValueError(f"car parameter {field.name} must be positive")
        if self.delta_max >= np.pi / 2:
            raise ValueError("delta_max must be below pi/2 (tan would blow up)")


DEFAULT_PARAMS = CarParams()


@dataclass(frozen=True)
class CarState:
    """The state of N cars: position ``x``, ``y``, heading ``psi``, speed ``v`` and ``alive``.

    All fields are arrays of length N. ``v`` is never negative (cars cannot reverse). Dead cars
    keep their last state forever.
    """

    x: Array
    y: Array
    psi: Array
    v: Array
    alive: NDArray[np.bool_]

    @property
    def n(self) -> int:
        """Number of cars."""
        return int(self.x.shape[0])

    @classmethod
    def from_pose(
        cls, x: ArrayLike, y: ArrayLike, psi: ArrayLike, v: ArrayLike = 0.0
    ) -> "CarState":
        """Make alive cars from positions, headings and speeds (scalars broadcast to N)."""
        x_, y_, psi_, v_ = np.broadcast_arrays(
            *(np.atleast_1d(np.asarray(a, float)) for a in (x, y, psi, v))
        )
        if x_.ndim != 1:
            raise ValueError("pose arrays must be scalars or 1-D")
        if np.any(v_ < 0):
            raise ValueError("speed must not be negative")
        return cls(
            x=x_.copy(), y=y_.copy(), psi=psi_.copy(), v=v_.copy(), alive=np.ones(x_.shape, bool)
        )

    def with_alive(self, alive: ArrayLike) -> "CarState":
        """Return a copy with the alive flags replaced."""
        flags = np.broadcast_to(np.asarray(alive, bool), self.x.shape).copy()
        return CarState(self.x, self.y, self.psi, self.v, flags)


def _action_array(values: ArrayLike, n: int, name: str) -> Array:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {arr.shape}")
    return np.clip(arr, -1.0, 1.0)  # the environment clips anything outside the range


def step_cars(
    state: CarState,
    steer: ArrayLike,
    throttle: ArrayLike,
    params: CarParams = DEFAULT_PARAMS,
    dt: float = DT,
) -> CarState:
    """Advance all cars by one time step and return the new state (the old one is unchanged).

    ``steer`` (s) and ``throttle`` (u) are in [-1, 1] (values outside are clipped); a scalar
    applies to every car. Symbols: a_acc, a_brk, k (drag), v_max, Lw (wheelbase), delta_max,
    a_lat_max are in :class:`CarParams`. For every alive car:

    1. Acceleration a = u * a_acc if u >= 0, otherwise u * a_brk.
    2. Speed v <- clamp(v + (a - k*v) * dt, 0, v_max).
    3. Requested curvature kappa_req = tan(s * delta_max) / Lw. Grip allows at most
       kappa_max = a_lat_max / max(v^2, 1), since sideways acceleration is v^2 * kappa. The
       actual curvature kappa is kappa_req clamped to [-kappa_max, +kappa_max]. Then
       psi <- psi - v * kappa * dt (positive steer turns right).
    4. Position: x <- x + v*cos(psi)*dt and y <- y + v*sin(psi)*dt, with the new v and psi.

    Dead cars are copied unchanged.
    """
    n = state.n
    s = _action_array(steer, n, "steer")
    u = _action_array(throttle, n, "throttle")

    accel = np.where(u >= 0.0, u * params.a_acc, u * params.a_brk)
    v = np.clip(state.v + (accel - params.drag * state.v) * dt, 0.0, params.v_max)

    kappa_req = np.tan(s * params.delta_max) / params.wheelbase
    kappa_max = params.a_lat_max / np.maximum(v * v, 1.0)
    kappa = np.clip(kappa_req, -kappa_max, kappa_max)
    psi = state.psi - v * kappa * dt
    psi = np.pi - np.mod(np.pi - psi, 2.0 * np.pi)  # wrap into (-pi, pi]

    x = state.x + v * np.cos(psi) * dt
    y = state.y + v * np.sin(psi) * dt

    live = state.alive
    return CarState(
        x=np.where(live, x, state.x),
        y=np.where(live, y, state.y),
        psi=np.where(live, psi, state.psi),
        v=np.where(live, v, state.v),
        alive=live.copy(),
    )
