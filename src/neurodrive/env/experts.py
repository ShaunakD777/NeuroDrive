"""The scripted expert: a hand-written driver with privileged state (ADR-009, ARCHITECTURE §6.7).

The expert reads what no network is given: each car's arc length ``s`` along the centreline and
the shape of the road ahead. It drives with two rules.

* **Steering, pure pursuit.** Aim at the centreline point a look-ahead distance
  ell = 8 m + 0.5 s * v ahead. With alpha the angle from the car's heading to that point
  (positive = to the left) and d the straight distance to it, the circle that leaves the car
  along its heading and passes through the point has curvature kappa = 2 sin(alpha) / d. The
  car follows curvature kappa with its wheels at delta = atan(kappa * Lw), so
  steer = -delta / delta_max (positive steer turns right), clipped to [-1, 1].
  Example: car at (0, 0) heading east, target (10, 5): alpha = 26.57 deg, d = 11.18 m,
  kappa = 0.08 /m (a 12.5 m circle), delta = atan(0.2) = 0.1974 rad, steer = -0.395.
* **Throttle, curvature speed control.** A speed profile v_ref(s) is built once per track: the
  bend speed min(v_max, sqrt(grip_share * a_lat / kappa)) with kappa averaged over 9 m, then a
  backward pass v_ref(s) <= sqrt(v_ref(s + ds)^2 + 2 * brake_share * a_brk * ds) so that the car
  starts slowing in time. Each step, throttle = speed_gain * (v_ref(s) - v), clipped.
  Example: a 13 m hairpin gives sqrt(0.8 * 9 * 13) = 9.67 m/s; braking at 0.7 * 16 = 11.2 m/s^2
  from 30 m/s starts (900 - 93.6) / 22.4 = 36.0 m before it.

The Policy protocol hands a policy only the 7 observations, which do not hold this state, so a
:class:`ScriptedExpert` is bound to the :class:`~neurodrive.env.world.World` it drives and reads
the state from it. ``label()`` is the teacher-label tap (any state, for imitation and DAgger);
``act(observations)`` makes it a Policy for ``run_episode`` and the exam.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.env.car import DEFAULT_PARAMS, CarParams
from neurodrive.env.sensors import OBSERVATION_SIZE
from neurodrive.env.track import Track
from neurodrive.env.world import World

Array = NDArray[np.float64]


@dataclass(frozen=True)
class ExpertParams:
    """The expert's settings (see the module docstring for the formulas they enter)."""

    look_base: float = 8.0  # look-ahead at standstill, metres
    look_per_speed: float = 0.5  # extra look-ahead per m/s of speed, seconds
    grip_share: float = 0.8  # share of the grip limit the speed profile plans to use
    brake_share: float = 0.7  # share of full braking the speed profile plans to use
    speed_gain: float = 0.5  # throttle per m/s of speed error
    curvature_window: int = 9  # samples (about metres) the curvature is averaged over

    def __post_init__(self) -> None:
        if self.look_base <= 0 or self.look_per_speed < 0:
            raise ValueError("look-ahead must be positive")
        if not (0 < self.grip_share <= 1 and 0 < self.brake_share <= 1):
            raise ValueError("grip_share and brake_share must be in (0, 1]")
        if self.speed_gain <= 0:
            raise ValueError("speed_gain must be positive")
        if self.curvature_window < 1 or self.curvature_window % 2 == 0:
            raise ValueError("curvature_window must be a positive odd number of samples")


DEFAULT_EXPERT = ExpertParams()


def speed_profile(
    track: Track, car: CarParams = DEFAULT_PARAMS, params: ExpertParams = DEFAULT_EXPERT
) -> Array:
    """Target speed v_ref (m/s) at every centreline sample of ``track``.

    1. Curvature at sample i: kappa_i = |turn angle to the next sample| / spacing, averaged over
       ``curvature_window`` samples centred on i (wrapping round the loop).
    2. Bend speed: min(v_max, sqrt(grip_share * a_lat / kappa_i)).
    3. Backward pass: walking round the loop backwards, v_i <= sqrt(v_(i+1)^2 + 2 b ds_i) with
       b = brake_share * a_brk, so the car can slow down in time for every bend. Two rounds let
       the start line's wrap-round settle.
    """
    turn = np.angle(np.exp(1j * np.diff(np.append(track.heading, track.heading[0]))))
    ds = np.diff(np.append(track.s, track.length))
    kappa = np.abs(turn) / ds
    half = params.curvature_window // 2
    padded = np.concatenate([kappa[len(kappa) - half :], kappa, kappa[:half]])
    window = np.ones(params.curvature_window) / params.curvature_window
    kappa = np.convolve(padded, window, mode="valid")
    v = np.minimum(car.v_max, np.sqrt(params.grip_share * car.a_lat_max / np.maximum(kappa, 1e-12)))
    brake = params.brake_share * car.a_brk
    n = len(v)
    for _ in range(2):
        for i in range(n - 1, -1, -1):
            v[i] = min(v[i], np.sqrt(v[(i + 1) % n] ** 2 + 2.0 * brake * ds[i]))
    return v


def pursuit_steer(
    x: ArrayLike,
    y: ArrayLike,
    psi: ArrayLike,
    target_x: ArrayLike,
    target_y: ArrayLike,
    car: CarParams = DEFAULT_PARAMS,
) -> Array:
    """Pure-pursuit steering towards (target_x, target_y) for cars at (x, y) with heading psi.

    alpha = wrap(atan2(target - car) - psi) and d = |target - car|; the arc through the target
    tangent to the heading has curvature kappa = 2 sin(alpha) / d; the wheel angle that drives
    it is atan(kappa * Lw); steer = -angle / delta_max (positive steer turns right), clipped.
    """
    dx = np.asarray(target_x, dtype=np.float64) - np.asarray(x, dtype=np.float64)
    dy = np.asarray(target_y, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    alpha = np.angle(np.exp(1j * (np.arctan2(dy, dx) - np.asarray(psi, dtype=np.float64))))
    distance = np.maximum(np.hypot(dx, dy), 1e-9)
    kappa = 2.0 * np.sin(alpha) / distance
    return np.clip(-np.arctan(kappa * car.wheelbase) / car.delta_max, -1.0, 1.0)


class ScriptedExpert:
    """The expert driver for every car of one world (privileged: it reads the world's state)."""

    def __init__(self, world: World, params: ExpertParams = DEFAULT_EXPERT) -> None:
        self.world = world
        self.params = params
        self.v_ref = speed_profile(world.track, world.params, params)

    def label(self) -> Array:
        """The expert's (steer, throttle) for every car in the world's current state, (N, 2)."""
        world, track, p = self.world, self.world.track, self.params
        cars = world.cars
        ahead = p.look_base + p.look_per_speed * cars.v
        target_x, target_y, _ = track.point_at(world.s + ahead)
        steer = pursuit_steer(cars.x, cars.y, cars.psi, target_x, target_y, world.params)
        sample = np.clip(np.searchsorted(track.s, world.s, side="right") - 1, 0, len(track.s) - 1)
        throttle = np.clip(p.speed_gain * (self.v_ref[sample] - cars.v), -1.0, 1.0)
        return np.stack([steer, throttle], axis=1)

    def act(self, observations: Array) -> Array:
        """Policy protocol: the expert's actions. The observations are checked, not used."""
        obs = np.asarray(observations, dtype=np.float64)
        expected = (self.world.n, OBSERVATION_SIZE)
        if obs.shape != expected:
            raise ValueError(f"observations must have shape {expected}, got {obs.shape}")
        return self.label()
