"""The World: N cars on one track, with progress, crashes and episode termination (§6.6).

A *world* owns a track and a batch of N cars. ``reset`` puts the cars on the road; each ``step``
moves them (car physics), then finds where each car now is relative to the road, and from that
updates:

* **Progress.** Each car's arc length ``s`` (distance along the road) is compared with its value
  one step ago. The difference ``ds`` is wrapped into (-L/2, L/2], where L is the loop length, so
  crossing the start line does not look like a jump of L metres. Cumulative progress ``P`` is the
  running total of ``ds``: driving forward adds to it, driving backwards subtracts from it.
  Completed laps are floor(P / L), never below 0.
* **Crash.** A car crashes when its lateral offset ``e`` (sideways distance from the centreline)
  satisfies |e| > width/2 - car half-width, i.e. the car's side has reached a road edge.
* **Termination.** A car stops running (it freezes) for the first of these reasons:
  crashed; finished the target number of laps; stalled (progress over the last ``stall_window``
  seconds below ``stall_distance`` metres, checked only after ``stall_grace`` seconds); or the
  time limit was reached.

A car's *status* says which happened. Cars that are no longer running never move or accumulate
progress, but the other cars keep going. The world uses no randomness: the same start and the same
actions always give the same trajectories.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.rng import Rng
from neurodrive.env.car import DEFAULT_PARAMS, DT, CarParams, CarState, step_cars
from neurodrive.env.track import Track

Array = NDArray[np.float64]

WINDOW_HALF_SEGMENTS = 15  # projection search window: +-15 segments (about 15 m) round the last one


class Status(IntEnum):
    """Why a car is (or is not) still running. Lower numbers win when several apply at once."""

    RUNNING = 0
    CRASHED = 1
    FINISHED = 2
    STALLED = 3
    TIMEOUT = 4


@dataclass(frozen=True)
class WorldConfig:
    """Episode rules. Defaults follow ARCHITECTURE §6.6."""

    time_limit: float = 90.0  # seconds
    target_laps: int = 2  # the episode ends successfully after this many completed laps
    stall_distance: float = 2.0  # metres of progress required ...
    stall_window: float = 3.0  # ... in every window of this many seconds ...
    stall_grace: float = 2.0  # ... checked only once this many seconds have passed

    def __post_init__(self) -> None:
        if not (self.time_limit > 0 and self.stall_window > 0 and self.stall_grace >= 0):
            raise ValueError("time_limit and stall_window must be positive, stall_grace >= 0")
        if self.target_laps < 1:
            raise ValueError("target_laps must be at least 1")


DEFAULT_CONFIG = WorldConfig()


def _wrap_ds(ds: Array, length: float) -> Array:
    """Wrap arc-length differences into (-length/2, length/2]."""
    return ds - length * np.ceil((ds - length / 2.0) / length)


class World:
    """N cars on a track. Call :meth:`reset`, then :meth:`step` repeatedly.

    After ``reset``/``step`` these hold one value per car: ``cars`` (the :class:`CarState`),
    ``progress`` (P, metres), ``laps`` (completed laps), ``status`` (a :class:`Status` value per
    car), ``s`` (arc length), ``lateral_offset`` (e), ``road_heading`` (tangent angle at the car's
    nearest centreline point), and ``last_ds`` (progress made in the latest step). ``step_count``
    and ``time`` count steps and seconds since reset.
    """

    def __init__(
        self,
        track: Track,
        config: WorldConfig = DEFAULT_CONFIG,
        params: CarParams = DEFAULT_PARAMS,
        dt: float = DT,
    ) -> None:
        self.track = track
        self.config = config
        self.params = params
        self.dt = dt
        self.limit_steps = round(config.time_limit / dt)
        self.stall_steps = round(config.stall_window / dt)
        self.grace_steps = round(config.stall_grace / dt)
        self.crash_offset = track.width / 2.0 - params.half_width
        self._ready = False

    # ── reading the state ───────────────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        """Number of cars."""
        return self.cars.n

    @property
    def running(self) -> NDArray[np.bool_]:
        """True for cars still driving (equals ``cars.alive``)."""
        return self.cars.alive

    @property
    def laps(self) -> NDArray[np.intp]:
        """Completed laps, floor(P / L_track), never below 0."""
        return np.maximum(np.floor(self.progress / self.track.length), 0).astype(np.intp)

    @property
    def time(self) -> float:
        """Seconds since reset."""
        return self.step_count * self.dt

    @property
    def done(self) -> bool:
        """True when no car is running any more."""
        return not bool(self.cars.alive.any())

    # ── reset ───────────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        n: int | None = None,
        s0: ArrayLike | None = None,
        lateral: ArrayLike = 0.0,
        heading_offset: ArrayLike = 0.0,
        speed: ArrayLike = 0.0,
        rng: Rng | None = None,
        random_s0: bool = False,
        lateral_jitter: float = 0.0,
        heading_jitter: float = 0.0,
    ) -> None:
        """Place N cars on the road and clear all progress.

        Car i starts at arc length ``s0[i]`` (default: the track's start, ``s0`` in the file),
        shifted sideways by ``lateral[i]`` metres (positive = left of the centreline), with
        heading equal to the road direction plus ``heading_offset[i]`` radians, and speed
        ``speed[i]``. Scalars apply to every car; ``n`` defaults to the length of the longest
        array argument (1 if all are scalars).

        Seeded random starts: pass a generator ``rng`` (from ``core/rng.py``) and any of

        * ``random_s0=True``: every car (that was not given an explicit ``s0``) starts at an
          arc length drawn uniformly from [0, L_track);
        * ``lateral_jitter`` (j, metres): a number drawn uniformly from [-j, +j] is added to
          each car's sideways offset;
        * ``heading_jitter`` (h, radians): likewise added to each car's heading offset.

        Draws happen in that order (arc length, lateral, heading), N numbers each, only for the
        options used, so the same seed and arguments always give the same start. Asking for
        randomness without an ``rng`` raises an error.
        """
        wants_random = random_s0 or lateral_jitter != 0.0 or heading_jitter != 0.0
        if wants_random and rng is None:
            raise ValueError("random_s0 / jitter need an rng (a seeded generator from core/rng)")
        if lateral_jitter < 0.0 or heading_jitter < 0.0:
            raise ValueError("jitter ranges must not be negative")
        explicit_s0 = s0 is not None
        arrays = [
            np.atleast_1d(np.asarray(a, float))
            for a in (self.track.start_s0 if s0 is None else s0, lateral, heading_offset, speed)
        ]
        count = n if n is not None else max(len(a) for a in arrays)
        s_start, lat, dpsi, v = (np.broadcast_to(a, (count,)).copy() for a in arrays)
        if rng is not None:
            if random_s0 and not explicit_s0:
                s_start = rng.uniform(0.0, self.track.length, size=count)
            if lateral_jitter > 0.0:
                lat = lat + rng.uniform(-lateral_jitter, lateral_jitter, size=count)
            if heading_jitter > 0.0:
                dpsi = dpsi + rng.uniform(-heading_jitter, heading_jitter, size=count)

        x, y, road = self.track.point_at(s_start)
        x = x - lat * np.sin(road)  # the left normal of heading h is (-sin h, cos h)
        y = y + lat * np.cos(road)
        psi = np.pi - np.mod(np.pi - (road + dpsi), 2.0 * np.pi)
        self.cars = CarState.from_pose(x, y, psi, v)

        proj = self.track.project(np.stack([x, y], axis=1))  # global search: no previous position
        self._index = proj.index
        self._s_prev = proj.s
        self.s, self.lateral_offset, self.road_heading = proj.s, proj.e, proj.heading
        self.progress = np.zeros(count)
        self.last_ds = np.zeros(count)
        self.status = np.full(count, Status.RUNNING, dtype=np.int8)
        self.step_count = 0
        self._history = np.zeros((self.stall_steps + 1, count))  # ring buffer of past P values
        self._ready = True

    # ── step ────────────────────────────────────────────────────────────────────────────────

    def step(self, steer: ArrayLike, throttle: ArrayLike) -> Array:
        """Advance every running car by one time step; return the progress ``ds`` each made.

        1. Move the cars with :func:`step_cars` (cars not running stay frozen).
        2. Find each car's new arc length ``s`` and lateral offset ``e`` with a windowed search
           around its previous position.
        3. ``ds`` = wrapped difference of ``s``; add it to progress ``P`` (running cars only).
        4. Crash if |e| > width/2 - half-width.
        5. Otherwise finish if laps >= target; stall if after the grace period progress over the
           last window is below the required distance; time out at the time limit.
        """
        if not self._ready:
            raise RuntimeError("call reset() before step()")
        was_running = self.cars.alive
        self.cars = step_cars(self.cars, steer, throttle, self.params, self.dt)
        self.step_count += 1

        proj = self.track.project_window(
            np.stack([self.cars.x, self.cars.y], axis=1), self._index, WINDOW_HALF_SEGMENTS
        )
        ds = np.where(was_running, _wrap_ds(proj.s - self._s_prev, self.track.length), 0.0)
        self.progress = self.progress + ds
        self.last_ds = ds
        self._index = np.where(was_running, proj.index, self._index)
        self._s_prev = np.where(was_running, proj.s, self._s_prev)
        self.s = self._s_prev
        self.lateral_offset = np.where(was_running, proj.e, self.lateral_offset)
        self.road_heading = np.where(was_running, proj.heading, self.road_heading)

        self._history[self.step_count % len(self._history)] = self.progress
        reference = self._history[
            (self.step_count - min(self.step_count, self.stall_steps)) % len(self._history)
        ]

        # Each rule overrides the one before it, so the priority is crash > finished > stalled
        # > timeout when several apply in the same step.
        ended = np.full(self.n, Status.RUNNING, dtype=np.int8)
        if self.step_count >= self.limit_steps:
            ended[:] = Status.TIMEOUT
        if self.step_count >= self.grace_steps:
            stalled = self.progress - reference < self.config.stall_distance
            ended = np.where(stalled, Status.STALLED, ended)
        ended = np.where(self.laps >= self.config.target_laps, Status.FINISHED, ended)
        ended = np.where(np.abs(self.lateral_offset) > self.crash_offset, Status.CRASHED, ended)

        newly_ended = was_running & (ended != Status.RUNNING)
        self.status = np.where(newly_ended, ended, self.status).astype(np.int8)
        self.cars = self.cars.with_alive(was_running & ~newly_ended)
        return ds
