"""Driving metrics: the numbers the Driving Exam takes from each episode (EXPERIMENT_PROTOCOL §3).

Symbols, for one car of a recording: dt is the time step, L the lap length, K the target laps
(from the recording's own world rules), P_t the cumulative progress at frame t, v_t the speed at
frame t and s_t the steering of step t. ``n`` is the number of steps the car took before its
episode ended: the frames where it was still alive, so frames 0 .. n are its own and step t takes
frame t to frame t + 1. Frozen frames after the end, and the actions a policy keeps returning for
a car that has stopped, are never counted.

Per episode:

* ``progress_m`` = P_n; ``progress_frac`` = P_n / (K L), kept in [0, 1] (driving backwards
  scores 0); ``laps`` = floor(max(P_n, 0) / L).
* ``status`` (finished, crashed, stalled, timeout, or running for a recording cut short);
  ``completed`` and ``crashed`` are status finished and crashed.
* ``lap_times_s``: lap k ends at the first moment P reaches k L. If that falls between frames
  j - 1 and j, linear interpolation puts it at t_k = (j - 1 + (k L - P_(j-1)) / (P_j - P_(j-1))) dt.
  Lap k takes t_k - t_(k-1), with t_0 = 0; ``best_lap_s`` is the shortest.
* ``mean_speed`` = mean of v_1 .. v_n. The car moves by its new speed each step, so this is the
  distance driven divided by the time driven.
* ``smoothness`` = mean of |s_t - s_(t-1)| over t = 1 .. n - 1 (0 with fewer than 2 steps).
* ``time_s`` = n dt; ``time_to_crash_s`` = time_s for a crash; ``end_s_m`` = where on the lap the
  episode ended, (start arc length + P_n) mod L.

Example (L = 100 m, dt = 0.5 s, K = 2): P = 0, 15, 30, ..., 210 at frames 0 .. 14. P passes 100
between frames 6 (90) and 7 (105): t_1 = (6 + 10/15) * 0.5 = 3.333 s; it passes 200 between
frames 13 (195) and 14 (210): t_2 = 6.667 s; both laps take 3.333 s. Whole frames would say 3.5 s.

Aggregates over a group of episodes: completion and crash rates, median and quartiles of
``progress_frac`` (the percentile rule of ``stats.py``), median best lap over the completed
episodes only, and median smoothness.
"""

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurodrive.env.episode import EpisodeRecording
from neurodrive.env.world import Status
from neurodrive.experiment.stats import percentile, summarise


@dataclass(frozen=True)
class EpisodeMetrics:
    """The driving metrics of one car's episode (see the module docstring for each definition)."""

    status: str
    steps: int
    time_s: float
    progress_m: float
    progress_frac: float
    laps: int
    completed: bool
    crashed: bool
    lap_times_s: tuple[float, ...]
    best_lap_s: float | None
    mean_speed: float
    smoothness: float
    time_to_crash_s: float | None
    end_s_m: float

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready values (lap times as a list)."""
        out = asdict(self)
        out["lap_times_s"] = list(self.lap_times_s)
        return out


def lap_crossing_times(
    progress: NDArray[np.float64], length: float, laps: int, dt: float
) -> list[float]:
    """Time (seconds from the start) at which progress first reaches 1, 2, ..., ``laps`` laps.

    ``progress`` holds P at frames 0, 1, ... For lap k the first frame j with P_j >= k L is
    found; P_(j-1) < k L <= P_j, so the crossing lies a fraction
    (k L - P_(j-1)) / (P_j - P_(j-1)) of the way from frame j - 1 to frame j, at time
    (j - 1 + fraction) dt. Every lap up to ``laps`` must have been reached.
    """
    times = []
    for k in range(1, laps + 1):
        target = k * length
        reached = progress >= target
        if not reached.any():
            raise ValueError(f"progress never reaches lap {k} ({target:.1f} m)")
        j = int(np.argmax(reached))  # the first frame at or past the line; j >= 1 since P_0 = 0
        before, after = progress[j - 1], progress[j]
        times.append((j - 1 + (target - before) / (after - before)) * dt)
    return times


def episode_metrics(recording: EpisodeRecording, track_length: float) -> list[EpisodeMetrics]:
    """The driving metrics of every car in ``recording``, driven on a track ``track_length`` long.

    dt and the target laps K are read from the recording's metadata, so an episode is scored by
    the rules it was driven under.
    """
    if not track_length > 0:
        raise ValueError(f"track_length must be positive, got {track_length}")
    dt = float(recording.meta["dt"])
    target_laps = int(recording.meta["world_config"]["target_laps"])
    steps_taken = recording.alive[: recording.n_steps].sum(axis=0)  # n for every car

    results = []
    for car in range(recording.n_cars):
        n = int(steps_taken[car])
        progress = recording.progress[: n + 1, car]
        p_end = float(progress[-1])
        laps = math.floor(max(p_end, 0.0) / track_length)
        crossings = lap_crossing_times(progress, track_length, laps, dt)
        lap_times = tuple(float(t) for t in np.diff([0.0, *crossings]))  # t_k - t_(k-1)
        steer = recording.action[:n, car, 0]
        status = Status(int(recording.status[car])).name.lower()
        time_s = n * dt
        results.append(
            EpisodeMetrics(
                status=status,
                steps=n,
                time_s=time_s,
                progress_m=p_end,
                progress_frac=min(max(p_end / (target_laps * track_length), 0.0), 1.0),
                laps=laps,
                completed=status == "finished",
                crashed=status == "crashed",
                lap_times_s=lap_times,
                best_lap_s=min(lap_times) if lap_times else None,
                mean_speed=float(recording.v[1 : n + 1, car].mean()) if n > 0 else 0.0,
                smoothness=float(np.abs(np.diff(steer)).mean()) if n >= 2 else 0.0,
                time_to_crash_s=time_s if status == "crashed" else None,
                end_s_m=float((recording.start_s0[car] + p_end) % track_length),
            )
        )
    return results


@dataclass(frozen=True)
class Aggregate:
    """Summary of a group of episodes (one track, or the whole exam)."""

    episodes: int
    completion_rate: float
    crash_rate: float
    progress_frac_median: float
    progress_frac_q1: float
    progress_frac_q3: float
    median_best_lap_s: float | None  # completed episodes only; None if none completed
    median_smoothness: float

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-ready values."""
        return asdict(self)


def aggregate(metrics: Sequence[EpisodeMetrics]) -> Aggregate:
    """Completion and crash rates, progress quartiles, median best lap and median smoothness.

    Rates are counts divided by the number of episodes. Medians and quartiles use the linear
    interpolation rule of ``stats.percentile``. The best lap counts completed episodes only, so
    a crash after one fast lap does not set the group's lap time.
    """
    if not metrics:
        raise ValueError("aggregate needs at least one episode")
    count = len(metrics)
    progress = summarise([m.progress_frac for m in metrics])
    best_laps = [m.best_lap_s for m in metrics if m.completed and m.best_lap_s is not None]
    return Aggregate(
        episodes=count,
        completion_rate=sum(m.completed for m in metrics) / count,
        crash_rate=sum(m.crashed for m in metrics) / count,
        progress_frac_median=progress.median,
        progress_frac_q1=progress.q1,
        progress_frac_q3=progress.q3,
        median_best_lap_s=percentile(best_laps, 50.0) if best_laps else None,
        median_smoothness=percentile([m.smoothness for m in metrics], 50.0),
    )
