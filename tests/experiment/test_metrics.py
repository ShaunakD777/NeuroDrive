"""Tests for experiment/metrics.py: hand-made recordings with answers worked out on paper, then
real drives whose metrics must agree with what the world itself recorded."""

from pathlib import Path

import numpy as np
import pytest

from neurodrive.env.episode import EpisodeRecording, run_episode
from neurodrive.env.policies import ConstantPolicy, NeuronPolicy
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig
from neurodrive.experiment.metrics import (
    EpisodeMetrics,
    aggregate,
    episode_metrics,
    lap_crossing_times,
)

TRACKS = Path(__file__).resolve().parents[2] / "tracks"
L = 100.0  # lap length of the hand-made recordings, metres
DT = 0.5  # their time step, seconds


def make_recording(
    progress: list[list[float]],
    steps: list[int],
    status: list[Status],
    steer: list[list[float]] | None = None,
    start_s0: list[float] | None = None,
    target_laps: int = 2,
) -> EpisodeRecording:
    """A recording with N cars built from progress alone (one list of T + 1 frames per car).

    Car i is alive at frames 0 .. steps[i] - 1 and frozen afterwards. Speed at frame t is the
    progress gained into that frame divided by dt (a car driving along the road); x is the
    progress, y is 0. ``steer`` gives T actions per car (default 0); throttle is always 0.
    """
    p = np.array(progress, dtype=float).T  # (T+1, N)
    frames, cars = p.shape
    alive = np.arange(frames)[:, None] < np.array(steps)[None, :]
    v = np.vstack([np.zeros(cars), np.abs(np.diff(p, axis=0)) / DT])
    act = np.zeros((frames - 1, cars, 2))
    if steer is not None:
        act[:, :, 0] = np.array(steer, dtype=float).T
    zeros = np.zeros(cars)
    return EpisodeRecording(
        meta={"dt": DT, "world_config": {"target_laps": target_laps}},
        start_s0=np.array(start_s0 if start_s0 is not None else zeros, dtype=float),
        start_lateral=zeros,
        start_heading_offset=zeros,
        start_speed=zeros,
        x=p.copy(),
        y=np.zeros_like(p),
        psi=np.zeros_like(p),
        v=v,
        alive=alive,
        progress=p,
        rays=np.zeros((frames, cars, 5)),
        observation=np.zeros((frames, cars, 7)),
        action=act,
        status=np.array(status, dtype=np.int8),
    )


# ── the Brief's worked example ──────────────────────────────────────────────────────────────

FINISHER = [15.0 * k for k in range(15)]  # 0, 15, ..., 210: 30 m/s, finishes at frame 14
CRASHER = [0.0, 10.0, 20.0, 30.0] + [30.0] * 11  # crashes into frame 3, frozen afterwards


@pytest.fixture
def two_cars() -> list[EpisodeMetrics]:
    # The crasher's policy kept steering wildly after the crash; that must not count.
    crasher_steer = [0.0, 0.5, 0.5] + [(-1.0) ** k for k in range(11)]
    rec = make_recording(
        [FINISHER, CRASHER],
        steps=[14, 3],
        status=[Status.FINISHED, Status.CRASHED],
        steer=[[0.0] * 14, crasher_steer],
        start_s0=[0.0, 80.0],
    )
    return episode_metrics(rec, L)


def test_lap_times_are_interpolated_between_frames(two_cars):
    car = two_cars[0]
    # P passes 100 between frames 6 (90) and 7 (105): (6 + 10/15) * 0.5 = 3.333 s; it passes
    # 200 between frames 13 (195) and 14 (210): 6.667 s, so lap 2 also takes 3.333 s.
    assert car.lap_times_s == pytest.approx((10 / 3, 10 / 3), abs=1e-12)
    assert car.best_lap_s == pytest.approx(10 / 3, abs=1e-12)
    assert car.laps == 2 and car.completed and not car.crashed and car.status == "finished"
    assert car.steps == 14 and car.time_s == pytest.approx(7.0)
    assert car.progress_m == 210.0 and car.progress_frac == 1.0  # capped at 1
    assert car.mean_speed == pytest.approx(30.0, abs=1e-12)
    assert car.time_to_crash_s is None and car.end_s_m == pytest.approx(10.0)


def test_a_crash_counts_only_the_steps_before_it(two_cars):
    car = two_cars[1]
    assert car.status == "crashed" and car.crashed and not car.completed
    assert car.steps == 3 and car.time_to_crash_s == pytest.approx(1.5)
    assert car.progress_m == 30.0 and car.progress_frac == pytest.approx(0.15)
    assert car.laps == 0 and car.lap_times_s == () and car.best_lap_s is None
    assert car.mean_speed == pytest.approx(20.0)  # 10 m every 0.5 s
    # Steering 0, 0.5, 0.5 before the crash: changes 0.5 and 0, mean 0.25. Counting the +-1
    # the policy kept sending afterwards would give far more.
    assert car.smoothness == pytest.approx(0.25)
    assert car.end_s_m == pytest.approx(10.0)  # started at 80 m, 30 m on: (80 + 30) mod 100


def test_smoothness_is_the_mean_absolute_change_in_steering():
    rec = make_recording(
        [[0.0, 5.0, 10.0, 15.0, 20.0]],
        steps=[4],
        status=[Status.STALLED],
        steer=[[0.0, 0.2, 0.1, 0.1]],
    )
    (car,) = episode_metrics(rec, L)
    assert car.smoothness == pytest.approx(0.1)  # changes 0.2, 0.1, 0.0
    assert car.status == "stalled" and not car.crashed and not car.completed


def test_driving_backwards_scores_zero_progress():
    rec = make_recording([[0.0, -5.0, -10.0]], steps=[2], status=[Status.STALLED])
    (car,) = episode_metrics(rec, L)
    assert car.progress_m == -10.0 and car.progress_frac == 0.0 and car.laps == 0
    assert car.end_s_m == pytest.approx(90.0)  # 10 m behind the start line


def test_a_recording_cut_short_counts_every_step_as_running():
    rec = make_recording([[0.0, 10.0, 20.0]], steps=[3], status=[Status.RUNNING])
    (car,) = episode_metrics(rec, L)
    assert car.status == "running" and car.steps == 2 and car.time_s == pytest.approx(1.0)


def test_one_lap_target_and_a_crossing_exactly_on_a_frame():
    assert lap_crossing_times(np.array([0.0, 50.0, 100.0, 150.0]), L, 1, DT) == [1.0]
    rec = make_recording([[0.0, 50.0, 100.0]], [2], [Status.FINISHED], target_laps=1)
    (car,) = episode_metrics(rec, L)
    assert car.lap_times_s == (1.0,) and car.progress_frac == 1.0


def test_a_lap_that_was_never_reached_is_an_error():
    with pytest.raises(ValueError, match="never reaches lap 2"):
        lap_crossing_times(np.array([0.0, 60.0, 120.0]), L, 2, DT)


def test_to_dict_is_plain_json(two_cars):
    out = two_cars[0].to_dict()
    assert out["lap_times_s"] == pytest.approx([10 / 3, 10 / 3]) and out["status"] == "finished"


# ── aggregates ──────────────────────────────────────────────────────────────────────────────


def episode(frac: float, status: str, best: float | None, smooth: float) -> EpisodeMetrics:
    return EpisodeMetrics(
        status=status,
        steps=1,
        time_s=1.0,
        progress_m=frac * 200,
        progress_frac=frac,
        laps=0,
        completed=status == "finished",
        crashed=status == "crashed",
        lap_times_s=() if best is None else (best,),
        best_lap_s=best,
        mean_speed=10.0,
        smoothness=smooth,
        time_to_crash_s=None,
        end_s_m=0.0,
    )


def test_aggregate_rates_quartiles_and_medians():
    group = [
        episode(1.0, "finished", 40.0, 0.02),
        episode(1.0, "finished", 44.0, 0.04),
        episode(0.5, "crashed", 30.0, 0.10),  # a fast first lap, then a crash: not counted
        episode(0.2, "stalled", None, 0.00),
    ]
    agg = aggregate(group)
    assert agg.episodes == 4 and agg.completion_rate == 0.5 and agg.crash_rate == 0.25
    # Sorted 0.2, 0.5, 1, 1: median h = 1.5 -> 0.75; Q1 h = 0.75 -> 0.425; Q3 h = 2.25 -> 1.
    assert agg.progress_frac_median == pytest.approx(0.75)
    assert agg.progress_frac_q1 == pytest.approx(0.425)
    assert agg.progress_frac_q3 == pytest.approx(1.0)
    assert agg.median_best_lap_s == pytest.approx(42.0)  # completed only: 40 and 44
    assert agg.median_smoothness == pytest.approx(0.03)  # 0, 0.02, 0.04, 0.10
    assert agg.to_dict()["completion_rate"] == 0.5


def test_aggregate_without_finishers_has_no_lap_time_and_empty_is_an_error():
    assert aggregate([episode(0.1, "crashed", None, 0.0)]).median_best_lap_s is None
    with pytest.raises(ValueError):
        aggregate([])


# ── real drives agree with the world ────────────────────────────────────────────────────────


def drive(policy, name: str = "train_loop_01") -> tuple[EpisodeRecording, World, Track]:
    track = Track.from_json(TRACKS / f"{name}.json")
    world = World(track, WorldConfig.for_track(track))
    world.reset()
    return run_episode(world, policy), world, track


def path_length(rec: EpisodeRecording, car: int, n: int) -> float:
    return float(np.hypot(np.diff(rec.x[: n + 1, car]), np.diff(rec.y[: n + 1, car])).sum())


def test_a_real_crash_matches_the_world():
    rec, world, track = drive(ConstantPolicy(steer=0.0, throttle=1.0))  # straight on, flat out
    (car,) = episode_metrics(rec, track.length)
    assert car.crashed and world.status[0] == Status.CRASHED
    assert car.steps == world.step_count and car.time_to_crash_s == pytest.approx(world.time)
    assert car.progress_m == world.progress[0]
    # Mean speed = distance driven / time driven, because each step moves the car by v_new dt.
    assert car.mean_speed == pytest.approx(path_length(rec, 0, car.steps) / car.time_s, rel=1e-12)
    assert car.end_s_m == pytest.approx(world.s[0], abs=0.05)


def test_the_hand_neuron_finishes_two_laps_and_its_lap_times_add_up(tmp_path):
    neuron = NeuronPolicy((-12.0, 12.0), 0.0, "tanh", 0.24)  # Study S1's baseline
    rec, world, track = drive(neuron)
    (car,) = episode_metrics(rec, track.length)
    assert car.completed and car.laps == 2 and len(car.lap_times_s) == 2
    # The finish line falls inside the last step, so the interpolated laps end at most one step
    # before the step on which the world declared the finish.
    assert car.time_s - world.dt <= sum(car.lap_times_s) <= car.time_s
    assert car.lap_times_s[0] > car.lap_times_s[1]  # the first lap starts from rest
    assert car.best_lap_s == car.lap_times_s[1]
    assert car.mean_speed == pytest.approx(path_length(rec, 0, car.steps) / car.time_s, rel=1e-12)
    assert 0.0 < car.smoothness < 0.1
    # A saved and reloaded recording scores the same.
    reloaded = EpisodeRecording.load(rec.save(tmp_path / "episode"))
    assert episode_metrics(reloaded, track.length) == [car]
