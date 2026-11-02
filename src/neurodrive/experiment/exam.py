"""The Driving Exam, version 1 (EXPERIMENT_PROTOCOL §3, ARCHITECTURE §6.14).

One fixed test for every driver, however it was trained:

* **Tracks** (fixed for exam v1, ADR-034): ``train_loop_01`` (the training track, reported
  separately) and the held-out tracks ``hairpins_01``, ``monza_ccw``, ``sweeper_01`` and
  ``train_loop_01_mirror``.
* **Starts:** on a track of length L, start i (0 .. 9) is at arc length s0 = f L with f in
  0, 0.2, 0.4, 0.6, 0.8, centred for i = 0 .. 4 and 1.5 m to the left for i = 5 .. 9, at rest and
  facing along the road. Example: ``train_loop_01`` (L = 730.7 m) starts at 0, 146.1, 292.3,
  438.4 and 584.6 m.
* **Rules:** 2 laps, the per-track time limit ceil(2 L / 16 m/s) (ADR-022; 92 s on the loop),
  the world's stall rule.
* **Scoring:** every episode by :func:`~neurodrive.experiment.metrics.episode_metrics`; each
  track's 10 episodes, the training track, and the 40 held-out episodes pooled are summarised
  by :func:`~neurodrive.experiment.metrics.aggregate`. Every track has the same 10 starts, so
  each held-out track weighs the same in the pool: completion rates 1.0, 0.0, 0.5 and 1.0 pool
  to (10 + 0 + 5 + 10) / 40 = 0.625.

Each track's 10 starts are driven as one batch of 10 cars, in start order, always: a network's
output for a car differs in the last bits (about 1e-16) between batch sizes, because the matrix
routines add in a different order, and a drive can grow such differences. A recording made the
same way (:func:`exam_starts`, 10 cars, ``run_episode``) gives exactly the card's numbers.
"""

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from neurodrive.env.episode import EpisodeRecording, run_episode
from neurodrive.env.policies import Policy
from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig
from neurodrive.experiment.metrics import EpisodeMetrics, aggregate, episode_metrics

EXAM_VERSION = 1
TRAINING_TRACK = "train_loop_01"
DEFAULT_TRACKS_DIR = Path(__file__).resolve().parents[3] / "tracks"

PolicyFactory = Callable[[World], Policy]


@dataclass(frozen=True)
class ExamSpec:
    """What the exam consists of. Changing any of it needs an ADR and a new exam version."""

    version: int
    tracks: tuple[str, ...]  # the training track first, then the held-out tracks
    start_fractions: tuple[float, ...]
    lateral_offsets: tuple[float, ...]  # metres, positive = left; each applies to every fraction
    target_laps: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "tracks": list(self.tracks),
            "start_fractions": list(self.start_fractions),
            "lateral_offsets": list(self.lateral_offsets),
            "target_laps": self.target_laps,
        }


EXAM_V1 = ExamSpec(
    version=EXAM_VERSION,
    tracks=(TRAINING_TRACK, "hairpins_01", "monza_ccw", "sweeper_01", "train_loop_01_mirror"),
    start_fractions=(0.0, 0.2, 0.4, 0.6, 0.8),
    lateral_offsets=(0.0, 1.5),
    target_laps=2,
)


def exam_starts(length: float, spec: ExamSpec = EXAM_V1) -> tuple[np.ndarray, np.ndarray]:
    """Arc lengths s0 and sideways offsets of the exam's starts on a track ``length`` long.

    Every offset is combined with every fraction, offsets in the outer loop: for exam v1 the
    starts are the five fractions centred, then the same five 1.5 m to the left.
    """
    fractions = np.asarray(spec.start_fractions, dtype=np.float64)
    s0 = np.tile(fractions * length, len(spec.lateral_offsets))
    lateral = np.repeat(np.asarray(spec.lateral_offsets, dtype=np.float64), len(fractions))
    return s0, lateral


@dataclass(frozen=True)
class TrackExam:
    """One track's part of the card: its rules, its episodes and their summary."""

    track: str
    split: str
    length_m: float
    time_limit_s: float
    episodes: list[dict[str, Any]]  # start_index, start_s0_m, start_lateral_m + the metrics
    aggregate: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "track": self.track,
            "split": self.split,
            "length_m": self.length_m,
            "time_limit_s": self.time_limit_s,
            "episodes": self.episodes,
            "aggregate": self.aggregate,
        }


def exam_track(
    track: Track, policy_for: PolicyFactory, spec: ExamSpec = EXAM_V1
) -> tuple[TrackExam, EpisodeRecording]:
    """Drive the exam's starts on ``track`` as one batch and score them.

    ``policy_for`` builds the policy for the world it will drive (the expert needs the world; a
    network policy can ignore it). Returns the track's part of the card and the recording.
    """
    world = World(track, WorldConfig.for_track(track, spec.target_laps))
    s0, lateral = exam_starts(track.length, spec)
    world.reset(s0=s0, lateral=lateral)
    recording = run_episode(
        world, policy_for(world), policy_ref={"kind": "exam", "track": track.name}
    )
    metrics = episode_metrics(recording, track.length)
    episodes = [
        {
            "start_index": i,
            "start_s0_m": float(s0[i]),
            "start_lateral_m": float(lateral[i]),
            **m.to_dict(),
        }
        for i, m in enumerate(metrics)
    ]
    part = TrackExam(
        track=track.name,
        split=track.split,
        length_m=track.length,
        time_limit_s=world.config.time_limit,
        episodes=episodes,
        aggregate=aggregate(metrics).to_dict(),
    )
    return part, recording


@dataclass(frozen=True)
class ExamResult:
    """An exam card (exam v1): the spec, the policy examined, every track and two summaries."""

    exam_version: int
    spec: dict[str, Any]
    policy: dict[str, Any]
    tracks: list[TrackExam]
    training: dict[str, Any]  # aggregate of the training track's episodes
    heldout: dict[str, Any]  # aggregate of every held-out episode, pooled
    env_steps: int  # car-steps the exam simulated (not part of any training budget)
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "exam_version": self.exam_version,
            "spec": self.spec,
            "policy": self.policy,
            "tracks": [t.to_dict() for t in self.tracks],
            "training": self.training,
            "heldout": self.heldout,
            "env_steps": self.env_steps,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ExamResult":
        if data.get("exam_version") != EXAM_VERSION:
            raise ValueError(f"unsupported exam version {data.get('exam_version')!r}")
        return cls(
            exam_version=data["exam_version"],
            spec=data["spec"],
            policy=data["policy"],
            tracks=[TrackExam(**t) for t in data["tracks"]],
            training=data["training"],
            heldout=data["heldout"],
            env_steps=data["env_steps"],
            extra=data.get("extra", {}),
        )

    def save(self, path: str | Path) -> Path:
        """Write the card as JSON (creating the folder if needed)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "ExamResult":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _metrics_of(episodes: Sequence[dict[str, Any]]) -> list[EpisodeMetrics]:
    names = EpisodeMetrics.__dataclass_fields__
    return [
        EpisodeMetrics(
            **{k: (tuple(v) if k == "lap_times_s" else v) for k, v in e.items() if k in names}
        )
        for e in episodes
    ]


def run_exam(
    policy_for: PolicyFactory,
    policy_ref: dict[str, Any] | None = None,
    tracks_dir: str | Path = DEFAULT_TRACKS_DIR,
    spec: ExamSpec = EXAM_V1,
) -> ExamResult:
    """Take the exam: every track of ``spec``, its starts as one batch, scored and summarised.

    ``policy_ref`` describes what was examined (for example ``{"run": ..., "unit": 3}``) and is
    stored on the card. There is no randomness: the same policy always gets the same card.
    """
    parts = []
    env_steps = 0
    for name in spec.tracks:
        track = Track.from_json(Path(tracks_dir) / f"{name}.json")
        part, recording = exam_track(track, policy_for, spec)
        parts.append(part)
        env_steps += int(recording.alive[: recording.n_steps].sum())
    training = [e for p in parts if p.track == spec.tracks[0] for e in p.episodes]
    heldout = [e for p in parts if p.split == "heldout" for e in p.episodes]
    return ExamResult(
        exam_version=spec.version,
        spec=spec.to_dict(),
        policy=policy_ref or {},
        tracks=parts,
        training=aggregate(_metrics_of(training)).to_dict(),
        heldout=aggregate(_metrics_of(heldout)).to_dict(),
        env_steps=env_steps,
    )


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def format_card(result: ExamResult) -> str:
    """The card as a plain-text table: one row per track, then the two summaries."""
    header = (
        f"{'track':31s} {'done':>5s} {'crash':>5s} {'progress median [Q1, Q3]':>26s}"
        f" {'best lap s':>10s} {'smooth':>7s}"
    )
    lines = [f"Driving Exam v{result.exam_version}  policy: {json.dumps(result.policy)}", header]

    def row(label: str, agg: dict[str, Any]) -> str:
        spread = (
            f"{agg['progress_frac_median']:.2f} [{agg['progress_frac_q1']:.2f},"
            f" {agg['progress_frac_q3']:.2f}]"
        )
        return (
            f"{label:31s} {agg['completion_rate']:5.2f} {agg['crash_rate']:5.2f} {spread:>26s}"
            f" {_fmt(agg['median_best_lap_s'], 1):>10s} {agg['median_smoothness']:7.4f}"
        )

    for part in result.tracks:
        lines.append(row(f"{part.track} ({part.split})", part.aggregate))
    lines.append(row("TRAINING (reported apart)", result.training))
    lines.append(row("HELD-OUT (pooled)", result.heldout))
    return "\n".join(lines)
