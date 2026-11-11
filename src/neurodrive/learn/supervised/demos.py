"""Demonstration datasets (dataset v1, ARCHITECTURE §6.16, ADR-044): what a teacher saw and did.

Behavioural cloning learns from pairs (o, y): o is the 7-number observation the student will see
and y the teacher's action in that moment, its *label*. A **demonstration frame** is one such
row: one car at one step while it was still driving. Frames are a budget currency (protocol §2
rule 4): an imitation learner is charged for the frames it reads.

Which frames count. Car i drove n_i steps before its episode ended (crash, finish, stall or time
limit); its rows are steps t = 0 .. n_i - 1, so a dataset of N episodes has
F = n_1 + ... + n_N frames. The episode runner keeps logging a stopped car's frozen state while
the others drive on; those steps are left out. Example: car A crashes after 3 steps, car B
finishes after 5, the episode lasted 5 steps: 10 recorded actions, F = 3 + 5 = 8 frames.

The label of a demonstration frame is the (clipped) action the teacher drove with, y_t = a_t; it
has its own column because DAgger (task 6.6, ``learn/supervised/dagger.py``) labels states that
someone else drove: its datasets have teacher kind ``dagger``, the network drove and the expert
labelled.

Keyboard demonstrations (task 6.1b, ADR-051): :class:`DemoRecorder` is filled one step at a time
by the live session while a person drives one car; its dataset has teacher ``keyboard`` and the
label is the clipped action the car was given.

A dataset is a folder ``<out_dir>/<YYYYMMDD-HHMMSS>-<teacher>-<tag>/`` with

* ``dataset.npz``: ``observation`` (F, 7), ``label`` (F, 2), ``episode`` and ``step`` (F,)
  whole numbers, and ``x``, ``y``, ``psi``, ``v`` (F,) for drawing (never inputs). Rows are in
  episode order, and within an episode in step order;
* ``meta.json``: dataset version, teacher, track, starts, every episode's start, outcome and
  frame count, ``demo_frames`` = F, the contract versions, the world rules and provenance.

The expert records from the Driving Exam's starts on a training track plus seeded random starts
(ADR-044), so the data is not tied to the ten starts the exam measures.

Finding a dataset by name (task 6.3, ADR-047): :func:`find_dataset` accepts a dataset id (the
folder name) or a selector ``<teacher>-<tag>``, meaning the newest dataset of that teacher and
tag; :func:`dataset_sha256` fingerprints ``dataset.npz`` so a run can tell whether the frames it
learned from have changed since.
"""

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

import neurodrive
from neurodrive.core.rng import child_rng, make_rng
from neurodrive.env.car import ACTION_NAMES, ACTION_SIZE, ACTION_VERSION
from neurodrive.env.episode import PHYSICS_VERSION, EpisodeRecording, run_episode
from neurodrive.env.experts import DEFAULT_EXPERT, ExpertParams, ScriptedExpert
from neurodrive.env.sensors import OBSERVATION_NAMES, OBSERVATION_SIZE, OBSERVATION_VERSION
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig

Array = NDArray[np.float64]

DATASET_VERSION = 1
DEFAULT_DATASETS_DIR = Path(__file__).resolve().parents[4] / "data" / "demos"  # git-ignored
DATASET_FILE = "dataset.npz"
META_FILE = "meta.json"
RANDOM_START_LATERAL = 1.0  # metres: random starts are shifted sideways by up to this (live server)
RANDOM_START_HEADING = 0.1  # radians: and turned by up to this
STRAIGHT_STEER = 0.1  # |steer| below this counts as "nearly straight" in straight_share

_FLOAT_FIELDS = ("observation", "label", "x", "y", "psi", "v")
_INT_FIELDS = ("episode", "step")
_ARRAY_FIELDS = _FLOAT_FIELDS + _INT_FIELDS
ARRAY_FIELDS = _ARRAY_FIELDS  # the arrays of a dataset, in file order (used by DAgger)
_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


@dataclass
class DemoDataset:
    """A demonstration dataset: one row per frame (see the module docstring) plus metadata."""

    meta: dict[str, Any]
    observation: Array
    label: Array
    episode: NDArray[np.int64]
    step: NDArray[np.int64]
    x: Array
    y: Array
    psi: Array
    v: Array

    def __post_init__(self) -> None:
        f = self.n_frames
        if self.observation.shape != (f, OBSERVATION_SIZE):
            raise ValueError(
                f"observation must be ({f}, {OBSERVATION_SIZE}), got {self.observation.shape}"
            )
        if self.label.shape != (f, ACTION_SIZE):
            raise ValueError(f"label must be ({f}, {ACTION_SIZE}), got {self.label.shape}")
        for name in ("episode", "step", "x", "y", "psi", "v"):
            if getattr(self, name).shape != (f,):
                raise ValueError(f"{name} must have shape ({f},), got {getattr(self, name).shape}")

    @property
    def n_frames(self) -> int:
        """F, the number of frames (rows): the dataset's demonstration budget."""
        return int(self.observation.shape[0])

    def save(self, folder: str | Path) -> Path:
        """Write ``dataset.npz`` and ``meta.json`` into ``folder`` (created if missing)."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(folder / DATASET_FILE, **{name: getattr(self, name) for name in _ARRAY_FIELDS})
        text = json.dumps(self.meta, indent=2, sort_keys=True) + "\n"
        (folder / META_FILE).write_text(text, encoding="utf-8")
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> "DemoDataset":
        """Read a dataset folder written by :meth:`save`; ``ValueError`` if it does not fit v1."""
        folder = Path(folder)
        if not (folder / META_FILE).is_file() or not (folder / DATASET_FILE).is_file():
            raise FileNotFoundError(f"{folder.as_posix()} is not a dataset folder")
        meta = json.loads((folder / META_FILE).read_text(encoding="utf-8"))
        if meta.get("dataset_version") != DATASET_VERSION:
            raise ValueError(
                f"unsupported dataset version {meta.get('dataset_version')!r}; "
                f"expected {DATASET_VERSION}"
            )
        with np.load(folder / DATASET_FILE, allow_pickle=False) as data:
            missing = [name for name in _ARRAY_FIELDS if name not in data]
            if missing:
                raise ValueError(f"dataset is missing array(s): {', '.join(missing)}")
            dataset = cls(meta=meta, **{name: data[name] for name in _ARRAY_FIELDS})
        if meta.get("demo_frames") != dataset.n_frames:
            raise ValueError(
                f"meta.json says {meta.get('demo_frames')} frames but the arrays hold "
                f"{dataset.n_frames}"
            )
        return dataset

    def identical_to(self, other: "DemoDataset") -> bool:
        """True when metadata and every array match exactly, bit for bit."""
        return self.meta == other.meta and all(
            np.array_equal(getattr(self, name), getattr(other, name)) for name in _ARRAY_FIELDS
        )


def driven_steps(recording: EpisodeRecording) -> NDArray[np.int64]:
    """n_i for every car: the number of steps it took before its episode ended.

    A car is running in frame t exactly when ``alive[t]`` is true (the world clears it in the
    step that ends the episode and never sets it again), so n_i counts its true frames among
    0 .. T - 1.
    """
    return recording.alive[: recording.n_steps].sum(axis=0).astype(np.int64)


def frames_from_recording(
    recording: EpisodeRecording, first_episode: int = 0, labels: Array | None = None
) -> dict[str, np.ndarray]:
    """The dataset rows of one recording: car i's steps 0 .. n_i - 1, labelled with its actions.

    Car i becomes episode ``first_episode + i``. Returns the arrays of :class:`DemoDataset`.
    ``labels`` (T, N, 2), when given, replaces the actions as the labels: DAgger (task 6.6)
    records what the teacher *would* have done in states the student drove into.
    """
    steps = driven_steps(recording)
    if labels is not None and labels.shape != recording.action.shape:
        raise ValueError(f"labels must be {recording.action.shape}, got {labels.shape}")
    source = recording.action if labels is None else labels
    parts: dict[str, list[np.ndarray]] = {name: [] for name in _ARRAY_FIELDS}
    for car, n in enumerate(steps):
        parts["observation"].append(recording.observation[:n, car])
        parts["label"].append(source[:n, car])
        parts["episode"].append(np.full(n, first_episode + car, dtype=np.int64))
        parts["step"].append(np.arange(n, dtype=np.int64))
        for name in ("x", "y", "psi", "v"):
            parts[name].append(getattr(recording, name)[:n, car])
    return {name: np.concatenate(chunks) for name, chunks in parts.items()}


def straight_share(label: Array, threshold: float = STRAIGHT_STEER) -> float:
    """q, the share of frames whose steering label is nearly straight (|steer| < threshold).

    Example: steers [0.02, -0.05, 0.4, 0.01, -0.6] give q = 3 / 5 = 0.6.
    """
    steer = np.asarray(label, dtype=np.float64)[:, 0]
    return float(np.mean(np.abs(steer) < threshold)) if len(steer) else 0.0


def exam_like_starts(
    length: float, start_fractions: tuple[float, ...], lateral_offsets: tuple[float, ...]
) -> tuple[Array, Array]:
    """Arc lengths and sideways offsets of every fraction x offset, offsets in the outer loop
    (the order of ``experiment/exam.py: exam_starts``)."""
    fractions = np.asarray(start_fractions, dtype=np.float64)
    s0 = np.tile(fractions * length, len(lateral_offsets))
    lateral = np.repeat(np.asarray(lateral_offsets, dtype=np.float64), len(fractions))
    return s0, lateral


def record_expert(
    track: Track,
    *,
    start_fractions: tuple[float, ...],
    lateral_offsets: tuple[float, ...],
    random_starts: int,
    seed: int,
    target_laps: int = 2,
    params: ExpertParams = DEFAULT_EXPERT,
    random_lateral_m: float = RANDOM_START_LATERAL,
    random_heading_rad: float = RANDOM_START_HEADING,
) -> DemoDataset:
    """Let the scripted expert drive and keep what it saw and did.

    1. Refuse a held-out track (protocol rule 7: they are never used for training).
    2. Group "exam": one car per fraction x offset (at rest, along the road), one batch.
    3. Group "random" (if ``random_starts`` > 0): that many cars at seeded random starts: arc
       length uniform on the lap, sideways uniform on ±``random_lateral_m`` (default 1 m),
       heading offset on ±``random_heading_rad`` (default 0.1 rad; wider ranges since the 6.6
       follow-up, ADR-050), from the ``starts`` stream of ``make_rng(seed)``; one batch.
    4. Each group is one world with the per-track rules for ``target_laps`` and its own
       :class:`ScriptedExpert`, driven by ``run_episode``; its frames are cut out by
       :func:`frames_from_recording`. Episodes are numbered exam starts first.

    The metadata has no ``created_at`` or provenance: :func:`save_dataset` adds them, so two
    recordings with the same arguments are identical.
    """
    if track.split != "train":
        raise ValueError(
            f"{track.name!r} is a held-out track; demonstrations may only use training tracks "
            "(EXPERIMENT_PROTOCOL rule 7)"
        )
    if random_starts < 0:
        raise ValueError(f"random_starts must be 0 or more, got {random_starts}")
    if not (random_lateral_m >= 0.0 and random_heading_rad >= 0.0):
        raise ValueError("the random starts' sideways and heading ranges must be 0 or more")
    config = WorldConfig.for_track(track, target_laps)
    groups: list[tuple[str, World]] = []
    exam_world = World(track, config)
    s0, lateral = exam_like_starts(track.length, start_fractions, lateral_offsets)
    if len(s0):
        exam_world.reset(s0=s0, lateral=lateral)
        groups.append(("exam", exam_world))
    if random_starts:
        random_world = World(track, config)
        random_world.reset(
            n=random_starts,
            rng=child_rng(make_rng(seed), "starts"),
            random_s0=True,
            lateral_jitter=random_lateral_m,
            heading_jitter=random_heading_rad,
        )
        groups.append(("random", random_world))
    if not groups:
        raise ValueError("no starts: give start fractions and offsets, or random starts")

    rows: list[dict[str, np.ndarray]] = []
    episodes: list[dict[str, Any]] = []
    for kind, world in groups:
        recording = run_episode(
            world, ScriptedExpert(world, params), seed=seed if kind == "random" else None
        )
        rows.append(frames_from_recording(recording, first_episode=len(episodes)))
        for car, n in enumerate(driven_steps(recording)):
            episodes.append(
                {
                    "index": len(episodes),
                    "start_kind": kind,
                    "start_s0_m": float(recording.start_s0[car]),
                    "start_lateral_m": float(recording.start_lateral[car]),
                    "start_heading_offset_rad": float(recording.start_heading_offset[car]),
                    "start_speed_mps": float(recording.start_speed[car]),
                    "status": Status(int(recording.status[car])).name.lower(),
                    "frames": int(n),
                }
            )
    arrays = {name: np.concatenate([r[name] for r in rows]) for name in _ARRAY_FIELDS}
    frames = int(arrays["observation"].shape[0])
    meta = {
        "dataset_version": DATASET_VERSION,
        "teacher": {"kind": "expert", "params": asdict(params)},
        "track": track.name,
        "track_split": track.split,
        "starts": {
            "start_fractions": list(start_fractions),
            "lateral_offsets": list(lateral_offsets),
            "random_starts": random_starts,
            "seed": seed,
            "random_lateral_m": random_lateral_m,
            "random_heading_rad": random_heading_rad,
        },
        "episodes": episodes,
        "demo_frames": frames,
        "env_steps": frames,  # car-steps driven while recording: one per frame
        "straight_share": straight_share(arrays["label"]),
        "observation_version": OBSERVATION_VERSION,
        "observation_names": list(OBSERVATION_NAMES),
        "action_version": ACTION_VERSION,
        "action_names": list(ACTION_NAMES),
        "physics_version": PHYSICS_VERSION,
        "dt": exam_world.dt,
        "world_config": asdict(config),
        "car_params": asdict(exam_world.params),
    }
    return DemoDataset(meta=meta, **arrays)


class DemoRecorder:
    """Collects the frames of a drive that someone else is steering, one step at a time.

    Used by the live session for keyboard demonstrations (task 6.1b, ADR-051). The session calls
    :meth:`begin_episode` at step 0 of a drive (with the start the world used), :meth:`add` once
    per physics step in which the car is still running, just before the world moves (the
    observation o_t the car saw and the action a_t it was given, clipped to [-1, 1]), and
    :meth:`end_episode` when the drive ends or is cut short. Rows therefore follow the dataset
    rule of ADR-044: one per step in which the car was driving. Example: a first drive that
    crashes after 3 steps and a second that is stopped after 5 give F = 3 + 5 = 8.

    The recorder computes nothing about driving; it stores what it is given, so the dataset is
    exactly what the session did.
    """

    def __init__(
        self,
        track: Track,
        config: WorldConfig,
        dt: float,
        car_params: Any,
        *,
        seed: int | None = None,
    ) -> None:
        if track.split != "train":
            raise ValueError(
                f"{track.name!r} is a held-out track; demonstrations may only use training tracks "
                "(EXPERIMENT_PROTOCOL rule 7)"
            )
        self.track, self.config, self.dt = track, config, dt
        self.car_params, self.seed = car_params, seed
        self._chunks: dict[str, list[np.ndarray]] = {name: [] for name in _ARRAY_FIELDS}
        self._episodes: list[dict[str, Any]] = []
        self._open: dict[str, Any] | None = None
        self._rows: list[tuple[np.ndarray, np.ndarray, float, float, float, float]] = []

    @property
    def is_open(self) -> bool:
        """True while an episode is being recorded."""
        return self._open is not None

    @property
    def n_frames(self) -> int:
        """Frames recorded so far, the open episode's included."""
        return sum(e["frames"] for e in self._episodes) + len(self._rows)

    @property
    def n_episodes(self) -> int:
        """Episodes with at least one frame, the open one included."""
        return len(self._episodes) + (1 if self._rows else 0)

    def begin_episode(self, start: dict[str, np.ndarray]) -> None:
        """Open an episode for car 0 at the world's start (``World.start``): s0, lateral,
        heading_offset and speed arrays. Refused while one is open."""
        if self._open is not None:
            raise RuntimeError("an episode is already open")
        self._open = {
            "start_s0_m": float(start["s0"][0]),
            "start_lateral_m": float(start["lateral"][0]),
            "start_heading_offset_rad": float(start["heading_offset"][0]),
            "start_speed_mps": float(start["speed"][0]),
        }
        self._rows = []

    def add(
        self, observation: Array, action: Array, x: float, y: float, psi: float, v: float
    ) -> None:
        """One frame: the observation (7,), the action given (clipped to [-1, 1] here, as the
        world does) and the car's pose for drawing."""
        if self._open is None:
            raise RuntimeError("no episode is open")
        clipped = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        self._rows.append((np.array(observation, dtype=np.float64), clipped, x, y, psi, v))

    def end_episode(self, status: str) -> None:
        """Close the open episode with its outcome (``crashed``, ``finished``, ``stalled``,
        ``timeout`` or, for a drive cut short, ``stopped``). An episode with no frames is
        dropped. Does nothing when none is open."""
        if self._open is None:
            return
        rows, start = self._rows, self._open
        self._open, self._rows = None, []
        if not rows:
            return
        index = len(self._episodes)
        n = len(rows)
        self._chunks["observation"].append(np.stack([r[0] for r in rows]))
        self._chunks["label"].append(np.stack([r[1] for r in rows]))
        self._chunks["episode"].append(np.full(n, index, dtype=np.int64))
        self._chunks["step"].append(np.arange(n, dtype=np.int64))
        for k, name in enumerate(("x", "y", "psi", "v"), start=2):
            self._chunks[name].append(np.array([r[k] for r in rows], dtype=np.float64))
        self._episodes.append(
            {"index": index, "start_kind": "live", **start, "status": status, "frames": n}
        )

    def to_dataset(self) -> DemoDataset:
        """The recorded frames as dataset v1 (teacher ``keyboard``). Episodes must be closed."""
        if self._open is not None:
            raise RuntimeError("close the open episode first")
        if not self._episodes:
            raise ValueError("nothing recorded: there are no frames")
        arrays = {name: np.concatenate(chunks) for name, chunks in self._chunks.items()}
        frames = int(arrays["observation"].shape[0])
        meta = {
            "dataset_version": DATASET_VERSION,
            "teacher": {"kind": "keyboard", "params": {}},
            "track": self.track.name,
            "track_split": self.track.split,
            "starts": {"kind": "live", "seed": self.seed},
            "episodes": self._episodes,
            "demo_frames": frames,
            "env_steps": frames,
            "straight_share": straight_share(arrays["label"]),
            "observation_version": OBSERVATION_VERSION,
            "observation_names": list(OBSERVATION_NAMES),
            "action_version": ACTION_VERSION,
            "action_names": list(ACTION_NAMES),
            "physics_version": PHYSICS_VERSION,
            "dt": self.dt,
            "world_config": asdict(self.config),
            "car_params": asdict(self.car_params),
        }
        return DemoDataset(meta=meta, **arrays)


def record_expert_from_starts(
    track: Track,
    source: DemoDataset,
    *,
    match_frames: bool,
    target_laps: int = 2,
    params: ExpertParams = DEFAULT_EXPERT,
) -> DemoDataset:
    """Let the scripted expert drive from the starts another dataset used (task 6.7, ADR-052).

    Episode i of ``source`` has a start pose (arc length, sideways offset, heading offset, speed).
    The expert drives from that same pose, all episodes as one batch, and its frames are cut out
    by :func:`frames_from_recording`. With ``match_frames`` only its first n_i frames of episode
    i are kept, n_i being the number of frames of ``source``'s episode i, so the two datasets
    have the same starts and the same size. Example: source episodes of 3 and 5 frames give the
    expert's first 3 and first 5 frames, F = 8 in both. They are the expert's own states, not the
    source teacher's. An episode cut short that way has status ``truncated``; without
    ``match_frames`` every drive runs to its own end. Held-out tracks are refused (rule 7).
    """
    if track.split != "train":
        raise ValueError(
            f"{track.name!r} is a held-out track; demonstrations may only use training tracks "
            "(EXPERIMENT_PROTOCOL rule 7)"
        )
    if source.meta.get("track") != track.name:
        raise ValueError(
            f"the source dataset was recorded on {source.meta.get('track')!r}, not {track.name!r}"
        )
    episodes = source.meta["episodes"]
    if not episodes:
        raise ValueError("the source dataset has no episodes")
    config = WorldConfig.for_track(track, target_laps)
    world = World(track, config)
    world.reset(
        s0=[e["start_s0_m"] for e in episodes],
        lateral=[e["start_lateral_m"] for e in episodes],
        heading_offset=[e["start_heading_offset_rad"] for e in episodes],
        speed=[e["start_speed_mps"] for e in episodes],
    )
    recording = run_episode(world, ScriptedExpert(world, params), seed=None)
    arrays = frames_from_recording(recording)
    own = driven_steps(recording)
    limit = np.array([e["frames"] for e in episodes], dtype=np.int64)
    if match_frames:
        keep = arrays["step"] < limit[arrays["episode"]]
        arrays = {name: value[keep] for name, value in arrays.items()}
    kept = np.bincount(arrays["episode"], minlength=len(episodes))
    records = []
    for i, e in enumerate(episodes):
        cut = match_frames and kept[i] < own[i]
        records.append(
            {
                "index": i,
                "start_kind": "from_dataset",
                "source_index": e["index"],
                "start_s0_m": float(recording.start_s0[i]),
                "start_lateral_m": float(recording.start_lateral[i]),
                "start_heading_offset_rad": float(recording.start_heading_offset[i]),
                "start_speed_mps": float(recording.start_speed[i]),
                "status": "truncated" if cut else Status(int(recording.status[i])).name.lower(),
                "frames": int(kept[i]),
            }
        )
    frames = int(arrays["observation"].shape[0])
    meta = {
        "dataset_version": DATASET_VERSION,
        "teacher": {"kind": "expert", "params": asdict(params)},
        "track": track.name,
        "track_split": track.split,
        "starts": {
            "kind": "from_dataset",
            "source_dataset": source.meta.get("dataset_id"),
            "match_frames": match_frames,
        },
        "episodes": records,
        "demo_frames": frames,
        "env_steps": frames,
        "straight_share": straight_share(arrays["label"]),
        "observation_version": OBSERVATION_VERSION,
        "observation_names": list(OBSERVATION_NAMES),
        "action_version": ACTION_VERSION,
        "action_names": list(ACTION_NAMES),
        "physics_version": PHYSICS_VERSION,
        "dt": world.dt,
        "world_config": asdict(config),
        "car_params": asdict(world.params),
    }
    return DemoDataset(meta=meta, **arrays)


_SAME_IN_ALL = (
    "track",
    "observation_version",
    "action_version",
    "physics_version",
    "dt",
    "world_config",
    "car_params",
)


def merge_datasets(datasets: list[DemoDataset]) -> DemoDataset:
    """Join several datasets of one teacher into one (task 6.7, ADR-052).

    1. Refuse fewer than one dataset, a mix of teachers, or datasets that differ in track,
       contract versions, time step, world rules or car parameters.
    2. Put their rows one after the other, renumbering episodes 0 .. k - 1 in order. Every episode
       keeps ``source_dataset`` (the source's id) and ``source_index`` (its number there).
    3. ``demo_frames`` and ``env_steps`` are the sums, ``straight_share`` is recomputed.

    Example: datasets of 3 + 5 frames (two episodes) and 4 frames (one episode) give 12 frames
    and episodes 0, 1, 2 with 3, 5 and 4 frames.
    """
    if not datasets:
        raise ValueError("nothing to merge: give at least one dataset")
    first = datasets[0].meta
    for d in datasets[1:]:
        if d.meta["teacher"] != first["teacher"]:
            raise ValueError(
                f"cannot merge different teachers: {first['teacher']['kind']!r} and "
                f"{d.meta['teacher']['kind']!r}"
            )
        for key in _SAME_IN_ALL:
            if d.meta.get(key) != first.get(key):
                raise ValueError(f"cannot merge datasets that differ in {key!r}")
    episodes: list[dict[str, Any]] = []
    parts: dict[str, list[np.ndarray]] = {name: [] for name in _ARRAY_FIELDS}
    for d in datasets:
        offset = len(episodes)
        for e in d.meta["episodes"]:
            episodes.append(
                {
                    **e,
                    "index": offset + e["index"],
                    "source_dataset": d.meta.get("dataset_id"),
                    "source_index": e["index"],
                }
            )
        for name in _ARRAY_FIELDS:
            values = getattr(d, name)
            parts[name].append(values + offset if name == "episode" else values)
    arrays = {name: np.concatenate(chunks) for name, chunks in parts.items()}
    frames = int(arrays["observation"].shape[0])
    meta = {
        "dataset_version": DATASET_VERSION,
        "teacher": first["teacher"],
        **{key: first[key] for key in _SAME_IN_ALL},
        "track_split": first["track_split"],
        "starts": {"kind": "merged", "sources": [d.meta.get("dataset_id") for d in datasets]},
        "episodes": episodes,
        "demo_frames": frames,
        "env_steps": frames,
        "straight_share": straight_share(arrays["label"]),
        "observation_names": first["observation_names"],
        "action_names": first["action_names"],
    }
    return DemoDataset(meta=meta, **arrays)


def save_dataset(
    dataset: DemoDataset,
    out_dir: str | Path,
    tag: str = "default",
    provenance: dict[str, Any] | None = None,
    when: datetime | None = None,
) -> Path:
    """Save ``dataset`` as a new folder ``<out_dir>/<YYYYMMDD-HHMMSS>-<teacher>-<tag>/``.

    Adds ``dataset_id`` (the folder name), ``tag``, ``created_at`` (UTC), the NeuroDrive version
    and ``provenance`` (for example the git commit, given by the caller: it is found in
    ``experiment/store.py``, which ``learn/`` may not import) to the metadata. An existing
    folder name gets ``-2``, ``-3``, ... appended, as run folders do.
    """
    if not _NAME.match(tag):
        raise ValueError(f"tag must be letters, digits, '_' or '-', got {tag!r}")
    when = when or datetime.now(UTC)
    base = Path(out_dir) / f"{when:%Y%m%d-%H%M%S}-{dataset.meta['teacher']['kind']}-{tag}"
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f"{base.name}-{n}")
    dataset.meta.update(
        {
            "dataset_id": path.name,
            "tag": tag,
            "created_at": when.isoformat(timespec="seconds"),
            "neurodrive_version": neurodrive.__version__,
            **(provenance or {}),
        }
    )
    return dataset.save(path)


def _is_dataset_folder(path: Path) -> bool:
    return (path / META_FILE).is_file() and (path / DATASET_FILE).is_file()


def find_dataset(datasets_dir: str | Path, name: str) -> Path:
    """The dataset folder ``name`` refers to in ``datasets_dir``.

    1. If a dataset folder is called exactly ``name``, that is the one (a dataset id).
    2. Otherwise ``name`` is a selector ``<teacher>-<tag>``: among the folders named
       ``<YYYYMMDD-HHMMSS>-<teacher>-<tag>``, or with ``-2``, ``-3``, ... after it (two recordings
       in the same second), the newest one: latest time first, then highest suffix.
    3. Nothing matches: ``FileNotFoundError`` listing the datasets that are there.

    Example: ``expert-default`` with folders ``20261104-084617-expert-default`` and
    ``20261105-090000-expert-default`` gives the second.
    """
    if not _NAME.match(name):
        raise ValueError(f"a dataset name must be letters, digits, '_' or '-', got {name!r}")
    folder = Path(datasets_dir)
    exact = folder / name
    if _is_dataset_folder(exact):
        return exact
    pattern = re.compile(rf"^(\d{{8}}-\d{{6}})-{re.escape(name)}(?:-(\d+))?$")
    found: list[tuple[str, int, Path]] = []
    present: list[str] = []
    if folder.is_dir():
        for path in folder.iterdir():
            if not _is_dataset_folder(path):
                continue
            present.append(path.name)
            match = pattern.match(path.name)
            if match:
                found.append((match.group(1), int(match.group(2) or 1), path))
    if not found:
        listed = ", ".join(sorted(present)) or "none"
        raise FileNotFoundError(
            f"no dataset {name!r} in {folder.name}/ (a dataset id, or <teacher>-<tag> for the "
            f"newest such dataset); datasets there: {listed}"
        )
    return max(found)[2]


def dataset_sha256(folder: str | Path) -> str:
    """The SHA-256 checksum of ``dataset.npz`` in ``folder``, as 64 hexadecimal digits.

    Any change to any frame changes it, so it tells whether a dataset is still the one a run
    learned from.
    """
    digest = hashlib.sha256()
    with open(Path(folder) / DATASET_FILE, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()
