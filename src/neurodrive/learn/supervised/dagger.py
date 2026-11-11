"""DAgger, Dataset Aggregation (task 6.6, ADR-049): the student drives, the teacher labels.

Behavioural cloning (``imitation.py``) learns only from the states the teacher visited; driving
on its own, the clone's small errors take it to states the teacher never showed, where its
guesses are worse, so the errors compound (Studies S5a, S5b). DAgger (Ross, Gordon and Bagnell,
2011) collects data where the *student* goes. With π* the scripted expert (it reads the
privileged world state, so it can label any state), π̂_i the network after round i and D the
aggregated training set:

* **round 1**: D = the demonstration dataset's training frames D_0; train E epochs. This is
  behavioural cloning, epoch for epoch the imitation learner with the same seed;
* **round i ≥ 2**: drive π̂_(i−1) from R fresh seeded random starts (:func:`collect`); at every
  step record each running car's observation o_t and ask the expert for its label
  y*_t = π*(state_t); the network's actions drive the cars, never the labels. These M_i frames
  form D_i; D ← D ∪ D_i (added, never swapped in); train E more epochs on all of D, continuing
  from the current network and the optimizer's state (ADR-049, D2A).

In the paper the round-i driver is a mix, the expert with probability β_i; here β_1 = 1 (round 1
is the expert's own data) and β_i = 0 afterwards, the usual practical choice.

**The budget.** A collected frame follows the dataset rule (ADR-044): car c at step t while it
still drives, so a car that drove n_c steps gives n_c frames and M_i = n_1 + ... + n_R. Each is
one environment step (the network drove it) and one **teacher query** (the expert labelled
it). Two cars, one crashing after 3 steps and one driving 5: M = 8, so the round reports
env_steps 8, teacher_queries 8, demo_frames 0. Round 1 reports the dataset's F demonstration
frames (as imitation) and no steps or queries.

**The on-policy error** of a round's drive, before training on it (S5b's L_on), over its M
frames and both outputs k: L_on = (1 / (2M)) Σ_t Σ_k (ŷ_t,k − y*_t,k)², ŷ_t what the network
did. Actions (0, 0.5), (0.1, 0.9), (0, 1) against labels (0, 0.4), (0.1, 0.3), (−0.2, −0.4):
2.37 / 6 = 0.395. The validation loss is still measured on the expert's held-back episodes.

The learner is resumable: its state is imitation's (network, optimizer, batch generator,
epochs) plus the collection generator, the round count and every collected frame.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurodrive.core.nn.network import Network
from neurodrive.core.rng import Rng, child_rng
from neurodrive.env.car import ACTION_NAMES, ACTION_VERSION
from neurodrive.env.episode import PHYSICS_VERSION, run_episode
from neurodrive.env.experts import DEFAULT_EXPERT, ExpertParams, ScriptedExpert
from neurodrive.env.policies import MLPPolicy, Policy
from neurodrive.env.sensors import OBSERVATION_NAMES, OBSERVATION_SIZE, OBSERVATION_VERSION
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig
from neurodrive.learn.base import (
    FeedbackKind,
    LearnerState,
    UnitResult,
    generator_state,
    set_generator_state,
)
from neurodrive.learn.supervised.demos import (
    ARRAY_FIELDS,
    DATASET_VERSION,
    RANDOM_START_HEADING,
    RANDOM_START_LATERAL,
    DemoDataset,
    driven_steps,
    frames_from_recording,
    straight_share,
)
from neurodrive.learn.supervised.imitation import ImitationLearner, ImitationSettings

Array = NDArray[np.float64]

# Metrics of a round's drive; null in round 1, which drives nothing.
COLLECT_METRICS = (
    "onpolicy_loss",
    "onpolicy_loss_steer",
    "onpolicy_loss_throttle",
    "collect_frames",
    "collect_finished",
    "collect_crashed",
)


@dataclass(frozen=True)
class DaggerSettings:
    """Behavioural cloning's settings plus DAgger's own.

    ``epochs_per_round`` E, ``random_starts`` R (cars per collection drive), ``target_laps``
    (each collection drive's goal, with the per-track time limit for it), ``start_lateral_m``
    a and ``start_heading_rad`` h (each collection start is moved sideways by up to ±a and
    turned by up to ±h; the defaults are the dataset's random-start recipe, ±1 m and ±0.1 rad;
    task 6.6 follow-up, ADR-050) and ``expert`` (the labelling teacher's settings).
    """

    imitation: ImitationSettings = field(default_factory=ImitationSettings)
    epochs_per_round: int = 30
    random_starts: int = 10
    target_laps: int = 2
    start_lateral_m: float = RANDOM_START_LATERAL
    start_heading_rad: float = RANDOM_START_HEADING
    expert: ExpertParams = DEFAULT_EXPERT

    def __post_init__(self) -> None:
        for name in ("epochs_per_round", "random_starts", "target_laps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a whole number >= 1, got {value!r}")
        for name in ("start_lateral_m", "start_heading_rad"):
            if not getattr(self, name) >= 0.0:
                raise ValueError(f"{name} must be 0 or more, got {getattr(self, name)!r}")


class ExpertLabelled:
    """A policy that drives with ``policy`` and, before every step, also records the expert's
    label for the state every car is in. The actions are the wrapped policy's, unchanged, so
    the drive is the plain drive (the labels never steer)."""

    def __init__(self, policy: Policy, expert: ScriptedExpert) -> None:
        self.policy = policy
        self.expert = expert
        self.labels: list[Array] = []

    def act(self, observations: Array) -> Array:
        self.labels.append(np.clip(self.expert.label(), -1.0, 1.0))
        return self.policy.act(observations)


def onpolicy_errors(action: Array, label: Array) -> dict[str, float | None]:
    """L_on and its steer and throttle parts for (M, 2) arrays (None for every part when M = 0).

    Example: actions (0, 0.5), (0.1, 0.9), (0, 1) against labels (0, 0.4), (0.1, 0.3),
    (−0.2, −0.4) give L_on = 2.37 / 6 = 0.395, steer 0.04 / 3, throttle 2.33 / 3.
    """
    sq = (np.asarray(action, dtype=np.float64) - np.asarray(label, dtype=np.float64)) ** 2
    if sq.shape[0] == 0:
        return {"onpolicy_loss": None, "onpolicy_loss_steer": None, "onpolicy_loss_throttle": None}
    return {
        "onpolicy_loss": float(np.mean(sq)),
        "onpolicy_loss_steer": float(np.mean(sq[:, 0])),
        "onpolicy_loss_throttle": float(np.mean(sq[:, 1])),
    }


@dataclass
class Collection:
    """One collection drive: its frames (the dataset arrays, labelled by the expert), what the
    network did in each (``action``, (M, 2)) and one record per car (episode)."""

    frames: dict[str, np.ndarray]
    action: Array
    episodes: list[dict[str, Any]]

    @property
    def n_frames(self) -> int:
        """M, the frames collected: the round's environment steps and teacher queries."""
        return int(self.action.shape[0])

    def metrics(self) -> dict[str, float | int | None]:
        """L_on and its parts, M, and the shares of the drives that finished and crashed."""
        statuses = [e["status"] for e in self.episodes]
        n = len(statuses)
        return {
            **onpolicy_errors(self.action, self.frames["label"]),
            "collect_frames": self.n_frames,
            "collect_finished": statuses.count("finished") / n,
            "collect_crashed": statuses.count("crashed") / n,
        }


def collect(
    network: Network,
    track: Track,
    rng: Rng,
    *,
    n_starts: int,
    target_laps: int,
    params: ExpertParams = DEFAULT_EXPERT,
    first_episode: int = 0,
    round_number: int | None = None,
    lateral_m: float = RANDOM_START_LATERAL,
    heading_rad: float = RANDOM_START_HEADING,
) -> Collection:
    """Drive ``network`` and keep every state it reached, labelled by the expert.

    1. One world with the per-track rules for ``target_laps``; ``n_starts`` cars at seeded
       random starts drawn from ``rng``: arc length uniform on the lap, sideways uniform on
       ±``lateral_m``, heading offset uniform on ±``heading_rad``, at rest (by default ±1 m and
       ±0.1 rad, the dataset's random-start recipe, ADR-044).
    2. Drive the network (:class:`ExpertLabelled` around an ``MLPPolicy``) until every car has
       stopped, recording the expert's label of every car's state at every step.
    3. Cut out each car's frames while it drove (:func:`frames_from_recording` with the labels;
       car c becomes episode ``first_episode + c``) and what the network did in them.
    """
    if track.split != "train":
        raise ValueError(
            f"{track.name!r} is a held-out track; DAgger may only drive training tracks "
            "(EXPERIMENT_PROTOCOL rule 7)"
        )
    world = World(track, WorldConfig.for_track(track, target_laps))
    world.reset(
        n=n_starts,
        rng=rng,
        random_s0=True,
        lateral_jitter=lateral_m,
        heading_jitter=heading_rad,
    )
    policy = ExpertLabelled(MLPPolicy(network), ScriptedExpert(world, params))
    recording = run_episode(world, policy, policy_ref={"kind": "dagger", "round": round_number})
    labels = np.array(policy.labels, dtype=np.float64).reshape(-1, world.n, 2)
    frames = frames_from_recording(recording, first_episode, labels=labels)
    steps = driven_steps(recording)
    action = np.concatenate([recording.action[:n, car] for car, n in enumerate(steps)])
    episodes = [
        {
            "index": first_episode + car,
            "start_kind": "dagger",
            "round": round_number,
            "start_s0_m": float(recording.start_s0[car]),
            "start_lateral_m": float(recording.start_lateral[car]),
            "start_heading_offset_rad": float(recording.start_heading_offset[car]),
            "start_speed_mps": float(recording.start_speed[car]),
            "status": Status(int(recording.status[car])).name.lower(),
            "frames": int(n),
        }
        for car, n in enumerate(steps)
    ]
    return Collection(frames=frames, action=action, episodes=episodes)


def _empty_frames() -> dict[str, np.ndarray]:
    shapes = {"observation": (0, OBSERVATION_SIZE), "label": (0, 2)}
    return {
        name: np.zeros(
            shapes.get(name, (0,)), dtype=np.int64 if name in ("episode", "step") else np.float64
        )
        for name in ARRAY_FIELDS
    }


class DaggerLearner(ImitationLearner):
    """DAgger as a learner (learner v1, resumable): one unit is one round (module docstring)."""

    name = "dagger"
    feedback_kind: FeedbackKind = "labels"

    def __init__(
        self, settings: DaggerSettings, dataset: DemoDataset, track: Track, rng: Rng
    ) -> None:
        """Behavioural cloning's set-up (split, network, optimizer, loader from the generators
        "split", "init" and "batches"), plus the collection track and the "collect" generator.
        """
        if track.split != "train":
            raise ValueError(
                f"{track.name!r} is a held-out track; DAgger may only drive training tracks "
                "(EXPERIMENT_PROTOCOL rule 7)"
            )
        super().__init__(settings.imitation, dataset, rng)
        self.dagger = settings
        self.track = track
        self.base_dataset = dataset.meta.get("dataset_id")
        self.collect_rng = child_rng(rng, "collect")
        self.base_x, self.base_y = self.x_train, self.y_train
        self.collected = _empty_frames()
        self.collected_action = np.zeros((0, 2))
        self.episodes: list[dict[str, Any]] = []
        self.rounds = 0

    @property
    def queries(self) -> int:
        """Teacher queries so far: every collected frame."""
        return int(self.collected_action.shape[0])

    def _use_collected(self) -> None:
        """Train on the expert's training frames followed by every collected frame."""
        self.set_training_frames(
            np.concatenate([self.base_x, self.collected["observation"]]),
            np.concatenate([self.base_y, self.collected["label"]]),
        )

    def _aggregate(self, found: Collection) -> None:
        """D ← D ∪ D_i: append the round's frames, never replace."""
        for name in ARRAY_FIELDS:
            self.collected[name] = np.concatenate([self.collected[name], found.frames[name]])
        self.collected_action = np.concatenate([self.collected_action, found.action])
        self.episodes.extend(found.episodes)
        self._use_collected()

    def train_unit(self) -> UnitResult:
        """One round: (from round 2) collect and aggregate, then E guarded epochs."""
        self.rounds += 1
        start_network = self.network
        metrics: dict[str, Any] = dict.fromkeys(COLLECT_METRICS)
        demo = steps = 0
        if self.rounds == 1:
            demo = self.demo_frames  # the dataset is read once, as by imitation
        else:
            found = collect(
                self.network,
                self.track,
                self.collect_rng,
                n_starts=self.dagger.random_starts,
                target_laps=self.dagger.target_laps,
                lateral_m=self.dagger.start_lateral_m,
                heading_rad=self.dagger.start_heading_rad,
                params=self.dagger.expert,
                first_episode=len(self.episodes),
                round_number=self.rounds,
            )
            self._aggregate(found)
            steps = found.n_frames
            metrics.update(found.metrics())
        first: dict | None = None
        last: dict | None = None
        for _ in range(self.dagger.epochs_per_round):
            self.epochs += 1
            last = self._epoch()
            if last is None:
                self.network = start_network  # the network of the last completed round
                metrics.update({"loss": None, "loss_before": None})
                return UnitResult(
                    metrics=metrics,
                    env_steps=steps,
                    demo_frames=demo,
                    teacher_queries=steps,
                    diverged=True,
                )
            first = first or last
        assert first is not None and last is not None
        metrics.update(last)
        metrics["loss_before"] = first["loss_before"]  # the round's data before its training
        metrics["grad_norm"] = first["grad_norm"]
        metrics["train_frames"] = int(self.x_train.shape[0])
        metrics["epochs"] = self.epochs
        return UnitResult(metrics=metrics, env_steps=steps, demo_frames=demo, teacher_queries=steps)

    def state(self) -> LearnerState:
        state = super().state()
        state.arrays.update({f"collected_{k}": v for k, v in self.collected.items()})
        state.arrays["collected_action"] = self.collected_action
        state.info.update(
            {
                "rounds": self.rounds,
                "episodes": self.episodes,
                "collect_rng": generator_state(self.collect_rng),
            }
        )
        return state

    def restore(self, state: LearnerState) -> None:
        self.collected = {k: state.arrays[f"collected_{k}"] for k in ARRAY_FIELDS}
        self.collected_action = state.arrays["collected_action"]
        self.episodes = list(state.info["episodes"])
        self.rounds = state.info["rounds"]
        set_generator_state(self.collect_rng, state.info["collect_rng"])
        self._use_collected()  # the loader over the aggregated frames, before its state is set
        super().restore(state)


def collected_dataset(learner: DaggerLearner) -> DemoDataset | None:
    """The frames the learner collected as a dataset v1 with teacher kind ``dagger`` (the
    network drove, the expert labelled), or None if it collected none (ADR-049, D4A).

    Its ``demo_frames``, ``env_steps`` and ``teacher_queries`` all equal the frame count. The
    metadata has no ``created_at`` or provenance: ``save_dataset`` adds them (the runner also
    passes the run id).
    """
    if learner.queries == 0:
        return None
    world = World(learner.track, WorldConfig.for_track(learner.track, learner.dagger.target_laps))
    frames = learner.queries
    meta = {
        "dataset_version": DATASET_VERSION,
        "teacher": {
            "kind": "dagger",
            "labeller": "expert",
            "params": asdict(learner.dagger.expert),
        },
        "track": learner.track.name,
        "track_split": learner.track.split,
        "starts": {
            "random_starts": learner.dagger.random_starts,
            "per_round": True,
            "rounds_collected": learner.rounds - 1,
            "random_lateral_m": learner.dagger.start_lateral_m,
            "random_heading_rad": learner.dagger.start_heading_rad,
        },
        "dagger": {
            "base_dataset": learner.base_dataset,
            "epochs_per_round": learner.dagger.epochs_per_round,
            "rounds": learner.rounds,
        },
        "episodes": learner.episodes,
        "demo_frames": frames,
        "env_steps": frames,
        "teacher_queries": frames,
        "straight_share": straight_share(learner.collected["label"]),
        "observation_version": OBSERVATION_VERSION,
        "observation_names": list(OBSERVATION_NAMES),
        "action_version": ACTION_VERSION,
        "action_names": list(ACTION_NAMES),
        "physics_version": PHYSICS_VERSION,
        "dt": world.dt,
        "world_config": asdict(world.config),
        "car_params": asdict(world.params),
    }
    return DemoDataset(meta=meta, **{k: v.copy() for k, v in learner.collected.items()})
