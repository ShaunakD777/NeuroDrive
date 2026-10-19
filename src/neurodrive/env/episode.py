"""Episode runner and recording (recording v1, ARCHITECTURE §6.11).

``run_episode`` plays a policy in a freshly reset :class:`World` until every car has stopped
(crashed, finished, stalled or timed out) and writes down what happened as an
:class:`EpisodeRecording`. A recording holds the *start* of the episode and every *action*, so
it can be replayed: feeding the same actions into a world reset to the same start reproduces the
same trajectory exactly (the world has no hidden randomness).

Layout of a recording with T steps and N cars. State arrays have T + 1 frames, because frame t
is the state before action t and frame T is the final state; actions have T entries:

* ``x, y, psi, v, progress`` (T+1, N); ``alive`` (T+1, N) booleans;
* ``rays`` (T+1, N, 5) and ``observation`` (T+1, N, 7) as the policy saw them;
* ``action`` (T, N, 2): steer and throttle after clipping; action t takes frame t to frame t+1;
* ``start_*`` (N,): the arguments that recreate the start, and ``status`` (N,): why each car's
  episode ended (see :class:`neurodrive.env.world.Status`).

Metadata (track name, seed, dt, policy reference, world rules, car parameters and the physics /
observation / recording versions) is stored as JSON text inside the same ``.npz`` file.
"""

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurodrive.env.car import CarParams
from neurodrive.env.policies import Policy
from neurodrive.env.sensors import edge_grid, observe
from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig

Array = NDArray[np.float64]

PHYSICS_VERSION = 1
OBSERVATION_VERSION = 1
RECORDING_VERSION = 1

_ARRAY_FIELDS = (
    "start_s0",
    "start_lateral",
    "start_heading_offset",
    "start_speed",
    "x",
    "y",
    "psi",
    "v",
    "alive",
    "progress",
    "rays",
    "observation",
    "action",
    "status",
)


@dataclass
class EpisodeRecording:
    """Everything needed to inspect and replay one episode (see the module docstring)."""

    meta: dict[str, Any]
    start_s0: Array
    start_lateral: Array
    start_heading_offset: Array
    start_speed: Array
    x: Array
    y: Array
    psi: Array
    v: Array
    alive: NDArray[np.bool_]
    progress: Array
    rays: Array
    observation: Array
    action: Array
    status: NDArray[np.int8]

    @property
    def n_steps(self) -> int:
        """Number of steps T taken."""
        return int(self.action.shape[0])

    @property
    def n_cars(self) -> int:
        """Number of cars N."""
        return int(self.x.shape[1])

    def save(self, path: str | Path) -> Path:
        """Write the recording to an ``.npz`` file (the suffix is added if missing)."""
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_name(path.name + ".npz")
        arrays = {name: getattr(self, name) for name in _ARRAY_FIELDS}
        np.savez(path, meta=np.array(json.dumps(self.meta, sort_keys=True)), **arrays)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeRecording":
        """Read a recording written by :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("recording_version") != RECORDING_VERSION:
                raise ValueError(
                    f"unsupported recording version {meta.get('recording_version')!r}; "
                    f"expected {RECORDING_VERSION}"
                )
            return cls(meta=meta, **{name: data[name] for name in _ARRAY_FIELDS})

    def identical_to(self, other: "EpisodeRecording") -> bool:
        """True when metadata and every array match exactly, bit for bit."""
        return self.meta == other.meta and all(
            np.array_equal(getattr(self, name), getattr(other, name)) for name in _ARRAY_FIELDS
        )


def run_episode(
    world: World,
    policy: Policy,
    *,
    seed: int | None = None,
    policy_ref: dict[str, Any] | None = None,
    max_steps: int | None = None,
) -> EpisodeRecording:
    """Play ``policy`` in ``world`` until no car is running (or ``max_steps``) and record it.

    ``world`` must have just been reset (zero steps taken). Each step: build every car's
    observation, record the frame, ask the policy for actions of shape (N, 2), clip them to
    [-1, 1], record them and step the world. ``seed`` and ``policy_ref`` are only written into
    the metadata (the seed used for random starts, and a description of the policy).
    """
    if not world.ready or world.step_count != 0:
        raise ValueError("run_episode needs a world that has just been reset")
    segments = edge_grid(world.track)  # only nearby walls: same readings, faster

    frames: dict[str, list[Array]] = {k: [] for k in ("x", "y", "psi", "v", "alive", "progress")}
    rays_log: list[Array] = []
    obs_log: list[Array] = []
    actions: list[Array] = []

    while True:
        cars = world.cars
        obs = observe(world, segments)
        rays_log.append(obs[:, :5].copy())  # the rays are the first five observation entries
        obs_log.append(obs)
        for name, value in (
            ("x", cars.x),
            ("y", cars.y),
            ("psi", cars.psi),
            ("v", cars.v),
            ("alive", cars.alive),
            ("progress", world.progress),
        ):
            frames[name].append(value.copy())
        if world.done or (max_steps is not None and world.step_count >= max_steps):
            break

        action = np.clip(np.asarray(policy.act(obs), dtype=np.float64), -1.0, 1.0)
        if action.shape != (world.n, 2):
            raise ValueError(
                f"policy must return actions of shape ({world.n}, 2), got {action.shape}"
            )
        actions.append(action)
        world.step(action[:, 0], action[:, 1])

    meta = {
        "recording_version": RECORDING_VERSION,
        "physics_version": PHYSICS_VERSION,
        "observation_version": OBSERVATION_VERSION,
        "track": world.track.name,
        "seed": seed,
        "dt": world.dt,
        "policy": policy_ref if policy_ref is not None else {"kind": type(policy).__name__},
        "world_config": asdict(world.config),
        "car_params": asdict(world.params),
    }
    return EpisodeRecording(
        meta=meta,
        start_s0=world.start["s0"],
        start_lateral=world.start["lateral"],
        start_heading_offset=world.start["heading_offset"],
        start_speed=world.start["speed"],
        x=np.stack(frames["x"]),
        y=np.stack(frames["y"]),
        psi=np.stack(frames["psi"]),
        v=np.stack(frames["v"]),
        alive=np.stack(frames["alive"]),
        progress=np.stack(frames["progress"]),
        rays=np.stack(rays_log),
        observation=np.stack(obs_log),
        action=np.stack(actions) if actions else np.zeros((0, world.n, 2)),
        status=world.status.copy(),
    )


class _ReplayPolicy:
    """Hands back the recorded actions one step at a time, ignoring what the cars see."""

    def __init__(self, actions: Array) -> None:
        self._actions = actions
        self._step = 0

    def act(self, observations: Array) -> Array:
        action = self._actions[self._step]
        self._step += 1
        return action


def replay(recording: EpisodeRecording, track: Track) -> EpisodeRecording:
    """Re-run a recording's actions from its start and return the resulting new recording.

    The world is rebuilt from the recording's stored rules, car parameters and time step, reset
    to the recorded start, and driven by the recorded actions. If the code is deterministic the
    result is identical to the original (``replay(r, track).identical_to(r)`` is True).
    """
    if track.name != recording.meta["track"]:
        raise ValueError(
            f"recording was made on track {recording.meta['track']!r}, got {track.name!r}"
        )
    world = World(
        track,
        WorldConfig(**recording.meta["world_config"]),
        CarParams(**recording.meta["car_params"]),
        recording.meta["dt"],
    )
    world.reset(
        s0=recording.start_s0,
        lateral=recording.start_lateral,
        heading_offset=recording.start_heading_offset,
        speed=recording.start_speed,
    )
    return run_episode(
        world,
        _ReplayPolicy(recording.action),
        seed=recording.meta["seed"],
        policy_ref=recording.meta["policy"],
        max_steps=recording.n_steps,
    )
