"""Live sessions: a simulation that runs on a clock while the browser sends commands (api v1.2).

``LiveSession`` is the whole live-driving logic with no sockets in it, which makes it easy to
test. It is used like this: feed it each text message the browser sends (``handle_text``) and call
``tick`` regularly with the current time; both return the messages to send back. The WebSocket
route in ``app.py`` only moves text between the socket and these two methods.

How time works. The physics always advances in fixed steps of ``dt`` = 1/30 s. The *speed factor*
changes how many steps are taken per second of real time, never their size: at speed 2 the world
needs 2 * 30 = 60 steps per real second. ``tick`` adds ``elapsed * 30 * speed`` to a running tally
of steps that are *due*. Once at least ``max(1, speed)`` steps are due it runs all the whole steps
and sends one frame, keeping the fraction. So a frame covers one step at speeds up to 1 and
``speed`` steps above it: frames per second = min(30, 30 * speed) on average, never more than 30,
and the pace stays smooth even when the timer ticks unevenly (Windows timers tick every ~15 ms).
Real time is passed in, never read, so tests can drive the clock by hand.
"""

import json
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from neurodrive.core.rng import make_rng
from neurodrive.env.policies import ConstantPolicy, KeyboardPolicy, NeuronPolicy, Policy
from neurodrive.env.sensors import RAY_ANGLES, RAY_RANGE, build_observation, edge_segments
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World
from neurodrive.server.schemas import (
    COMMAND_ADAPTER,
    CarColumns,
    EpisodeEndMessage,
    ErrorMessage,
    FrameMessage,
    KeyState,
    LoadScenario,
    NeuronReadout,
    Pause,
    Play,
    Reset,
    ScenarioMessage,
    SelectCar,
    SelectedCar,
    SetParam,
    SetSpeed,
    Step,
    TrackGeometry,
    TrackInfo,
    WireModel,
)

MAX_CATCH_UP = 0.25  # real seconds of simulation owed after a stall; the rest is dropped
CONSTANT_THROTTLE = 0.3  # throttle of the "constant" policy
RANDOM_START_LATERAL = 1.0  # metres: random-start sideways range for seeded scenarios
RANDOM_START_HEADING = 0.1  # radians: random-start heading range for seeded scenarios


class TrackLibrary:
    """The tracks the server can load: every ``*.json`` file in one folder, by file name.

    Names are looked up in the listing of the folder, never turned into paths, so a request can
    only ever reach a file that is in that folder.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)
        self._cache: dict[str, Track] = {}

    def names(self) -> list[str]:
        """Names of all available tracks, sorted."""
        return sorted(p.stem for p in self.directory.glob("*.json"))

    def get(self, name: str) -> Track:
        """Load a track by name (cached). Raises ``KeyError`` for a name that is not available."""
        if name not in self.names():
            raise KeyError(name)
        if name not in self._cache:
            self._cache[name] = Track.from_json(self.directory / f"{name}.json")
        return self._cache[name]

    def info(self) -> list[TrackInfo]:
        """Name and split of every track."""
        # Only the "split" field is needed, so read it from the file instead of building the
        # whole track (a long track takes a moment to build).
        infos = []
        for n in self.names():
            data = json.loads((self.directory / f"{n}.json").read_text(encoding="utf-8"))
            infos.append(TrackInfo(name=n, split=data["split"]))
        return infos

    def geometry(self, name: str) -> TrackGeometry:
        """Everything the browser needs to draw a track (see :class:`TrackGeometry`)."""
        track = self.get(name)
        return TrackGeometry(
            name=track.name,
            split=track.split,  # type: ignore[arg-type]
            width=track.width,
            length=round(track.length, 3),
            start_s0=track.start_s0,
            centerline=np.round(track.points, 2).tolist(),
            left=np.round(track.left, 2).tolist(),
            right=np.round(track.right, 2).tolist(),
        )


class LiveSession:
    """One browser connection's simulation: a loaded scenario, a clock and a selected car."""

    def __init__(self, tracks: TrackLibrary) -> None:
        self.tracks = tracks
        self.playing = False
        self.speed = 1.0
        self.selected = 0
        self._world: World | None = None
        self._policy: Policy | None = None
        self._keys: KeyboardPolicy | None = None
        self._neuron: NeuronPolicy | None = None
        # The neuron driver's settings live for the whole session: Reset, a new track, more cars,
        # a random start or a detour through another driver all keep the same weights.
        self._saved_neuron: NeuronPolicy | None = None
        self._scenario: LoadScenario | None = None
        self._segments = np.zeros((0, 2, 2))
        self._action = np.zeros((0, 2))
        self._obs = np.zeros((0, 7))
        self._obs_step = -1
        self._due = 0.0
        self._last_tick: float | None = None

    # ── incoming messages ───────────────────────────────────────────────────────────────────

    def handle_text(self, text: str) -> list[WireModel]:
        """Parse one message from the browser and run it; return the messages to send back."""
        try:
            command = COMMAND_ADAPTER.validate_json(text)
        except ValidationError as exc:
            first = exc.errors()[0]
            where = ".".join(str(part) for part in first["loc"])
            return [ErrorMessage(message=f"invalid message ({where}): {first['msg']}")]
        return self.handle(command)

    def handle(self, command: Any) -> list[WireModel]:
        """Run one validated command and return the messages to send back."""
        if isinstance(command, LoadScenario):
            return self._load(command)
        if isinstance(command, KeyState):
            if self._keys is not None:  # ignored unless the keyboard is driving
                self._keys.set_input(command.steer, command.throttle)
            return []
        if self._world is None:
            return [ErrorMessage(message="no scenario loaded: send load_scenario first")]
        if isinstance(command, Play):
            if self._world.done:
                return [ErrorMessage(message="the episode has ended: send reset to start again")]
            self.playing = True
            self._restart_clock()
            return [self._frame()]
        if isinstance(command, Pause):
            self.playing = False
            return [self._frame()]
        if isinstance(command, SetSpeed):
            self.speed = command.speed
            return [self._frame()]
        if isinstance(command, SelectCar):
            if command.index >= self._world.n:
                return [ErrorMessage(message=f"no car {command.index}: there are {self._world.n}")]
            self.selected = command.index
            return [self._frame()]
        if isinstance(command, SetParam):
            if self._neuron is None:
                return [ErrorMessage(message="set_param needs the neuron driver (policy 'neuron')")]
            try:
                self._neuron.set_param(command.name, command.value)
            except ValueError as exc:
                return [ErrorMessage(message=str(exc))]
            return [self._frame()]
        if isinstance(command, Reset):
            self._reset_world()
            self._restart_clock()
            return [self._frame()]
        if isinstance(command, Step):
            if self._world.done:
                return [ErrorMessage(message="the episode has ended: send reset to start again")]
            self._advance(1)
            return self._frame_and_end()
        return [ErrorMessage(message=f"unsupported command {type(command).__name__}")]

    # ── the clock ───────────────────────────────────────────────────────────────────────────

    def tick(self, now: float) -> list[WireModel]:
        """Advance the simulation to real time ``now`` (seconds); return any messages to send.

        Does nothing unless playing. A frame is produced only once ``max(1, speed)`` steps are due,
        which keeps the frame rate at or below 30 per second whatever the speed.
        """
        if not self.playing or self._world is None:
            return []
        if self._last_tick is None:
            self._last_tick = now
            return []
        elapsed = min(max(now - self._last_tick, 0.0), MAX_CATCH_UP)
        self._last_tick = now
        self._due += elapsed * (1.0 / self._world.dt) * self.speed
        if self._due < max(1.0, self.speed):  # one frame per max(1, speed) steps: at most 30 fps
            return []
        steps = int(self._due)
        self._due -= steps
        self._advance(steps)
        return self._frame_and_end()

    # ── internals ───────────────────────────────────────────────────────────────────────────

    def _restart_clock(self) -> None:
        self._due = 0.0
        self._last_tick = None

    def _load(self, command: LoadScenario) -> list[WireModel]:
        try:
            track = self.tracks.get(command.track)
        except KeyError:
            return [ErrorMessage(message=f"unknown track {command.track!r}")]
        self._scenario = command
        self._world = World(track)
        self._segments = edge_segments(track)
        self._keys = KeyboardPolicy() if command.policy == "keyboard" else None
        if command.policy == "neuron":
            if self._saved_neuron is None:
                self._saved_neuron = NeuronPolicy()  # the first neuron of the session: all zeros
            self._neuron = self._saved_neuron
        else:
            self._neuron = None
        self._policy = self._keys or self._neuron or ConstantPolicy(0.0, CONSTANT_THROTTLE)
        self.playing = False
        self.selected = 0
        self._reset_world()
        self._restart_clock()
        world = self._world
        scenario = ScenarioMessage(
            track=track.name,
            policy=command.policy,
            n_cars=command.n_cars,
            seed=command.seed,
            dt=world.dt,
            time_limit=world.config.time_limit,
            target_laps=world.config.target_laps,
            v_max=world.params.v_max,
            wheelbase=world.params.wheelbase,
            car_half_width=world.params.half_width,
            ray_angles=[round(float(a), 6) for a in RAY_ANGLES],
            ray_range=RAY_RANGE,
        )
        return [scenario, self._frame()]

    def _reset_world(self) -> None:
        assert self._world is not None and self._scenario is not None
        scenario = self._scenario
        if scenario.seed is None:
            self._world.reset(n=scenario.n_cars)
        else:
            self._world.reset(
                n=scenario.n_cars,
                rng=make_rng(scenario.seed),
                random_s0=True,
                lateral_jitter=RANDOM_START_LATERAL,
                heading_jitter=RANDOM_START_HEADING,
            )
        self._action = np.zeros((scenario.n_cars, 2))
        self._obs_step = -1

    def _observe(self) -> np.ndarray:
        """The (N, 7) observation of every car now (computed once per step)."""
        world = self._world
        assert world is not None
        if self._obs_step != world.step_count:
            cars = world.cars
            self._obs = build_observation(
                self._segments, cars.x, cars.y, cars.psi, cars.v, world.road_heading,
                world.params.v_max,
            )  # fmt: skip
            self._obs_step = world.step_count
        return self._obs

    def _advance(self, steps: int) -> None:
        """Run up to ``steps`` physics steps: observe, let the policy act, step the world."""
        world, policy = self._world, self._policy
        assert world is not None and policy is not None
        for _ in range(steps):
            if world.done:
                break
            action = np.clip(np.asarray(policy.act(self._observe()), dtype=np.float64), -1.0, 1.0)
            world.step(action[:, 0], action[:, 1])
            self._action = action

    def _frame(self) -> FrameMessage:
        world = self._world
        assert world is not None
        cars = world.cars
        obs = self._observe()
        i = self.selected
        return FrameMessage(
            step=world.step_count,
            time=round(world.time, 4),
            playing=self.playing,
            speed=self.speed,
            cars=CarColumns(
                x=np.round(cars.x, 3).tolist(),
                y=np.round(cars.y, 3).tolist(),
                psi=np.round(cars.psi, 4).tolist(),
                v=np.round(cars.v, 3).tolist(),
                alive=cars.alive.tolist(),
                progress=np.round(world.progress, 3).tolist(),
                status=world.status.astype(int).tolist(),
                action=np.round(self._action, 3).tolist(),
            ),
            selected=SelectedCar(
                index=i,
                rays=np.round(obs[i, :5] * RAY_RANGE, 3).tolist(),
                observation=np.round(obs[i], 4).tolist(),
                neuron=self._neuron_readout(obs[i]),
            ),
        )

    def _neuron_readout(self, observation: np.ndarray) -> NeuronReadout | None:
        """The neuron driver's numbers for one car (None for the other drivers)."""
        if self._neuron is None:
            return None
        r = self._neuron.readout(observation)
        return NeuronReadout(
            inputs=[round(v, 4) for v in r.inputs],
            input_rays=list(r.input_rays),
            weights=list(r.weights),
            bias=r.bias,
            contributions=[round(v, 4) for v in r.contributions],
            z=round(r.z, 4),
            output=round(r.output, 4),
            activation=r.activation,  # type: ignore[arg-type]
            throttle=r.throttle,
        )

    def _frame_and_end(self) -> list[WireModel]:
        """A frame, followed by ``episode_end`` (and a pause) if every car has now stopped."""
        world = self._world
        assert world is not None
        if world.done:
            self.playing = False  # so the frame already says "paused"
        messages: list[WireModel] = [self._frame()]
        if world.done:
            messages.append(
                EpisodeEndMessage(
                    step=world.step_count,
                    time=round(world.time, 4),
                    status=[Status(int(s)).name.lower() for s in world.status],
                    progress=np.round(world.progress, 3).tolist(),
                    laps=world.laps.astype(int).tolist(),
                )
            )
        return messages
