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

from neurodrive.core.nn.forward import ForwardTrace, forward
from neurodrive.core.nn.init import init_population
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import child_rng, make_rng
from neurodrive.env.car import CarState, step_cars
from neurodrive.env.policies import (
    ConstantPolicy,
    KeyboardPolicy,
    MLPPolicy,
    NeuronPolicy,
    Policy,
    PopulationPolicy,
)
from neurodrive.env.sensors import RAY_ANGLES, RAY_RANGE, edge_grid, observe
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig
from neurodrive.explain.contributions import neuron_contributions
from neurodrive.explain.sensitivity import sensitivity
from neurodrive.server.schemas import (
    COMMAND_ADAPTER,
    CarColumns,
    EpisodeEndMessage,
    ErrorMessage,
    Explanation,
    ExplanationMessage,
    FrameMessage,
    GhostPaths,
    InspectStep,
    KeyState,
    LoadScenario,
    MomentMessage,
    NetworkMessage,
    NetworkTrace,
    NeuronReadout,
    OutputReceipt,
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
    TrailPoints,
    WhatIf,
    WireModel,
)

MAX_CATCH_UP = 0.25  # real seconds of simulation owed after a stall; the rest is dropped
CONSTANT_THROTTLE = 0.3  # throttle of the "constant" policy
RANDOM_START_LATERAL = 1.0  # metres: random-start sideways range for seeded scenarios
RANDOM_START_HEADING = 0.1  # radians: random-start heading range for seeded scenarios
GHOST_SECONDS = 2.0  # how far ahead the what-if ghost car is drawn (slow cars need time to part)
TRAIL_SECONDS = 60.0  # how much of the drive the session remembers for the decision trail (ADR-027)


class DriveMemory:
    """The last ``capacity`` steps of a live drive, for every car (ADR-027).

    Before each physics step t the session calls :meth:`record` with every car's state at step t:
    pose (x, y, psi), speed v, cumulative progress, status code (0 = still running),
    observation o_t (7 numbers), the action a_t chosen from it, and the time. Step t lives in
    row ``t % capacity`` of fixed-size arrays, so the oldest step is overwritten once
    ``capacity`` steps are stored and memory never grows (60 s of 100 cars is about 20 MB, ADR-028
    added progress and status). The observation is stored at full precision, so a
    past forward pass recomputed from it is bit-identical to the one made live.
    """

    def __init__(self, capacity: int, n_cars: int) -> None:
        self.capacity = capacity
        self.x = np.zeros((capacity, n_cars))
        self.y = np.zeros((capacity, n_cars))
        self.psi = np.zeros((capacity, n_cars))
        self.v = np.zeros((capacity, n_cars))
        self.progress = np.zeros((capacity, n_cars))
        self.status = np.zeros((capacity, n_cars), dtype=np.int8)
        self.observation = np.zeros((capacity, n_cars, 7))
        self.action = np.zeros((capacity, n_cars, 2))
        self.time = np.zeros(capacity)
        self.end = 0  # one past the last recorded step

    @property
    def first(self) -> int:
        """The oldest step still remembered."""
        return max(0, self.end - self.capacity)

    def has(self, step: int) -> bool:
        return self.first <= step < self.end

    def record(self, world: World, observation: np.ndarray, action: np.ndarray) -> None:
        """Store step ``world.step_count``: the state before the action, and the action."""
        t = world.step_count
        assert t == self.end, "steps must be recorded in order"
        row, cars = t % self.capacity, world.cars
        self.x[row], self.y[row], self.psi[row], self.v[row] = cars.x, cars.y, cars.psi, cars.v
        self.observation[row] = observation
        self.action[row] = action
        self.progress[row] = world.progress
        self.status[row] = world.status
        self.time[row] = world.time
        self.end = t + 1

    def trail(self, car: int, start: int, *, replace: bool) -> TrailPoints:
        """Car ``car``'s remembered steps from ``start`` on (only those where it was running)."""
        steps = np.arange(max(start, self.first), self.end)
        rows = steps % self.capacity
        keep = self.status[rows, car] == Status.RUNNING
        steps, rows = steps[keep], rows[keep]
        return TrailPoints(
            replace=replace,
            steps=steps.tolist(),
            x=np.round(self.x[rows, car], 3).tolist(),
            y=np.round(self.y[rows, car], 3).tolist(),
            action=np.round(self.action[rows, car], 3).tolist(),
        )


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
        self._memory = DriveMemory(1, 0)
        self._trail_sent = 0  # the next step whose trail point the browser has not been sent
        self._trail_full = True  # send the whole trail with the next frame (new car or new drive)

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
            self._trail_full = True  # the new car's whole trail goes with the next frame
            return [*self._network_messages(), self._frame()]
        if isinstance(command, SetParam):
            if self._neuron is None:
                return [ErrorMessage(message="set_param needs the neuron driver (policy 'neuron')")]
            try:
                self._neuron.set_param(command.name, command.value)
            except ValueError as exc:
                return [ErrorMessage(message=str(exc))]
            return [self._frame()]
        if isinstance(command, WhatIf):
            return self._what_if(command)
        if isinstance(command, InspectStep):
            return self._inspect(command)
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
        self._world = World(track, WorldConfig.for_track(track))  # per-track time limit (ADR-022)
        self._segments = edge_grid(track)
        self._keys = KeyboardPolicy() if command.policy == "keyboard" else None
        if command.policy == "neuron":
            if self._saved_neuron is None:
                self._saved_neuron = NeuronPolicy()  # the first neuron of the session: all zeros
            self._neuron = self._saved_neuron
        else:
            self._neuron = None
        if command.policy == "random_networks":
            # One Xavier-initialised network per car, drawn from the seed's "init" stream (seed 0
            # when none is given). Built once per load, so Reset replays the same brains.
            rng = child_rng(make_rng(command.seed or 0), "init")
            self._policy = PopulationPolicy(init_population(NetworkSpec(), command.n_cars, rng))
        else:
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
            trail_seconds=TRAIL_SECONDS,
        )
        return [scenario, *self._network_messages(), self._frame()]

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
        capacity = round(TRAIL_SECONDS / self._world.dt)
        self._memory = DriveMemory(capacity, scenario.n_cars)  # a new drive: forget the old one
        self._trail_sent = 0
        self._trail_full = True

    def _observe(self) -> np.ndarray:
        """The (N, 7) observation of every car now (computed once per step)."""
        world = self._world
        assert world is not None
        if self._obs_step != world.step_count:
            self._obs = observe(world, self._segments)
            self._obs_step = world.step_count
        return self._obs

    def _advance(self, steps: int) -> None:
        """Run up to ``steps`` physics steps: observe, let the policy act, step the world."""
        world, policy = self._world, self._policy
        assert world is not None and policy is not None
        for _ in range(steps):
            if world.done:
                break
            obs = self._observe()
            action = np.clip(np.asarray(policy.act(obs), dtype=np.float64), -1.0, 1.0)
            self._memory.record(world, obs, action)  # step t: the state, then what was decided
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
                trace=self._network_trace(obs, i),
                explain=self._explain(obs, i),
                trail=self._trail_update(),
            ),
        )

    def _trail_update(self) -> TrailPoints:
        """The selected car's trail points the browser does not have yet (all of them after a
        load, a reset or a car change; otherwise the steps taken since the last frame)."""
        replace = self._trail_full
        start = self._memory.first if replace else self._trail_sent
        points = self._memory.trail(self.selected, start, replace=replace)
        self._trail_sent = self._memory.end
        self._trail_full = False
        return points

    def _inspect(self, command: InspectStep) -> list[WireModel]:
        """The selected car at a remembered step, recomputed from the stored observation."""
        if self.playing:
            return [ErrorMessage(message="inspect_step works while paused: send pause first")]
        memory, t = self._memory, command.step
        if not memory.has(t):
            span = "nothing yet" if memory.end == 0 else f"steps {memory.first} to {memory.end - 1}"
            return [ErrorMessage(message=f"step {t} is not remembered ({span})")]
        row, car = t % memory.capacity, self.selected
        observations = memory.observation[row]  # every car's observation at step t, full precision
        obs = observations[car]
        return [
            MomentMessage(
                car=car,
                step=t,
                time=round(float(memory.time[row]), 4),
                x=round(float(memory.x[row, car]), 3),
                y=round(float(memory.y[row, car]), 3),
                psi=round(float(memory.psi[row, car]), 4),
                v=round(float(memory.v[row, car]), 3),
                progress=round(float(memory.progress[row, car]), 3),
                status=int(memory.status[row, car]),
                rays=np.round(obs[:5] * RAY_RANGE, 3).tolist(),
                observation=np.round(obs, 4).tolist(),
                action=np.round(memory.action[row, car], 3).tolist(),
                # the same functions a live frame uses, on the same numbers: the same trace
                trace=self._network_trace(observations, car),
                explain=self._explain(observations, car),
            )
        ]

    def _network_messages(self) -> list[WireModel]:
        """The selected car's network as a ``network`` message (none for non-network drivers)."""
        policy = self._policy
        if not isinstance(policy, MLPPolicy | PopulationPolicy):
            return []
        net = policy.network_of(self.selected)
        spec = net.spec
        return [
            NetworkMessage(
                car=self.selected,
                layer_sizes=list(spec.layer_sizes),
                activations=[spec.activation_name(i) for i in range(spec.n_layers)],  # type: ignore[misc]
                weights=[np.round(w, 4).tolist() for w in net.weights],
                biases=[np.round(b, 4).tolist() for b in net.biases],
            )
        ]

    def _network_trace(self, observations: np.ndarray, car: int) -> NetworkTrace | None:
        """Every value of the selected car's forward pass on its current observation."""
        policy = self._policy
        if not isinstance(policy, MLPPolicy | PopulationPolicy):
            return None
        return _wire_trace(policy.trace_of(observations, car))

    def _explain(self, observations: np.ndarray, car: int) -> Explanation | None:
        """Contributions and sensitivity of the selected car's two outputs, right now."""
        policy = self._policy
        if not isinstance(policy, MLPPolicy | PopulationPolicy):
            return None
        network = policy.network_of(car)
        return _explanation(network, policy.trace_of(observations, car), observations[car])

    def _what_if(self, command: WhatIf) -> list[WireModel]:
        """The selected car's network on a changed observation; the simulation is untouched."""
        policy, world = self._policy, self._world
        assert world is not None
        if not isinstance(policy, MLPPolicy | PopulationPolicy):
            return [
                ErrorMessage(message="what_if needs a network driver (policy 'random_networks')")
            ]
        if self.playing:
            return [ErrorMessage(message="what_if works while paused: send pause first")]
        car = self.selected
        real = self._observe()[car]
        tried = real.copy()
        for k, value in command.overrides.items():
            if not 0 <= k < len(real):
                return [
                    ErrorMessage(message=f"what_if input {k} does not exist (0 to {len(real) - 1})")
                ]
            if not np.isfinite(value):
                return [ErrorMessage(message="what_if values must be finite numbers")]
            tried[k] = value  # the other inputs keep their exact values
        network = policy.network_of(car)
        base = forward(
            network, real
        )  # same function as the what-if, so equal inputs give equal bits
        slopes = sensitivity(network, real)
        predicted = base.output + slopes @ (tried - real)  # the straight-line guess
        trace = forward(network, tried)  # the car's own network on the tried observation
        return [
            ExplanationMessage(
                car=car,
                step=world.step_count,
                observation=np.round(tried, 4).tolist(),
                trace=_wire_trace(trace),
                explain=_explanation(network, trace, tried),
                base_outputs=np.round(base.output, 4).tolist(),
                predicted_outputs=np.round(predicted, 4).tolist(),
                ghost=self._ghost(car, base.output, trace.output),
            )
        ]

    def _ghost(self, car: int, real_action: np.ndarray, what_if_action: np.ndarray) -> GhostPaths:
        """Roll the selected car forward twice with the car physics, once per held action.

        Two copies of the car start at its pose now; copy 0 holds the real action and copy 1 the
        what-if action (each clipped to [-1, 1], as the world does) for GHOST_SECONDS. Only the car
        moves: no track, crash or progress rules apply, so a path can cross the road edge.
        """
        world = self._world
        assert world is not None
        cars = world.cars
        actions = np.clip(np.stack([real_action, what_if_action]), -1.0, 1.0)
        state = CarState.from_pose(cars.x[car], cars.y[car], cars.psi[car], [cars.v[car]] * 2)
        points = [np.column_stack([state.x, state.y])]
        for _ in range(round(GHOST_SECONDS / world.dt)):
            state = step_cars(state, actions[:, 0], actions[:, 1], world.params, world.dt)
            points.append(np.column_stack([state.x, state.y]))
        path = np.round(np.stack(points), 3)  # (steps + 1, 2 copies, 2)
        return GhostPaths(
            seconds=GHOST_SECONDS,
            real_action=np.round(actions[0], 4).tolist(),
            what_if_action=np.round(actions[1], 4).tolist(),
            real=path[:, 0].tolist(),
            what_if=path[:, 1].tolist(),
            real_end=[*path[-1, 0].tolist(), round(float(state.psi[0]), 4)],
            what_if_end=[*path[-1, 1].tolist(), round(float(state.psi[1]), 4)],
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


def _wire_trace(trace: ForwardTrace) -> NetworkTrace:
    """A single-observation trace, rounded for the wire."""
    return NetworkTrace(
        activations=[np.round(a, 4).tolist() for a in trace.activations],
        weighted_sums=[np.round(z, 4).tolist() for z in trace.weighted_sums],
    )


def _explanation(network: Network, trace: ForwardTrace, observation: np.ndarray) -> Explanation:
    """Receipts of both outputs and the output-by-input sensitivity table (explain/, task 2.9)."""
    last = network.spec.n_layers - 1
    receipts = []
    for m in range(network.spec.n_outputs):
        r = neuron_contributions(network, trace, last, m)
        receipts.append(
            OutputReceipt(
                output=m,
                inputs=np.round(r.inputs, 4).tolist(),
                contributions=np.round(r.contributions, 4).tolist(),
                bias=round(r.bias, 4),
                z=round(r.z, 4),
                value=round(r.output, 4),
            )
        )
    return Explanation(
        receipts=receipts,
        sensitivity=np.round(sensitivity(network, observation), 4).tolist(),
    )
