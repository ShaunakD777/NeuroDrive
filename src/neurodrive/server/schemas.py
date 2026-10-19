"""Message shapes sent between the server and the browser (api v1.1, ADR-016).

Every shape is a Pydantic model, and the TypeScript types for the web app are generated from
these (task 0.4), so the two sides cannot drift apart.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

API_VERSION = "v1.4"


class WireModel(BaseModel):
    """Base class for everything the server sends.

    The server always fills in every field, so the generated schema marks fields that have a
    default value as required too. Without this, TypeScript would treat them as maybe-missing.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True)


class HealthResponse(WireModel):
    """Answer to ``GET /api/health``: the server is up, and which NeuroDrive it is running."""

    status: Literal["ok"] = "ok"
    version: str


class HelloMessage(WireModel):
    """First message on ``/ws/live``, sent as soon as a client connects.

    ``version`` is the NeuroDrive package version and ``api`` the message-contract version.
    """

    type: Literal["hello"] = "hello"
    version: str
    api: str = API_VERSION


class TrackInfo(WireModel):
    """One entry of ``GET /api/tracks``."""

    name: str
    split: Literal["train", "heldout"]


class TrackListResponse(WireModel):
    """Answer to ``GET /api/tracks``: the tracks the server can load."""

    tracks: list[TrackInfo]


class TrackGeometry(WireModel):
    """Answer to ``GET /api/tracks/{name}``: what the browser needs to draw a track.

    ``centerline``, ``left`` and ``right`` are closed loops of [x, y] points in metres, rounded to
    2 decimals, about 1 m apart (the first point is not repeated at the end). ``left`` and
    ``right`` are the road edges as seen when driving along the centreline. ``length`` is the loop
    length and ``start_s0`` the arc length of the start line.
    """

    name: str
    split: Literal["train", "heldout"]
    width: float
    length: float
    start_s0: float
    centerline: list[list[float]]
    left: list[list[float]]
    right: list[list[float]]


# ── client → server ────────────────────────────────────────────────────────────────────────────


class CommandModel(WireModel):
    """Base of every message the browser sends over ``/ws/live``: unknown fields are errors."""

    model_config = ConfigDict(json_schema_serialization_defaults_required=True, extra="forbid")


class LoadScenario(CommandModel):
    """Load a track and a driver; the session starts paused at step 0.

    ``policy`` is ``"keyboard"`` (driven by ``key_state``), ``"constant"`` (every car holds steer 0,
    throttle 0.3), ``"neuron"`` (one shared neuron steers; change it with ``set_param``) or
    ``"random_networks"`` (every car is driven by its own random [7, 16, 16, 2] network).
    ``seed`` None starts every car at the track's start; an integer gives a seeded random start
    (position, sideways offset and heading). For ``"random_networks"`` the same ``seed`` also picks
    the networks (seed 0 when None), so a seed always gives the same 20 drivers.
    """

    type: Literal["load_scenario"] = "load_scenario"
    track: str
    policy: Literal["keyboard", "constant", "neuron", "random_networks"] = "keyboard"
    n_cars: int = Field(default=1, ge=1, le=100)
    seed: int | None = Field(default=None, ge=0)


class Play(CommandModel):
    """Start or resume the simulation clock."""

    type: Literal["play"] = "play"


class Pause(CommandModel):
    """Stop the simulation clock."""

    type: Literal["pause"] = "pause"


class Step(CommandModel):
    """Advance exactly one physics step (works while paused) and answer with a frame."""

    type: Literal["step"] = "step"


class Reset(CommandModel):
    """Return to step 0 of the loaded scenario (same seed, so the same start) and send a frame."""

    type: Literal["reset"] = "reset"


class SetSpeed(CommandModel):
    """Set the speed factor: physics steps per real second = 30 * speed (0.1 to 8)."""

    type: Literal["set_speed"] = "set_speed"
    speed: float = Field(ge=0.1, le=8.0)


class SelectCar(CommandModel):
    """Choose which car's rays and observation appear in ``frame.selected``."""

    type: Literal["select_car"] = "select_car"
    index: int = Field(ge=0)


class KeyState(CommandModel):
    """The manual-driving input held right now: steer and throttle, each in [-1, 1]."""

    type: Literal["key_state"] = "key_state"
    steer: float = Field(ge=-1.0, le=1.0)
    throttle: float = Field(ge=-1.0, le=1.0)


class SetParam(CommandModel):
    """Change one parameter of the neuron driver while it drives; answered with a frame.

    ``name`` is ``w_left``, ``w_right``, ``bias`` or ``throttle`` (then ``value`` is a number;
    weights and bias within +-30, throttle within +-1) or ``activation`` (then ``value`` is
    ``"tanh"``, ``"sigmoid"``, ``"relu"`` or ``"linear"``). Only valid for the neuron policy.
    """

    type: Literal["set_param"] = "set_param"
    name: Literal["w_left", "w_right", "bias", "throttle", "activation"]
    value: float | str


Command = Annotated[
    LoadScenario | Play | Pause | Step | Reset | SetSpeed | SelectCar | KeyState | SetParam,
    Field(discriminator="type"),
]
COMMAND_ADAPTER: TypeAdapter[Command] = TypeAdapter(Command)


# ── server → client ────────────────────────────────────────────────────────────────────────────


class ScenarioMessage(WireModel):
    """Sent after ``load_scenario``: what was loaded and the constants needed to draw it.

    ``dt`` is the physics step in seconds, ``ray_angles`` the sensor directions in radians
    relative to the heading (left positive), ``ray_range`` the sensor range in metres. Track
    geometry is not included: fetch ``GET /api/tracks/{track}``.
    """

    type: Literal["scenario"] = "scenario"
    track: str
    policy: Literal["keyboard", "constant", "neuron", "random_networks"]
    n_cars: int
    seed: int | None
    dt: float
    time_limit: float
    target_laps: int
    v_max: float
    wheelbase: float
    car_half_width: float
    ray_angles: list[float]
    ray_range: float


class CarColumns(WireModel):
    """The state of every car as one array per quantity (entry i belongs to car i).

    ``status``: 0 running, 1 crashed, 2 finished, 3 stalled, 4 timeout. ``action`` holds the last
    [steer, throttle] applied to each car ([0, 0] before the first step).
    """

    x: list[float]
    y: list[float]
    psi: list[float]
    v: list[float]
    alive: list[bool]
    progress: list[float]
    status: list[int]
    action: list[list[float]]


class NeuronReadout(WireModel):
    """What the selected car's neuron saw and computed (neuron policy only).

    ``inputs`` are the two ray readings (0 to 1) it reads and ``input_rays`` which of the five rays
    they are. ``contributions`` are weight times input for each, ``z`` is their total plus the
    bias, and ``output`` is the activation of ``z``: the steering command. All computed by Python.
    """

    inputs: list[float]
    input_rays: list[int]
    weights: list[float]
    bias: float
    contributions: list[float]
    z: float
    output: float
    activation: Literal["tanh", "sigmoid", "relu", "linear"]
    throttle: float


class SelectedCar(WireModel):
    """Extra detail for the selected car: rays in metres, the observation and the neuron readout."""

    index: int
    rays: list[float]
    observation: list[float]
    neuron: NeuronReadout | None


class ActivationCurves(WireModel):
    """Answer to ``GET /api/activations``: each activation function sampled on the same z values.

    ``curves`` maps a name (tanh, sigmoid, relu, linear) to the list of f(z) for every z in ``z``.
    """

    z: list[float]
    curves: dict[str, list[float]]


class FrameMessage(WireModel):
    """One snapshot of the world: while playing (at most 30 per second) and after commands."""

    type: Literal["frame"] = "frame"
    step: int
    time: float
    playing: bool
    speed: float
    cars: CarColumns
    selected: SelectedCar | None


class EpisodeEndMessage(WireModel):
    """Sent once when every car has stopped; the session then pauses.

    ``status`` has one name per car: crashed, finished, stalled or timeout.
    """

    type: Literal["episode_end"] = "episode_end"
    step: int
    time: float
    status: list[str]
    progress: list[float]
    laps: list[int]


class ErrorMessage(WireModel):
    """A bad message or a command that cannot run now. The connection stays open."""

    type: Literal["error"] = "error"
    message: str


# Every model the browser sends or receives. `npm run gen:types` turns each one into a TypeScript
# type, so a new message shape is added here and nowhere else.
WIRE_MODELS: tuple[type[WireModel], ...] = (
    HealthResponse,
    HelloMessage,
    TrackInfo,
    TrackListResponse,
    TrackGeometry,
    LoadScenario,
    Play,
    Pause,
    Step,
    Reset,
    SetSpeed,
    SelectCar,
    KeyState,
    SetParam,
    ScenarioMessage,
    CarColumns,
    NeuronReadout,
    SelectedCar,
    ActivationCurves,
    FrameMessage,
    EpisodeEndMessage,
    ErrorMessage,
)
