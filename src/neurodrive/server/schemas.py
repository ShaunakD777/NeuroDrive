"""Message shapes sent between the server and the browser (api v1.1, ADR-016).

Every shape is a Pydantic model, and the TypeScript types for the web app are generated from
these (task 0.4), so the two sides cannot drift apart.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

API_VERSION = "v1.9"


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


class WhatIf(CommandModel):
    """Ask what the selected car's network would output for a changed observation (paused only).

    ``overrides`` maps an input index (0 to 6) to the value to try for it; every other input keeps
    the car's exact current value (so an empty map asks about the car exactly as it is). Answered
    with an ``explanation`` message; nothing in the simulation changes. Only valid for the
    random-networks driver and while the session is paused (api v1.7, ADR-026).
    """

    type: Literal["what_if"] = "what_if"
    overrides: dict[int, float] = Field(default_factory=dict)


class InspectStep(CommandModel):
    """Look at the selected car at an earlier step of the drive (paused only; api v1.8, ADR-027).

    ``step`` must be one of the steps the session still remembers (the last ``trail_seconds`` of
    the drive). Answered with a ``moment`` message; the simulation itself stays where it is.
    """

    type: Literal["inspect_step"] = "inspect_step"
    step: int = Field(ge=0)


Command = Annotated[
    LoadScenario
    | Play
    | Pause
    | Step
    | Reset
    | SetSpeed
    | SelectCar
    | KeyState
    | SetParam
    | WhatIf
    | InspectStep,
    Field(discriminator="type"),
]
COMMAND_ADAPTER: TypeAdapter[Command] = TypeAdapter(Command)


# ── server → client ────────────────────────────────────────────────────────────────────────────


class ScenarioMessage(WireModel):
    """Sent after ``load_scenario``: what was loaded and the constants needed to draw it.

    ``dt`` is the physics step in seconds, ``ray_angles`` the sensor directions in radians
    relative to the heading (left positive), ``ray_range`` the sensor range in metres. Track
    geometry is not included: fetch ``GET /api/tracks/{track}``. ``trail_seconds`` is how much
    of the drive the session remembers for the decision trail (ADR-027).
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
    trail_seconds: float


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


class NetworkMessage(WireModel):
    """The weights of one car's network, sent when a scenario loads and when a car is selected.

    ``car`` is the car the network belongs to. ``layer_sizes`` are the sizes of every layer from
    the inputs to the outputs, e.g. [7, 16, 16, 2]; ``activations[l]`` is the activation name of
    weight layer ``l``. ``weights[l]`` is the matrix W_l as ``n_out`` rows of ``n_in`` numbers (row
    j holds the weights of neuron j of the next layer) and ``biases[l]`` is b_l. Numbers are
    rounded to 4 decimals. They do not change while driving, so they are sent once, not per frame.
    """

    type: Literal["network"] = "network"
    car: int
    layer_sizes: list[int]
    activations: list[Literal["tanh", "sigmoid", "relu", "linear"]]
    weights: list[list[list[float]]]
    biases: list[list[float]]


class NetworkTrace(WireModel):
    """Every value of one forward pass of the selected car's network, for the current state.

    ``activations[0]`` is the observation (the inputs), ``activations[l]`` is a_l, the outputs of
    weight layer ``l``, so the last entry is the action. ``weighted_sums[l - 1]`` is z_l, the
    sums before the activation. Rounded to 4 decimals; all computed by Python.
    """

    activations: list[list[float]]
    weighted_sums: list[list[float]]


class OutputReceipt(WireModel):
    """Why one output has its value: the contributions of the last hidden layer (task 2.9).

    ``output`` is 0 for steer and 1 for throttle. ``inputs`` are the last hidden layer's values
    a_i, ``contributions`` the products W[output, i] * a_i (one per hidden neuron), ``bias`` the
    output's bias, ``z`` the weighted sum (the contributions plus the bias) and ``value`` the
    activation of z: the output itself. Rounded to 4 decimals.
    """

    output: int
    inputs: list[float]
    contributions: list[float]
    bias: float
    z: float
    value: float


class Explanation(WireModel):
    """The explanation of the selected car's two outputs for one observation (task 2.9).

    ``receipts[m]`` explains output m (steer, then throttle). ``sensitivity[m][k]`` is how much
    output m moves per unit change of input k, computed exactly by backpropagation (task 3.6): +0.4
    means "if input k rose by 0.1, output m would rise by about 0.04". Rounded to 4 decimals.
    """

    receipts: list[OutputReceipt]
    sensitivity: list[list[float]]


class GhostPaths(WireModel):
    """Where the selected car would go in the next ``seconds`` if it held one action (ADR-026).

    Both paths start at the car's pose now and come from the car physics in Python: ``real`` holds
    the action the network chooses for its real observation, ``what_if`` the action it chooses for
    the tried one. Each is a list of [x, y] points in metres, one per physics step (the first is
    the car's position now); ``real_end`` and ``what_if_end`` are the final [x, y, heading] for
    drawing the car there. Holding one action for the whole time is a simplification: a driving
    network decides again at every step. Rounded to 3 decimals (positions) and 4 (headings).
    """

    seconds: float
    real_action: list[float]
    what_if_action: list[float]
    real: list[list[float]]
    what_if: list[list[float]]
    real_end: list[float]
    what_if_end: list[float]


class ExplanationMessage(WireModel):
    """Answer to ``what_if``: the selected car's network on a changed observation.

    ``car`` and ``step`` say which car and which moment the question was about. ``observation``
    is the observation that was tried, ``trace`` and ``explain`` the network's forward pass and
    explanation for it, ``base_outputs`` what the car outputs for its real observation now, and
    ``predicted_outputs`` what the real observation's sensitivity predicts for the tried one:
    base + sum over k of sensitivity[m][k] * (tried_k - real_k). The prediction is a straight-line
    guess; ``trace``'s last activations are the network's real answer. Rounded to 4 decimals.
    """

    type: Literal["explanation"] = "explanation"
    car: int
    step: int
    observation: list[float]
    trace: NetworkTrace
    explain: Explanation
    base_outputs: list[float]
    predicted_outputs: list[float]
    ghost: GhostPaths


class TrailPoints(WireModel):
    """Points of the selected car's decision trail (ADR-027): where it was and what it decided.

    Entry i says: at step ``steps[i]`` the car was at (``x[i]``, ``y[i]``) and chose ``action[i]``
    ([steer, throttle]) from there. Only steps at which the car was still running are listed.
    ``replace`` true means "these are all the points the session remembers for this car" (after a
    load, a reset or a car change); false means "add these new points to the ones you have".
    Rounded to 3 decimals.
    """

    replace: bool
    steps: list[int]
    x: list[float]
    y: list[float]
    action: list[list[float]]


class MomentMessage(WireModel):
    """Answer to ``inspect_step``: the selected car at an earlier step, recomputed by Python.

    ``x``, ``y``, ``psi`` and ``v`` are its pose and speed then, ``progress`` its cumulative
    progress in metres and ``status`` its status code (as in ``CarColumns``), ``rays`` the five
    ray lengths in metres, ``observation`` what its network read, ``action`` what it chose there
    (api v1.9, ADR-028: progress and status let the HUD show that moment too). ``trace`` and
    ``explain`` are its network's forward pass and explanation for that observation, exactly as a
    frame at that step showed them (null for drivers without a network). Rounded like a frame.
    """

    type: Literal["moment"] = "moment"
    car: int
    step: int
    time: float
    x: float
    y: float
    psi: float
    v: float
    progress: float
    status: int
    rays: list[float]
    observation: list[float]
    action: list[float]
    trace: NetworkTrace | None
    explain: Explanation | None


class SelectedCar(WireModel):
    """Extra detail for the selected car: rays in metres, the observation, the neuron readout
    (neuron driver) or the network trace (random networks); the other is null."""

    index: int
    rays: list[float]
    observation: list[float]
    neuron: NeuronReadout | None
    trace: NetworkTrace | None = None
    explain: Explanation | None = None
    trail: TrailPoints | None = None


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
    NetworkMessage,
    NetworkTrace,
    OutputReceipt,
    Explanation,
    GhostPaths,
    ExplanationMessage,
    WhatIf,
    InspectStep,
    TrailPoints,
    MomentMessage,
    SelectedCar,
    ActivationCurves,
    FrameMessage,
    EpisodeEndMessage,
    ErrorMessage,
)
