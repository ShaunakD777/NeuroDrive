"""Message shapes sent between the server and the browser (api v1.1, ADR-016).

Every shape is a Pydantic model, and the TypeScript types for the web app are generated from
these (task 0.4), so the two sides cannot drift apart.
"""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

API_VERSION = "v1.16"


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


class CheckpointRef(CommandModel):
    """A saved network: checkpoint ``unit`` of run ``run`` (a run folder name)."""

    run: str
    unit: int = Field(ge=0)


class LoadScenario(CommandModel):
    """Load a track and a driver; the session starts paused at step 0.

    ``policy`` is ``"keyboard"`` (driven by ``key_state``), ``"constant"`` (every car holds steer 0,
    throttle 0.3), ``"neuron"`` (one shared neuron steers; change it with ``set_param``),
    ``"random_networks"`` (every car is driven by its own random [7, 16, 16, 2] network) or
    ``"expert"`` (the scripted expert of ``env/experts.py``, which reads the track; api v1.12).
    ``seed`` None starts every car at the track's start; an integer gives a seeded random start
    (position, sideways offset and heading). For ``"random_networks"`` the same ``seed`` also picks
    the networks (seed 0 when None), so a seed always gives the same 20 drivers.

    ``"checkpoint"`` (api v1.14, task 5.9, ADR-042) replays a saved network's Driving Exam drive
    on ``track``: ``checkpoint`` names it, and the cars are placed at the exam's starts on that
    track in the exam's order (car k = exam start k, 10 cars), all driven by that one network as
    one batch, exactly as the exam drove them. ``n_cars`` and ``seed`` are then ignored.
    """

    type: Literal["load_scenario"] = "load_scenario"
    track: str
    policy: Literal["keyboard", "constant", "neuron", "random_networks", "expert", "checkpoint"] = (
        "keyboard"
    )
    n_cars: int = Field(default=1, ge=1, le=100)
    seed: int | None = Field(default=None, ge=0)
    checkpoint: CheckpointRef | None = None


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
    of the drive the session remembers for the decision trail (ADR-027). ``checkpoint`` is the
    replayed network for the ``"checkpoint"`` driver (api v1.14), else null.
    """

    type: Literal["scenario"] = "scenario"
    track: str
    policy: Literal["keyboard", "constant", "neuron", "random_networks", "expert", "checkpoint"]
    n_cars: int
    seed: int | None
    checkpoint: CheckpointRef | None = None
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
class RunSummary(WireModel):
    """One entry of ``GET /api/runs`` (api v1.11, ADR-031, ADR-033): a run folder, its state.

    ``run_id`` is the folder name (``<YYYYMMDD-HHMMSS>-<method>-<tag>``), ``status`` one of
    running, completed, diverged, failed. ``checkpoints`` is how many checkpoints it holds and
    ``last_unit`` the latest one (null when there are none). ``sweep`` is the sweep id of a run
    made by ``nd sweep`` (null otherwise).
    """

    run_id: str
    method: str
    tag: str
    status: Literal["running", "completed", "diverged", "failed"]
    started_at: str
    ended_at: str | None
    checkpoints: int
    last_unit: int | None
    sweep: str | None = None


class RunListResponse(WireModel):
    """Answer to ``GET /api/runs``: every run folder, newest first."""

    runs: list[RunSummary]


class RunDetail(WireModel):
    """Answer to ``GET /api/runs/{id}``: the run's ``config.yaml`` and ``meta.json`` as stored."""

    run_id: str
    config: dict[str, Any]
    meta: dict[str, Any]


class RunMetrics(WireModel):
    """Answer to ``GET /api/runs/{id}/metrics``: ``metrics.jsonl`` as columns (entry i = one unit).

    ``loss`` is the loss after unit ``unit[i]`` (null if it diverged), ``loss_before`` the loss
    before that unit's step, so ``loss_before[0]`` is the untrained network's loss (unit 0).
    ``grad_norm[l][i]`` is the gradient size of weight layer ``l`` in that unit. Numbers keep 6
    significant digits.
    """

    run_id: str
    unit: list[int]
    loss: list[float | None]
    loss_before: list[float | None]
    grad_norm: list[list[float | None]]
    diverged: list[bool]


class CheckpointList(WireModel):
    """Answer to ``GET /api/runs/{id}/checkpoints``: the units that have a saved network."""

    run_id: str
    units: list[int]


class CheckpointNetwork(WireModel):
    """Answer to ``GET /api/runs/{id}/checkpoints/{unit}``: the network saved at that unit.

    Same layout as the live ``network`` message: ``weights[l]`` is W_l as ``n_out`` rows of
    ``n_in`` numbers, ``biases[l]`` is b_l, ``activations[l]`` the activation of weight layer l.
    Rounded to 4 decimals.
    """

    run_id: str
    unit: int
    layer_sizes: list[int]
    activations: list[Literal["tanh", "sigmoid", "relu", "linear"]]
    weights: list[list[list[float]]]
    biases: list[list[float]]


class ToyCheckpointResponse(WireModel):
    """Answer to ``GET /api/runs/{id}/checkpoints/{unit}/toy``: the toy lab's view of a checkpoint.

    ``loss`` is the loss on the run's dataset (``data_x`` rows, targets ``data_y``) for this
    checkpoint's ``network``. For a 1-input task ``prediction_curve[k]`` is the network's output
    at x = ``grid_x[k]``; for a 2-input task ``prediction_map[i][k]`` is its output at
    (``grid_x[k]``, ``grid_y[i]``); the other is null. ``weight_gradients`` and
    ``bias_gradients`` are dL/dW and dL/db at this checkpoint (6 significant digits);
    ``glow[l][j][i]`` = |dL/dW_l[j][i]| / ``max_gradient``, the largest |dL/dW| (3 decimals).
    Computed by ``experiment/toy_view.py``.
    """

    run_id: str
    unit: int
    network: CheckpointNetwork
    loss: float
    data_x: list[list[float]]
    data_y: list[float]
    grid_x: list[float]
    grid_y: list[float] | None
    prediction_curve: list[float] | None
    prediction_map: list[list[float]] | None
    weight_gradients: list[list[list[float]]]
    bias_gradients: list[list[float]]
    glow: list[list[list[float]]]
    max_gradient: float


class CompareRun(WireModel):
    """One run's curve in ``GET /api/runs/compare`` (api v1.11, ADR-033).

    ``unit[i]`` is the epoch of ``values[i]`` (from 0 for ``loss``: the loss before training;
    from 1 for ``val_loss``). A value is null where the run diverged. ``final`` is the last
    finite value (null if there is none); ``seed`` comes from the run's config.
    """

    run_id: str
    group_id: str
    seed: int | None
    status: str
    diverged: bool
    unit: list[int]
    values: list[float | None]
    final: float | None


class CompareGroup(WireModel):
    """One group (a sweep, or a single run) in ``GET /api/runs/compare``.

    ``unit``, ``median``, ``q1``, ``q3``, ``n_present``, ``n_finite``: per epoch, across the group's
    runs (``experiment/stats.curve_bands``; a diverged run counts as +infinity from its divergence
    on, sent as **null** because JSON has no infinity). ``final_*``: the median and quartiles of
    the runs' final values (a diverged run counts as +infinity, null), ``n_diverged`` how many
    diverged.
    """

    group_id: str
    run_ids: list[str]
    unit: list[int]
    median: list[float | None]
    q1: list[float | None]
    q3: list[float | None]
    n_present: list[int]
    n_finite: list[int]
    final_median: float | None
    final_q1: float | None
    final_q3: float | None
    n_diverged: int


# ── launching and following training runs (api v1.13, task 5.8, ADR-041) ──────────────────────


class ConfigInfo(WireModel):
    """One training config of ``GET /api/configs``: a YAML file in ``configs/`` that ``nd train``
    accepts.

    ``length_field`` names what the run's length counts (``epochs`` for toy configs, ``units``
    for random search) and ``length`` is the file's value (null when a random-search budget has
    no unit limit). ``task`` is the toy task (null for random search).
    """

    name: str
    method: Literal["toy", "random_search"]
    task: str | None
    seed: int
    tag: str
    length_field: Literal["epochs", "units"]
    length: int | None


class ConfigListResponse(WireModel):
    """Answer to ``GET /api/configs``: the training configs, by file name."""

    configs: list[ConfigInfo]


class LaunchRequest(WireModel):
    """Body of ``POST /api/runs``: a config from ``configs/`` and the overrides of ``nd train``.

    Each override left out (null) keeps the file's value. ``epochs`` is for toy configs and
    ``units`` for random search, exactly as on the command line.
    """

    model_config = ConfigDict(json_schema_serialization_defaults_required=True, extra="forbid")

    config: str
    seed: int | None = None
    tag: str | None = None
    epochs: int | None = None
    units: int | None = None


class LaunchResponse(WireModel):
    """Answer to ``POST /api/runs``: the id of the run folder the new training process made."""

    run_id: str


class StopResponse(WireModel):
    """Answer to ``POST /api/runs/{id}/stop``: the stop request was written into the run folder;
    the run ends after its current unit."""

    run_id: str
    stop_requested: bool = True


class WatchRun(CommandModel):
    """``/ws/runs``, browser → server: follow one run (null: none). The server answers with the
    run's history (``metrics`` with ``replace``, ``checkpoint``, one ``exam_done`` per card) and
    then sends that run's new events as they appear."""

    type: Literal["watch"] = "watch"
    run_id: str | None


class RunStatusMessage(WireModel):
    """``/ws/runs``: a run appeared or changed (sent to every connection).

    ``status`` and ``stop_reason`` come from the run's ``meta.json``; ``units_done`` is the number
    of metrics lines, ``budget_units`` the unit limit (null without one). ``stop_requested``: a
    stop request is waiting for the current unit to end. ``launched``: this server started the
    training process; ``exit_code`` is that process's exit code once it has ended (null while it
    runs, and for runs this server did not start). A run whose status is still "running" but whose
    process has ended was killed; ``nd resume`` can continue it (resumable methods).
    """

    type: Literal["run_status"] = "run_status"
    run_id: str
    method: str
    tag: str
    status: Literal["running", "completed", "diverged", "failed"]
    stop_reason: str | None
    units_done: int
    budget_units: int | None
    stop_requested: bool
    launched: bool
    exit_code: int | None


class RunMetricsMessage(WireModel):
    """``/ws/runs``: new lines of the watched run's ``metrics.jsonl``, as stored (one object per
    unit: ``unit``, the budget totals, ``wall_clock_s`` and the learner's own metrics).

    ``replace`` true: these are all the lines so far, and the browser starts the run's history
    again from them (after ``watch``, and when a resumed run was cut back); its exam rows then
    follow again as ``exam_done`` messages. Otherwise the lines are to be added.
    """

    type: Literal["metrics"] = "metrics"
    run_id: str
    replace: bool
    records: list[dict[str, Any]]


class CheckpointMessage(WireModel):
    """``/ws/runs``: the watched run's checkpoint units, all of them, whenever the list changes."""

    type: Literal["checkpoint"] = "checkpoint"
    run_id: str
    units: list[int]


class ExamHeadline(WireModel):
    """The headline of one part (training or held-out) of an exam card: completion rate, crash
    rate and median progress (share of the target laps), as in the card's summaries."""

    completion_rate: float
    crash_rate: float
    progress_frac_median: float


class ExamDoneMessage(WireModel):
    """``/ws/runs``: an exam card of the watched run was saved (``exams/unit_<n>.json``)."""

    type: Literal["exam_done"] = "exam_done"
    run_id: str
    unit: int
    training: ExamHeadline
    heldout: ExamHeadline


# ── replay and the run timeline (api v1.14, task 5.9, ADR-042) ─────────────────────────────────


class CheckpointDiff(WireModel):
    """Answer to ``GET /api/runs/{id}/checkpoints/{unit}/diff?from={from_unit}``: what changed.

    ``weight_delta[l][j][i]`` = W_l[j, i] at ``unit`` minus the same weight at ``from_unit``
    (``bias_delta`` likewise); ``max_abs_delta`` = M, the largest |delta| over every weight and
    bias; ``weight_change`` = delta / M, so between -1 and 1: the sign says up or down, the size
    how strongly to draw it (all 0 when nothing changed). Rounded to 4 decimals (deltas to 6
    significant digits).
    """

    run_id: str
    unit: int
    from_unit: int
    layer_sizes: list[int]
    weight_delta: list[list[list[float]]]
    bias_delta: list[list[float]]
    max_abs_delta: float
    weight_change: list[list[list[float]]]
    bias_change: list[list[float]]


class ExamCardResponse(WireModel):
    """Answer to ``GET /api/runs/{id}/exams/{unit}``: the exam card as stored (exam v1, §6.14)."""

    run_id: str
    unit: int
    card: dict[str, Any]


class CompareResponse(WireModel):
    """Answer to ``GET /api/runs/compare?ids=a,b,...&metric=loss|val_loss`` (api v1.11, ADR-033).

    Groups in the order their first run was asked for. ``skipped`` lists the runs that have no
    values for ``metric`` (for example ``val_loss`` of a run without a validation set).
    """

    metric: Literal["loss", "val_loss"]
    groups: list[CompareGroup]
    runs: list[CompareRun]
    skipped: list[str]


# ── demonstration datasets (api v1.15, task 6.2, ADR-045) ──────────────────────────────────────


class DatasetSummary(WireModel):
    """One entry of ``GET /api/datasets``: a dataset folder's headline from its ``meta.json``.

    ``finished`` is how many of the ``episodes`` ended by completing their laps; ``straight_share``
    is q, the share of frames with |steer| < 0.1.
    """

    dataset_id: str
    teacher: str
    tag: str
    track: str
    created_at: str
    demo_frames: int
    episodes: int
    finished: int
    straight_share: float


class DatasetListResponse(WireModel):
    """Answer to ``GET /api/datasets``: every readable dataset folder, newest first."""

    datasets: list[DatasetSummary]


class DatasetHistogram(WireModel):
    """One quantity's histogram (``experiment/dataset_view.py``).

    ``edges`` are the B + 1 bin edges from ``lo`` to ``hi`` (6 decimals); ``counts[b]`` is the
    number of frames in [edges[b], edges[b + 1]) (the last bin includes ``hi``), so the counts add
    up to the dataset's frames. ``min`` and ``max`` are the smallest and largest value (null for
    an empty dataset) and ``covered`` = (max − min) / (hi − lo), the share of the range reached.
    ``dense`` (api v1.16, ADR-046) is the dense coverage d ≤ ``covered``: the summed length of the
    bins holding at least ``dense_threshold`` = τ·F frames (τ = 1 %), each clipped to [min, max],
    divided by (hi − lo).
    """

    name: str
    lo: float
    hi: float
    edges: list[float]
    counts: list[int]
    min: float | None
    max: float | None
    covered: float
    dense: float
    dense_threshold: float


class DatasetMap(WireModel):
    """Every ``stride``-th frame of a dataset, for the track map: its position (metres, 2
    decimals), its steer and throttle label (4 decimals) and its episode number."""

    stride: int
    x: list[float]
    y: list[float]
    steer: list[float]
    throttle: list[float]
    episode: list[int]


class DatasetDetail(WireModel):
    """Answer to ``GET /api/datasets/{id}``: the metadata as stored, the 2 label histograms
    (steer, throttle), the 7 observation histograms (observation v1 order) and the map."""

    dataset_id: str
    meta: dict[str, Any]
    labels: list[DatasetHistogram]
    observations: list[DatasetHistogram]
    map: DatasetMap


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
    RunSummary,
    RunListResponse,
    RunDetail,
    RunMetrics,
    CompareRun,
    CompareGroup,
    CompareResponse,
    CheckpointList,
    CheckpointNetwork,
    ToyCheckpointResponse,
    ConfigInfo,
    ConfigListResponse,
    LaunchRequest,
    LaunchResponse,
    StopResponse,
    WatchRun,
    RunStatusMessage,
    RunMetricsMessage,
    CheckpointMessage,
    ExamHeadline,
    ExamDoneMessage,
    CheckpointRef,
    CheckpointDiff,
    ExamCardResponse,
    DatasetSummary,
    DatasetListResponse,
    DatasetHistogram,
    DatasetMap,
    DatasetDetail,
)
