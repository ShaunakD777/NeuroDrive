"""Config files: the written recipe of an experiment, validated before anything runs (ADR-032).

A config file is YAML. Its ``method`` field says which kind of run it describes: ``toy`` (task
4.6b), ``random_search`` (task 5.7, ``configs/random_search.yaml``), ``imitation`` (task 6.3,
``configs/imitation.yaml``) or ``dagger`` (task 6.6, ``configs/dagger.yaml``). A toy config
(config v1, docs/ARCHITECTURE.md §6.15) looks like::

    method: toy
    task: sine              # line, sine, xor or two_spirals
    seed: 0
    tag: default            # labels the run folder
    network:
      layer_sizes: [1, 16, 1]
      hidden_activation: tanh
      output_activation: linear
      init_scheme: xavier
    training:
      epochs: 3000
      loss: mse
      batch_size: null      # null: the whole training set (full batch)
      val_fraction: 0.0     # share of examples held back for validation
    optimizer:
      name: sgd             # sgd, momentum, rmsprop or adam
      learning_rate: 0.05   # plus only that optimizer's own settings

Validation refuses unknown fields, wrong types, unknown names and impossible numbers, and lists
every problem with its field path (``ConfigError``). The *resolved* config, every default written
out, is what a run folder stores, so ``nd train --config runs/<id>/config.yaml`` re-creates the run.
Run folders written before this format (Phase 3: the learner's flat ``ToyConfig`` dictionary) are
still read: they are converted on load.

A random-search config has the sections ``network`` (a driving network: 7 inputs, 2 outputs),
``search`` (candidates per unit), ``fitness`` (a training track and the evaluation starts; a
held-out track is refused, EXPERIMENT_PROTOCOL rule 7), ``budget`` (units, env_steps,
wall_clock_s: at least one) and ``exams``.

An imitation config (behavioural cloning, ADR-047) has the sections ``network`` (a driving
network), ``data`` (the demonstration dataset: an id or a ``<teacher>-<tag>`` selector, its
optional ``sha256`` checksum, and the share of episodes held back for validation), ``training``
(epochs, loss, batch size), ``optimizer`` (as for toys), ``checkpoint_every`` (also checkpoint and
examine every k-th epoch) and ``exams``. Validation does not look for the dataset: the runner
finds it, and the run folder's config then names the exact dataset and its checksum.

A DAgger config (task 6.6, ADR-049) has imitation's ``network``, ``data``, ``optimizer``,
``checkpoint_every`` and ``exams``, a ``training`` section with ``epochs_per_round`` instead of
``epochs``, and a ``dagger`` section: ``rounds`` (the run's units; ``--units`` overrides it), the
collection ``track`` (a training track), ``random_starts`` per round and ``target_laps``.

The learners keep their own plain settings (``learn/`` does not use Pydantic); this module
converts between the file and them. Every list of valid names comes from the module that owns it.
"""

import math
from functools import cache
from pathlib import Path
from typing import Annotated, Any, Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    ValidationError,
    field_validator,
    model_validator,
)

from neurodrive.core.activations import ACTIVATIONS
from neurodrive.core.nn.init import SCHEMES
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.optim import OPTIMIZERS
from neurodrive.core.rng import make_rng
from neurodrive.env.car import ACTION_SIZE
from neurodrive.env.sensors import OBSERVATION_SIZE
from neurodrive.env.track import Track
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, EXAM_V1, TRAINING_TRACK
from neurodrive.learn.evolution.random_search import RandomSearchSettings
from neurodrive.learn.supervised.dagger import DaggerSettings
from neurodrive.learn.supervised.demos import RANDOM_START_HEADING, RANDOM_START_LATERAL
from neurodrive.learn.supervised.imitation import ImitationSettings
from neurodrive.learn.supervised.toy_tasks import TOY_TASKS, make_toy_dataset
from neurodrive.learn.supervised.train_toy import LOSSES, ToyConfig

CONFIG_VERSION = 1


class ConfigError(ValueError):
    """A config that cannot be used. ``problems`` lists each one as "field.path: message"."""

    def __init__(self, problems: list[str], source: str = "config") -> None:
        self.problems = problems
        super().__init__(f"invalid {source}:\n" + "\n".join(f"  - {p}" for p in problems))


class _Section(BaseModel):
    """Base for every part of a config: unknown fields are errors, numbers must be finite."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _one_of(value: str, valid, what: str) -> str:
    if value not in valid:
        raise ValueError(f"unknown {what} {value!r}; valid: {', '.join(sorted(valid))}")
    return value


class NetworkSection(_Section):
    """The network's shape: input count, hidden layer sizes, output count, and activations."""

    layer_sizes: list[Annotated[StrictInt, Field(ge=1)]] = Field(min_length=2)
    hidden_activation: str = "tanh"
    output_activation: str = "linear"
    init_scheme: str = "xavier"

    @field_validator("hidden_activation", "output_activation")
    @classmethod
    def _activation(cls, v: str) -> str:
        return _one_of(v, ACTIVATIONS, "activation")

    @field_validator("init_scheme")
    @classmethod
    def _scheme(cls, v: str) -> str:
        return _one_of(v, SCHEMES, "init scheme")


class TrainingSection(_Section):
    """How long and on what: epochs, loss, batch size (null = full batch), validation share."""

    epochs: Annotated[StrictInt, Field(ge=1)]
    loss: str = "mse"
    batch_size: Annotated[StrictInt, Field(ge=1)] | None = None
    val_fraction: float = Field(0.0, ge=0.0, lt=1.0)

    @field_validator("loss")
    @classmethod
    def _loss(cls, v: str) -> str:
        return _one_of(v, LOSSES, "loss")


LearningRate = Annotated[float, Field(gt=0.0)]
Decay = Annotated[float, Field(ge=0.0, lt=1.0)]
Epsilon = Annotated[float, Field(gt=0.0)]


class SGDSection(_Section):
    name: Literal["sgd"]
    learning_rate: LearningRate


class MomentumSection(_Section):
    name: Literal["momentum"]
    learning_rate: LearningRate
    beta: Decay = 0.9


class RMSPropSection(_Section):
    name: Literal["rmsprop"]
    learning_rate: LearningRate
    rho: Decay = 0.9
    eps: Epsilon = 1e-8


class AdamSection(_Section):
    name: Literal["adam"]
    learning_rate: LearningRate
    beta1: Decay = 0.9
    beta2: Decay = 0.999
    eps: Epsilon = 1e-8


OPTIMIZER_SECTIONS: dict[str, type[_Section]] = {
    "sgd": SGDSection,
    "momentum": MomentumSection,
    "rmsprop": RMSPropSection,
    "adam": AdamSection,
}

OptimizerSection = Annotated[
    SGDSection | MomentumSection | RMSPropSection | AdamSection, Field(discriminator="name")
]


@cache
def _task_io(task: str) -> tuple[int, int]:
    """(inputs, outputs) of a toy task, read from a generated dataset."""
    data = make_toy_dataset(task, make_rng(0))
    return data.n_inputs, data.n_outputs


class ToyRunConfig(_Section):
    """A complete toy run: the task, the seed, and the network, training and optimizer sections."""

    method: Literal["toy"]
    task: str
    seed: Annotated[StrictInt, Field(ge=0)] = 0
    tag: str = Field("default", pattern=r"^[A-Za-z0-9_-]+$")
    network: NetworkSection
    training: TrainingSection
    optimizer: OptimizerSection

    @field_validator("task")
    @classmethod
    def _task(cls, v: str) -> str:
        return _one_of(v, TOY_TASKS, "toy task")

    @model_validator(mode="after")
    def _network_fits_task(self) -> "ToyRunConfig":
        n_in, n_out = _task_io(self.task)
        sizes = self.network.layer_sizes
        if (sizes[0], sizes[-1]) != (n_in, n_out):
            raise ValueError(
                f"network.layer_sizes {sizes} do not fit the {self.task} data: it needs "
                f"{n_in} input(s) first and {n_out} output(s) last"
            )
        return self


class DrivingNetworkSection(NetworkSection):
    """A network that drives: 7 observation inputs first, 2 outputs (steer, throttle) last."""

    layer_sizes: list[Annotated[StrictInt, Field(ge=1)]] = Field(
        default_factory=lambda: list(NetworkSpec().layer_sizes), min_length=2
    )
    output_activation: str = "tanh"

    @field_validator("layer_sizes")
    @classmethod
    def _drives(cls, v: list[int]) -> list[int]:
        if (v[0], v[-1]) != (OBSERVATION_SIZE, ACTION_SIZE):
            raise ValueError(
                f"a driving network needs {OBSERVATION_SIZE} inputs first and {ACTION_SIZE} "
                f"outputs last, got {v}"
            )
        return v


class SearchSection(_Section):
    """Random search: how many fresh networks one unit draws and scores."""

    candidates_per_unit: Annotated[StrictInt, Field(ge=1)] = 20


@cache
def _track_split(name: str) -> str:
    """The split (train / heldout) of the track file ``name``; ValueError if there is none."""
    path = DEFAULT_TRACKS_DIR / f"{name}.json"
    if not path.is_file():
        raise ValueError(f"no track {name!r} in tracks/")
    return Track.from_json(path).split


class FitnessSection(_Section):
    """Where fitness is measured: a training track and the starts (the exam's layout by default)."""

    track: str = TRAINING_TRACK
    start_fractions: list[Annotated[float, Field(ge=0.0, lt=1.0)]] = Field(
        default_factory=lambda: list(EXAM_V1.start_fractions), min_length=1
    )
    lateral_offsets: list[Annotated[float, Field(gt=-2.5, lt=2.5)]] = Field(
        default_factory=lambda: list(EXAM_V1.lateral_offsets), min_length=1
    )
    target_laps: Annotated[StrictInt, Field(ge=1)] = 2

    @field_validator("track")
    @classmethod
    def _training_track(cls, v: str) -> str:
        if _track_split(v) != "train":
            raise ValueError(
                f"{v!r} is a held-out track; fitness may only use training tracks "
                "(EXPERIMENT_PROTOCOL rule 7)"
            )
        return v


class BudgetSection(_Section):
    """When the run stops: any limit reached ends it after that unit. At least one is needed."""

    units: Annotated[StrictInt, Field(ge=1)] | None = None
    env_steps: Annotated[StrictInt, Field(ge=1)] | None = None
    wall_clock_s: Annotated[float, Field(gt=0.0)] | None = None

    @model_validator(mode="after")
    def _some_limit(self) -> "BudgetSection":
        if self.units is None and self.env_steps is None and self.wall_clock_s is None:
            raise ValueError("give at least one of units, env_steps, wall_clock_s")
        return self


class RandomSearchRunConfig(_Section):
    """A complete random-search run (task 5.7)."""

    method: Literal["random_search"]
    seed: Annotated[StrictInt, Field(ge=0)] = 0
    tag: str = Field("default", pattern=r"^[A-Za-z0-9_-]+$")
    network: DrivingNetworkSection = Field(default_factory=DrivingNetworkSection)
    search: SearchSection = Field(default_factory=SearchSection)
    fitness: FitnessSection = Field(default_factory=FitnessSection)
    budget: BudgetSection
    exams: bool = True


Name = Annotated[str, Field(pattern=r"^[A-Za-z0-9_-]+$")]


class DataSection(_Section):
    """Imitation: which demonstration dataset, and the share of its episodes held back.

    ``dataset`` is a dataset id or a ``<teacher>-<tag>`` selector (the newest such dataset);
    ``sha256`` (optional) is the checksum the dataset must have, filled in by the runner.
    """

    dataset: Name
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")] | None = None
    val_fraction: float = Field(0.2, ge=0.0, lt=1.0)


class ImitationTrainingSection(_Section):
    """Imitation: how many epochs, the loss and the minibatch size."""

    epochs: Annotated[StrictInt, Field(ge=1)]
    loss: str = "mse"
    batch_size: Annotated[StrictInt, Field(ge=1)] = 256

    @field_validator("loss")
    @classmethod
    def _loss(cls, v: str) -> str:
        return _one_of(v, LOSSES, "loss")


class ImitationRunConfig(_Section):
    """A complete behavioural-cloning run (task 6.3)."""

    method: Literal["imitation"]
    seed: Annotated[StrictInt, Field(ge=0)] = 0
    tag: str = Field("default", pattern=r"^[A-Za-z0-9_-]+$")
    network: DrivingNetworkSection = Field(default_factory=DrivingNetworkSection)
    data: DataSection
    training: ImitationTrainingSection
    optimizer: OptimizerSection
    checkpoint_every: Annotated[StrictInt, Field(ge=1)] | None = None
    exams: bool = True


class DaggerTrainingSection(_Section):
    """DAgger: the epochs of every round, the loss and the minibatch size."""

    epochs_per_round: Annotated[StrictInt, Field(ge=1)] = 30
    loss: str = "mse"
    batch_size: Annotated[StrictInt, Field(ge=1)] = 256

    @field_validator("loss")
    @classmethod
    def _loss(cls, v: str) -> str:
        return _one_of(v, LOSSES, "loss")


class DaggerSection(_Section):
    """DAgger: how many rounds (the run's units), where the network drives when collecting
    (a training track, R fresh random starts per round, moved sideways by up to
    ±``start_lateral_m`` and turned by up to ±``start_heading_rad``) and each drive's goal in
    laps. The start ranges came with the task 6.6 follow-up (ADR-050)."""

    rounds: Annotated[StrictInt, Field(ge=1)]
    track: str = TRAINING_TRACK
    random_starts: Annotated[StrictInt, Field(ge=1)] = 10
    target_laps: Annotated[StrictInt, Field(ge=1)] = 2
    start_lateral_m: float = Field(RANDOM_START_LATERAL, ge=0.0, le=4.5)
    start_heading_rad: float = Field(RANDOM_START_HEADING, ge=0.0, le=math.pi / 4)

    @field_validator("track")
    @classmethod
    def _training_track(cls, v: str) -> str:
        if _track_split(v) != "train":
            raise ValueError(
                f"{v!r} is a held-out track; DAgger may only drive training tracks "
                "(EXPERIMENT_PROTOCOL rule 7)"
            )
        return v


class DaggerRunConfig(_Section):
    """A complete DAgger run (task 6.6, ADR-049)."""

    method: Literal["dagger"]
    seed: Annotated[StrictInt, Field(ge=0)] = 0
    tag: str = Field("default", pattern=r"^[A-Za-z0-9_-]+$")
    network: DrivingNetworkSection = Field(default_factory=DrivingNetworkSection)
    data: DataSection
    training: DaggerTrainingSection = Field(default_factory=DaggerTrainingSection)
    dagger: DaggerSection
    optimizer: OptimizerSection
    checkpoint_every: Annotated[StrictInt, Field(ge=1)] | None = None
    exams: bool = True


RunConfig = ToyRunConfig | RandomSearchRunConfig | ImitationRunConfig | DaggerRunConfig
METHODS = ("toy", "random_search", "imitation", "dagger")
_MODELS: dict[str, type[_Section]] = {
    "random_search": RandomSearchRunConfig,
    "imitation": ImitationRunConfig,
    "dagger": DaggerRunConfig,
}


# --- reading ---


def _format_errors(exc: ValidationError) -> list[str]:
    """One "field.path: message" line per problem, in plain words."""
    problems = []
    for err in exc.errors():
        loc = [str(p) for p in err["loc"]]
        if len(loc) > 1 and loc[0] == "optimizer" and loc[1] in OPTIMIZER_SECTIONS:
            loc.pop(1)  # Pydantic adds the chosen optimizer's name; the file has no such level
        valid = ", ".join(OPTIMIZERS)
        if err["type"] == "extra_forbidden":
            message = "unknown field (not allowed here)"
        elif err["type"] == "union_tag_invalid":  # optimizer.name is not a known optimizer
            loc.append("name")
            message = f"unknown optimizer {err['ctx']['tag']!r}; valid: {valid}"
        elif err["type"] == "union_tag_not_found":  # optimizer.name is missing
            loc.append("name")
            message = f"Field required (one of {valid})"
        else:
            message = err["msg"].removeprefix("Value error, ")
        problems.append(f"{'.'.join(loc) or 'config'}: {message}")
    return problems


def _is_flat_toy_config(data: dict) -> bool:
    """A Phase 3 run-folder config: the learner's flat dictionary, without ``method``."""
    return "method" not in data and "layer_sizes" in data


def parse_config(
    data: Any,
    *,
    seed: int | None = None,
    tag: str | None = None,
    epochs: int | None = None,
    units: int | None = None,
    source: str = "config",
) -> RunConfig:
    """Validate ``data`` (a parsed YAML mapping) and return the config.

    ``seed``, ``tag``, ``epochs`` (toy and imitation: training.epochs) and ``units`` (random
    search: budget.units; DAgger: dagger.rounds) override the file's values when given. A
    Phase 3 flat config is converted first. Raises ``ConfigError`` listing every problem.
    """
    if not isinstance(data, dict):
        raise ConfigError(["the file must contain a mapping (field: value lines)"], source)
    if _is_flat_toy_config(data):
        try:
            data = resolved_from_toy_config(ToyConfig.from_dict(data))
        except (TypeError, ValueError) as exc:
            raise ConfigError([f"old flat toy config: {exc}"], source) from None
    data = {**data}
    if seed is not None:
        data["seed"] = seed
    if tag is not None:
        data["tag"] = tag
    method = data.get("method")
    if epochs is not None:
        if method in ("random_search", "dagger"):
            raise ConfigError(["--epochs is for toy and imitation configs; use --units"], source)
        training = data.get("training")
        data["training"] = (
            {**training, "epochs": epochs} if isinstance(training, dict) else training
        )
    if units is not None and method == "dagger":  # a DAgger run's units are its rounds
        section = data.get("dagger")
        data["dagger"] = (
            {**section, "rounds": units} if isinstance(section, dict) else {"rounds": units}
        )
    elif units is not None:
        if method != "random_search":
            raise ConfigError(
                ["--units is for random_search and dagger configs; use --epochs"], source
            )
        budget = data.get("budget")
        data["budget"] = (
            {**budget, "units": units} if isinstance(budget, dict) else {"units": units}
        )
    # Any other method is read as a toy config, so a missing or unknown method gets the same
    # message as before random search existed.
    model = _MODELS.get(method, ToyRunConfig) if isinstance(method, str) else ToyRunConfig
    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_errors(exc), source) from None


def load_config(
    path: str | Path,
    *,
    seed: int | None = None,
    tag: str | None = None,
    epochs: int | None = None,
    units: int | None = None,
) -> RunConfig:
    """Read and validate the YAML config file at ``path`` (overrides as in :func:`parse_config`)."""
    path = Path(path)
    source = path.name
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError([f"cannot read the file: {exc.strerror or exc}"], source) from None
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        where = f"line {mark.line + 1}" if mark is not None else "the file"
        problem = getattr(exc, "problem", None) or exc
        raise ConfigError([f"not valid YAML at {where}: {problem}"], source) from None
    if data is None:
        raise ConfigError(["the file is empty"], source)
    return parse_config(data, seed=seed, tag=tag, epochs=epochs, units=units, source=source)


# --- converting ---


def resolved(config: RunConfig) -> dict[str, Any]:
    """The config as plain data with every default written out (what a run folder stores)."""
    return config.model_dump(mode="json")


def to_toy_config(config: ToyRunConfig) -> ToyConfig:
    """The learner's flat ``ToyConfig`` for this config."""
    opt = config.optimizer
    settings = {k: v for k, v in opt.model_dump().items() if k not in ("name", "learning_rate")}
    return ToyConfig(
        task=config.task,
        layer_sizes=tuple(config.network.layer_sizes),
        hidden_activation=config.network.hidden_activation,
        output_activation=config.network.output_activation,
        learning_rate=opt.learning_rate,
        epochs=config.training.epochs,
        seed=config.seed,
        loss=config.training.loss,
        init_scheme=config.network.init_scheme,
        tag=config.tag,
        optimizer=opt.name,
        optimizer_settings=settings,
        batch_size=config.training.batch_size,
        val_fraction=config.training.val_fraction,
    )


def to_random_search_settings(config: RandomSearchRunConfig) -> RandomSearchSettings:
    """The learner's ``RandomSearchSettings`` for this config."""
    net = config.network
    return RandomSearchSettings(
        spec=NetworkSpec(
            layer_sizes=tuple(net.layer_sizes),
            hidden_activation=net.hidden_activation,
            output_activation=net.output_activation,
        ),
        init_scheme=net.init_scheme,
        candidates=config.search.candidates_per_unit,
        start_fractions=tuple(config.fitness.start_fractions),
        lateral_offsets=tuple(config.fitness.lateral_offsets),
        target_laps=config.fitness.target_laps,
    )


def to_imitation_settings(config: ImitationRunConfig) -> ImitationSettings:
    """The learner's ``ImitationSettings`` for this config."""
    net, opt = config.network, config.optimizer
    settings = {k: v for k, v in opt.model_dump().items() if k not in ("name", "learning_rate")}
    return ImitationSettings(
        spec=NetworkSpec(
            layer_sizes=tuple(net.layer_sizes),
            hidden_activation=net.hidden_activation,
            output_activation=net.output_activation,
        ),
        init_scheme=net.init_scheme,
        batch_size=config.training.batch_size,
        loss=config.training.loss,
        optimizer=opt.name,
        learning_rate=opt.learning_rate,
        optimizer_settings=settings,
        val_fraction=config.data.val_fraction,
    )


def to_dagger_settings(config: DaggerRunConfig) -> DaggerSettings:
    """The learner's ``DaggerSettings`` for this config (the expert labels with its defaults)."""
    net, opt = config.network, config.optimizer
    settings = {k: v for k, v in opt.model_dump().items() if k not in ("name", "learning_rate")}
    return DaggerSettings(
        imitation=ImitationSettings(
            spec=NetworkSpec(
                layer_sizes=tuple(net.layer_sizes),
                hidden_activation=net.hidden_activation,
                output_activation=net.output_activation,
            ),
            init_scheme=net.init_scheme,
            batch_size=config.training.batch_size,
            loss=config.training.loss,
            optimizer=opt.name,
            learning_rate=opt.learning_rate,
            optimizer_settings=settings,
            val_fraction=config.data.val_fraction,
        ),
        epochs_per_round=config.training.epochs_per_round,
        random_starts=config.dagger.random_starts,
        target_laps=config.dagger.target_laps,
        start_lateral_m=config.dagger.start_lateral_m,
        start_heading_rad=config.dagger.start_heading_rad,
    )


def resolved_from_toy_config(toy: ToyConfig) -> dict[str, Any]:
    """The config-file form (resolved) of a learner ``ToyConfig``."""
    config = ToyRunConfig.model_validate(
        {
            "method": "toy",
            "task": toy.task,
            "seed": toy.seed,
            "tag": toy.tag,
            "network": {
                "layer_sizes": list(toy.layer_sizes),
                "hidden_activation": toy.hidden_activation,
                "output_activation": toy.output_activation,
                "init_scheme": toy.init_scheme,
            },
            "training": {
                "epochs": toy.epochs,
                "loss": toy.loss,
                "batch_size": toy.batch_size,
                "val_fraction": toy.val_fraction,
            },
            "optimizer": {
                "name": toy.optimizer,
                "learning_rate": toy.learning_rate,
                **toy.optimizer_settings,
            },
        }
    )
    return resolved(config)


def toy_config_from_run(data: dict[str, Any]) -> ToyConfig:
    """The ``ToyConfig`` of a run folder's ``config.yaml``, in either the new or the flat form."""
    if _is_flat_toy_config(data):
        return ToyConfig.from_dict(data)
    return to_toy_config(parse_config(data, source="run config"))
