"""Config files: the written recipe of an experiment, validated before anything runs (ADR-032).

A config file is YAML. Its ``method`` field says which kind of run it describes; only ``toy``
exists so far. A toy config (config v1, docs/ARCHITECTURE.md §6.15) looks like::

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

The learner keeps its own flat ``ToyConfig`` (``learn/`` does not use Pydantic); this module
converts between the file and it. Every list of valid names comes from the module that owns it.
"""

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
from neurodrive.core.optim import OPTIMIZERS
from neurodrive.core.rng import make_rng
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
    source: str = "config",
) -> ToyRunConfig:
    """Validate ``data`` (a parsed YAML mapping) and return the config.

    ``seed``, ``tag`` and ``epochs`` override the file's values when given. A Phase 3 flat
    config is converted first. Raises ``ConfigError`` listing every problem.
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
    if epochs is not None:
        training = data.get("training")
        data["training"] = (
            {**training, "epochs": epochs} if isinstance(training, dict) else training
        )
    try:
        return ToyRunConfig.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(_format_errors(exc), source) from None


def load_config(
    path: str | Path,
    *,
    seed: int | None = None,
    tag: str | None = None,
    epochs: int | None = None,
) -> ToyRunConfig:
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
    return parse_config(data, seed=seed, tag=tag, epochs=epochs, source=source)


# --- converting ---


def resolved(config: ToyRunConfig) -> dict[str, Any]:
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
