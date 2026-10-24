"""Plain gradient descent on the toy tasks: the first learner (learner contract v0, ADR-030).

Each unit is one *epoch*: one pass over the whole toy dataset (full-batch gradient descent).
With θ all the network's parameters, η the learning rate and L(θ) the loss over the dataset:

1. forward the whole dataset and compute L and the upstream gradient ∂L/∂output;
2. backpropagate to get ∂L/∂θ for every weight and bias (``core/nn/backward.py``);
3. step downhill: θ ← θ − η · ∂L/∂θ.

The learner reports ``loss_before`` (L before the step, measured during the epoch), ``loss``
(L after the step, so it belongs to the network a checkpoint of this unit holds) and
``grad_norm`` (for each layer, the length √(Σ ∂L/∂W² + Σ ∂L/∂b²) of its gradient; small values
in early layers are a sign of vanishing gradients).

If a loss, a gradient or a stepped parameter is not finite (a too-large η can make the
parameters grow without bound), the unit is reported as *diverged*, the step is not applied and
the network keeps its last finite parameters. This module does no file input/output; the runner
(``experiment/runner.py``) writes the run folder.
"""

from dataclasses import asdict, dataclass, replace

import numpy as np

from neurodrive.core.losses import huber, huber_grad, mse, mse_grad
from neurodrive.core.nn.backward import backward
from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import Rng, child_rng
from neurodrive.learn.base import FeedbackKind, UnitResult
from neurodrive.learn.supervised.toy_tasks import TOY_TASKS, ToyDataset, make_toy_dataset

LOSSES = {"mse": (mse, mse_grad), "huber": (huber, huber_grad)}


@dataclass(frozen=True)
class ToyConfig:
    """Everything that defines a toy run (it is written to the run folder's ``config.yaml``).

    ``layer_sizes`` must start with the task's input count and end with its output count.
    ``learning_rate`` is η, ``epochs`` the number of units, ``seed`` the master seed (the data and
    the starting weights come from separate child generators of it). ``init_scheme`` is passed to
    ``init_network`` ("xavier", "uniform" or "zeros"). ``tag`` labels the run folder.
    """

    task: str
    layer_sizes: tuple[int, ...]
    hidden_activation: str = "tanh"
    output_activation: str = "linear"
    learning_rate: float = 0.1
    epochs: int = 1000
    seed: int = 0
    loss: str = "mse"
    init_scheme: str = "xavier"
    tag: str = "default"

    def __post_init__(self) -> None:
        if self.task not in TOY_TASKS:
            raise ValueError(f"unknown toy task {self.task!r}; valid: {', '.join(TOY_TASKS)}")
        if self.loss not in LOSSES:
            raise ValueError(f"unknown loss {self.loss!r}; valid: {', '.join(LOSSES)}")
        if not (np.isfinite(self.learning_rate) and self.learning_rate > 0):
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        object.__setattr__(self, "layer_sizes", tuple(int(s) for s in self.layer_sizes))

    @property
    def spec(self) -> NetworkSpec:
        return NetworkSpec(
            self.layer_sizes,
            hidden_activation=self.hidden_activation,
            output_activation=self.output_activation,
        )

    def to_dict(self) -> dict:
        """Plain dictionary (lists, not tuples) for writing to YAML."""
        d = asdict(self)
        d["layer_sizes"] = list(self.layer_sizes)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ToyConfig":
        return cls(**{**d, "layer_sizes": tuple(d["layer_sizes"])})

    @classmethod
    def for_task(cls, task: str, **overrides) -> "ToyConfig":
        """The default config for ``task`` (values chosen in task 3.7), with ``overrides``."""
        return replace(DEFAULTS[task], **overrides)


DEFAULTS = {
    "line": ToyConfig("line", (1, 1), "linear", "linear", learning_rate=0.5, epochs=200),
    "sine": ToyConfig("sine", (1, 16, 1), "tanh", "linear", learning_rate=0.05, epochs=3000),
    "xor": ToyConfig("xor", (2, 4, 1), "tanh", "sigmoid", learning_rate=1.0, epochs=3000),
    "two_spirals": ToyConfig(
        "two_spirals", (2, 16, 16, 1), "tanh", "sigmoid", learning_rate=0.5, epochs=5000
    ),
}


def _all_finite(arrays: list[np.ndarray]) -> bool:
    return all(bool(np.all(np.isfinite(a))) for a in arrays)


class ToyLearner:
    """Full-batch gradient descent on one toy task (follows ``learn.base.Learner`` v0)."""

    feedback_kind: FeedbackKind = "labels"

    def __init__(self, config: ToyConfig, rng: Rng, dataset: ToyDataset | None = None) -> None:
        """Build the dataset and the starting network from ``rng`` (``make_rng(config.seed)``).

        ``dataset`` replaces the task's generated data (used for hand-worked examples).
        """
        self.config = config
        self.name = f"toy_{config.task}"
        self.dataset = (
            dataset
            if dataset is not None
            else make_toy_dataset(config.task, child_rng(rng, "data"))
        )
        spec = config.spec
        if (self.dataset.n_inputs, self.dataset.n_outputs) != (spec.n_inputs, spec.n_outputs):
            raise ValueError(
                f"layer_sizes {config.layer_sizes} do not fit the {config.task} data "
                f"({self.dataset.n_inputs} inputs, {self.dataset.n_outputs} outputs)"
            )
        self.network = init_network(spec, child_rng(rng, "init"), scheme=config.init_scheme)
        self._loss, self._loss_grad = LOSSES[config.loss]

    def loss(self, network: Network | None = None) -> float:
        """The loss of ``network`` (default: the current one) on the whole dataset."""
        net = self.network if network is None else network
        return self._loss(forward(net, self.dataset.x).output, self.dataset.y)

    def train_unit(self) -> UnitResult:
        """One epoch of plain gradient descent (see the module docstring)."""
        net, x, y = self.network, self.dataset.x, self.dataset.y
        diverged = UnitResult(metrics={"loss": None, "loss_before": None}, diverged=True)
        with np.errstate(all="ignore"):  # overflow is detected below, not warned about
            trace = forward(net, x)
            loss_before = self._loss(trace.output, y)
            grads = backward(net, trace, self._loss_grad(trace.output, y))
            if not (np.isfinite(loss_before) and _all_finite(grads.weights + grads.biases)):
                return diverged
            lr = self.config.learning_rate
            new_w = [w - lr * g for w, g in zip(net.weights, grads.weights, strict=True)]
            new_b = [b - lr * g for b, g in zip(net.biases, grads.biases, strict=True)]
            if not _all_finite(new_w + new_b):
                return diverged
            stepped = Network(net.spec, new_w, new_b)  # θ − η ∂L/∂θ
            loss_after = self.loss(stepped)
        if not np.isfinite(loss_after):
            return diverged
        self.network = stepped  # kept only when everything stayed finite
        grad_norm = [
            float(np.sqrt(np.sum(gw * gw) + np.sum(gb * gb)))
            for gw, gb in zip(grads.weights, grads.biases, strict=True)
        ]
        return UnitResult(
            metrics={"loss": loss_after, "loss_before": loss_before, "grad_norm": grad_norm}
        )

    def current_networks(self) -> list[Network]:
        return [self.network]
