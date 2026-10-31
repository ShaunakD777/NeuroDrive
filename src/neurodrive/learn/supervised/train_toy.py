"""Gradient descent on the toy tasks: the first learner (learner contract v0, ADR-030, ADR-032).

Each unit is one *epoch*: one pass over the training examples. With θ all the network's
parameters, L(θ) the loss, N the number of training examples and B the batch size (B = N when no
batch size is set: full-batch gradient descent), an epoch is K = ceil(N / B) optimizer steps:

1. forward all N training examples, compute L and its gradient ∂L/∂θ by backpropagation
   (``core/nn/backward.py``); these give ``loss_before`` and ``grad_norm``;
2. for each of the K minibatches (``learn/supervised/datasets.py``): forward the batch, compute
   the batch loss's gradient (an estimate of the full gradient, since the batch loss is the
   average over the batch), backpropagate, and let the optimizer step (``core/optim``: SGD
   θ ← θ − η · ∂L/∂θ, or Momentum, RMSProp, Adam). With a full batch the gradient of step 1 is
   used directly;
3. forward again to measure ``loss`` (L after the epoch, so it belongs to the network a checkpoint
   of this unit holds) and, when a validation set was held back, ``val_loss`` on it.

``grad_norm`` is, for each layer, the length √(Σ ∂L/∂W² + Σ ∂L/∂b²) of the full gradient at the
start of the epoch; small values in early layers are a sign of vanishing gradients.

If a loss, a gradient or a stepped parameter is not finite (a too-large η can make the
parameters grow without bound), the unit is reported as *diverged* and none of its steps are
kept: the network stays as it was after the last completed epoch, which is the network the runner
checkpoints under that epoch's number. The defaults (SGD, full batch, no
validation set) give exactly the runs of Phase 3. This module does no file input/output; the
runner (``experiment/runner.py``) writes the run folder.
"""

from dataclasses import asdict, dataclass, field, replace

import numpy as np

from neurodrive.core.losses import huber, huber_grad, mse, mse_grad
from neurodrive.core.nn.backward import NetworkGradients, backward
from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.optim import make_optimizer
from neurodrive.core.optim.base import grads_of, network_from_params, params_of
from neurodrive.core.rng import Rng, child_rng
from neurodrive.learn.base import FeedbackKind, UnitResult
from neurodrive.learn.supervised.datasets import MinibatchLoader, train_val_split
from neurodrive.learn.supervised.toy_tasks import TOY_TASKS, ToyDataset, make_toy_dataset

LOSSES = {"mse": (mse, mse_grad), "huber": (huber, huber_grad)}


@dataclass(frozen=True)
class ToyConfig:
    """Everything that defines a toy run, in the learner's own flat form.

    ``layer_sizes`` must start with the task's input count and end with its output count.
    ``learning_rate`` is η, ``epochs`` the number of units, ``seed`` the master seed (the data,
    the validation split, the starting weights and the batch order come from separate child
    generators of it). ``init_scheme`` is passed to ``init_network`` ("xavier", "uniform" or
    "zeros"). ``optimizer`` is a name from ``core.optim.OPTIMIZERS`` and ``optimizer_settings``
    its other settings (e.g. ``{"beta1": 0.8}``). ``batch_size`` B is the minibatch size (None:
    the whole training set, full batch). ``val_fraction`` is the share of examples held back for
    validation (0: none). ``tag`` labels the run folder. Config files (``experiment/config.py``)
    are converted to this form.
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
    optimizer: str = "sgd"
    optimizer_settings: dict[str, float] = field(default_factory=dict)
    batch_size: int | None = None
    val_fraction: float = 0.0

    def __post_init__(self) -> None:
        if self.task not in TOY_TASKS:
            raise ValueError(f"unknown toy task {self.task!r}; valid: {', '.join(TOY_TASKS)}")
        if self.loss not in LOSSES:
            raise ValueError(f"unknown loss {self.loss!r}; valid: {', '.join(LOSSES)}")
        if not (np.isfinite(self.learning_rate) and self.learning_rate > 0):
            raise ValueError(f"learning_rate must be positive, got {self.learning_rate}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.batch_size is not None and (
            isinstance(self.batch_size, bool)
            or int(self.batch_size) != self.batch_size
            or self.batch_size < 1
        ):
            raise ValueError(
                f"batch_size must be a whole number >= 1 or None, got {self.batch_size}"
            )
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        object.__setattr__(self, "layer_sizes", tuple(int(s) for s in self.layer_sizes))
        object.__setattr__(self, "optimizer_settings", dict(self.optimizer_settings))
        make_optimizer(self.optimizer, self.learning_rate, self.optimizer_settings)  # validates

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
        """Inverse of :meth:`to_dict`. Fields added later (optimizer, ...) may be missing."""
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
    # The overfitting set-up of Study S3 (task 4.8): a big network on 48 noisy training points.
    "noisy_sine": ToyConfig(
        "noisy_sine",
        (1, 64, 64, 1),
        "tanh",
        "linear",
        learning_rate=0.01,
        epochs=5000,
        optimizer="adam",
        optimizer_settings={"beta1": 0.9, "beta2": 0.999, "eps": 1e-8},
        val_fraction=0.25,
    ),
}


def _all_finite(arrays: list[np.ndarray]) -> bool:
    return all(bool(np.all(np.isfinite(a))) for a in arrays)


class ToyLearner:
    """Gradient descent with a chosen optimizer on one toy task (``learn.base.Learner`` v0)."""

    feedback_kind: FeedbackKind = "labels"

    def __init__(self, config: ToyConfig, rng: Rng, dataset: ToyDataset | None = None) -> None:
        """Build the data, the split, the starting network and the optimizer from ``rng``.

        ``rng`` is ``make_rng(config.seed)``. ``dataset`` replaces the task's generated data
        (used for hand-worked examples); it is still split when ``val_fraction`` > 0.
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
        x, y = self.dataset.x, self.dataset.y
        if config.val_fraction > 0.0:
            split = train_val_split(x, y, config.val_fraction, child_rng(rng, "split"))
            self.x_train, self.y_train = split.x_train, split.y_train
            self.x_val, self.y_val = split.x_val, split.y_val
        else:
            self.x_train, self.y_train = x, y
            self.x_val = self.y_val = None
        self.network = init_network(spec, child_rng(rng, "init"), scheme=config.init_scheme)
        self.optimizer = make_optimizer(
            config.optimizer, config.learning_rate, config.optimizer_settings
        )
        n = self.x_train.shape[0]
        batch = n if config.batch_size is None else min(config.batch_size, n)
        # A full batch keeps the rows in their original order (no shuffle), so the arithmetic,
        # and therefore every bit of the result, is the same as in Phase 3.
        self.loader = MinibatchLoader(
            self.x_train, self.y_train, batch, child_rng(rng, "batches"), shuffle=batch < n
        )
        self._loss, self._loss_grad = LOSSES[config.loss]

    def loss(self, network: Network | None = None) -> float:
        """The loss of ``network`` (default: the current one) on the training examples."""
        net = self.network if network is None else network
        return self._loss(forward(net, self.x_train).output, self.y_train)

    def val_loss(self, network: Network | None = None) -> float | None:
        """The loss on the validation examples, or None when none were held back."""
        if self.x_val is None or self.y_val is None:
            return None
        net = self.network if network is None else network
        return self._loss(forward(net, self.x_val).output, self.y_val)

    def _gradients(self, net: Network, x: np.ndarray, y: np.ndarray) -> NetworkGradients:
        trace = forward(net, x)
        return backward(net, trace, self._loss_grad(trace.output, y))

    def _step(self, net: Network, grads: NetworkGradients) -> Network | None:
        """One optimizer step from ``net``; None if a gradient or a new parameter is not finite."""
        g = grads_of(grads)
        if not _all_finite(g):
            return None
        new = self.optimizer.step(params_of(net), g)
        return network_from_params(net, new) if _all_finite(new) else None

    def train_unit(self) -> UnitResult:
        """One epoch (see the module docstring)."""
        net, x, y = self.network, self.x_train, self.y_train
        diverged = UnitResult(metrics={"loss": None, "loss_before": None}, diverged=True)
        with np.errstate(all="ignore"):  # overflow is detected below, not warned about
            trace = forward(net, x)
            loss_before = self._loss(trace.output, y)
            grads = backward(net, trace, self._loss_grad(trace.output, y))
            if not (np.isfinite(loss_before) and _all_finite(grads.weights + grads.biases)):
                return diverged
            if len(self.loader) == 1 and not self.loader.shuffle:
                stepped = self._step(net, grads)  # full batch: the gradient is already known
                if stepped is None:
                    return diverged
                net = stepped
            else:
                for xb, yb in self.loader:
                    stepped = self._step(net, self._gradients(net, xb, yb))
                    if stepped is None:
                        return diverged
                    net = stepped
            loss_after = self.loss(net)
            val = self.val_loss(net)
        if not np.isfinite(loss_after) or (val is not None and not np.isfinite(val)):
            return diverged
        self.network = net  # kept only when everything stayed finite
        grad_norm = [
            float(np.sqrt(np.sum(gw * gw) + np.sum(gb * gb)))
            for gw, gb in zip(grads.weights, grads.biases, strict=True)
        ]
        metrics: dict = {"loss": loss_after, "loss_before": loss_before, "grad_norm": grad_norm}
        if val is not None:
            metrics["val_loss"] = val
        return UnitResult(metrics=metrics)

    def current_networks(self) -> list[Network]:
        return [self.network]
