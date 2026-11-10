"""Behavioural cloning (task 6.3, ADR-047): learn to drive by copying a teacher's actions.

The data is a demonstration dataset (``learn/supervised/demos.py``): F frames, each an
observation o_i (the 7 sensor readings) and a label y_i = (y_i,steer, y_i,throttle), what the
teacher did in that moment. The driving network f_θ (θ: all its weights and biases) is trained,
exactly like the toy tasks of Phase 3, to output the label for each observation. With
ŷ_i = f_θ(o_i) its two outputs (tanh, so each in (−1, 1)) and N training frames, the loss is the
mean squared error over all 2N numbers:

    L = (1 / (2N)) · Σ_i Σ_k (ŷ_i,k − y_i,k)²

L_steer and L_throttle are the same average over one output only, so L = (L_steer + L_throttle) / 2.
Two frames with labels (0.1, 1.0), (−0.2, 0.5) and outputs (0.0, 0.8), (−0.2, 0.9) have squared
errors 0.01, 0.04, 0.00, 0.16: L = 0.21 / 4 = 0.0525, L_steer = 0.005, L_throttle = 0.1.

**The split by episode** (:func:`episode_split`). Frames 1/30 s apart are almost identical, so a
validation set of random frames would have a near twin of every frame in the training set. Whole
episodes (drives) are held back instead: with E episodes and validation share v, round(v · E)
episodes (at least 1, and at least one left for training), chosen by a seeded shuffle of the
episode numbers. E = 20, v = 0.2: 4 episodes.

**One unit is one epoch**, as in ``train_toy.py``: the loss and gradient on all training frames
(``loss_before``, ``grad_norm``), then K = ceil(N / B) minibatch steps of the optimizer (Adam by
default), then the losses after the epoch on the training and the validation frames. If any
loss, gradient or parameter stops being finite the unit is reported as diverged and the network
stays as it was after the last completed epoch.

**The budget.** The learner reads every frame of the dataset, validation frames included (they
were recorded too), so it reports ``demo_frames`` = F once, in unit 1. It drives no car while
learning (``env_steps`` 0; the exam is never counted) and asks no teacher (``teacher_queries`` 0).

The learner is resumable: its state is the network, the optimizer's running averages and step
count, the batch generator's state and the epoch count.
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from neurodrive.core.nn.backward import NetworkGradients, backward
from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.optim import make_optimizer
from neurodrive.core.optim.base import grads_of, network_from_params, params_of
from neurodrive.core.rng import Rng, child_rng
from neurodrive.env.car import ACTION_SIZE, ACTION_VERSION
from neurodrive.env.sensors import OBSERVATION_SIZE, OBSERVATION_VERSION
from neurodrive.learn.base import (
    FeedbackKind,
    LearnerState,
    UnitResult,
    generator_state,
    set_generator_state,
)
from neurodrive.learn.supervised.datasets import MinibatchLoader
from neurodrive.learn.supervised.demos import DemoDataset
from neurodrive.learn.supervised.train_toy import LOSSES

IntArray = NDArray[np.int64]

# The running averages an optimizer keeps between steps (core/optim), saved with the state.
_OPTIMIZER_BUFFERS = ("velocities", "square_averages", "first_moments", "second_moments")
_OUTPUTS = ("steer", "throttle")


@dataclass(frozen=True)
class ImitationSettings:
    """Everything behavioural cloning needs besides the dataset and the generator.

    ``spec`` is the driving network (7 inputs, 2 outputs), ``init_scheme`` its starting weights,
    ``batch_size`` B, ``loss`` a name from ``train_toy.LOSSES``, ``optimizer`` a name from
    ``core.optim.OPTIMIZERS`` with ``learning_rate`` η and its other ``optimizer_settings``,
    ``val_fraction`` v the share of episodes held back (0: none).
    """

    spec: NetworkSpec = field(default_factory=NetworkSpec)
    init_scheme: str = "xavier"
    batch_size: int = 256
    loss: str = "mse"
    optimizer: str = "adam"
    learning_rate: float = 0.003
    optimizer_settings: dict[str, float] = field(default_factory=dict)
    val_fraction: float = 0.2

    def __post_init__(self) -> None:
        if (self.spec.n_inputs, self.spec.n_outputs) != (OBSERVATION_SIZE, ACTION_SIZE):
            raise ValueError(
                f"a driving network needs {OBSERVATION_SIZE} inputs and {ACTION_SIZE} outputs, "
                f"got {list(self.spec.layer_sizes)}"
            )
        if self.loss not in LOSSES:
            raise ValueError(f"unknown loss {self.loss!r}; valid: {', '.join(LOSSES)}")
        if isinstance(self.batch_size, bool) or self.batch_size < 1:
            raise ValueError(f"batch_size must be a whole number >= 1, got {self.batch_size}")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError(f"val_fraction must be in [0, 1), got {self.val_fraction}")
        object.__setattr__(self, "optimizer_settings", dict(self.optimizer_settings))
        make_optimizer(self.optimizer, self.learning_rate, self.optimizer_settings)  # validates


@dataclass(frozen=True)
class EpisodeSplit:
    """Row numbers of the training and validation frames, and the held-back episode numbers."""

    train_idx: IntArray
    val_idx: IntArray
    val_episodes: list[int]


def episode_split(episode: IntArray, val_fraction: float, rng: Rng) -> EpisodeSplit:
    """Hold back whole episodes for validation (see the module docstring).

    1. List the E distinct episode numbers in ``episode`` (one per frame).
    2. n_val = round(v · E), at least 1 when v > 0; refuse if that leaves no training episode.
    3. Shuffle the episode numbers with ``rng``; the first n_val are held back.
    4. A frame is a validation frame exactly when its episode is held back. Rows keep their order.
    """
    episode = np.asarray(episode, dtype=np.int64)
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in [0, 1), got {val_fraction}")
    episodes = np.unique(episode)
    n_val = int(round(val_fraction * len(episodes)))
    if val_fraction > 0.0:
        n_val = max(n_val, 1)
    if n_val >= len(episodes):
        raise ValueError(
            f"val_fraction {val_fraction} of {len(episodes)} episode(s) leaves no training episode"
        )
    held = np.sort(rng.permutation(episodes)[:n_val])
    is_val = np.isin(episode, held)
    return EpisodeSplit(
        train_idx=np.flatnonzero(~is_val),
        val_idx=np.flatnonzero(is_val),
        val_episodes=[int(e) for e in held],
    )


def check_trainable(dataset: DemoDataset) -> None:
    """Refuse a dataset a driving network must not learn from: one recorded on a held-out track
    (EXPERIMENT_PROTOCOL rule 7) or with other observation or action contracts."""
    meta = dataset.meta
    if meta.get("track_split") != "train":
        raise ValueError(
            f"dataset track {meta.get('track')!r} is not a training track; imitation may only "
            "learn from training tracks (EXPERIMENT_PROTOCOL rule 7)"
        )
    for key, version in (
        ("observation_version", OBSERVATION_VERSION),
        ("action_version", ACTION_VERSION),
    ):
        if meta.get(key) != version:
            raise ValueError(f"dataset {key} is {meta.get(key)!r}; this code uses {version}")


def _all_finite(arrays: list[np.ndarray]) -> bool:
    return all(bool(np.all(np.isfinite(a))) for a in arrays)


class ImitationLearner:
    """Behavioural cloning as a learner (learner v1, resumable)."""

    name = "imitation"
    feedback_kind: FeedbackKind = "labels"

    def __init__(self, settings: ImitationSettings, dataset: DemoDataset, rng: Rng) -> None:
        """Split the dataset, build the starting network, the optimizer and the batch loader.

        ``rng`` is ``make_rng(seed)``; the split, the starting weights and the batch order come
        from its child generators "split", "init" and "batches", so changing one never shifts
        the others.
        """
        check_trainable(dataset)
        self.settings = settings
        self.demo_frames = dataset.n_frames
        x, y = dataset.observation, dataset.label
        if settings.val_fraction > 0.0:
            split = episode_split(dataset.episode, settings.val_fraction, child_rng(rng, "split"))
            self.val_episodes = split.val_episodes
            self.x_train, self.y_train = x[split.train_idx], y[split.train_idx]
            self.x_val, self.y_val = x[split.val_idx], y[split.val_idx]
        else:
            self.val_episodes = []
            self.x_train, self.y_train = x, y
            self.x_val = self.y_val = None
        self.network = init_network(
            settings.spec, child_rng(rng, "init"), scheme=settings.init_scheme
        )
        self.optimizer = make_optimizer(
            settings.optimizer, settings.learning_rate, settings.optimizer_settings
        )
        batch = min(settings.batch_size, self.x_train.shape[0])
        self.loader = MinibatchLoader(self.x_train, self.y_train, batch, child_rng(rng, "batches"))
        self._loss, self._loss_grad = LOSSES[settings.loss]
        self.epochs = 0

    def losses(self, x: np.ndarray, y: np.ndarray, network: Network | None = None) -> dict:
        """L, L_steer and L_throttle of ``network`` (default: the current one) on (x, y)."""
        out = forward(self.network if network is None else network, x).output
        found = {"loss": self._loss(out, y)}
        for k, name in enumerate(_OUTPUTS):
            found[f"loss_{name}"] = self._loss(out[:, k : k + 1], y[:, k : k + 1])
        return found

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

    def set_training_frames(self, x: np.ndarray, y: np.ndarray) -> None:
        """Train on (x, y) from now on (DAgger, task 6.6, adds the frames it collects).

        The batch loader is rebuilt over the new frames with the same generator, which simply
        continues: the next epoch's shuffle is its next draw.
        """
        self.x_train, self.y_train = x, y
        batch = min(self.settings.batch_size, x.shape[0])
        self.loader = MinibatchLoader(x, y, batch, self.loader.rng)

    def _epoch(self) -> dict | None:
        """One guarded epoch on the training frames (see the module docstring).

        Returns the epoch's metrics and keeps the new network, or returns None, leaving the
        network as it was, if any loss, gradient or parameter stopped being finite.
        """
        net, x, y = self.network, self.x_train, self.y_train
        with np.errstate(all="ignore"):  # overflow is detected below, not warned about
            trace = forward(net, x)
            loss_before = self._loss(trace.output, y)
            grads = backward(net, trace, self._loss_grad(trace.output, y))
            if not (np.isfinite(loss_before) and _all_finite(grads.weights + grads.biases)):
                return None
            for xb, yb in self.loader:
                stepped = self._step(net, self._gradients(net, xb, yb))
                if stepped is None:
                    return None
                net = stepped
            metrics: dict = self.losses(x, y, net)
            if self.x_val is not None and self.y_val is not None:
                val = self.losses(self.x_val, self.y_val, net)
                metrics.update({f"val_{key}": value for key, value in val.items()})
        if not all(np.isfinite(v) for v in metrics.values()):
            return None
        self.network = net  # kept only when everything stayed finite
        metrics["loss_before"] = loss_before
        metrics["grad_norm"] = [
            float(np.sqrt(np.sum(gw * gw) + np.sum(gb * gb)))
            for gw, gb in zip(grads.weights, grads.biases, strict=True)
        ]
        return metrics

    def train_unit(self) -> UnitResult:
        """One epoch (see the module docstring)."""
        self.epochs += 1
        frames = self.demo_frames if self.epochs == 1 else 0  # the dataset is read once
        metrics = self._epoch()
        if metrics is None:
            return UnitResult(
                metrics={"loss": None, "loss_before": None}, demo_frames=frames, diverged=True
            )
        return UnitResult(metrics=metrics, demo_frames=frames)

    def current_networks(self) -> list[Network]:
        return [self.network]

    def state(self) -> LearnerState:
        arrays = {"genome": self.network.flatten()}
        buffers = {}
        for name in _OPTIMIZER_BUFFERS:
            values = getattr(self.optimizer, name, None)
            if values is not None:
                buffers[name] = len(values)
                arrays.update({f"opt_{name}_{i}": v for i, v in enumerate(values)})
        return LearnerState(
            arrays=arrays,
            info={
                "epochs": self.epochs,
                "optimizer_buffers": buffers,
                "optimizer_t": getattr(self.optimizer, "t", None),
                "batches_rng": generator_state(self.loader.rng),
            },
        )

    def restore(self, state: LearnerState) -> None:
        info = state.info
        self.network = Network.unflatten(self.settings.spec, state.arrays["genome"])
        for name, count in info["optimizer_buffers"].items():
            setattr(self.optimizer, name, [state.arrays[f"opt_{name}_{i}"] for i in range(count)])
        if info["optimizer_t"] is not None:
            self.optimizer.t = info["optimizer_t"]
        set_generator_state(self.loader.rng, info["batches_rng"])
        self.epochs = info["epochs"]
