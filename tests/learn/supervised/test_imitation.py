"""Tests for behavioural cloning (task 6.3, ADR-047): the loss, the split by episode, the
overfitting sanity check, the budget, divergence and an exact resume; and finding a dataset."""

import math

import numpy as np
import pytest

from neurodrive.core.losses import mse_grad
from neurodrive.core.nn.gradcheck import gradient_check
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.env.car import ACTION_VERSION
from neurodrive.env.sensors import OBSERVATION_VERSION
from neurodrive.learn.base import Learner, Resumable
from neurodrive.learn.supervised.demos import (
    DATASET_FILE,
    DemoDataset,
    dataset_sha256,
    find_dataset,
)
from neurodrive.learn.supervised.imitation import (
    ImitationLearner,
    ImitationSettings,
    episode_split,
)


def hand_dataset(
    episodes: int = 20, frames: int = 6, seed: int = 0, **meta_overrides
) -> DemoDataset:
    """``episodes`` drives of ``frames`` frames each: random observations in [-1, 1] and labels
    that a network can learn (steer from the first two rays, throttle from the third)."""
    rng = make_rng(seed)
    n = episodes * frames
    obs = rng.uniform(-1.0, 1.0, size=(n, 7))
    label = np.stack([0.5 * np.tanh(obs[:, 0] - obs[:, 1]), 0.6 - 0.2 * obs[:, 2]], axis=1)
    meta = {
        "dataset_version": 1,
        "track": "circle",
        "track_split": "train",
        "observation_version": OBSERVATION_VERSION,
        "action_version": ACTION_VERSION,
        "demo_frames": n,
        **meta_overrides,
    }
    zeros = np.zeros(n)
    return DemoDataset(
        meta=meta,
        observation=obs,
        label=label,
        episode=np.repeat(np.arange(episodes), frames),
        step=np.tile(np.arange(frames), episodes),
        x=zeros,
        y=zeros,
        psi=zeros,
        v=zeros,
    )


def train(learner: ImitationLearner, epochs: int) -> list:
    return [learner.train_unit() for _ in range(epochs)]


# ── the maths: loss, split, budget ──────────────────────────────────────────────────────────


def test_loss_worked_example():
    """Brief §4a: labels (0.1, 1.0), (−0.2, 0.5); outputs (0.0, 0.8), (−0.2, 0.9)."""
    spec = NetworkSpec((7, 2), hidden_activation="tanh", output_activation="tanh")
    weights = np.zeros((2, 7))  # frame 1 has input 0 = 1, frame 2 input 1 = 1: tanh(W) = output
    weights[:, 0] = np.arctanh([0.0, 0.8])
    weights[:, 1] = np.arctanh([-0.2, 0.9])
    net = Network(spec, [weights], [np.zeros(2)])
    x = np.zeros((2, 7))
    x[0, 0] = x[1, 1] = 1.0
    y = np.array([[0.1, 1.0], [-0.2, 0.5]])
    learner = ImitationLearner(
        ImitationSettings(spec=spec, val_fraction=0.0), hand_dataset(1, 2), make_rng(0)
    )
    found = learner.losses(x, y, net)
    assert found["loss"] == pytest.approx(0.0525)
    assert found["loss_steer"] == pytest.approx(0.005)
    assert found["loss_throttle"] == pytest.approx(0.1)
    assert found["loss"] == pytest.approx((found["loss_steer"] + found["loss_throttle"]) / 2)
    # The gradient that starts backpropagation: 2 (ŷ − y) / (2N).
    output = np.array([[0.0, 0.8], [-0.2, 0.9]])
    assert np.allclose(mse_grad(output, y), [[-0.05, -0.1], [0.0, 0.2]])


def test_episode_split_holds_back_whole_episodes():
    episode = np.repeat(np.arange(20), 6)
    split = episode_split(episode, 0.2, make_rng(3))
    assert len(split.val_episodes) == 4  # round(0.2 · 20)
    assert len(split.val_idx) == 24 and len(split.train_idx) == 96
    assert sorted(np.concatenate([split.train_idx, split.val_idx])) == list(range(120))
    assert set(episode[split.val_idx]) == set(split.val_episodes)
    assert not set(episode[split.train_idx]) & set(split.val_episodes)  # no drive on both sides
    again = episode_split(episode, 0.2, make_rng(3))
    assert again.val_episodes == split.val_episodes  # the same seed, the same episodes
    others = {tuple(episode_split(episode, 0.2, make_rng(s)).val_episodes) for s in range(10)}
    assert len(others) > 1  # the seed chooses them


def test_episode_split_edges():
    episode = np.repeat(np.arange(3), 4)
    assert len(episode_split(episode, 0.01, make_rng(0)).val_episodes) == 1  # at least one
    assert episode_split(episode, 0.0, make_rng(0)).val_episodes == []
    with pytest.raises(ValueError, match="no training episode"):
        episode_split(np.zeros(5, dtype=np.int64), 0.5, make_rng(0))  # one episode only


def test_the_dataset_is_charged_once_and_nothing_else():
    data = hand_dataset()
    results = train(ImitationLearner(ImitationSettings(), data, make_rng(0)), 3)
    assert [r.demo_frames for r in results] == [data.n_frames, 0, 0]  # F = 120, in unit 1
    assert all(r.env_steps == 0 and r.teacher_queries == 0 for r in results)


def test_metrics_of_an_epoch():
    data = hand_dataset()
    learner = ImitationLearner(ImitationSettings(batch_size=32), data, make_rng(0))
    assert learner.x_train.shape[0] == 96 and learner.x_val.shape[0] == 24
    first = learner.train_unit().metrics
    assert learner.optimizer.t == math.ceil(96 / 32)  # K = ceil(N / B) Adam steps per epoch
    for prefix in ("", "val_"):
        assert first[f"{prefix}loss"] == pytest.approx(
            (first[f"{prefix}loss_steer"] + first[f"{prefix}loss_throttle"]) / 2
        )
    assert len(first["grad_norm"]) == 3
    later = train(learner, 30)[-1].metrics
    assert later["loss"] < first["loss"] and later["loss_before"] < first["loss_before"]


# ── the sanity check (EXPERIMENT_PROTOCOL §1) ──────────────────────────────────────────────


def test_imitation_overfits_ten_frames():
    data = hand_dataset(episodes=2, frames=5)
    settings = ImitationSettings(batch_size=10, learning_rate=0.01, val_fraction=0.0)
    learner = ImitationLearner(settings, data, make_rng(0))
    results = train(learner, 1500)
    assert results[-1].metrics["loss"] < 1e-4


def test_gradients_match_the_gradient_checker_on_a_driving_batch():
    learner = ImitationLearner(ImitationSettings(), hand_dataset(), make_rng(1))
    check = gradient_check(learner.network, learner.x_train[:16], learner.y_train[:16])
    assert check.passed()


# ── determinism, divergence, resuming ───────────────────────────────────────────────────────


def test_same_seed_same_training_other_seed_differs():
    data = hand_dataset()
    a = [r.metrics for r in train(ImitationLearner(ImitationSettings(), data, make_rng(4)), 3)]
    b = [r.metrics for r in train(ImitationLearner(ImitationSettings(), data, make_rng(4)), 3)]
    c = [r.metrics for r in train(ImitationLearner(ImitationSettings(), data, make_rng(5)), 3)]
    assert a == b and a != c


def test_a_non_finite_loss_is_divergence_and_keeps_the_last_network():
    learner = ImitationLearner(ImitationSettings(), hand_dataset(), make_rng(0))
    train(learner, 2)
    before = learner.network.clone()
    learner.y_train[0, 0] = np.nan  # the loader reads the same array
    result = learner.train_unit()
    assert result.diverged and result.metrics["loss"] is None
    assert learner.network.equals(before)


@pytest.mark.parametrize("optimizer", ["sgd", "momentum", "rmsprop", "adam"])
def test_state_and_restore_continue_exactly(optimizer):
    data = hand_dataset()
    settings = ImitationSettings(optimizer=optimizer, learning_rate=0.01, batch_size=16)
    whole = ImitationLearner(settings, data, make_rng(2))
    train(whole, 3)
    state = whole.state()
    expected = [r.metrics for r in train(whole, 2)]
    resumed = ImitationLearner(settings, data, make_rng(2))
    resumed.restore(state)
    assert [r.metrics for r in train(resumed, 2)] == expected
    assert resumed.network.equals(whole.network) and resumed.epochs == whole.epochs == 5


def test_follows_the_learner_protocol():
    learner = ImitationLearner(ImitationSettings(), hand_dataset(), make_rng(0))
    assert isinstance(learner, Learner) and isinstance(learner, Resumable)
    assert (learner.name, learner.feedback_kind) == ("imitation", "labels")


def test_untrainable_datasets_and_settings_are_refused():
    with pytest.raises(ValueError, match="rule 7"):
        ImitationLearner(ImitationSettings(), hand_dataset(track_split="heldout"), make_rng(0))
    with pytest.raises(ValueError, match="observation_version"):
        ImitationLearner(ImitationSettings(), hand_dataset(observation_version=0), make_rng(0))
    with pytest.raises(ValueError, match="driving network"):
        ImitationSettings(spec=NetworkSpec((2, 4, 1)))
    with pytest.raises(ValueError, match="unknown loss"):
        ImitationSettings(loss="hinge")
    with pytest.raises(ValueError):
        ImitationSettings(optimizer="adam", learning_rate=-1.0)


# ── finding a dataset by name (ADR-047) ─────────────────────────────────────────────────────


def test_find_dataset_by_id_and_by_newest_selector(tmp_path):
    data = hand_dataset(2, 3)
    for name in (
        "20261104-084617-expert-default",
        "20261105-090000-expert-default",
        "20261105-090000-expert-default-2",  # recorded in the same second: newer
        "20261106-000000-expert-default-b",  # another tag: never matches expert-default
        "20261106-000000-keyboard-default",
    ):
        data.save(tmp_path / name)
    (tmp_path / "20261107-000000-expert-default").mkdir()  # not a dataset folder: ignored
    assert find_dataset(tmp_path, "expert-default").name == "20261105-090000-expert-default-2"
    assert find_dataset(tmp_path, "keyboard-default").name == "20261106-000000-keyboard-default"
    exact = "20261104-084617-expert-default"
    assert find_dataset(tmp_path, exact).name == exact
    with pytest.raises(FileNotFoundError, match="no dataset 'expert-other'"):
        find_dataset(tmp_path, "expert-other")
    with pytest.raises(ValueError):
        find_dataset(tmp_path, "../expert-default")


def test_the_checksum_sees_any_change(tmp_path):
    folder = hand_dataset(2, 3).save(tmp_path / "d")
    before = dataset_sha256(folder)
    assert len(before) == 64 and before == dataset_sha256(folder)
    raw = bytearray((folder / DATASET_FILE).read_bytes())
    raw[-1] ^= 1  # one bit of the file
    (folder / DATASET_FILE).write_bytes(bytes(raw))
    assert dataset_sha256(folder) != before
