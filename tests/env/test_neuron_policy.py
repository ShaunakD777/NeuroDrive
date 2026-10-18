"""Tests for NeuronPolicy: the maths, parameter changes, and that a hand-wired neuron can drive."""

import math
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.env.episode import run_episode
from neurodrive.env.policies import NEURON_INPUTS, NeuronPolicy, Policy
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig

TRACK_FILE = Path(__file__).resolve().parents[2] / "tracks" / "train_loop_01.json"


def observation(d_left: float, d_right: float) -> np.ndarray:
    """A 7-number observation with the two rays the neuron reads set and everything else 0."""
    obs = np.zeros(7)
    obs[0], obs[4] = d_left, d_right
    return obs


# ── the maths: the Brief's worked example ───────────────────────────────────────────────────


def test_worked_example_wall_near_on_the_left_steers_right():
    # w_left = -4, w_right = 4, bias 0.1; d_left = 0.2, d_right = 0.5:
    # z = (-4)(0.2) + (4)(0.5) + 0.1 = 1.3 and tanh(1.3) = 0.8617
    policy = NeuronPolicy((-4.0, 4.0), 0.1)
    actions = policy.act(observation(0.2, 0.5)[None, :])
    assert actions[0, 0] == pytest.approx(math.tanh(1.3), abs=1e-9)
    assert actions[0, 0] == pytest.approx(0.8617, abs=1e-4)
    assert actions[0, 1] == pytest.approx(0.15)  # the fixed default throttle


def test_worked_example_wall_near_on_the_right_steers_left():
    # d_left = 0.5, d_right = 0.2: z = -2.0 + 0.8 + 0.1 = -1.1 and tanh(-1.1) = -0.8005
    policy = NeuronPolicy((-4.0, 4.0), 0.1)
    assert policy.act(observation(0.5, 0.2)[None, :])[0, 0] == pytest.approx(
        math.tanh(-1.1), abs=1e-9
    )


def test_the_same_inputs_through_sigmoid_never_steer_left():
    policy = NeuronPolicy((-4.0, 4.0), 0.1, "sigmoid")
    right = policy.act(observation(0.2, 0.5)[None, :])[0, 0]
    left_case = policy.act(observation(0.5, 0.2)[None, :])[0, 0]
    assert right == pytest.approx(1 / (1 + math.exp(-1.3)), abs=1e-9)  # 0.7858
    assert left_case == pytest.approx(1 / (1 + math.exp(1.1)), abs=1e-9)  # 0.2497, still > 0
    assert left_case > 0.0


def test_sigmoid_steer_is_never_negative_whatever_the_weights_and_inputs():
    rng = make_rng(3)
    for _ in range(30):
        w = rng.uniform(-30, 30, size=2)
        policy = NeuronPolicy((w[0], w[1]), float(rng.uniform(-30, 30)), "sigmoid")
        obs = np.zeros((40, 7))
        obs[:, [0, 4]] = rng.uniform(0, 1, size=(40, 2))
        assert np.all(policy.act(obs)[:, 0] >= 0.0)


def test_only_the_far_left_and_far_right_rays_matter():
    policy = NeuronPolicy((-4.0, 4.0), 0.1)
    assert NEURON_INPUTS == (0, 4)
    base = observation(0.2, 0.5)
    other = base.copy()
    other[[1, 2, 3, 5, 6]] = [0.9, 0.9, 0.9, 0.7, -0.3]  # centre rays, speed, heading error
    assert policy.act(base[None, :])[0, 0] == policy.act(other[None, :])[0, 0]


def test_batch_of_cars_equals_single_cars():
    rng = make_rng(5)
    policy = NeuronPolicy((-7.5, 6.0), -0.2, "tanh", 0.3)
    obs = rng.uniform(0.0, 1.0, size=(6, 7))
    together = policy.act(obs)
    assert together.shape == (6, 2)
    for i in range(6):
        assert policy.act(obs[i : i + 1])[0] == pytest.approx(together[i], abs=1e-12)


def test_zero_weights_steer_the_bias_activation_output():
    obs = np.full((3, 7), 0.4)
    assert NeuronPolicy().act(obs)[:, 0] == pytest.approx(0.0)  # tanh(0) = 0: straight ahead
    assert NeuronPolicy(bias=0.5).act(obs)[:, 0] == pytest.approx(math.tanh(0.5))


def test_follows_the_policy_protocol_and_validates_the_observation_shape():
    assert isinstance(NeuronPolicy(), Policy)
    with pytest.raises(ValueError, match="shape"):
        NeuronPolicy().act(np.zeros((2, 5)))


# ── readout ─────────────────────────────────────────────────────────────────────────────────


def test_readout_reports_inputs_contributions_sum_and_output():
    policy = NeuronPolicy((-4.0, 4.0), 0.1)
    r = policy.readout(observation(0.2, 0.5))
    assert r.inputs == (0.2, 0.5) and r.input_rays == (0, 4)
    assert r.weights == (-4.0, 4.0) and r.bias == 0.1
    assert r.contributions == pytest.approx((-0.8, 2.0))
    assert r.z == pytest.approx(1.3) and r.z == pytest.approx(sum(r.contributions) + r.bias)
    assert r.output == pytest.approx(math.tanh(1.3), abs=1e-12)
    assert r.activation == "tanh" and r.throttle == 0.15
    with pytest.raises(ValueError, match="shape"):
        policy.readout(np.zeros((1, 7)))


# ── changing parameters ─────────────────────────────────────────────────────────────────────


def test_set_param_changes_one_thing_and_the_next_action_uses_it():
    policy = NeuronPolicy((-4.0, 4.0), 0.1)
    obs = observation(0.2, 0.5)[None, :]
    before = policy.act(obs)[0, 0]
    policy.set_param("w_right", 6.0)  # z = -0.8 + 3.0 + 0.1 = 2.3
    assert policy.neuron.weights.tolist() == [-4.0, 6.0] and policy.neuron.bias == 0.1
    assert policy.act(obs)[0, 0] == pytest.approx(math.tanh(2.3), abs=1e-9) != before
    policy.set_param("w_left", 0.0)  # z = 3.1
    policy.set_param("bias", 0.0)  # z = 3.0
    assert policy.act(obs)[0, 0] == pytest.approx(math.tanh(3.0), abs=1e-9)
    policy.set_param("throttle", 0.4)
    assert policy.act(obs)[0, 1] == 0.4


def test_set_param_activation_switches_the_function_and_keeps_the_numbers():
    policy = NeuronPolicy((-4.0, 4.0), 0.1)
    policy.set_param("activation", "sigmoid")
    assert policy.neuron.activation == "sigmoid"
    assert policy.neuron.weights.tolist() == [-4.0, 4.0] and policy.neuron.bias == 0.1
    assert policy.act(observation(0.2, 0.5)[None, :])[0, 0] == pytest.approx(0.78583, abs=1e-5)
    for name in ("relu", "linear", "tanh"):
        policy.set_param("activation", name)
        assert policy.neuron.activation == name


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("w_left", 31.0, "between"),
        ("bias", -31.0, "between"),
        ("w_right", float("nan"), "finite"),
        ("bias", float("inf"), "finite"),
        ("w_left", "3", "finite"),
        ("throttle", 1.5, "throttle"),
        ("activation", "gelu", "activation must be"),
        ("activation", 3.0, "activation must be"),
        ("speed", 1.0, "unknown parameter"),
    ],
)
def test_set_param_rejects_bad_names_and_values_without_changing_anything(name, value, message):
    policy = NeuronPolicy((-4.0, 4.0), 0.1)
    with pytest.raises(ValueError, match=message):
        policy.set_param(name, value)
    assert policy.neuron.weights.tolist() == [-4.0, 4.0]
    assert policy.neuron.bias == 0.1 and policy.neuron.activation == "tanh"
    assert policy.throttle == 0.15


# ── driving ─────────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def loop() -> Track:
    return Track.from_json(TRACK_FILE)


def drive(loop: Track, policy: NeuronPolicy):
    world = World(loop, WorldConfig())
    world.reset()
    return run_episode(world, policy), world


def test_weights_are_behaviour_all_zeros_crash_but_opposite_signs_drive_on(loop):
    zeros, zero_world = drive(loop, NeuronPolicy())
    assert zeros.status[0] == Status.CRASHED and zeros.progress[-1, 0] < 60.0  # straight off

    wired, wired_world = drive(loop, NeuronPolicy((-8.0, 8.0), 0.0))
    assert wired.status[0] == Status.TIMEOUT  # never crashed in the whole 90 s
    assert wired.progress[-1, 0] > 700.0  # about a lap, with a slow fixed throttle
    assert wired_world.laps[0] >= 1


def test_the_same_weights_with_the_signs_flipped_crash(loop):
    flipped, _ = drive(loop, NeuronPolicy((8.0, -8.0), 0.0))
    assert flipped.status[0] == Status.CRASHED and flipped.progress[-1, 0] < 60.0
