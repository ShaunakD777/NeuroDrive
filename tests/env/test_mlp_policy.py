"""Tests for MLPPolicy and PopulationPolicy (task 2.6): one brain per car, traces, determinism."""

from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.init import init_network, init_population
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.population import Population
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import child_rng, make_rng
from neurodrive.env.episode import run_episode
from neurodrive.env.policies import MLPPolicy, Policy, PopulationPolicy
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig

TRACK_FILE = Path(__file__).resolve().parents[2] / "tracks" / "train_loop_01.json"
SPEC = NetworkSpec()


@pytest.fixture(scope="module")
def loop() -> Track:
    return Track.from_json(TRACK_FILE)


def random_obs(n: int, seed: int = 0) -> np.ndarray:
    return make_rng(seed).uniform(0.0, 1.0, size=(n, 7))


def population(n: int, seed: int) -> Population:
    return init_population(SPEC, n, child_rng(make_rng(seed), "init"))


def drive(loop: Track, seed: int, n: int = 20, steps: int = 300):
    """Record ``n`` cars driven by the seed's random networks, all from the start line."""
    world = World(loop, WorldConfig.for_track(loop))
    world.reset(n=n)
    return run_episode(world, PopulationPolicy(population(n, seed)), max_steps=steps)


# ── MLPPolicy: one network for every car ────────────────────────────────────────────────────


def test_mlp_policy_is_a_policy_with_actions_n_by_2():
    policy = MLPPolicy(init_network(SPEC, make_rng(0)))
    assert isinstance(policy, Policy)
    assert policy.act(random_obs(5)).shape == (5, 2)


def test_mlp_policy_rows_equal_single_forward_passes():
    net = init_network(SPEC, make_rng(1))
    obs = random_obs(6, seed=2)
    actions = MLPPolicy(net).act(obs)
    for i in range(6):
        np.testing.assert_allclose(actions[i], forward(net, obs[i]).output, atol=1e-9)


def test_mlp_policy_shares_one_brain_equal_inputs_give_equal_actions():
    policy = MLPPolicy(init_network(SPEC, make_rng(3)))
    actions = policy.act(np.tile(random_obs(1)[0], (4, 1)))
    assert np.all(actions == actions[0])


def test_mlp_actions_lie_in_the_tanh_range():
    actions = MLPPolicy(init_network(SPEC, make_rng(4))).act(random_obs(50))
    assert np.all(np.abs(actions) <= 1.0)


def test_mlp_policy_sees_weight_changes_because_it_holds_the_network_itself():
    net = Network.zeros(SPEC)
    policy = MLPPolicy(net)
    obs = random_obs(3)
    assert not policy.act(obs).any()
    net.biases[-1][0] = 0.5  # steer bias
    np.testing.assert_allclose(policy.act(obs)[:, 0], np.tanh(0.5))


# ── PopulationPolicy: one network per car ───────────────────────────────────────────────────


def test_hand_example_each_car_is_driven_by_its_own_network():
    spec = NetworkSpec((7, 2), output_activation="linear")
    w0, w1 = np.zeros((2, 7)), np.zeros((2, 7))
    w0[0, :2] = [1.0, 2.0]  # network 0: steer = o1 + 2*o2
    w1[0, :2] = [-1.0, 1.0]  # network 1: steer = -o1 + o2 + 0.5
    nets = [Network(spec, [w0], [[0.0, 0.0]]), Network(spec, [w1], [[0.5, 0.0]])]
    obs = np.zeros((2, 7))
    obs[0, :2] = [1.0, 1.0]
    obs[1, :2] = [2.0, 3.0]
    actions = PopulationPolicy(Population.from_networks(nets)).act(obs)
    np.testing.assert_allclose(actions[:, 0], [3.0, 1.5], atol=1e-9)  # not clipped by the policy
    np.testing.assert_allclose(actions[:, 1], [0.0, 0.0], atol=1e-9)


def test_population_policy_equals_each_network_alone_for_20_cars():
    pop = population(20, seed=5)
    obs = random_obs(20, seed=6)
    actions = PopulationPolicy(pop).act(obs)
    assert actions.shape == (20, 2)
    for i in range(20):
        np.testing.assert_allclose(actions[i], forward(pop.network(i), obs[i]).output, atol=1e-9)


def test_population_of_identical_networks_equals_the_shared_mlp_policy():
    net = init_network(SPEC, make_rng(7))
    obs = random_obs(8, seed=8)
    same = PopulationPolicy(Population.from_networks([net] * 8)).act(obs)
    np.testing.assert_allclose(same, MLPPolicy(net).act(obs), atol=1e-9)


def test_different_networks_give_different_actions_for_the_same_observation():
    actions = PopulationPolicy(population(5, seed=9)).act(np.tile(random_obs(1)[0], (5, 1)))
    assert len({tuple(row) for row in np.round(actions, 6)}) == 5


def test_population_policy_needs_one_row_per_network():
    policy = PopulationPolicy(population(4, seed=0))
    with pytest.raises(ValueError, match="4 networks but got 3"):
        policy.act(random_obs(3))


@pytest.mark.parametrize("sizes", [(5, 16, 2), (7, 16, 3), (7, 16, 1)])
def test_networks_that_do_not_fit_the_contract_are_rejected(sizes):
    spec = NetworkSpec(sizes)
    with pytest.raises(ValueError, match="7 inputs and 2 outputs"):
        MLPPolicy(Network.zeros(spec))
    with pytest.raises(ValueError, match="7 inputs and 2 outputs"):
        PopulationPolicy(Population.from_networks([Network.zeros(spec)]))


def test_wrong_observation_shape_is_rejected():
    with pytest.raises(ValueError, match="observations must have shape"):
        MLPPolicy(Network.zeros(SPEC)).act(np.zeros((3, 6)))
    with pytest.raises(ValueError, match="observations must have shape"):
        PopulationPolicy(population(3, 0)).act(np.zeros(7))


# ── traces ──────────────────────────────────────────────────────────────────────────────────


def test_no_trace_before_the_first_act():
    for policy in (MLPPolicy(Network.zeros(SPEC)), PopulationPolicy(population(2, 0))):
        assert policy.last_trace is None
        with pytest.raises(RuntimeError, match="act"):
            policy.trace_for(0)


def test_population_trace_for_car_matches_that_networks_own_trace():
    pop = population(6, seed=10)
    obs = random_obs(6, seed=11)
    policy = PopulationPolicy(pop)
    policy.act(obs)
    for i in (0, 3, 5):
        mine, alone = policy.trace_for(i), forward(pop.network(i), obs[i])
        assert [a.shape for a in mine.activations] == [(7,), (16,), (16,), (2,)]
        assert [z.shape for z in mine.weighted_sums] == [(16,), (16,), (2,)]
        for a, b in zip(
            mine.activations + mine.weighted_sums,
            alone.activations + alone.weighted_sums,
            strict=True,
        ):
            np.testing.assert_allclose(a, b, atol=1e-9)


def test_mlp_trace_for_car_and_trace_output_equals_the_actions():
    net = init_network(SPEC, make_rng(12))
    obs = random_obs(4, seed=13)
    policy = MLPPolicy(net)
    actions = policy.act(obs)
    np.testing.assert_array_equal(policy.last_trace.output, actions)
    np.testing.assert_allclose(policy.trace_for(2).output, actions[2])
    np.testing.assert_array_equal(policy.trace_for(2).activations[0], obs[2])


def test_trace_for_returns_copies_and_checks_the_index():
    policy = PopulationPolicy(population(3, seed=14))
    policy.act(random_obs(3))
    first = policy.trace_for(1)
    first.activations[1][:] = 99.0
    assert not np.any(policy.trace_for(1).activations[1] == 99.0)
    for bad in (3, -1):
        with pytest.raises(IndexError):
            policy.trace_for(bad)


def test_the_trace_follows_the_latest_step():
    policy = MLPPolicy(init_network(SPEC, make_rng(15)))
    policy.act(random_obs(2, seed=1))
    before = policy.trace_for(0).activations[0].copy()
    policy.act(random_obs(2, seed=2))
    assert not np.array_equal(policy.trace_for(0).activations[0], before)


# ── 20 random brains driving: reproducible, and different from each other ───────────────────


def test_same_seed_reproduces_the_same_drive_bit_for_bit(loop):
    a, b = drive(loop, seed=3), drive(loop, seed=3)
    for name in ("x", "y", "psi", "v", "alive", "progress", "observation", "action"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name), err_msg=name)


def test_a_different_seed_gives_a_different_drive(loop):
    assert not np.array_equal(drive(loop, seed=3).x, drive(loop, seed=4).x)


def test_twenty_random_brains_choose_differently_from_the_same_start(loop):
    rec = drive(loop, seed=0)
    assert rec.x.shape[1] == 20
    assert len({tuple(row) for row in np.round(rec.action[0], 6)}) == 20  # 20 different actions
    ends = {(round(x, 3), round(y, 3)) for x, y in zip(rec.x[-1], rec.y[-1], strict=True)}
    assert len(ends) >= 8  # and the cars end up in many different places


def test_random_brains_are_poor_drivers_stalling_or_crashing_never_finishing(loop):
    # Seed 0: about two thirds brake from the start and never move (a car at speed 0 that is told
    # to brake stays at 0); the rest crash within seconds. Nothing finishes or even laps.
    world = World(loop, WorldConfig.for_track(loop))
    world.reset(n=20)
    run_episode(world, PopulationPolicy(population(20, seed=0)))
    status = [Status(int(s)) for s in world.status]
    assert status.count(Status.STALLED) >= 5 and status.count(Status.CRASHED) >= 3
    assert all(s in (Status.STALLED, Status.CRASHED) for s in status)
    assert world.progress.max() < loop.length / 2


# ── trace_of and network_of: the current state, without touching last_trace ─────────────────


def test_trace_of_is_that_cars_forward_pass_on_the_given_observations():
    pop = population(5, seed=20)
    obs = random_obs(5, seed=21)
    policy = PopulationPolicy(pop)
    for car in (0, 2, 4):
        mine, alone = policy.trace_of(obs, car), forward(pop.network(car), obs[car])
        mine_all = mine.activations + mine.weighted_sums
        alone_all = alone.activations + alone.weighted_sums
        for a, b in zip(mine_all, alone_all, strict=True):
            np.testing.assert_array_equal(a, b)


def test_trace_of_does_not_change_last_trace_and_works_before_any_act():
    policy = PopulationPolicy(population(3, seed=22))
    policy.trace_of(random_obs(3), 1)
    assert policy.last_trace is None
    policy.act(random_obs(3, seed=1))
    kept = policy.last_trace
    policy.trace_of(random_obs(3, seed=2), 0)
    assert policy.last_trace is kept


def test_mlp_trace_of_uses_the_shared_network():
    net = init_network(SPEC, make_rng(23))
    obs = random_obs(4, seed=24)
    policy = MLPPolicy(net)
    np.testing.assert_array_equal(policy.trace_of(obs, 3).output, forward(net, obs[3]).output)


def test_network_of_gives_each_cars_own_network():
    pop = population(4, seed=25)
    policy = PopulationPolicy(pop)
    for car in range(4):
        assert policy.network_of(car).equals(pop.network(car))
    shared = init_network(SPEC, make_rng(26))
    assert MLPPolicy(shared).network_of(2) is shared


def test_trace_of_and_network_of_check_the_car_index():
    pop_policy, mlp = PopulationPolicy(population(3, 0)), MLPPolicy(Network.zeros(SPEC))
    for bad in (3, -1):
        with pytest.raises(IndexError):
            pop_policy.trace_of(random_obs(3), bad)
        with pytest.raises(IndexError):
            mlp.trace_of(random_obs(3), bad)
    with pytest.raises(IndexError):
        pop_policy.network_of(3)
    with pytest.raises(IndexError):
        mlp.network_of(-1)
