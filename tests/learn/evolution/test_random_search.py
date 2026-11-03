"""Tests for random search (task 5.7, ADR-040): the fitness formula, the batched evaluation (checked
against the Driving Exam's independent scoring), keep-the-best, determinism and save/restore."""

import numpy as np
import pytest

from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.env.policies import MLPPolicy
from neurodrive.env.sensors import build_observation, edge_grid, observe
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig
from neurodrive.experiment.baselines import hand_neuron_network, load_hand_neuron
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, EXAM_V1, exam_starts, exam_track
from neurodrive.learn.base import Resumable
from neurodrive.learn.evolution.random_search import (
    RandomSearchLearner,
    RandomSearchSettings,
    evaluate,
    start_fitness,
)

TRACK = Track.from_json(DEFAULT_TRACKS_DIR / "train_loop_01.json")
SMALL = RandomSearchSettings(candidates=6, start_fractions=(0.0, 0.5), lateral_offsets=(0.0,))


# ── fitness ─────────────────────────────────────────────────────────────────────────────────


def test_fitness_worked_example():
    # start A finishes at 69 s of 92 s: 1 + (1 - 0.75) = 1.25; start B crashes at 30 %: 0.30
    f = start_fitness([1.0, 0.30], [True, False], [69.0, np.nan], 92.0)
    assert f.tolist() == [1.25, 0.30] and f.mean() == pytest.approx(0.775)


def test_fitness_edges():
    f = start_fitness([-0.2, 1.4, 1.0, 0.5], [False, False, True, False], [0, 0, 92.0, 0], 92.0)
    assert f.tolist() == [0.0, 1.0, 1.0, 0.5]  # backwards = 0, capped at 1, finish at T: no bonus


def test_the_starts_are_the_exam_layout():
    s0, lateral = RandomSearchSettings().starts(TRACK.length)
    exam_s0, exam_lateral = exam_starts(TRACK.length, EXAM_V1)
    assert np.array_equal(s0, exam_s0) and np.array_equal(lateral, exam_lateral)


# ── evaluation ──────────────────────────────────────────────────────────────────────────────


def test_rays_of_running_cars_only_equal_the_full_rays():
    world = World(TRACK, WorldConfig.for_track(TRACK))
    nets = [init_network(NetworkSpec(), make_rng(s)) for s in range(12)]
    policy = MLPPolicy(nets[0])
    world.reset(s0=np.linspace(0, TRACK.length, 12, endpoint=False))
    segments, checked = edge_grid(TRACK), 0
    for _ in range(400):
        run, cars = world.running, world.cars
        full = observe(world, segments)
        part = build_observation(
            segments,
            cars.x[run],
            cars.y[run],
            cars.psi[run],
            cars.v[run],
            world.road_heading[run],
            world.params.v_max,
        )
        assert np.array_equal(part, full[run])
        checked += int((~run).any())
        a = np.clip(policy.act(full), -1, 1)
        world.step(a[:, 0], a[:, 1])
        if world.done:
            break
    assert checked > 0  # some cars had stopped while others ran


def test_evaluation_agrees_with_the_exams_scoring_of_the_hand_neuron():
    """The neuron's network on the exam's 10 training-track starts: the learner's fitness equals
    the formula applied to the exam's own episode metrics (written independently in 5.2)."""
    network = hand_neuron_network(load_hand_neuron())
    settings = RandomSearchSettings(spec=network.spec)
    ours = evaluate([network], TRACK, settings)
    part, recording = exam_track(TRACK, lambda world: MLPPolicy(network), EXAM_V1)
    eps = part.episodes
    theirs = start_fitness(
        [e["progress_frac"] for e in eps],
        [e["completed"] for e in eps],
        [e["time_s"] for e in eps],
        part.time_limit_s,
    )
    assert [e["status"] for e in eps] == ["finished"] * 10
    assert (ours.status[0] == Status.FINISHED).all()
    assert ours.per_start[0] == pytest.approx(theirs, rel=1e-9)
    assert ours.fitness[0] == pytest.approx(1.0 + (1 - 86.2 / 92.0), abs=0.01)  # the Brief's ≈1.07
    assert ours.env_steps == int(recording.alive[: recording.n_steps].sum())


def test_a_network_scores_the_same_in_a_batch_as_alone():
    nets = [init_network(NetworkSpec(), make_rng(s)) for s in range(5)]
    batch = evaluate(nets, TRACK, SMALL)
    for i, net in enumerate(nets):
        alone = evaluate([net], TRACK, SMALL)
        assert alone.per_start[0] == pytest.approx(batch.per_start[i], rel=1e-12, abs=1e-12)


# ── the learner ─────────────────────────────────────────────────────────────────────────────


def test_unit_zero_is_the_random_network_baseline_of_the_same_seed():
    learner = RandomSearchLearner(RandomSearchSettings(), TRACK, make_rng(7))
    assert learner.current_networks()[0].equals(init_network(NetworkSpec(), make_rng(7)))
    assert learner.best_fitness is None and isinstance(learner, Resumable)


@pytest.fixture(scope="module")
def eight_units():
    learner = RandomSearchLearner(SMALL, TRACK, make_rng(3))
    results = [learner.train_unit() for _ in range(8)]
    return learner, results


def test_best_so_far_never_decreases(eight_units):
    learner, results = eight_units
    best = [r.metrics["best_fitness"] for r in results]
    assert all(b2 >= b1 for b1, b2 in zip(best, best[1:], strict=False))
    for k, r in enumerate(results):
        m = r.metrics
        previous = best[k - 1] if k else None
        assert m["improved"] == (previous is None or m["unit_best_fitness"] > previous)
        assert m["best_fitness"] == max(x.metrics["unit_best_fitness"] for x in results[: k + 1])
        assert m["candidates_total"] == 6 * (k + 1) and r.env_steps > 0
    assert len(set(best)) > 1  # it did improve at least once after unit 1
    kept = evaluate(learner.current_networks(), TRACK, SMALL)
    assert kept.fitness[0] == pytest.approx(learner.best_fitness, rel=1e-12)


def test_same_seed_same_search(eight_units):
    _, results = eight_units
    again = RandomSearchLearner(SMALL, TRACK, make_rng(3))
    for r in results[:3]:
        other = again.train_unit()
        assert other.metrics == r.metrics and other.env_steps == r.env_steps


def test_state_and_restore_continue_exactly():
    a = RandomSearchLearner(SMALL, TRACK, make_rng(11))
    a.train_unit()
    a.train_unit()
    saved = a.state()
    b = RandomSearchLearner(SMALL, TRACK, make_rng(999))  # a different generator, then restored
    b.restore(saved)
    ra, rb = a.train_unit(), b.train_unit()
    assert ra.metrics == rb.metrics and ra.env_steps == rb.env_steps
    assert a.current_networks()[0].equals(b.current_networks()[0])
