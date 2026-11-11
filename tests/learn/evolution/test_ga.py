"""Tests for learn/evolution/ga.py (task 7.2, ADR-055): a generation's fitness and ranking, checked
against the Driving Exam's independent episode summaries, plus order, determinism and errors."""

import numpy as np
import pytest

from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.env.policies import MLPPolicy
from neurodrive.env.track import Track
from neurodrive.experiment.baselines import hand_neuron_network, load_hand_neuron
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, EXAM_V1, exam_track
from neurodrive.learn.evolution.ga import GASettings, evaluate_genomes, ranking
from neurodrive.learn.evolution.genome import genomes_of, networks_of, random_genomes
from neurodrive.learn.evolution.random_search import start_fitness

TRACK = Track.from_json(DEFAULT_TRACKS_DIR / "train_loop_01.json")
NOISE = (0.0, 0.02, 0.05, 0.1, 0.3, 1.0, 3.0, 10.0)  # the neuron, then ever noisier copies


@pytest.fixture(scope="module")
def neuron_family():
    """The hand-wired neuron (a 7 -> 2 network) and 7 copies with Gaussian noise on every gene."""
    base = hand_neuron_network(load_hand_neuron())
    g0 = base.flatten()
    rng = make_rng(17)
    genomes = np.stack([g0 + s * rng.standard_normal(g0.size) for s in NOISE])
    settings = GASettings(spec=base.spec, population=len(NOISE))
    return genomes, settings, evaluate_genomes(genomes, TRACK, settings)


# ── the formula and the ranking ────────────────────────────────────────────────────────────


def test_the_briefs_worked_example_speed_can_outweigh_a_near_miss():
    x = start_fitness([1.0, 1.0], [True, True], [90.0, 90.0], 92.0).mean()
    y = start_fitness([1.0, 0.99], [True, False], [60.0, np.nan], 92.0).mean()
    assert x == pytest.approx(1.022, abs=5e-4)
    assert y == pytest.approx(1.169, abs=5e-4)
    order, _ = ranking([x, y])
    assert order.tolist() == [1, 0]  # Y (fast but fragile) above X (reliable but slow)


def test_ranking_is_a_stable_descending_sort_and_ranks_invert_it():
    order, places = ranking([0.3, 1.2, 0.3, 1.9])
    assert order.tolist() == [3, 1, 0, 2]  # the tie (0.3, 0.3) keeps index order
    assert places.tolist() == [2, 1, 3, 0]
    assert np.array_equal(order[places], np.arange(4))
    with pytest.raises(ValueError):
        ranking([0.1, np.nan])
    with pytest.raises(ValueError):
        ranking([[0.1, 0.2]])


# ── the roadmap's check: the ranking matches the episode summaries ─────────────────────────


def test_fitness_and_ranking_match_the_exams_episode_summaries(neuron_family):
    genomes, settings, gen = neuron_family
    summary_fitness = []
    for i, net in enumerate(networks_of(settings.spec, genomes)):
        part, _ = exam_track(TRACK, lambda world, net=net: MLPPolicy(net), EXAM_V1)
        eps = part.episodes
        per_start = start_fitness(
            [e["progress_frac"] for e in eps],
            [e["completed"] for e in eps],
            [e["time_s"] for e in eps],
            part.time_limit_s,
        )
        assert gen.evaluation.per_start[i] == pytest.approx(per_start, rel=1e-9, abs=1e-12)
        summary_fitness.append(per_start.mean())
    order, places = ranking(summary_fitness)
    assert gen.ranking.tolist() == order.tolist()
    assert gen.ranks.tolist() == places.tolist()
    # the family is a real test: finishers and non-finishers, the neuron near the top
    finished = gen.evaluation.finish_time == gen.evaluation.finish_time  # not NaN
    assert finished.all(axis=1).any() and (~finished).all(axis=1).any()
    assert gen.fitness[0] == pytest.approx(1.0 + (1 - 86.2 / 92.0), abs=0.01)
    assert gen.ranks[0] <= 2
    assert gen.best == int(np.argmax(gen.fitness))


# ── one batched world: order, independence, determinism, budget ────────────────────────────

SMALL = GASettings(population=6, start_fractions=(0.0, 0.5), lateral_offsets=(0.0,))


def test_reordering_the_genomes_reorders_fitness_and_ranking():
    genomes = random_genomes(SMALL.spec, SMALL.population, make_rng(3))
    first = evaluate_genomes(genomes, TRACK, SMALL)
    perm = np.array([4, 0, 5, 2, 1, 3])
    again = evaluate_genomes(genomes[perm], TRACK, SMALL)
    assert again.evaluation.per_start == pytest.approx(
        first.evaluation.per_start[perm], rel=1e-12, abs=1e-12
    )
    assert again.env_steps == first.env_steps
    # Random networks often stall at exactly 0 and tie; a tie keeps index order, so the tied
    # networks swap places with the rows. What must hold: the same scores in ranked order, and
    # the new ranking is the stable ranking of the permuted fitness.
    assert np.array_equal(again.fitness[again.ranking], first.fitness[first.ranking])
    assert again.ranking.tolist() == ranking(first.fitness[perm])[0].tolist()
    assert int(perm[again.best]) == first.best  # the best (no tie here) is the same network


def test_a_network_scores_the_same_beside_any_partner():
    genomes = random_genomes(SMALL.spec, SMALL.population, make_rng(4))
    batch = evaluate_genomes(genomes, TRACK, SMALL)
    pair = GASettings(population=2, start_fractions=(0.0, 0.5), lateral_offsets=(0.0,))
    for i in (0, 3):
        twice = evaluate_genomes(np.stack([genomes[i], genomes[i]]), TRACK, pair)
        assert twice.fitness[0] == twice.fitness[1]
        assert twice.fitness[0] == pytest.approx(batch.fitness[i], rel=1e-12, abs=1e-12)
        assert twice.ranking.tolist() == [0, 1]  # a tie keeps index order


def test_same_genomes_same_scores_and_the_budget_is_the_car_steps():
    genomes = random_genomes(SMALL.spec, SMALL.population, make_rng(5))
    a = evaluate_genomes(genomes, TRACK, SMALL)
    b = evaluate_genomes(genomes, TRACK, SMALL)
    assert a.fitness.tobytes() == b.fitness.tobytes()
    assert a.evaluation.per_start.tobytes() == b.evaluation.per_start.tobytes()
    assert a.ranking.tolist() == b.ranking.tolist()
    assert 0 < a.env_steps <= SMALL.population * SMALL.starts_per_network * 92 * 30


def test_the_default_starts_are_random_searchs_exam_starts():
    settings = GASettings()
    assert settings.population == 50 and settings.starts_per_network == 10
    rs = settings.evaluation()
    assert (rs.start_fractions, rs.lateral_offsets) == ((0.0, 0.2, 0.4, 0.6, 0.8), (0.0, 1.5))
    assert genomes_of(networks_of(settings.spec, np.zeros((2, 434)))).shape == (2, 434)


# ── errors ─────────────────────────────────────────────────────────────────────────────────


def test_bad_settings_matrices_and_tracks_are_refused():
    with pytest.raises(ValueError, match="at least 2"):
        GASettings(population=1)
    with pytest.raises(ValueError, match="whole number"):
        GASettings(population=2.0)
    with pytest.raises(ValueError, match="inputs"):
        GASettings(spec=NetworkSpec((3, 2)))
    with pytest.raises(ValueError):
        GASettings(lateral_offsets=())
    with pytest.raises(ValueError, match="shape"):
        evaluate_genomes(np.zeros((6, 433)), TRACK, SMALL)
    with pytest.raises(ValueError, match="shape"):
        evaluate_genomes(np.zeros((5, 434)), TRACK, SMALL)
    heldout = Track.from_json(DEFAULT_TRACKS_DIR / "hairpins_01.json")
    with pytest.raises(ValueError, match="held-out"):
        evaluate_genomes(np.zeros((6, 434)), heldout, SMALL)
