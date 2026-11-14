"""Tests for task 7.5 (ADR-056): the GA's generation loop and learner (``learn/evolution/ga.py``).

The roadmap's check: with elitism the best fitness never decreases on a fixed evaluation, and
the learner is deterministic. Also the Brief's worked example, a trivial fitness that needs no
driving (the protocol's learner sanity check), the elite's exact copy, and an exact resume.
"""

import numpy as np
import pytest

from neurodrive.core.rng import child_rng, make_rng
from neurodrive.env.track import Track
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR
from neurodrive.learn.base import Resumable
from neurodrive.learn.evolution.ga import (
    BreedingSettings,
    GAGenerators,
    GALearner,
    GASettings,
    diversity,
    evaluate_genomes,
    next_generation,
    ranking,
)
from neurodrive.learn.evolution.genome import random_genomes

TRACK = Track.from_json(DEFAULT_TRACKS_DIR / "train_loop_01.json")
HELD_OUT = Track.from_json(DEFAULT_TRACKS_DIR / "hairpins_01.json")
SMALL = GASettings(
    population=6,
    start_fractions=(0.0, 0.5),
    lateral_offsets=(0.0,),
    target_laps=1,
    breeding=BreedingSettings(elite=1, mutation_rate=0.1, mutation_strength=0.2),
)


class ScriptedRng:
    """A stand-in generator that hands out prepared arrays in order (checking kind and shape)."""

    def __init__(self, *draws: tuple[str, list]) -> None:
        self.draws = list(draws)

    def _next(self, kind: str, size) -> np.ndarray:
        expected_kind, values = self.draws.pop(0)
        assert kind == expected_kind, f"asked for {kind}, the script has {expected_kind}"
        out = np.asarray(values)
        assert out.shape == tuple(np.atleast_1d(size)), f"shape {size} vs {out.shape}"
        return out

    def random(self, size):
        return self._next("random", size)

    def standard_normal(self, size):
        return self._next("normal", size)

    def integers(self, low, high, size):
        return self._next("integers", size)


def generators(seed: int) -> GAGenerators:
    return GAGenerators.from_master(make_rng(seed))


# ── the Brief's worked example ─────────────────────────────────────────────────────────────

G = [[0.0, 0.0], [0.5, -0.5], [1.0, 1.0], [0.2, 0.4]]
F = [0.3, 1.2, 0.3, 1.9]


def test_the_briefs_worked_example():
    # Tournaments of size 1 make each pick the drawn row: (3, 1), (1, 0), (3, 3).
    scripted = GAGenerators(
        selection=ScriptedRng(("integers", [[3], [1], [1], [0], [3], [3]])),
        crossover=ScriptedRng(
            ("random", [0.3, 0.8, 0.9]), ("random", [[0.7, 0.2], [0.1, 0.1], [0.5, 0.5]])
        ),
        mutation=ScriptedRng(
            ("random", [[0.1, 0.9], [0.7, 0.6], [0.2, 0.3]]),
            ("normal", [[1.0, 0.0], [0.0, 0.0], [-2.0, 1.0]]),
        ),
    )
    breeding = BreedingSettings(
        elite=1, tournament_size=1, crossover_rate=0.5, mutation_rate=0.5, mutation_strength=0.1
    )
    bred = next_generation(G, F, breeding, scripted)
    expected = [[0.2, 0.4], [0.6, 0.4], [0.5, -0.5], [0.0, 0.5]]
    assert bred.genomes == pytest.approx(np.array(expected), abs=1e-15)
    assert bred.elite.tolist() == [3]
    assert bred.parents_a.tolist() == [3, 1, 3] and bred.parents_b.tolist() == [1, 0, 3]
    assert bred.crossover.crossed.tolist() == [True, False, False]
    assert bred.mutation.mutated.sum(axis=1).tolist() == [1, 0, 2]


def test_the_diversity_example():
    assert diversity([[0, 0], [3, 4], [0, 4]]) == pytest.approx(4.0)
    assert diversity([[1.0, 2.0]]) == 0.0
    assert diversity(np.ones((5, 3))) == 0.0
    rng = make_rng(0)
    g = rng.standard_normal((7, 4))
    pairs = [np.linalg.norm(g[i] - g[j]) for i in range(7) for j in range(i + 1, 7)]
    assert diversity(g) == pytest.approx(np.mean(pairs), rel=1e-12)


# ── the generation step ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("elite", [0, 1, 3])
def test_the_elite_is_an_exact_copy_of_the_best_rows_in_ranking_order(elite):
    rng = make_rng(4)
    g = rng.standard_normal((10, 6))
    f = np.array([0.5, 0.9, 0.0, 0.9, 0.2, 0.0, 1.5, 0.1, 0.9, 0.3])  # ties at 0.9 and 0
    strong = BreedingSettings(elite=elite, mutation_rate=1.0, mutation_strength=1.0)
    bred = next_generation(g, f, strong, generators(1))
    order, _ = ranking(f)
    assert order[:4].tolist() == [6, 1, 3, 8]  # the tie keeps the lower row first
    assert bred.elite.tolist() == order[:elite].tolist()
    assert np.array_equal(bred.genomes[:elite], g[order[:elite]])  # bit for bit, never mutated
    assert bred.genomes.shape == g.shape
    assert bred.mutation.mutated.all()  # p = 1: every child gene moved
    assert not any(np.array_equal(row, old) for row in bred.genomes[elite:] for old in g)


def test_the_parents_come_from_selection():
    g = make_rng(2).standard_normal((8, 5))
    f = np.zeros(8)
    f[5] = 1.0
    # tournaments of 200 draws: the best is in practically every one (1 - (7/8)^200 > 0.9999999)
    sure = BreedingSettings(elite=1, tournament_size=200, crossover_rate=1.0, mutation_rate=0.0)
    bred = next_generation(g, f, sure, generators(3))
    assert set(bred.parents_a) == {5} and set(bred.parents_b) == {5}
    assert all(np.array_equal(row, g[5]) for row in bred.genomes)  # p = 0: copies of the best
    flat = BreedingSettings(elite=0, selection="rank", rank_pressure=1.0, mutation_rate=0.0)
    picked = next_generation(g, f, flat, generators(3))
    assert len(set(picked.parents_a) | set(picked.parents_b)) > 1  # s = 1: no preference


def test_the_same_generators_give_the_same_generation_and_operators_do_not_share_numbers():
    g = make_rng(5).standard_normal((12, 4))
    f = make_rng(6).random(12)
    a = next_generation(g, f, BreedingSettings(), generators(9))
    b = next_generation(g, f, BreedingSettings(), generators(9))
    assert np.array_equal(a.genomes, b.genomes)
    # a different mutation rate leaves the parents and the crossover exactly as they were
    c = next_generation(g, f, BreedingSettings(mutation_rate=0.5), generators(9))
    assert np.array_equal(a.parents_a, c.parents_a) and np.array_equal(a.parents_b, c.parents_b)
    assert np.array_equal(a.crossover.genomes, c.crossover.genomes)
    assert not np.array_equal(a.genomes, c.genomes)


def quadratic_run(breeding: BreedingSettings, generations: int = 60, seed: int = 0):
    """The trivial fitness F = -||g - t||^2 (no driving): the best fitness per generation."""
    rng = make_rng(seed)
    target = rng.uniform(-1.0, 1.0, 10)
    g = rng.standard_normal((30, 10))
    gens = generators(seed)
    best = []
    for _ in range(generations):
        f = -((g - target) ** 2).sum(axis=1)
        best.append(float(f.max()))
        g = next_generation(g, f, breeding, gens).genomes
    return np.array(best)


@pytest.mark.parametrize("selection", ["tournament", "rank"])
def test_a_trivial_fitness_improves_and_with_an_elite_never_decreases(selection):
    breeding = BreedingSettings(
        elite=2, selection=selection, mutation_rate=0.2, mutation_strength=0.1
    )
    best = quadratic_run(breeding)
    assert np.all(np.diff(best) >= 0.0)  # elitism: never worse than the generation before
    assert -best[-1] < 0.1 * -best[0]  # squared distance to the target shrank more than 10x


def test_without_an_elite_the_best_can_be_lost():
    strong = BreedingSettings(elite=0, mutation_rate=0.5, mutation_strength=0.3)
    best = quadratic_run(strong)
    assert (np.diff(best) < 0).any()  # the best of some generation is worse than before


def test_bad_breeding_settings_are_refused():
    for change in (
        {"elite": -1},
        {"elite": True},
        {"selection": "roulette"},
        {"tournament_size": 0},
        {"rank_pressure": 2.5},
        {"crossover_rate": 1.5},
        {"mutation_rate": -0.1},
        {"mutation_strength": -1.0},
        {"mutation_strength": float("nan")},
    ):
        with pytest.raises(ValueError):
            BreedingSettings(**change)
    four_elites = GASettings(population=4, breeding=BreedingSettings(elite=4))  # scoring is fine
    with pytest.raises(ValueError, match="smaller than the population"):
        GALearner(four_elites, TRACK, make_rng(0))  # breeding is not
    with pytest.raises(ValueError):
        next_generation(np.zeros((3, 2)), np.zeros(4), BreedingSettings(elite=1), generators(0))
    with pytest.raises(ValueError):
        next_generation(np.zeros((2, 2)), np.zeros(2), BreedingSettings(elite=2), generators(0))


# ── the learner on the track ───────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def five_generations():
    """A small driving GA for 5 generations, with each generation's genomes before it was scored."""
    learner = GALearner(SMALL, TRACK, make_rng(3))
    before, results = [], []
    for _ in range(5):
        before.append(learner.genomes.copy())
        results.append(learner.train_unit())
    return learner, before, results


def test_unit_zero_is_row_zero_of_the_first_generation():
    learner = GALearner(SMALL, TRACK, make_rng(7))
    first = random_genomes(SMALL.spec, 6, child_rng(make_rng(7), "init"))
    assert np.array_equal(learner.genomes, first)
    assert np.array_equal(learner.current_networks()[0].flatten(), first[0])
    assert learner.best_fitness is None and isinstance(learner, Resumable)
    assert (learner.name, learner.feedback_kind) == ("ga", "episode_score")


def test_with_an_elite_the_best_never_decreases_and_the_elite_keeps_its_fitness(five_generations):
    learner, before, results = five_generations
    gen_best = [r.metrics["generation_best_fitness"] for r in results]
    assert all(b2 >= b1 for b1, b2 in zip(gen_best, gen_best[1:], strict=False))
    assert gen_best[-1] > gen_best[0]  # it improved at least once in 5 generations
    assert results[0].metrics["elite_drift"] is None
    assert all(r.metrics["elite_drift"] == 0.0 for r in results[1:])  # exactly, not roughly
    for k, r in enumerate(results):
        m = r.metrics
        assert m["best_fitness"] == max(gen_best[: k + 1])
        assert m["improved"] == (k == 0 or gen_best[k] > max(gen_best[:k]))
        assert r.new_best == m["improved"]
        if k > 0 and not m["improved"]:
            assert r.extra["population"]["champion"] == 0  # the old champion keeps the title
        if k > 0:  # row 0 is last generation's champion, copied exactly
            previous = results[k - 1].extra["population"]["champion"]
            assert np.array_equal(before[k][0], before[k - 1][previous])


def test_the_scores_and_budget_are_the_evaluations(five_generations):
    _, before, results = five_generations
    for genomes, r in zip(before, results, strict=True):
        again = evaluate_genomes(genomes, TRACK, SMALL)
        line = r.extra["population"]
        assert line["fitness"] == again.fitness.tolist()
        assert r.env_steps == again.env_steps > 0
        assert line["champion"] == again.best
        m = r.metrics
        assert m["generation_best_fitness"] == again.fitness.max()
        assert m["mean_fitness"] == pytest.approx(again.fitness.mean(), rel=1e-12)
        assert m["worst_fitness"] == again.fitness.min()
        assert m["zero_fitness"] == int((again.fitness == 0).sum())
        assert m["diversity"] == pytest.approx(diversity(genomes), rel=1e-12)
        assert len(line["finished"]) == len(line["crashed"]) == 6
        assert 0.0 <= m["finished_share"] + m["crashed_share"] <= 1.0
    assert results[0].extra["population"]["elite"] == []
    assert all(r.extra["population"]["elite"] == [0] for r in results[1:])


def test_each_unit_hands_back_the_matrix_it_scored(five_generations):
    # task 7.6 (ADR-057): the runner saves it as extra/generations/unit_<g>.npz
    learner, before, results = five_generations
    for genomes, r in zip(before, results, strict=True):
        arrays = r.arrays["generations"]
        # since the 7.7 follow-up (ADR-059) also every drive from every start, each (N, S)
        assert list(arrays) == ["genomes", "start_fitness", "status", "progress", "finish_time"]
        saved = arrays["genomes"]
        assert np.array_equal(saved, genomes)  # generation g, not the bred generation g + 1
        n, s = SMALL.population, SMALL.starts_per_network
        assert all(arrays[k].shape == (n, s) for k in ["start_fitness", "status", "progress"])
        assert arrays["start_fitness"].mean(axis=1).tolist() == r.extra["population"]["fitness"]
    assert not np.array_equal(results[-1].arrays["generations"]["genomes"], learner.genomes)
    # a copy: later breeding or restoring never changes what was handed back
    assert results[-1].arrays["generations"]["genomes"] is not learner.genomes


def test_the_checkpoint_is_the_generations_best(five_generations):
    learner, before, results = five_generations
    champion = results[-1].extra["population"]["champion"]
    assert np.array_equal(learner.current_networks()[0].flatten(), before[-1][champion])
    assert np.array_equal(learner.genomes[0], before[-1][champion])  # and the next elite


def test_same_seed_same_evolution_other_seed_other(five_generations):
    learner, _, results = five_generations
    again = GALearner(SMALL, TRACK, make_rng(3))
    for r in results:
        other = again.train_unit()
        assert other.metrics == r.metrics and other.extra == r.extra
        assert other.env_steps == r.env_steps
    assert np.array_equal(again.genomes, learner.genomes)
    other_seed = GALearner(SMALL, TRACK, make_rng(4))
    assert not np.array_equal(other_seed.genomes, GALearner(SMALL, TRACK, make_rng(3)).genomes)


def test_state_and_restore_continue_exactly():
    a = GALearner(SMALL, TRACK, make_rng(11))
    a.train_unit()
    a.train_unit()
    saved = a.state()
    b = GALearner(SMALL, TRACK, make_rng(999))  # different generators, then restored
    b.restore(saved)
    for _ in range(2):
        ra, rb = a.train_unit(), b.train_unit()
        assert ra.metrics == rb.metrics and ra.extra == rb.extra and ra.env_steps == rb.env_steps
    assert np.array_equal(a.genomes, b.genomes)
    assert a.current_networks()[0].equals(b.current_networks()[0])
    fresh = GALearner(SMALL, TRACK, make_rng(11))
    fresh.restore(GALearner(SMALL, TRACK, make_rng(11)).state())  # an unscored learner too
    assert fresh.elite_fitness is None and fresh.best_fitness is None


def test_held_out_tracks_are_refused():
    with pytest.raises(ValueError, match="held-out"):
        GALearner(SMALL, HELD_OUT, make_rng(0))
