"""The genetic algorithm, part 1 (task 7.2, ADR-055): scoring and ranking a whole generation.

A generation is a genome matrix G of shape (N, P): row i is network i's genome (task 7.1,
``learn/evolution/genome.py``). Evaluating it means driving every network from every evaluation
start and turning each drive into one number, its fitness, because selection (7.3) and elitism
(7.5) only ever compare those numbers.

Fitness (fitness v2, docs/ARCHITECTURE.md §6.7; the same as random search, ADR-040). Per start,
with L the track length, T its time limit (ADR-022) and 2 target laps:

    f = p + c * (1 - t / T)

where p = progress / (2 L) kept in [0, 1], c = 1 for a finish (else 0) and t its time. A
network's fitness F is the mean of f over its S starts, so 0 <= F < 2: progress counts up to 1,
and a finisher gets up to 1 more for time left over. Example: a network that finishes one start
at 60 s (f = 1 + (1 - 60/92) = 1.348) and runs out of time on another at 99 % (f = 0.99) has
F = 1.169; one that finishes both at 90 s has F = 1.022, so speed on one start can outweigh a
near miss.

Evaluation starts (ADR-055 D2A, as random search's): the Driving Exam's 10 starts on a training
track (5 places x lateral offsets 0 and +1.5 m), the same every generation, so a network's
fitness never changes between generations and elitism can keep the best for sure.

All N x S cars drive in one world (``random_search.evaluate``: the same code, so the GA and
random search measure exactly the same thing): car j belongs to network j // S, and every step
computes the sensors of the running cars, one population forward pass and one physics step.

Ranking: networks sorted by F, largest first; equal F keeps the lower index first (a stable
sort), so the ranking is deterministic. ``ranks[i]`` is network i's place (0 = best).

**The generation loop** (task 7.5, ADR-056). Symbols: N networks of P genes, E elites
(0 <= E < N), the ranking r (best first), c, p and sigma the crossover rate, mutation rate and
mutation strength (``variation.py``), k the tournament size and s the rank pressure
(``selection.py``). One step from generation g (matrix G_g, fitness F) to g + 1:

1. Elite: rows 0 .. E - 1 of G_{g+1} are rows r_0 .. r_{E-1} of G_g, copied exactly;
2. Parents: 2 (N - E) picks by tournament or rank selection; child j has parents a = pick 2j
   and b = pick 2j + 1;
3. Crossover (rate c), then Gaussian mutation (rate p, strength sigma) of the children only;
4. Rows E .. N - 1 of G_{g+1} are the children.

The evaluation is the same every generation and deterministic, so an elite scores again exactly
what it scored before (its *elite drift*, the largest change, is 0) and, with E >= 1, the best
fitness of a generation never falls below the previous generation's best. The old champion sits
in row 0 and ties go to the lower row, so an equally good child never displaces it.

Worked example (N = 4, E = 1, P = 2, c = 0.5, p = 0.5, sigma = 0.1): genomes (0, 0), (0.5, -0.5),
(1, 1), (0.2, 0.4) with F = (0.3, 1.2, 0.3, 1.9). The elite is row 3. Tournaments give the pairs
(3, 1), (1, 0), (3, 3). Child 1 is crossed (w = 0.3) with coins u = (0.7, 0.2): (0.5, 0.4), then
gene 0 is mutated (v = 0.1, z = 1.0): (0.6, 0.4). Child 2 is not crossed (w = 0.8): a copy of row
1, and v = (0.7, 0.6) mutates nothing. Child 3 is row 3 with itself; v = (0.2, 0.3) and
z = (-2, 1) give (0.0, 0.5). Generation g + 1 = ((0.2, 0.4), (0.6, 0.4), (0.5, -0.5), (0.0, 0.5)).

*Diversity* D is the mean Euclidean distance between the genomes of all N (N - 1) / 2 pairs:
(0, 0), (3, 4), (0, 4) are 5, 3 and 4 apart, so D = 4.

Every unit also hands back the genome matrix it scored (``UnitResult.arrays["generations"]``,
task 7.6, ADR-057): the runner saves it as ``extra/generations/unit_<g>.npz``, row i being the
network whose fitness is entry i of that unit's ``extra/population.jsonl`` line, so any
generation can race again in the live view. Next to the genomes it keeps how every network drove
from every evaluation start (task 7.7 follow-up, ADR-059), each an array (N, S): ``start_fitness``
f_ik (so F_i is the mean of row i), ``status`` the ``Status`` code of the drive, ``progress`` the
share of the target distance covered and ``finish_time`` the seconds to the finish (NaN where it
did not finish). The evolution charts read them to show one start at a time.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import Rng, child_rng
from neurodrive.env.car import ACTION_SIZE
from neurodrive.env.sensors import OBSERVATION_SIZE
from neurodrive.env.track import Track
from neurodrive.env.world import Status
from neurodrive.learn.base import LearnerState, UnitResult, generator_state, set_generator_state
from neurodrive.learn.evolution.genome import networks_of, random_genomes
from neurodrive.learn.evolution.random_search import (
    Evaluation,
    RandomSearchSettings,
    evaluate,
)
from neurodrive.learn.evolution.selection import rank_selection, tournament_selection
from neurodrive.learn.evolution.variation import (
    CrossoverResult,
    MutationResult,
    gaussian_mutation,
    uniform_crossover,
)

Array = NDArray[np.float64]

SELECTIONS = ("tournament", "rank")


def _whole(value: Any, name: str, low: int) -> None:
    """Check: a whole number of at least ``low`` (booleans are refused)."""
    if isinstance(value, bool) or not isinstance(value, int | np.integer) or value < low:
        raise ValueError(f"{name} must be a whole number of at least {low}, got {value!r}")


def _between(value: Any, name: str, low: float, high: float) -> None:
    """Check: a number between ``low`` and ``high`` (booleans are refused)."""
    if isinstance(value, bool) or not isinstance(value, int | float | np.number):
        raise ValueError(f"the {name} must be a number between {low} and {high}, got {value!r}")
    if not low <= float(value) <= high:
        raise ValueError(f"the {name} must be between {low} and {high}, got {value!r}")


@dataclass(frozen=True)
class BreedingSettings:
    """How one generation makes the next: the elite E, the selection operator and its setting
    (tournament size k or rank pressure s), the crossover rate c, the mutation rate p and the
    mutation strength sigma. The defaults are the shipped ``configs/ga.yaml`` (ADR-056, D3B)."""

    elite: int = 2
    selection: str = "tournament"
    tournament_size: int = 3
    rank_pressure: float = 1.5
    crossover_rate: float = 0.5
    mutation_rate: float = 0.02
    mutation_strength: float = 0.05

    def __post_init__(self) -> None:
        _whole(self.elite, "the elite", 0)
        if self.selection not in SELECTIONS:
            raise ValueError(
                f"unknown selection {self.selection!r}; valid: {', '.join(SELECTIONS)}"
            )
        _whole(self.tournament_size, "the tournament size", 1)
        _between(self.rank_pressure, "rank selection pressure", 1.0, 2.0)
        _between(self.crossover_rate, "crossover rate", 0.0, 1.0)
        _between(self.mutation_rate, "mutation rate", 0.0, 1.0)
        sigma = self.mutation_strength
        if isinstance(sigma, bool) or not np.isfinite(float(sigma)) or float(sigma) < 0:
            raise ValueError(f"the mutation strength must be a finite number >= 0, got {sigma!r}")


@dataclass(frozen=True)
class GASettings:
    """What a GA generation is, how it is scored and how it breeds.

    ``population`` is N, the number of networks per generation; ``start_fractions`` and
    ``lateral_offsets`` give the evaluation starts exactly as for random search (every offset with
    every fraction of the track length, offsets in the outer loop: the exam's layout);
    ``breeding`` makes the next generation (task 7.5).
    """

    spec: NetworkSpec = field(default_factory=NetworkSpec)
    population: int = 50
    init_scheme: str = "xavier"
    start_fractions: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    lateral_offsets: tuple[float, ...] = (0.0, 1.5)
    target_laps: int = 2
    breeding: BreedingSettings = field(default_factory=BreedingSettings)

    def __post_init__(self) -> None:
        if isinstance(self.population, bool) or not isinstance(self.population, int):
            raise ValueError(f"population must be a whole number, got {self.population!r}")
        if self.population < 2:
            raise ValueError("a population needs at least 2 networks")
        # elite < population is checked where a generation is bred (GALearner, next_generation),
        # so settings used only to score a generation (task 7.2) may have any population
        if (self.spec.n_inputs, self.spec.n_outputs) != (OBSERVATION_SIZE, ACTION_SIZE):
            raise ValueError(
                f"a driving network needs {OBSERVATION_SIZE} inputs and {ACTION_SIZE} outputs, "
                f"got {list(self.spec.layer_sizes)}"
            )
        # random search's settings check the starts and the target laps
        self.evaluation()

    @property
    def starts_per_network(self) -> int:
        """S, the number of evaluation starts each network drives."""
        return len(self.start_fractions) * len(self.lateral_offsets)

    def evaluation(self) -> RandomSearchSettings:
        """The same evaluation expressed as random search's settings (its ``evaluate`` runs it)."""
        return RandomSearchSettings(
            spec=self.spec,
            init_scheme=self.init_scheme,
            candidates=self.population,
            start_fractions=self.start_fractions,
            lateral_offsets=self.lateral_offsets,
            target_laps=self.target_laps,
        )


@dataclass(frozen=True)
class GenerationEvaluation:
    """One generation's scores. ``evaluation`` holds random search's per-network and per-start
    arrays (fitness F, per-start f, progress, status, finish times, env steps); ``ranking`` lists
    network indices best first and ``ranks[i]`` is network i's place in it (0 = best)."""

    evaluation: Evaluation
    ranking: NDArray[np.intp]
    ranks: NDArray[np.intp]

    @property
    def fitness(self) -> Array:
        """F, one number per network, shape (N,)."""
        return self.evaluation.fitness

    @property
    def env_steps(self) -> int:
        """Car-steps driven: the generation's environment budget."""
        return self.evaluation.env_steps

    @property
    def best(self) -> int:
        """Index of the best network (the lowest index among equals)."""
        return int(self.ranking[0])


def ranking(fitness: ArrayLike) -> tuple[NDArray[np.intp], NDArray[np.intp]]:
    """The order of the networks by fitness, best first, and each network's place in it.

    A stable sort of -F: equal fitness keeps the lower index first. Example: F = (0.3, 1.2, 0.3,
    1.9) gives the order (3, 1, 0, 2) and the places (2, 1, 3, 0).
    """
    f = np.asarray(fitness, dtype=np.float64)
    if f.ndim != 1:
        raise ValueError(f"fitness must be 1-D, got shape {f.shape}")
    if np.isnan(f).any():
        raise ValueError("fitness contains NaN")
    order = np.argsort(-f, kind="stable").astype(np.intp)
    places = np.empty_like(order)
    places[order] = np.arange(len(order))
    return order, places


def evaluate_genomes(
    genomes: ArrayLike, track: Track, settings: GASettings
) -> GenerationEvaluation:
    """Score and rank one generation (genome matrix (N, P)) on a training track.

    1. Check the matrix: one row per network of the settings' population, P columns.
    2. Turn the rows into networks (``networks_of``).
    3. Drive all N x S cars in one world and score them (``random_search.evaluate``).
    4. Rank the networks by their fitness.

    Raises ``ValueError`` for a wrong matrix or a held-out track (protocol rule 7: fitness may
    only use training tracks).
    """
    g = np.asarray(genomes, dtype=np.float64)
    want = (settings.population, settings.spec.n_parameters)
    if g.shape != want:
        raise ValueError(f"genomes must have shape {want}, got {g.shape}")
    if track.split != "train":
        raise ValueError(
            f"{track.name!r} is a held-out track; fitness may only use training tracks"
        )
    result = evaluate(networks_of(settings.spec, g), track, settings.evaluation())
    order, places = ranking(result.fitness)
    return GenerationEvaluation(result, order, places)


# ── the generation loop (task 7.5, ADR-056) ──────────────────────────────────────────────────

_GENERATOR_NAMES = ("selection", "crossover", "mutation")


@dataclass
class GAGenerators:
    """One seeded generator per random operator, so changing one setting (say the mutation rate)
    never changes the numbers another operator draws."""

    selection: Rng
    crossover: Rng
    mutation: Rng

    @classmethod
    def from_master(cls, rng: Rng) -> "GAGenerators":
        """The named children "selection", "crossover" and "mutation" of ``rng`` (``child_rng``)."""
        return cls(*(child_rng(rng, name) for name in _GENERATOR_NAMES))

    def state(self) -> dict[str, Any]:
        """Every generator's state as plain JSON values."""
        return {name: generator_state(getattr(self, name)) for name in _GENERATOR_NAMES}

    def set_state(self, state: dict[str, Any]) -> None:
        """Put every generator back to ``state``."""
        for name in _GENERATOR_NAMES:
            set_generator_state(getattr(self, name), state[name])


@dataclass(frozen=True)
class Breeding:
    """The next generation and how it was made.

    ``genomes`` (N, P): rows 0 .. E - 1 the elite, then the N - E children. ``elite`` (E,): the
    old rows the elite was copied from, best first. ``parents_a`` / ``parents_b`` (N - E,): each
    child's parents (old rows). ``crossover`` and ``mutation``: the operators' records for the
    children (which parent each gene came from, which genes were mutated).
    """

    genomes: NDArray[np.float64]
    elite: NDArray[np.intp]
    parents_a: NDArray[np.intp]
    parents_b: NDArray[np.intp]
    crossover: CrossoverResult
    mutation: MutationResult


def next_generation(
    genomes: ArrayLike, fitness: ArrayLike, breeding: BreedingSettings, generators: GAGenerators
) -> Breeding:
    """Make generation g + 1 from generation g's genome matrix (N, P) and fitness (N,).

    1. Rank the rows by fitness (best first, ties to the lower row) and copy the best E rows,
       unchanged, into rows 0 .. E - 1.
    2. Pick 2 (N - E) parents by the chosen selection operator ("selection" generator) and pair
       them in pick order: child j gets picks 2j and 2j + 1.
    3. Cross each pair (rate c, "crossover" generator), then mutate the children (rate p,
       strength sigma, "mutation" generator). The elite is never mutated.
    4. Stack the elite on top of the children.

    Only the fitness order matters, never driving: any fitness works (the tests use a quadratic).
    Raises ``ValueError`` for mismatched shapes or E >= N.
    """
    g = np.asarray(genomes, dtype=np.float64)
    f = np.asarray(fitness, dtype=np.float64)
    if g.ndim != 2 or f.shape != (g.shape[0],):
        raise ValueError(
            f"need a genome matrix (N, P) and N fitness values, got {g.shape} and {f.shape}"
        )
    n, e = g.shape[0], breeding.elite
    if e >= n:
        raise ValueError(f"the elite ({e}) must be smaller than the population ({n})")
    order, _ = ranking(f)
    elite = order[:e]
    picks_needed = 2 * (n - e)
    if breeding.selection == "tournament":
        picks = tournament_selection(
            f, picks_needed, generators.selection, size=breeding.tournament_size
        )
    else:
        picks = rank_selection(
            f, picks_needed, generators.selection, pressure=breeding.rank_pressure
        )
    a, b = picks[0::2], picks[1::2]
    crossed = uniform_crossover(g[a], g[b], generators.crossover, rate=breeding.crossover_rate)
    mutated = gaussian_mutation(
        crossed.genomes,
        generators.mutation,
        rate=breeding.mutation_rate,
        strength=breeding.mutation_strength,
    )
    children = np.vstack([g[elite], mutated.genomes])
    return Breeding(children, elite, a, b, crossed, mutated)


def diversity(genomes: ArrayLike) -> float:
    """D, the mean Euclidean distance between the genomes of every pair of rows i < j.

    Row by row (the distances from row i to every later row), so memory stays N x P. A single
    genome has no pairs: D = 0. Example: (0, 0), (3, 4), (0, 4) give (5 + 3 + 4) / 3 = 4.
    """
    g = np.asarray(genomes, dtype=np.float64)
    if g.ndim != 2:
        raise ValueError(f"genomes must be a matrix (N, P), got shape {g.shape}")
    n = g.shape[0]
    if n < 2:
        return 0.0
    total = 0.0
    for i in range(n - 1):
        total += float(np.sqrt(((g[i + 1 :] - g[i]) ** 2).sum(axis=1)).sum())
    return total / (n * (n - 1) / 2)


class GALearner:
    """The genetic algorithm as a learner (learner v1, resumable; task 7.5, ADR-056).

    One unit is one generation: score the current generation (``evaluate_genomes``), note its
    best network (the checkpoint) and statistics, then breed the next generation
    (``next_generation``). Generation 1 is N Xavier genomes from the "init" child generator;
    before it is scored, the network to checkpoint (unit 0) is its row 0. The elites are scored
    again every generation (ADR-056, D1A): their change in fitness, the *elite drift*, must be 0.
    """

    name = "ga"
    feedback_kind = "episode_score"

    def __init__(self, settings: GASettings, track: Track, rng: Rng) -> None:
        if track.split != "train":
            raise ValueError(
                f"{track.name!r} is a held-out track; fitness may only use training tracks"
            )
        if settings.breeding.elite >= settings.population:
            raise ValueError(
                f"the elite ({settings.breeding.elite}) must be smaller than the population "
                f"({settings.population}), or no child is ever made"
            )
        self.settings, self.track = settings, track
        self.generators = GAGenerators.from_master(rng)
        self.genomes = random_genomes(
            settings.spec, settings.population, child_rng(rng, "init"), settings.init_scheme
        )
        self.champion = self.genomes[0].copy()  # the generation's best; row 0 before unit 1
        self.champion_fitness: float | None = None
        self.best_fitness: float | None = None  # the best of every generation so far
        self.best_unit = 0
        self.units = 0
        self.elite_fitness: Array | None = None  # rows 0 .. E - 1's fitness when they were kept

    def train_unit(self) -> UnitResult:
        """Score this generation, record its best and statistics, breed the next one."""
        self.units += 1
        scored = evaluate_genomes(self.genomes, self.track, self.settings)
        f, ev = scored.fitness, scored.evaluation
        e = self.settings.breeding.elite
        drift = None
        if self.elite_fitness is not None and e > 0:
            drift = float(np.max(np.abs(f[:e] - self.elite_fitness)))
        champion = scored.best
        self.champion = self.genomes[champion].copy()
        self.champion_fitness = float(f[champion])
        improved = self.best_fitness is None or self.champion_fitness > self.best_fitness
        if improved:
            self.best_fitness, self.best_unit = self.champion_fitness, self.units
        finished = (ev.status == Status.FINISHED).sum(axis=1)
        crashed = (ev.status == Status.CRASHED).sum(axis=1)
        drives = f.size * self.settings.starts_per_network
        metrics: dict[str, Any] = {
            "best_fitness": self.best_fitness,
            "generation_best_fitness": self.champion_fitness,
            "mean_fitness": float(f.mean()),
            "median_fitness": float(np.median(f)),
            "worst_fitness": float(f.min()),
            "fitness_std": float(f.std()),
            "zero_fitness": int((f == 0.0).sum()),
            "finished_share": float(finished.sum() / drives),
            "crashed_share": float(crashed.sum() / drives),
            "finishers": int((finished == self.settings.starts_per_network).sum()),
            "diversity": diversity(self.genomes),
            "elite_drift": drift,
            "improved": improved,
            "best_unit": self.best_unit,
        }
        line: dict[str, Any] = {
            "fitness": f.tolist(),
            "finished": finished.tolist(),
            "crashed": crashed.tolist(),
            "elite": list(range(e)) if self.elite_fitness is not None else [],
            "champion": champion,
        }
        scored_genomes = self.genomes  # generation g, the matrix just scored (task 7.6)
        bred = next_generation(scored_genomes, f, self.settings.breeding, self.generators)
        self.genomes, self.elite_fitness = bred.genomes, f[bred.elite].copy()
        return UnitResult(
            metrics=metrics,
            env_steps=scored.env_steps,
            new_best=improved,
            extra={"population": line},
            arrays={
                "generations": {
                    "genomes": scored_genomes.copy(),
                    "start_fitness": ev.per_start.copy(),  # (N, S), task 7.7 follow-up, ADR-059
                    "status": ev.status.astype(np.int8),
                    "progress": ev.progress_frac.copy(),
                    "finish_time": ev.finish_time.copy(),
                }
            },
        )

    def current_networks(self) -> list[Network]:
        """The best network of the latest scored generation (row 0 before the first)."""
        return [Network.unflatten(self.settings.spec, self.champion)]

    def state(self) -> LearnerState:
        """The next generation, the champion, the elites' fitness, counters and generators."""
        known = self.elite_fitness is not None
        return LearnerState(
            arrays={
                "genomes": self.genomes,
                "champion": self.champion,
                "elite_fitness": self.elite_fitness if known else np.zeros(0),
            },
            info={
                "units": self.units,
                "champion_fitness": self.champion_fitness,
                "best_fitness": self.best_fitness,
                "best_unit": self.best_unit,
                "elite_fitness_known": known,
                "generators": self.generators.state(),
            },
        )

    def restore(self, state: LearnerState) -> None:
        """Become exactly the learner that produced ``state``."""
        arrays, info = state.arrays, state.info
        self.genomes = np.array(arrays["genomes"], dtype=np.float64)
        self.champion = np.array(arrays["champion"], dtype=np.float64)
        known = info["elite_fitness_known"]
        self.elite_fitness = np.array(arrays["elite_fitness"], dtype=np.float64) if known else None
        self.units, self.champion_fitness = info["units"], info["champion_fitness"]
        self.best_fitness, self.best_unit = info["best_fitness"], info["best_unit"]
        self.generators.set_state(info["generators"])
