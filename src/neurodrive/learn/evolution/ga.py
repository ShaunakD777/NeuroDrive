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
"""

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.env.car import ACTION_SIZE
from neurodrive.env.sensors import OBSERVATION_SIZE
from neurodrive.env.track import Track
from neurodrive.learn.evolution.genome import networks_of
from neurodrive.learn.evolution.random_search import (
    Evaluation,
    RandomSearchSettings,
    evaluate,
)

Array = NDArray[np.float64]


@dataclass(frozen=True)
class GASettings:
    """What a GA generation is and how it is scored.

    ``population`` is N, the number of networks per generation; ``start_fractions`` and
    ``lateral_offsets`` give the evaluation starts exactly as for random search (every offset with
    every fraction of the track length, offsets in the outer loop: the exam's layout).
    """

    spec: NetworkSpec = field(default_factory=NetworkSpec)
    population: int = 50
    init_scheme: str = "xavier"
    start_fractions: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    lateral_offsets: tuple[float, ...] = (0.0, 1.5)
    target_laps: int = 2

    def __post_init__(self) -> None:
        if isinstance(self.population, bool) or not isinstance(self.population, int):
            raise ValueError(f"population must be a whole number, got {self.population!r}")
        if self.population < 2:
            raise ValueError("a population needs at least 2 networks")
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
