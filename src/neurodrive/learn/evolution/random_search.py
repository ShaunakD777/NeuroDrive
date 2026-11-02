"""Random search (task 5.7, ADR-040): try fresh random networks, keep the best one so far.

The simplest gradient-free learner and the null model for every later one: it uses nothing but
a score per network (its *fitness*) and selection by that score. It has no memory of where good
networks were: every candidate is a fresh draw from the initial distribution (Xavier), like a
new lottery ticket. A genetic algorithm (Phase 7) must beat it at an equal budget of
environment steps, or mutation and crossover bought nothing.

**Fitness** (one number per network, from fixed starts on a training track). For one start,
with L the track length, T the time limit, p = progress / (laps · L) kept in [0, 1], c = 1 if the
car finished the target laps (else 0) and t its finishing time:

    f = p + c · (1 − t / T)

so progress counts up to 1 and a finisher gets a bonus for the time left over. The fitness F is
the mean of f over the starts; 0 ≤ F < 2. Two starts, one finishing at t = 69 s of T = 92 s
(f = 1.25) and one crashing at 30 % (f = 0.30), give F = 0.775.

**One unit**: draw N networks; drive every network from every start in one batched world
(N · S cars, car j belongs to network j // S); score them; if the best of the batch beats the
best fitness so far (strictly), keep it. The evaluation is fixed and deterministic, so the best
fitness so far never decreases. The environment steps of a unit are the car-steps driven.

Sensors are computed only for cars still running (a stopped car's actions are ignored), which
gives the same readings and about 9× less work: most random networks stop within seconds.
"""

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.population import Population
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import Rng
from neurodrive.env.policies import PopulationPolicy
from neurodrive.env.sensors import OBSERVATION_SIZE, build_observation, edge_grid
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig
from neurodrive.learn.base import LearnerState, UnitResult, generator_state, set_generator_state

Array = NDArray[np.float64]


@dataclass(frozen=True)
class RandomSearchSettings:
    """Everything random search needs besides the track and the generator.

    ``start_fractions`` and ``lateral_offsets`` give the evaluation starts: every offset with
    every fraction of the track length, offsets in the outer loop (the Driving Exam's layout).
    """

    spec: NetworkSpec = field(default_factory=NetworkSpec)
    init_scheme: str = "xavier"
    candidates: int = 20
    start_fractions: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8)
    lateral_offsets: tuple[float, ...] = (0.0, 1.5)
    target_laps: int = 2

    def __post_init__(self) -> None:
        if self.candidates < 1:
            raise ValueError("candidates must be at least 1")
        if not self.start_fractions or not self.lateral_offsets:
            raise ValueError("need at least one start fraction and one lateral offset")
        if self.target_laps < 1:
            raise ValueError("target_laps must be at least 1")

    def starts(self, length: float) -> tuple[Array, Array]:
        """Arc lengths and sideways offsets of the evaluation starts on a track ``length`` long."""
        fractions = np.asarray(self.start_fractions, dtype=np.float64)
        s0 = np.tile(fractions * length, len(self.lateral_offsets))
        lateral = np.repeat(np.asarray(self.lateral_offsets, dtype=np.float64), len(fractions))
        return s0, lateral


def start_fitness(
    progress_frac: ArrayLike, completed: ArrayLike, finish_time: ArrayLike, time_limit: float
) -> Array:
    """f = p + c · (1 − t / T) per start (see the module docstring); t is ignored when c = 0."""
    p = np.clip(np.asarray(progress_frac, dtype=np.float64), 0.0, 1.0)
    c = np.asarray(completed, dtype=bool)
    t = np.where(c, np.asarray(finish_time, dtype=np.float64), time_limit)
    return p + c * (1.0 - t / time_limit)


@dataclass(frozen=True)
class Evaluation:
    """How a batch of networks drove: per network F, per start f, outcomes, and car-steps."""

    fitness: Array  # (N,): mean of f over the starts
    per_start: Array  # (N, S): f of every start
    progress_frac: Array  # (N, S)
    status: NDArray[np.int8]  # (N, S): Status values
    finish_time: Array  # (N, S): seconds, NaN where the car did not finish
    env_steps: int  # car-steps driven (cars running at the start of each step)


def evaluate(networks: list[Network], track: Track, settings: RandomSearchSettings) -> Evaluation:
    """Drive every network from every start of ``settings`` on ``track`` in one world, and score.

    Steps: place N · S cars (car j belongs to network j // S); each step, compute the sensors of
    the cars still running, run every network once, step the physics and note when a car
    finishes; when no car runs any more, turn progress, finish and time into fitness.
    """
    n, (s0, lateral) = len(networks), settings.starts(track.length)
    s = len(s0)
    world = World(track, WorldConfig.for_track(track, settings.target_laps))
    world.reset(s0=np.tile(s0, n), lateral=np.tile(lateral, n))
    genomes = np.stack([net.flatten() for net in networks])
    policy = PopulationPolicy(Population.from_genomes(settings.spec, np.repeat(genomes, s, 0)))
    segments = edge_grid(track)
    obs = np.zeros((world.n, OBSERVATION_SIZE))
    finish_step = np.full(world.n, -1)
    env_steps = 0
    while not world.done:
        run = world.running
        env_steps += int(run.sum())
        cars = world.cars
        obs[run] = build_observation(
            segments,
            cars.x[run],
            cars.y[run],
            cars.psi[run],
            cars.v[run],
            world.road_heading[run],
            world.params.v_max,
        )
        action = np.clip(policy.act(obs), -1.0, 1.0)
        world.step(action[:, 0], action[:, 1])
        finished = (world.status == Status.FINISHED) & (finish_step < 0)
        finish_step[finished] = world.step_count
    progress_frac = np.clip(world.progress / (settings.target_laps * track.length), 0.0, 1.0)
    completed = world.status == Status.FINISHED
    finish_time = np.where(completed, finish_step * world.dt, np.nan)
    per_start = start_fitness(progress_frac, completed, finish_time, world.config.time_limit)
    shape = (n, s)
    return Evaluation(
        fitness=per_start.reshape(shape).mean(axis=1),
        per_start=per_start.reshape(shape),
        progress_frac=progress_frac.reshape(shape),
        status=world.status.reshape(shape).copy(),
        finish_time=finish_time.reshape(shape),
        env_steps=env_steps,
    )


class RandomSearchLearner:
    """Random search as a learner (learner v1, resumable).

    The unit-0 network is the learner's first draw, not yet evaluated: with seed k it is the
    random-network baseline of seed k. Each unit draws ``settings.candidates`` new networks.
    """

    name = "random_search"
    feedback_kind = "episode_score"

    def __init__(self, settings: RandomSearchSettings, track: Track, rng: Rng) -> None:
        self.settings, self.track, self.rng = settings, track, rng
        self.best = init_network(settings.spec, rng, settings.init_scheme)
        self.best_fitness: float | None = None  # None until the first unit evaluated anything
        self.best_unit = 0
        self.units = 0
        self.candidates_total = 0

    def train_unit(self) -> UnitResult:
        self.units += 1
        n, spec, scheme = self.settings.candidates, self.settings.spec, self.settings.init_scheme
        batch = [init_network(spec, self.rng, scheme) for _ in range(n)]
        result = evaluate(batch, self.track, self.settings)
        self.candidates_total += n
        winner = int(np.argmax(result.fitness))  # the first of equal bests
        unit_best = float(result.fitness[winner])
        improved = self.best_fitness is None or unit_best > self.best_fitness
        if improved:
            self.best, self.best_fitness, self.best_unit = batch[winner], unit_best, self.units
        metrics: dict[str, Any] = {
            "best_fitness": self.best_fitness,
            "unit_best_fitness": unit_best,
            "unit_median_fitness": float(np.median(result.fitness)),
            "improved": improved,
            "best_unit": self.best_unit,
            "candidates_total": self.candidates_total,
        }
        return UnitResult(metrics=metrics, env_steps=result.env_steps)

    def current_networks(self) -> list[Network]:
        return [self.best]

    def state(self) -> LearnerState:
        return LearnerState(
            arrays={"best_genome": self.best.flatten()},
            info={
                "units": self.units,
                "best_fitness": self.best_fitness,
                "best_unit": self.best_unit,
                "candidates_total": self.candidates_total,
                "rng": generator_state(self.rng),
            },
        )

    def restore(self, state: LearnerState) -> None:
        self.best = Network.unflatten(self.settings.spec, state.arrays["best_genome"])
        info = state.info
        self.units, self.best_fitness = info["units"], info["best_fitness"]
        self.best_unit, self.candidates_total = info["best_unit"], info["candidates_total"]
        set_generator_state(self.rng, info["rng"])
