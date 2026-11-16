"""Task 7.0: how long does one generation take, and where does the time go?

Run from the repository root:

    uv run python scripts/profile_generation.py                       # worst + typical case
    uv run python scripts/profile_generation.py --scaling --profile   # also scaling, functions
    uv run python scripts/profile_generation.py --out perf.json       # also write the numbers

The target (docs/ARCHITECTURE.md §9): one GA generation, N = 50 cars x T = 60 s simulated, in at
most B = 5 s of wall-clock time. With the physics step dt = 1/30 s that is S = T / dt = 1,800
steps, so the budget is B / S = 2.78 ms per step, or 55.6 microseconds per car-step for 50 cars.

What is measured (nothing here changes the simulation; every random number is seeded):

* **Worst case.** A generation lasts as long as its longest-surviving car, and a good population
  drives the whole time. Random networks crash within seconds, so instead the scripted expert
  first drives the N cars from seeded random starts (an untimed recording pass: it never crashes
  and needs about 66 s for 2 laps of train_loop_01, so every car still runs at 60 s). The timed
  pass then does a generation's work for every step: the sensors of the running cars, a forward
  pass of N different Xavier-initialised [7, 16, 16, 2] networks, and the physics, but it steers
  with the recorded expert actions. The networks' outputs are computed and not used, so every
  car survives the full T and the expert's own cost stays out of the timing.
* **Typical early generation.** The same loop, but the N fresh random networks steer themselves
  (most crash early). And, for scale, one unit of the real random-search code
  (``learn/evolution/random_search.evaluate``): N networks x its 10 evaluation starts.
* **Breakdown.** Each step's time is split into sensors (``build_observation``), network
  (``PopulationPolicy.act``), world step (physics, projection onto the centreline, termination)
  and the rest (bookkeeping).
* **Scaling** (``--scaling``): the worst case for 1 to 500 cars over the first 10 s.
* **Functions** (``--profile``): the worst case once under Python's ``cProfile``.

Timings vary from run to run, so the main cases report the median of ``--repeats`` runs (the
breakdown is that run's) and the best run. Noise only ever adds time, so the best run is the
closest to what the code itself costs. On a CPU with performance and efficiency cores (task 7.0
measured an Intel i5-13420H: 4 + 4 cores) the same generation takes about twice as long on an
efficiency core, and the operating system may move the process between them, so the repeats can
differ by up to 2x; pinning the process to one core (Windows: ``start /AFFINITY <mask>``) makes
them steady.
"""

import argparse
import cProfile
import json
import os
import platform
import pstats
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from neurodrive.core.nn.init import init_network, init_population
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import child_rng, make_rng
from neurodrive.env.car import DT
from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.policies import PopulationPolicy
from neurodrive.env.sensors import OBSERVATION_SIZE, build_observation, edge_grid
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig
from neurodrive.learn.evolution.random_search import RandomSearchSettings, evaluate

ROOT = Path(__file__).resolve().parents[1]
TRACKS = ROOT / "tracks"
TARGET_S = 5.0  # ARCHITECTURE §9: one generation of 50 cars x 60 s in at most 5 s
LIVE_FRAME_MS = 1000.0 / 30.0  # ARCHITECTURE §9: the live view at 30 frames per second
START_LATERAL = 1.0  # metres: seeded random starts, as DAgger's collection drives use
START_HEADING = 0.1  # radians
PHASES = ("sensors", "network", "world_step", "rest")
SCALING_CARS = (1, 10, 50, 100, 200, 500)
SCALING_SECONDS = 10.0


@dataclass
class Generation:
    """What one timed generation did and how long each part took (wall-clock seconds)."""

    n_cars: int
    steps: int  # world steps until no car was running
    env_steps: int  # car-steps driven: cars running at the start of each step
    sensor_rows: int  # car-steps whose sensors were computed (must equal env_steps)
    total_s: float
    phase_s: dict[str, float]
    status: dict[str, int]  # how the cars ended
    final: NDArray[np.float64] = field(repr=False)  # (N, 4): x, y, psi, v at the end

    @property
    def ms_per_step(self) -> float:
        return 1000.0 * self.total_s / max(self.steps, 1)

    @property
    def us_per_car_step(self) -> float:
        return 1e6 * self.total_s / max(self.env_steps, 1)


def load_track(name: str) -> Track:
    """A track from ``tracks/<name>.json``."""
    return Track.from_json(TRACKS / f"{name}.json")


def make_world(track: Track, n_cars: int, seconds: float, seed: int) -> World:
    """A world with ``n_cars`` cars at seeded random starts and a time limit of ``seconds``.

    The time limit is the only rule changed from the training worlds; the starts are anywhere
    on the track, up to START_LATERAL metres sideways and START_HEADING radians turned, at rest.
    """
    world = World(track, WorldConfig(time_limit=seconds))
    world.reset(
        n=n_cars,
        rng=child_rng(make_rng(seed), "starts"),
        random_s0=True,
        lateral_jitter=START_LATERAL,
        heading_jitter=START_HEADING,
    )
    return world


def expert_actions(track: Track, n_cars: int, seconds: float, seed: int) -> NDArray[np.float64]:
    """The untimed recording pass: the expert's (steer, throttle) per step, shape (S, N, 2)."""
    world = make_world(track, n_cars, seconds, seed)
    expert = ScriptedExpert(world)
    actions = []
    while not world.done:
        a = expert.label()
        actions.append(a)
        world.step(a[:, 0], a[:, 1])
    return np.array(actions)


def run_generation(
    track: Track,
    n_cars: int,
    seconds: float,
    seed: int,
    actions: NDArray[np.float64] | None = None,
) -> Generation:
    """One timed generation: sensors, N networks and physics, every step until all cars stop.

    With ``actions`` (from :func:`expert_actions`, same track, cars, seconds and seed) the cars
    are steered by the recorded expert: the worst case. Without, the networks steer themselves.
    Steps: (1) the sensors of the running cars, (2) every network once on its car's
    observation, (3) the world step, (4) the bookkeeping that ``random_search.evaluate`` does
    (step count, finish moments). Each part is timed with ``time.perf_counter``.
    """
    world = make_world(track, n_cars, seconds, seed)
    nets = init_population(NetworkSpec(), n_cars, child_rng(make_rng(seed), "networks"))
    policy = PopulationPolicy(nets)
    segments = edge_grid(track)
    obs = np.zeros((n_cars, OBSERVATION_SIZE))
    finish_step = np.full(n_cars, -1)
    phase = dict.fromkeys(PHASES, 0.0)
    env_steps = sensor_rows = 0
    clock = time.perf_counter
    start = clock()
    while not world.done:
        t0 = clock()
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
        sensor_rows += int(run.sum())
        t1 = clock()
        action = np.clip(policy.act(obs), -1.0, 1.0)
        t2 = clock()
        if actions is not None:
            action = actions[world.step_count]
        world.step(action[:, 0], action[:, 1])
        t3 = clock()
        finished = (world.status == Status.FINISHED) & (finish_step < 0)
        finish_step[finished] = world.step_count
        t4 = clock()
        phase["sensors"] += t1 - t0
        phase["network"] += t2 - t1
        phase["world_step"] += t3 - t2
        phase["rest"] += t4 - t3
    total = clock() - start
    phase["rest"] += total - sum(phase.values())  # the loop's own overhead
    counts = np.bincount(world.status, minlength=len(Status))
    cars = world.cars
    return Generation(
        n_cars=n_cars,
        steps=world.step_count,
        env_steps=env_steps,
        sensor_rows=sensor_rows,
        total_s=total,
        phase_s=phase,
        status={s.name.lower(): int(counts[s]) for s in Status if counts[s]},
        final=np.stack([cars.x, cars.y, cars.psi, cars.v], axis=1),
    )


def median_generation(runs: list[Generation]) -> Generation:
    """The run with the median total time (its phases belong to one real run)."""
    ordered = sorted(runs, key=lambda g: g.total_s)
    return ordered[len(ordered) // 2]


def time_random_search_unit(track: Track, n_networks: int, seed: int) -> dict[str, Any]:
    """Wall-clock time of one random-search unit's evaluation (real code, its 10 starts)."""
    rng = child_rng(make_rng(seed), "networks")
    settings = RandomSearchSettings(candidates=n_networks)
    networks = [init_network(settings.spec, rng, settings.init_scheme) for _ in range(n_networks)]
    start = time.perf_counter()
    result = evaluate(networks, track, settings)
    total = time.perf_counter() - start
    n_starts = len(settings.start_fractions) * len(settings.lateral_offsets)
    return {
        "networks": n_networks,
        "starts_per_network": n_starts,
        "cars": n_networks * n_starts,
        "time_limit_s": WorldConfig.for_track(track, settings.target_laps).time_limit,
        "env_steps": result.env_steps,
        "total_s": total,
        "us_per_car_step": 1e6 * total / max(result.env_steps, 1),
    }


def scaling(track: Track, seed: int, cars: tuple[int, ...] = SCALING_CARS) -> list[dict[str, Any]]:
    """The worst case for each car count over SCALING_SECONDS: ms per step, us per car-step."""
    rows = []
    for n in cars:
        g = run_generation(
            track, n, SCALING_SECONDS, seed, expert_actions(track, n, SCALING_SECONDS, seed)
        )
        rows.append(
            {
                "cars": n,
                "steps": g.steps,
                "ms_per_step": g.ms_per_step,
                "us_per_car_step": g.us_per_car_step,
                "sensors_share": g.phase_s["sensors"] / g.total_s,
            }
        )
    return rows


def function_profile(
    track: Track, n_cars: int, seconds: float, seed: int, top: int = 12
) -> list[dict[str, Any]]:
    """The worst case once under cProfile: the ``top`` functions by their own time."""
    actions = expert_actions(track, n_cars, seconds, seed)
    profiler = cProfile.Profile()
    profiler.enable()
    run_generation(track, n_cars, seconds, seed, actions)
    profiler.disable()
    stats = pstats.Stats(profiler).stats  # {(file, line, name): (cc, nc, tottime, cumtime, ...)}
    rows = sorted(stats.items(), key=lambda kv: kv[1][2], reverse=True)[:top]
    return [
        {  # only the file's name: absolute paths stay out of every printout
            "function": f"{Path(file).name}:{line}({name})" if line else name,
            "calls": int(nc),
            "own_s": float(tt),
            "cumulative_s": float(ct),
        }
        for (file, line, name), (_cc, nc, tt, ct, _callers) in rows
    ]


def machine() -> dict[str, Any]:
    """What the numbers were measured on (no host or user names)."""
    return {
        "cpu_count": os.cpu_count(),
        "processor": platform.processor() or platform.machine(),
        "system": platform.system(),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }


def summary(g: Generation) -> dict[str, Any]:
    """A generation as plain JSON-friendly numbers."""
    return {
        "cars": g.n_cars,
        "steps": g.steps,
        "env_steps": g.env_steps,
        "total_s": g.total_s,
        "ms_per_step": g.ms_per_step,
        "us_per_car_step": g.us_per_car_step,
        "phase_s": g.phase_s,
        "status": g.status,
    }


def print_generation(title: str, g: Generation, repeats: list[float] | None = None) -> None:
    print(f"\n{title}")
    line = f"  {g.n_cars} cars, {g.steps} steps, {g.env_steps:,} car-steps: {g.total_s:.2f} s"
    if repeats:
        line += f" (median), best {min(repeats):.2f} s"
        line += f"  (repeats: {', '.join(f'{t:.2f}' for t in repeats)} s)"
    print(line)
    print(f"  {g.ms_per_step:.2f} ms per step, {g.us_per_car_step:.1f} us per car-step")
    print(f"  ended: {g.status}")
    for name in PHASES:
        share = g.phase_s[name] / g.total_s
        print(f"    {name:<11} {g.phase_s[name]:7.2f} s  {100 * share:5.1f} %")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cars", type=int, default=50)
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--track", default="train_loop_01")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--scaling", action="store_true", help="worst case for 1 to 500 cars")
    parser.add_argument("--profile", action="store_true", help="cProfile the worst case")
    parser.add_argument("--out", type=Path, help="also write the numbers as JSON")
    args = parser.parse_args()

    track = load_track(args.track)
    n, secs, seed = args.cars, args.seconds, args.seed
    budget_ms = 1000.0 * TARGET_S / round(60.0 / DT)
    print(f"Track {args.track}, {n} cars x {secs:g} s, dt = 1/{round(1 / DT)} s, seed {seed}")
    print(f"Target: <= {TARGET_S:g} s per generation of 50 cars x 60 s ({budget_ms:.2f} ms/step)")
    print(f"Machine: {machine()}")

    actions = expert_actions(track, n, secs, seed)
    worst_runs = [run_generation(track, n, secs, seed, actions) for _ in range(args.repeats)]
    worst = median_generation(worst_runs)
    print_generation(
        "Worst case (every car drives the whole time)", worst, [g.total_s for g in worst_runs]
    )
    best_s = min(g.total_s for g in worst_runs)
    for label, value in (("median", worst.total_s), ("best run", best_s)):
        verdict = "meets" if value <= TARGET_S else "MISSES"
        print(
            f"  -> {label} {verdict} the target: {value:.2f} s vs {TARGET_S:g} s"
            f" ({value / TARGET_S:.2f} x the budget)"
        )
    print(
        f"  -> live view: {worst.ms_per_step:.2f} ms per step vs {LIVE_FRAME_MS:.1f} ms per frame"
    )

    typical_runs = [run_generation(track, n, secs, seed) for _ in range(args.repeats)]
    typical = median_generation(typical_runs)
    print_generation(
        "Typical early generation (random networks steer)",
        typical,
        [g.total_s for g in typical_runs],
    )

    unit = time_random_search_unit(track, n, seed)
    print(
        f"\nOne random-search unit (real code): {unit['networks']} networks x "
        f"{unit['starts_per_network']} starts = {unit['cars']} cars, limit "
        f"{unit['time_limit_s']:g} s: {unit['total_s']:.2f} s, {unit['env_steps']:,} car-steps, "
        f"{unit['us_per_car_step']:.1f} us per car-step"
    )

    results: dict[str, Any] = {
        "track": args.track,
        "seed": seed,
        "target_s": TARGET_S,
        "machine": machine(),
        "worst_case": summary(worst)
        | {"repeats_s": [g.total_s for g in worst_runs], "best_s": best_s},
        "typical_case": summary(typical) | {"repeats_s": [g.total_s for g in typical_runs]},
        "random_search_unit": unit,
    }
    if args.scaling:
        rows = scaling(track, seed)
        print(f"\nScaling (worst case, first {SCALING_SECONDS:g} s)")
        print("   cars   ms/step   us/car-step   sensors share")
        for r in rows:
            print(
                f"  {r['cars']:5d}  {r['ms_per_step']:8.2f}  {r['us_per_car_step']:12.1f}"
                f"   {100 * r['sensors_share']:6.1f} %"
            )
        results["scaling"] = rows
    if args.profile:
        rows = function_profile(track, n, secs, seed)
        print("\nFunctions by own time (worst case, under cProfile, which slows everything)")
        for r in rows:
            print(
                f"  {r['own_s']:7.2f} s own  {r['cumulative_s']:7.2f} s cum  "
                f"{r['calls']:7d} calls  {r['function']}"
            )
        results["functions"] = rows
    if args.out:
        args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
