"""Study S2a: what do random networks do, and can the views explain why?

Run from the repository root:

    uv run python scripts/study_s2a.py                              # numbers only
    uv run --with matplotlib python scripts/study_s2a.py --figures  # also draw the figures

Every car is driven by its own Xavier-initialised [7, 16, 16, 2] tanh network, built exactly as
the live session builds the "Random networks" driver with "Random start" and a seed (task 2.6):
the networks come from the seed's "init" stream and the starts from the seed itself. So car c of
seed s here is car c of seed s in the app, and every case below can be replayed there. The guard
test ``tests/server/test_study_s2a.py`` checks that.

The rules are the exam's: 2 laps of ``train_loop_01``, the stall rule, and the 92 s limit of
ADR-022. Everything is deterministic. The numbers used in ``docs/studies/S2a.md`` are printed and
written to ``docs/studies/S2a_results.json``; with ``--figures`` the pictures go to
``docs/studies/figures/`` (matplotlib is needed only for that).
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.init import init_population
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.population import Population
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import child_rng, make_rng
from neurodrive.env.car import CarParams, CarState, step_cars
from neurodrive.env.episode import EpisodeRecording, run_episode
from neurodrive.env.policies import PopulationPolicy
from neurodrive.env.sensors import OBSERVATION_NAMES, RAY_RANGE
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig
from neurodrive.explain.contributions import neuron_contributions
from neurodrive.explain.sensitivity import sensitivity

ROOT = Path(__file__).resolve().parent.parent
TRACK = "train_loop_01"
RESULTS_FILE = ROOT / "docs" / "studies" / "S2a_results.json"
FIGURES_DIR = ROOT / "docs" / "studies" / "figures"

SEEDS = tuple(range(100))  # 100 seeds x 20 cars = 2,000 random brains
N_CARS = 20
# The live session's random-start ranges (server/live.py); the guard test checks they match.
RANDOM_START_LATERAL = 1.0  # metres
RANDOM_START_HEADING = 0.1  # radians
MOVED = 0.05  # m/s: a car whose speed never exceeds this never moved
FULL_LOCK = 0.9  # |steer| above this counts as (nearly) full lock
SPIN = 1.5 * np.pi  # turning this far relative to the road would count as circling
LEFT_RAYS, RIGHT_RAYS = (0, 1), (3, 4)  # +60 and +30 degrees; -30 and -60 degrees

# The thoroughness grid (added at the user's request): every shape on every track. All tanh,
# Xavier-initialised, biases 0. Only BASE_SHAPE can be loaded in the app.
TRACKS = ("train_loop_01", "monza_cw", "monza_ccw")
BASE_SHAPE = (7, 16, 16, 2)
SHAPES = ((7, 2), (7, 4, 2), (7, 16, 2), BASE_SHAPE, (7, 64, 64, 2), (7, 16, 16, 16, 16, 2))

# The cars explained in the report (all BASE_SHAPE, so each can be replayed in the app):
# (track, seed, car index, the steps to look at, why it was chosen).
CASES = {
    "brakes_at_standstill": (TRACK, 1, 0, (0,), "the most common behaviour: never moves"),
    "stops_itself": (TRACK, 5, 17, (0, 85), "moves, then its own throttle brings it to a stop"),
    "crash": (TRACK, 1, 11, (0, 280, 310), "drives 78 m and leaves the road (task 2.11)"),
    "finisher": (TRACK, 10, 10, None, "the only base-shape brain that finishes the loop"),
    "monza_finisher": ("monza_cw", 51, 7, None, "finishes 2 laps of Monza clockwise"),
}


# ── building and running the scenario ─────────────────────────────────────────────────────


def load_track(name: str = TRACK) -> Track:
    return Track.from_json(ROOT / "tracks" / f"{name}.json")


def scenario(
    track: Track, seed: int, shape: tuple[int, ...] = BASE_SHAPE
) -> tuple[World, Population]:
    """The live session's "Random networks, Random start, seed ``seed``" scenario, at step 0
    (with networks of another ``shape`` when asked; the app itself only has BASE_SHAPE)."""
    spec = NetworkSpec(layer_sizes=shape)
    population = init_population(spec, N_CARS, child_rng(make_rng(seed), "init"))
    world = World(track, WorldConfig.for_track(track))
    world.reset(
        n=N_CARS,
        rng=make_rng(seed),
        random_s0=True,
        lateral_jitter=RANDOM_START_LATERAL,
        heading_jitter=RANDOM_START_HEADING,
    )
    return world, population


def drive(
    track: Track, seed: int, shape: tuple[int, ...] = BASE_SHAPE
) -> tuple[EpisodeRecording, Population, float]:
    world, population = scenario(track, seed, shape)
    return run_episode(world, PopulationPolicy(population), seed=seed), population, world.dt


# ── what each car did ──────────────────────────────────────────────────────────────────────


def wiring(network: Network, observation: np.ndarray) -> dict:
    """The signs of steer's sensitivity to the four side rays at ``observation``.

    A wall-avoiding driver has s < 0 for the left rays (a shorter left ray, meaning a nearer left
    wall, raises steer: turn right) and s > 0 for the right rays.
    """
    s = sensitivity(network, observation)[0]
    good = [bool(s[k] < 0) for k in LEFT_RAYS] + [bool(s[k] > 0) for k in RIGHT_RAYS]
    return {
        "steer_sensitivity": [round(float(v), 4) for v in s],
        "right_signs": sum(good),
        "inner_pair_right": good[1] and good[2],  # the +-30 degree rays both wired to avoid walls
    }


def describe_car(rec: EpisodeRecording, population: Population, dt: float, seed: int, i: int):
    """One car's outcome and the numbers used to sort and explain it."""
    status = Status(int(rec.status[i])).name.lower()
    n = int(rec.alive[:-1, i].sum())  # steps at which it was still running (it decided n times)
    action, obs = rec.action[:n, i], rec.observation[:n, i]
    speed = rec.v[: n + 1, i]
    heading_error = np.unwrap(rec.observation[: n + 1, i, 6] * np.pi)
    car: dict = {
        "seed": seed,
        "car": i + 1,  # as the app numbers cars
        "status": status,
        "steps": n,
        "seconds": round(n * dt, 2),
        "distance": round(float(np.sum(speed[:-1]) * dt), 2),
        "progress": round(float(rec.progress[n, i]), 2),
        "top_speed": round(float(speed.max()), 2),
        "final_speed": round(float(speed[-1]), 3),
        "moved": bool(speed.max() > MOVED),
        "first_throttle": round(float(action[0, 1]), 4),
        "first_abs_steer": round(float(abs(action[0, 0])), 4),
        "full_lock_share": round(float(np.mean(np.abs(action[:, 0]) > FULL_LOCK)), 4),
        "spin_deg": round(float(np.degrees(heading_error.max() - heading_error.min())), 1),
    }
    if status == "crashed":
        last = obs[-1]
        wall_left = min(last[k] for k in LEFT_RAYS) < min(last[k] for k in RIGHT_RAYS)
        steer = float(action[-1, 0])
        car["wall"] = "left" if wall_left else "right"
        car["last_steer"] = round(steer, 4)
        car["steering_towards_wall"] = (wall_left and steer < 0) or (not wall_left and steer > 0)
        car["crash_speed"] = round(float(speed[n - 1]), 2)
    if status in ("crashed", "timeout", "finished"):
        car.update(wiring(population.network(i), obs[-1]))  # at its last decision
    if status == "stalled":
        # The stall rule ends a car whose progress over the last 3 s is under 2 m, checked from
        # 2 s on (60 steps). Why did it fall short?
        if not car["moved"]:
            car["stall_kind"] = "never_moved"
        elif speed[-1] <= MOVED:
            car["stall_kind"] = "stopped_itself"  # it moved, then braked to a standstill
            car["throttle_when_stopping"] = round(float(action[-1, 1]), 4)
        elif n == round(2.0 / dt):
            car["stall_kind"] = "too_slow_at_2s"  # still speeding up, but under 2 m in 2 s
        else:
            car["stall_kind"] = "crawling"  # moving, but under 2 m in some 3 s
    return car


def census(track: Track, shape: tuple[int, ...] = BASE_SHAPE) -> list[dict]:
    cars = []
    for seed in SEEDS:
        rec, population, dt = drive(track, seed, shape)
        cars += [describe_car(rec, population, dt, seed, i) for i in range(N_CARS)]
    return cars


def quartiles(values) -> list[float] | None:
    """[median, 25th percentile, 75th percentile] (None for no values)."""
    if len(values) == 0:
        return None
    q1, med, q3 = np.percentile(np.asarray(values, dtype=float), [25, 50, 75])
    return [round(float(med), 2), round(float(q1), 2), round(float(q3), 2)]


def summarise(cars: list[dict]) -> dict:
    status = Counter(c["status"] for c in cars)
    stalled = [c for c in cars if c["status"] == "stalled"]
    never = [c for c in stalled if not c["moved"]]
    stopped = [c for c in stalled if c["stall_kind"] == "stopped_itself"]
    crashed = [c for c in cars if c["status"] == "crashed"]
    movers = [c for c in cars if c["moved"]]
    return {
        "brains": len(cars),
        "status": dict(status),
        "stall_kinds": dict(Counter(c["stall_kind"] for c in stalled)),
        "never_moved_braking_at_step_0": sum(c["first_throttle"] <= 0 for c in never),
        "stopped_itself_braking": sum(c["throttle_when_stopping"] < 0 for c in stopped),
        "stopped_itself_top_speed_quartiles": quartiles([c["top_speed"] for c in stopped]),
        "first_throttle_positive": sum(c["first_throttle"] > 0 for c in cars),
        "crash_steering_towards_wall": sum(c["steering_towards_wall"] for c in crashed),
        "crash_wall_side": dict(Counter(c["wall"] for c in crashed)),
        "crash_right_signs_histogram": [
            sum(c["right_signs"] == k for c in crashed) for k in range(5)
        ],
        "crash_inner_pair_right": sum(c["inner_pair_right"] for c in crashed),
        "long_drivers_inner_pair_right": sum(
            c["inner_pair_right"] for c in cars if c["status"] in ("timeout", "finished")
        ),
        "long_drivers_right_signs": [
            c["right_signs"] for c in cars if c["status"] in ("timeout", "finished")
        ],
        # what four fair coin flips would give: C(4, k) / 16 of the crashes
        "coin_flip_histogram": [round(len(crashed) * comb / 16, 1) for comb in (1, 4, 6, 4, 1)],
        "crash_speed_quartiles": quartiles([c["crash_speed"] for c in crashed]),
        "crash_seconds_quartiles": quartiles([c["seconds"] for c in crashed]),
        "first_abs_steer_quartiles": quartiles([c["first_abs_steer"] for c in cars]),
        "first_abs_steer_over_full_lock": sum(c["first_abs_steer"] > FULL_LOCK for c in cars),
        "mover_distance_quartiles": quartiles([c["distance"] for c in movers]),
        "mover_top_speed_quartiles": quartiles([c["top_speed"] for c in movers]),
        "movers_full_lock_share_max": max((c["full_lock_share"] for c in movers), default=0.0),
        "movers_full_lock_share_mean": 0.0
        if not movers
        else round(float(np.mean([c["full_lock_share"] for c in movers])), 4),
        "spin_deg_max": max(c["spin_deg"] for c in cars),
        # the brain that spent the largest share of its steps at (nearly) full lock
        "most_full_lock": max(cars, key=lambda c: (c["full_lock_share"], -c["seed"], -c["car"])),
        "circling": sum(c["spin_deg"] > np.degrees(SPIN) for c in cars),
        "finished": [c for c in cars if c["status"] == "finished"],
        "timeout": [c for c in cars if c["status"] == "timeout"],
    }


# ── why no car circles ─────────────────────────────────────────────────────────────────────


def full_lock_circle(params: CarParams | None = None, dt: float = 1 / 30) -> dict:
    """Hold full lock at a slow, steady speed for 30 s with the car physics alone (no walls).

    Returns the circle's measured diameter and the formula's, 2 * wheelbase / tan(delta_max).
    """
    params = CarParams() if params is None else params
    state = CarState.from_pose(0.0, 0.0, 0.0, [3.0])
    xs, ys = [0.0], [0.0]
    throttle = params.drag * 3.0 / params.a_acc  # just enough to hold 3 m/s against the drag
    for _ in range(round(30 / dt)):
        state = step_cars(state, np.array([1.0]), np.array([throttle]), params, dt)
        xs.append(float(state.x[0]))
        ys.append(float(state.y[0]))
    measured = max(max(xs) - min(xs), max(ys) - min(ys))
    return {
        "speed": round(float(state.v[0]), 3),
        "diameter_measured": round(measured, 3),
        "diameter_formula": round(2 * params.wheelbase / np.tan(params.delta_max), 3),
        "road_width": None,  # filled in by main()
    }


# ── the cases the report explains ──────────────────────────────────────────────────────────


def receipt(network: Network, observation: np.ndarray, m: int, top: int = 3) -> dict:
    """Output m's receipt at ``observation``: z, bias, the largest contributions and the rest."""
    trace = forward(network, observation)
    r = neuron_contributions(network, trace, network.spec.n_layers - 1, m)
    order = np.argsort(-np.abs(r.contributions), kind="stable")
    return {
        "value": round(float(r.output), 4),
        "z": round(float(r.z), 4),
        "bias": round(float(r.bias), 4),
        "largest": [
            {"hidden_neuron": int(j) + 1, "contribution": round(float(r.contributions[j]), 4)}
            for j in order[:top]
        ],
        "rest": round(float(np.sum(r.contributions[order[top:]])), 4),
    }


def moment(rec: EpisodeRecording, population: Population, i: int, t: int) -> dict:
    """What the views show for car i at step t: inputs, both receipts, the sensitivity table."""
    network = population.network(i)
    obs = rec.observation[t, i]
    return {
        "step": t,
        "time": round(t / 30, 2),
        "speed": round(float(rec.v[t, i]), 3),
        "observation": dict(zip(OBSERVATION_NAMES, np.round(obs, 4).tolist(), strict=True)),
        "rays_m": np.round(obs[:5] * RAY_RANGE, 2).tolist(),
        "action": np.round(rec.action[t, i], 4).tolist(),
        "steer": receipt(network, obs, 0),
        "throttle": receipt(network, obs, 1),
        "sensitivity": np.round(sensitivity(network, obs), 4).tolist(),
    }


def what_changed(rec: EpisodeRecording, population: Population, i: int, a: int, b: int) -> dict:
    """Why output values moved between steps a and b: for each input k, the straight-line share
    sensitivity_b[m][k] * (o_b[k] - o_a[k]) (the what-if panel's prediction, read backwards)."""
    o_a, o_b = rec.observation[a, i], rec.observation[b, i]
    s = sensitivity(population.network(i), o_b)
    shares = s * (o_b - o_a)  # (2 outputs, 7 inputs)
    return {
        "from_step": a,
        "to_step": b,
        "actual_change": np.round(rec.action[b, i] - rec.action[a, i], 4).tolist(),
        "predicted_change": np.round(shares.sum(axis=1), 4).tolist(),
        "throttle_shares": dict(
            zip(OBSERVATION_NAMES, np.round(shares[1], 4).tolist(), strict=True)
        ),
        "steer_shares": dict(zip(OBSERVATION_NAMES, np.round(shares[0], 4).tolist(), strict=True)),
    }


def case(track: Track, seed: int, i: int, steps) -> dict:
    rec, population, dt = drive(track, seed)
    car = describe_car(rec, population, dt, seed, i)
    n = car["steps"]
    if steps is None:  # the start, the middle and the last decision
        steps = (0, n // 2, n - 1)
    throttle = rec.action[:n, i, 1]
    steer = rec.action[:n, i, 0]
    cruising = rec.v[min(300, n) : n + 1, i]  # after the first 10 s
    settled = slice(min(900, n), n)  # after the first 30 s
    centre = rec.observation[settled, i, 2]
    ahead_short = throttle[settled][centre < 0.5]
    ahead_clear = throttle[settled][centre > 0.99]
    result = {
        **car,
        "steer_range": [round(float(steer.min()), 4), round(float(steer.max()), 4)],
        "throttle_range": [round(float(throttle.min()), 4), round(float(throttle.max()), 4)],
        "speed_after_10s": [round(float(cruising.min()), 2), round(float(cruising.max()), 2)]
        if len(cruising) > 1
        else None,
        "speed_after_30s_percentiles_5_50_95": np.round(
            np.percentile(rec.v[settled, i], [5, 50, 95]), 2
        ).tolist()
        if n > 900
        else None,
        # mean throttle after 30 s when the centre ray is under half its range, and when it is clear
        "throttle_road_ahead_short_vs_clear": [
            round(float(ahead_short.mean()), 3) if len(ahead_short) else None,
            round(float(ahead_clear.mean()), 3) if len(ahead_clear) else None,
        ],
        "moments": [moment(rec, population, i, t) for t in steps],
    }
    if len(steps) > 1:
        result["what_changed"] = what_changed(rec, population, i, steps[-2], steps[-1])
    return result


# ── figures (optional) ─────────────────────────────────────────────────────────────────────


def figures(track: Track, cars: list[dict], cases: dict, grid: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    colours = {
        "stalled": "#8a94a6",
        "crashed": "#c2410c",
        "timeout": "#2563eb",
        "finished": "#15803d",
    }

    # 1. How far each brain got, by outcome.
    fig, ax = plt.subplots(figsize=(8, 4))
    bins = np.concatenate([[0, 1], np.geomspace(2, 2000, 30)])
    for status in ("stalled", "crashed", "timeout", "finished"):
        d = [max(c["distance"], 0.5) for c in cars if c["status"] == status]
        if d:
            ax.hist(d, bins=bins, color=colours[status], alpha=0.8, label=f"{status} ({len(d)})")
    ax.set_xscale("log")
    ax.set_xlabel("distance driven (m, log scale; never moved shown at 0.5 m)")
    ax.set_ylabel("brains")
    ax.set_title("2,000 random brains on train_loop_01: how far each one got")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s2a_distance.png", dpi=110)
    plt.close(fig)

    # 2. One seed's 20 drives on the track (seed 10, with the finisher).
    rec, _, _ = drive(track, 10)
    fig, ax = plt.subplots(figsize=(7, 7))
    for edge in (track.left, track.right):
        loop = np.vstack([edge, edge[:1]])
        ax.plot(loop[:, 0], loop[:, 1], color="#444", lw=1)
    for i in range(N_CARS):
        n = int(rec.alive[:-1, i].sum()) + 1
        status = Status(int(rec.status[i])).name.lower()
        ax.plot(
            rec.x[:n, i], rec.y[:n, i], color=colours[status], lw=1.5 if status != "stalled" else 3
        )
        ax.plot(rec.x[0, i], rec.y[0, i], "o", color=colours[status], ms=3)
    ax.set_aspect("equal")
    ax.set_title("Seed 10: 20 random brains (green = car 11, the finisher)")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s2a_seed10_paths.png", dpi=110)
    plt.close(fig)

    # 3. Steer and throttle over time for the moving cases.
    moving = ("stops_itself", "crash", "finisher", "monza_finisher")
    fig, axes = plt.subplots(len(moving), 1, figsize=(8, 9), sharex=False)
    for ax, name in zip(axes, moving, strict=True):
        where, seed, i, _, _ = CASES[name]
        rec, _, dt = drive(load_track(where), seed)
        n = int(rec.alive[:-1, i].sum())
        t = np.arange(n) * dt
        ax.plot(t, rec.action[:n, i, 0], label="steer", color="#2563eb")
        ax.plot(t, rec.action[:n, i, 1], label="throttle", color="#c2410c")
        ax.plot(t, rec.v[:n, i] / 30, label="speed / v_max", color="#15803d", ls="--")
        ax.axhline(0, color="#999", lw=0.8)
        ax.set_ylim(-1, 1)
        ax.set_title(f"{name.replace('_', ' ')}: {where}, seed {seed}, car {i + 1}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s2a_cases.png", dpi=110)
    plt.close(fig)

    # 4. The grid: outcomes and the size of the first steering command, per shape and track.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))
    labels = ["[" + ",".join(map(str, s)) + "]" for s in SHAPES]
    width = 0.27
    for k, name in enumerate(TRACKS):
        xs = np.arange(len(SHAPES)) + (k - 1) * width
        rows = [grid[config_name(name, s)] for s in SHAPES]
        total = np.array([r["brains"] for r in rows], dtype=float)
        stalled = np.array([r["status"].get("stalled", 0) for r in rows]) / total * 100
        crashed = np.array([r["status"].get("crashed", 0) for r in rows]) / total * 100
        ax1.bar(xs, stalled, width, color=colours["stalled"], edgecolor="white")
        ax1.bar(xs, crashed, width, bottom=stalled, color=colours["crashed"], edgecolor="white")
        steer = [r["first_abs_steer_quartiles"] for r in rows]
        ax2.errorbar(
            xs,
            [q[0] for q in steer],
            yerr=[[q[0] - q[1] for q in steer], [q[2] - q[0] for q in steer]],
            fmt="o",
            capsize=3,
            label=name,
        )
    for ax in (ax1, ax2):
        ax.set_xticks(np.arange(len(SHAPES)), labels, rotation=20, fontsize=8)
    ax1.set_ylabel("% of 2,000 brains (grey stalled, orange crashed)")
    ax1.set_title("Outcomes; bars left to right: loop, Monza cw, Monza ccw", fontsize=10)
    ax1.set_ylim(0, 100)
    ax2.set_ylabel("|steer| at step 0 (median and IQR)")
    ax2.set_title("How hard random brains steer at the start", fontsize=10)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s2a_grid.png", dpi=110)
    plt.close(fig)


def survivors_across_tracks(grid: dict) -> dict:
    """Is surviving a property of the brain or of the track? Car c of seed s has the same network
    on every track (only its start differs), so compare the long drivers (timeout or finished)."""
    result = {}
    for shape in SHAPES:
        sets = {}
        for name in TRACKS:
            s = grid[config_name(name, shape)]
            sets[name] = sorted({(c["seed"], c["car"]) for c in s["finished"] + s["timeout"]})
        loop = set(sets[TRACK])
        monza = set(sets["monza_cw"]) | set(sets["monza_ccw"])
        result[str(list(shape))] = {
            "long_drivers": {name: [list(p) for p in v] for name, v in sets.items()},
            "on_all_tracks": sorted(set.intersection(*(set(v) for v in sets.values()))),
            "on_any_track": len(set.union(*(set(v) for v in sets.values()))),
            "loop_long_drivers_also_long_on_monza": len(loop & monza),
        }
    return result


def config_name(track: str, shape: tuple[int, ...]) -> str:
    return f"{track} [{', '.join(map(str, shape))}]"


def grid_line(track: str, shape: tuple[int, ...], s: dict) -> str:
    """One printed row of the grid table."""
    st = s["status"]
    return (
        f"{track:14s} {str(list(shape)):22s} stalled {st.get('stalled', 0):4d} "
        f"crashed {st.get('crashed', 0):4d} timeout {st.get('timeout', 0):3d} "
        f"finished {st.get('finished', 0):2d} | first throttle > 0: {s['first_throttle_positive']} "
        f"| |steer| at step 0 {s['first_abs_steer_quartiles']} | circling {s['circling']} "
        f"| crash signs {s['crash_right_signs_histogram']} "
        f"| distance {s['mover_distance_quartiles']}"
    )


# ── main ────────────────────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures", action="store_true", help="also draw the figures")
    args = parser.parse_args()

    track = load_track()
    # The grid: every shape on every track (the base shape on train_loop_01 is the main census).
    grid, cars = {}, []
    for name in TRACKS:
        grid_track = load_track(name)
        for shape in SHAPES:
            these = census(grid_track, shape)
            grid[config_name(name, shape)] = summarise(these)
            print(grid_line(name, shape, grid[config_name(name, shape)]), flush=True)
            if name == TRACK and shape == BASE_SHAPE:
                cars = these
    summary = grid[config_name(TRACK, BASE_SHAPE)]
    circle = full_lock_circle()
    circle["road_width"] = track.width
    cases = {
        name: case(load_track(where), seed, i, steps)
        for name, (where, seed, i, steps, _) in CASES.items()
    }
    consistency = survivors_across_tracks(grid)
    results = {
        "track": TRACK,
        "seeds": [SEEDS[0], SEEDS[-1]],
        "cars_per_seed": N_CARS,
        "summary": summary,
        "grid": grid,
        "survivors_across_tracks": consistency,
        "full_lock_circle": circle,
        "cases": cases,
    }
    text = json.dumps(results, indent=1, default=lambda v: v.item())  # NumPy scalars as numbers
    RESULTS_FILE.write_text(text + "\n", encoding="utf-8")
    shown = {k: v for k, v in summary.items() if k not in ("finished", "timeout")}
    print(json.dumps(shown, indent=1, default=lambda v: v.item()))
    print("finished:", [(c["seed"], c["car"], c["seconds"]) for c in summary["finished"]])
    print("timeout:", [(c["seed"], c["car"], c["progress"]) for c in summary["timeout"]])
    print("full lock:", circle)
    for name, c in cases.items():
        print(f"\n== {name}: seed {c['seed']} car {c['car']}: {c['status']} after {c['seconds']} s")
        for m in c["moments"]:
            print(json.dumps(m, default=lambda v: v.item()))
    if args.figures:
        figures(track, cars, cases, grid)
        print(f"figures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
