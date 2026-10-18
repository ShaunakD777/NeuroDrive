"""Study S1b: the hand-wired neuron of S1 on half-size Monza, clockwise and counter-clockwise.

Run from the repository root:

    uv run python scripts/study_s1b.py                              # numbers only
    uv run --with matplotlib python scripts/study_s1b.py --figures  # also draw the figures

Every episode is deterministic, so the same code gives the same numbers. The tables used in
``docs/studies/S1b.md`` are printed and written to ``docs/studies/S1b_results.json``; with
``--figures`` the pictures go to ``docs/studies/figures/`` (matplotlib is needed only for that).

The rules are the Driving Exam's: 2 laps, the stall rule, and the per-track time limit of ADR-022,
ceil(2 laps x L / 16 m/s), which is 363 s on Monza. The ten exam-like starts are those of S1
(0, 20, 40, 60 and 80 % of the lap, centred and 1.5 m to the left).
"""

import argparse
import json
from pathlib import Path

import numpy as np
from make_tracks import MONZA_PIXELS, monza_centerline
from study_s1 import STATUS_NAMES, exam_starts, load_baseline, neuron, quartiles, summarise

from neurodrive.core.rng import make_rng
from neurodrive.env.episode import EpisodeRecording, run_episode
from neurodrive.env.policies import NeuronPolicy
from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig

ROOT = Path(__file__).resolve().parent.parent
TRACKS = ("monza_cw", "monza_ccw")
RESULTS_FILE = ROOT / "docs" / "studies" / "S1b_results.json"
FIGURES_DIR = ROOT / "docs" / "studies" / "figures"
S1_TRACK = "train_loop_01"

WEIGHTS = (4, 8, 12, 24)  # |w|: the neuron is (-w, +w)
THROTTLES = tuple(round(0.13 + 0.01 * i, 2) for i in range(10))  # 0.13 .. 0.22
FINE_THROTTLES = tuple(round(0.180 + 0.0025 * i, 4) for i in range(9))  # 0.180 .. 0.200
SAFE_THROTTLE = 0.17  # slow enough to survive both directions in the exploratory sweep
SEEDS = tuple(range(10))
LONG_LIMIT = 900.0  # seconds: "what if there were no time pressure?"
GRIP = 9.0  # a_lat_max, m/s^2 (car physics v1)

# The circuit in named sections, each from one traced corner point to another (indices into
# MONZA_PIXELS, in the racing direction; the names follow the comments in make_tracks.py).
# Together they cover the whole lap.
SECTIONS = {
    "main straight": (75, 5),
    "T1-T2 chicane": (5, 14),
    "Curva Grande": (14, 25),
    "Roggia chicane": (25, 33),
    "Lesmo 1": (33, 39),
    "Lesmo 2": (39, 45),
    "the diagonal": (45, 51),
    "Ascari chicane": (51, 60),
    "back straight": (60, 65),
    "Parabolica": (65, 75),
}
assert max(max(pair) for pair in SECTIONS.values()) < len(MONZA_PIXELS)


# ── helpers ─────────────────────────────────────────────────────────────────────────────────


def load(name: str) -> Track:
    return Track.from_json(ROOT / "tracks" / f"{name}.json")


def rules(track: Track, time_limit: float | None = None) -> WorldConfig:
    """The exam rules for ``track`` (per-track limit), or the same with another time limit."""
    config = WorldConfig.for_track(track)
    return config if time_limit is None else WorldConfig(time_limit, config.target_laps)


def exam_run(track: Track, policy: NeuronPolicy, time_limit: float | None = None):
    """The ten exam-like starts at once (a batch of 10 cars)."""
    world = World(track, rules(track, time_limit))
    s0, lateral = exam_starts(track.length)
    world.reset(s0=s0, lateral=lateral)
    rec = run_episode(world, policy)
    return summarise(track, world, rec), rec, world


def single_run(track: Track, policy: NeuronPolicy, time_limit: float | None = None):
    """One car from the start line."""
    world = World(track, rules(track, time_limit))
    world.reset()
    rec = run_episode(world, policy)
    return summarise(track, world, rec), rec, world


def sections(track: Track) -> dict[str, tuple[float, float]]:
    """Each named section as (where it starts, its length), in metres along this track.

    Both end points are projected onto the track; the section is the shorter way round between
    them, which works in either driving direction because every section is under half a lap.
    """
    points = np.array(monza_centerline(+1))
    out = {}
    for name, (first, last) in SECTIONS.items():
        a, b = track.project(points[[first, last]]).s
        d = (b - a) % track.length
        start, length = (a, d) if d <= track.length / 2 else (b, track.length - d)
        out[name] = (round(float(start), 1), round(float(length), 1))
    return out


def corner_positions(track: Track) -> dict[str, float]:
    """The middle of each named section, in metres from the start line along this track."""
    return {
        name: round((start + length / 2) % track.length, 1)
        for name, (start, length) in sections(track).items()
    }


def bend_radius(track: Track, s: float, before: float = 40.0, after: float = 10.0) -> float:
    """Tightest centreline radius (metres) from ``before`` m behind ``s`` to ``after`` m past it.

    The radius at a point is 1 / |curvature|, and curvature is the change of road heading per
    metre, measured over 5 m either side so the 1 m sampling does not add noise.
    """
    ds = 5.0
    probe = (s + np.arange(-before, after + 0.5, 1.0)) % track.length
    _, _, h_ahead = track.point_at((probe + ds) % track.length)
    _, _, h_behind = track.point_at((probe - ds) % track.length)
    turn = np.abs(np.angle(np.exp(1j * (h_ahead - h_behind))))
    return round(float(2.0 * ds / max(turn.max(), 1e-9)), 1)


def place_of(track: Track, s: float) -> str:
    """The named section that contains arc length ``s``."""
    for name, (start, length) in sections(track).items():
        if (s - start) % track.length <= length:
            return name
    return "?"


def crash_report(track: Track, rec: EpisodeRecording, world: World, car: int = 0) -> dict:
    """Where and how fast one car ended, and what the grip limit allowed in that bend."""
    alive = np.flatnonzero(rec.alive[:, car])
    last = int(alive[-1]) + 1 if not rec.alive[-1, car] else rec.n_steps  # first frozen frame
    s = float(track.project(np.array([[rec.x[last, car], rec.y[last, car]]])).s[0])
    radius = bend_radius(track, s)
    return {
        "status": STATUS_NAMES[int(rec.status[car])],
        "time_s": round(last * world.dt, 2),
        "progress_m": round(float(rec.progress[-1, car]), 1),
        "lap_position_m": round(s, 1),
        "place": place_of(track, s),
        "speed_m_s": round(float(rec.v[last, car]), 1),
        "tightest_radius_m": radius,
        "grip_speed_m_s": round(float(np.sqrt(GRIP * radius)), 1),
    }


# ── the experiments ─────────────────────────────────────────────────────────────────────────


def experiment_grid(track: Track) -> list[dict]:
    """E1: opposite-sign weights, tanh, bias 0: |w| against throttle, ten exam-like starts."""
    rows = []
    for throttle in THROTTLES:
        for w in WEIGHTS:
            result, _, _ = exam_run(track, neuron(w, throttle=throttle))
            rows.append({"throttle": throttle, "w": w, **result})
    return rows


def experiment_fine(track: Track) -> list[dict]:
    """E2: a fine throttle sweep with |w| = 12 round the edge between surviving and crashing."""
    rows = []
    for throttle in FINE_THROTTLES:
        result, _, _ = exam_run(track, neuron(12, throttle=throttle))
        rows.append({"throttle": throttle, **result})
    return rows


def fastest_safe(fine: list[dict]) -> float:
    """The largest throttle of the fine sweep at which none of the ten cars crashed."""
    safe = [r["throttle"] for r in fine if r["crashed"] == 0 and r["stalled"] == 0]
    return max(safe)


def experiment_no_time_pressure(track: Track, throttle: float) -> dict:
    """E3: the fastest safe setting with a 900 s limit: how long would it need for 2 laps?"""
    result, rec, world = single_run(track, neuron(12, throttle=throttle), LONG_LIMIT)
    return {"throttle": throttle, "time_limit_s": LONG_LIMIT, **result}


def experiment_baseline(track: Track) -> dict:
    """E4: the S1 baseline (throttle 0.24) and the first throttle that crashes: where, and why."""
    out = {}
    baseline = load_baseline()
    exam, rec, world = exam_run(track, baseline)
    out["baseline_exam_like"] = exam
    out["baseline_exam_like_cars"] = [crash_report(track, rec, world, i) for i in range(rec.n_cars)]
    _, rec, world = single_run(track, baseline)
    out["baseline_from_start_line"] = crash_report(track, rec, world)
    return out


def experiment_first_crash(track: Track, throttle: float) -> dict:
    """E4b: from the start line at the first throttle above the fastest safe one."""
    _, rec, world = single_run(track, neuron(12, throttle=throttle))
    return {"throttle": throttle, **crash_report(track, rec, world)}


def experiment_controls(track: Track) -> dict:
    """E5: negative controls at a safe throttle (0.17): zero weights, flipped signs, sigmoid."""
    cases = {
        "tanh_-12_+12": neuron(12, throttle=SAFE_THROTTLE),
        "zero_weights": NeuronPolicy((0.0, 0.0), 0.0, "tanh", SAFE_THROTTLE),
        "flipped_signs_+12_-12": NeuronPolicy((12.0, -12.0), 0.0, "tanh", SAFE_THROTTLE),
        "sigmoid_-12_+12_bias0": neuron(12, 0.0, "sigmoid", SAFE_THROTTLE),
        "sigmoid_-12_+12_bias-6": neuron(12, -6.0, "sigmoid", SAFE_THROTTLE),
    }
    out = {}
    for name, policy in cases.items():
        exam, _, _ = exam_run(track, policy)
        _, rec, world = single_run(track, policy)
        out[name] = {"exam_like": exam, "from_start_line": crash_report(track, rec, world)}
    return out


def experiment_sigmoid(track: Track) -> dict:
    """E6: a sigmoid grid from the start line: how far does it get, and does it ever steer left?"""
    rows = []
    for bias in (-6.0, -4.0, -2.0, 0.0):
        for w in (4, 12, 24):
            for throttle in (0.14, SAFE_THROTTLE):
                result, rec, _ = single_run(track, neuron(w, bias, "sigmoid", throttle))
                rows.append(
                    {
                        "bias": bias,
                        "w": w,
                        "throttle": throttle,
                        "progress_m": round(float(rec.progress[-1, 0]), 1),
                        "status": result,
                    }
                )
    progress = np.array([r["progress_m"] for r in rows])
    best = rows[int(progress.argmax())]
    return {
        "runs": len(rows),
        "any_finished": any(r["status"]["finished"] for r in rows),
        "progress_m_median": round(float(np.median(progress)), 1),
        "progress_m_max": best["progress_m"],
        "best": {k: best[k] for k in ("bias", "w", "throttle")},
        "most_negative_steer_ever": min(r["status"]["min_steer"] for r in rows),
        "rows": rows,
    }


def experiment_seeds(track: Track, throttle: float) -> dict:
    """E7: the fastest safe setting from ten seeded random starts."""
    outcomes, fractions = [], []
    for seed in SEEDS:
        world = World(track, rules(track))
        world.reset(rng=make_rng(seed), random_s0=True, lateral_jitter=1.5, heading_jitter=0.1)
        rec = run_episode(world, neuron(12, throttle=throttle), seed=seed)
        summary = summarise(track, world, rec)
        fractions.append(summary["progress_frac_median"])
        outcomes.append(STATUS_NAMES[int(rec.status[0])])
    median, q1, q3 = quartiles(np.array(fractions))
    return {
        "throttle": throttle,
        "outcomes": outcomes,
        "progress_frac_median": round(median, 4),
        "progress_frac_q1": round(q1, 4),
        "progress_frac_q3": round(q3, 4),
    }


def experiment_s1_recheck() -> dict:
    """E8: S1's throttle window on train_loop_01 again, under the new per-track limit (92 s)."""
    track = load(S1_TRACK)
    rows = []
    for throttle in tuple(round(0.200 + 0.005 * i, 3) for i in range(11)):  # 0.200 .. 0.250
        result, _, _ = exam_run(track, neuron(12, throttle=throttle))
        rows.append({"throttle": throttle, **result})
    return {"track": S1_TRACK, "time_limit_s": rules(track).time_limit, "rows": rows}


# ── figures (need matplotlib) ───────────────────────────────────────────────────────────────


def make_figures(tracks: dict[str, Track], results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: median progress (fraction of 2 laps) by throttle and weight size, both directions.
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, name in zip(axes, TRACKS, strict=True):
        grid = results[name]["grid"]
        values = np.zeros((len(WEIGHTS), len(THROTTLES)))
        for r in grid:
            values[WEIGHTS.index(r["w"]), THROTTLES.index(r["throttle"])] = r[
                "progress_frac_median"
            ]
        image = ax.imshow(values, origin="lower", cmap="viridis", vmin=0, vmax=1, aspect="auto")
        for r in grid:
            mark = "C" if r["crashed"] else ("S" if r["stalled"] else "")
            if mark:
                i, j = WEIGHTS.index(r["w"]), THROTTLES.index(r["throttle"])
                ax.text(j, i, mark, ha="center", va="center", color="white", fontsize=8)
        ax.set_xticks(range(len(THROTTLES)), [f"{t:.2f}" for t in THROTTLES], rotation=60)
        ax.set_yticks(range(len(WEIGHTS)), [str(w) for w in WEIGHTS])
        ax.set_xlabel("fixed throttle")
        ax.set_title(f"{name} (C = crashed, S = stalled)", fontsize=10)
    axes[0].set_ylabel("|w| (left −|w|, right +|w|)")
    fig.colorbar(image, ax=axes, label="fraction of 2 laps in 363 s")
    fig.savefig(FIGURES_DIR / "s1b_grid.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # Figure 2: progress against throttle in the fine sweep, both directions.
    fig, ax = plt.subplots(figsize=(7, 3.6))
    for name, colour in zip(TRACKS, ("tab:blue", "tab:orange"), strict=True):
        fine = results[name]["fine"]
        ax.plot(
            [r["throttle"] for r in fine],
            [r["progress_frac_median"] for r in fine],
            "o-",
            color=colour,
            label=name,
        )
    ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
    ax.text(0.1805, 1.02, "2 laps finished", fontsize=8)
    ax.set_xlabel("fixed throttle (|w| = 12, tanh)")
    ax.set_ylabel("median progress (fraction of 2 laps)")
    ax.set_ylim(0, 1.1)
    ax.set_title("S1b: faster survives longer, until it crashes; neither reaches 1.0")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s1b_throttle_edge.png", dpi=110)
    plt.close(fig)

    # Figure 3: where the cars drive and end, both directions.
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    closed = lambda a: np.vstack([a, a[:1]])  # noqa: E731
    for ax, name in zip(axes, TRACKS, strict=True):
        track = tracks[name]
        ax.fill(
            np.r_[closed(track.left)[:, 0], closed(track.right)[::-1, 0]],
            np.r_[closed(track.left)[:, 1], closed(track.right)[::-1, 1]],
            color="0.88",
        )
        ax.plot(*closed(track.left).T, "k-", lw=0.5)
        ax.plot(*closed(track.right).T, "k-", lw=0.5)
        safe = results[name]["fastest_safe_throttle"]
        cases = {
            "S1 baseline, throttle 0.24": (load_baseline(), "tab:red"),
            f"throttle {safe} (fastest safe)": (neuron(12, throttle=safe), "tab:green"),
            f"sigmoid, throttle {SAFE_THROTTLE}": (
                neuron(12, 0.0, "sigmoid", SAFE_THROTTLE),
                "tab:purple",
            ),
        }
        for label, (policy, colour) in cases.items():
            _, rec, _ = single_run(track, policy)
            ax.plot(rec.x[:, 0], rec.y[:, 0], color=colour, lw=1.2, label=label)
            ax.plot(rec.x[-1, 0], rec.y[-1, 0], "x", color=colour, ms=10, mew=2.5)
        ax.plot(*track.points[0], "k>", ms=9)
        for corner, s in corner_positions(track).items():
            x, y, _ = track.point_at(s)
            ax.annotate(corner, (float(x[0]), float(y[0])), fontsize=7, color="0.25")
        ax.set_aspect("equal")
        ax.set_title(f"{name} (▶ start line, × end of the episode)", fontsize=10)
        ax.legend(loc="lower left", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s1b_paths.png", dpi=110)
    plt.close(fig)


# ── printing ────────────────────────────────────────────────────────────────────────────────


def print_tables(results: dict) -> None:
    for name in TRACKS:
        r = results[name]
        print(f"\n=== {name}: {r['track_length_m']} m, limit {r['time_limit_s']} s ===")
        print("corners:", r["corners"])
        print("E1 grid: finished/crashed/stalled/timeout, median progress")
        for throttle in THROTTLES:
            cells = []
            for w in WEIGHTS:
                g = next(x for x in r["grid"] if x["throttle"] == throttle and x["w"] == w)
                cells.append(
                    f"|w|={w}: {g['finished']}/{g['crashed']}/{g['stalled']}/{g['timeout']}"
                    f" {g['progress_frac_median']:.3f}"
                )
            print(f"  {throttle:.2f}  " + "   ".join(cells))
        print("E2 fine (|w| = 12)")
        for g in r["fine"]:
            print(
                f"  {g['throttle']:.4f}: {g['finished']}/{g['crashed']}/{g['stalled']}/"
                f"{g['timeout']} progress {g['progress_frac_median']:.3f}"
                f" crash time {g['median_crash_time_s']}"
            )
        print("fastest safe throttle:", r["fastest_safe_throttle"])
        print("E3 no time pressure:", json.dumps(r["no_time_pressure"]))
        print("E4 baseline from start line:", json.dumps(r["baseline"]["baseline_from_start_line"]))
        print("E4 baseline exam-like:", json.dumps(r["baseline"]["baseline_exam_like"]))
        for car in r["baseline"]["baseline_exam_like_cars"]:
            print("    ", json.dumps(car))
        print("E4b first crash:", json.dumps(r["first_crash"]))
        print("E5 controls")
        for case, c in r["controls"].items():
            e = c["exam_like"]
            print(
                f"  {case:24s} exam {e['finished']}/{e['crashed']}/{e['stalled']}/{e['timeout']}"
                f" frac {e['progress_frac_median']:.3f}"
                f" | start line {json.dumps(c['from_start_line'])}"
            )
        sig = r["sigmoid"]
        print(
            f"E6 sigmoid: {sig['runs']} runs, finished {sig['any_finished']}, progress median"
            f" {sig['progress_m_median']} m, max {sig['progress_m_max']} m {sig['best']},"
            f" most negative steer {sig['most_negative_steer_ever']}"
        )
        print("E7 seeds:", json.dumps(r["seeds"]))
    s1 = results["s1_recheck"]
    print(f"\nE8 {s1['track']} at {s1['time_limit_s']} s (|w| = 12)")
    for g in s1["rows"]:
        print(
            f"  {g['throttle']:.3f}: {g['finished']}/{g['crashed']}/{g['stalled']}/{g['timeout']}"
            f" progress {g['progress_frac_median']:.3f} finish {g['median_finish_time_s']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--figures", action="store_true", help="also write the figures (needs matplotlib)"
    )
    args = parser.parse_args()

    tracks = {name: load(name) for name in TRACKS}
    results: dict = {"rules": "2 laps, stall rule, time limit ceil(2 L / 16 m/s) (ADR-022)"}
    for name, track in tracks.items():
        print(f"running {name} ...", flush=True)
        fine = experiment_fine(track)
        safe = fastest_safe(fine)
        above = min(r["throttle"] for r in fine if r["throttle"] > safe)
        results[name] = {
            "track_length_m": round(track.length, 2),
            "time_limit_s": rules(track).time_limit,
            "corners": corner_positions(track),
            "grid": experiment_grid(track),
            "fine": fine,
            "fastest_safe_throttle": safe,
            "no_time_pressure": experiment_no_time_pressure(track, safe),
            "baseline": experiment_baseline(track),
            "first_crash": experiment_first_crash(track, above),
            "controls": experiment_controls(track),
            "sigmoid": experiment_sigmoid(track),
            "seeds": experiment_seeds(track, safe),
        }
    results["s1_recheck"] = experiment_s1_recheck()
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print_tables(results)
    if args.figures:
        make_figures(tracks, results)
        print(f"\nfigures written to {FIGURES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
