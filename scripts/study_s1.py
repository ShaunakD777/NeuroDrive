"""Study S1: how far can a hand-wired single neuron go, and why does sigmoid fail to steer?

Run from the repository root:

    uv run python scripts/study_s1.py                          # numbers only
    uv run --with matplotlib python scripts/study_s1.py --figures   # also draw the figures

It runs every episode of the study (all deterministic: the same code gives the same numbers), prints
the tables used in ``docs/studies/S1.md``, and writes them to ``docs/studies/S1_results.json``.
With ``--figures`` it also writes the pictures to ``docs/studies/figures/`` (matplotlib is only
needed for that; it is not a dependency of the project). It also saves one lap of the baseline
controller as ``recordings/s1_hand_neuron_lap.npz`` (git-ignored) and checks that replaying it gives
the identical trajectory.

The "exam-like" starts are the ten starts of the Driving Exam v1 (docs/EXPERIMENT_PROTOCOL.md §3):
5 positions at 0, 20, 40, 60 and 80 % of the track, centred, and the same 5 shifted 1.5 m left. The
real exam arrives in Phase 5; here the same starts are used with the same limits (90 s, 2 laps).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from neurodrive.core.rng import make_rng
from neurodrive.env.episode import EpisodeRecording, replay, run_episode
from neurodrive.env.policies import NeuronPolicy
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig

ROOT = Path(__file__).resolve().parent.parent
TRACK_FILE = ROOT / "tracks" / "train_loop_01.json"
BASELINE_FILE = ROOT / "configs" / "baselines" / "hand_neuron.yaml"
RESULTS_FILE = ROOT / "docs" / "studies" / "S1_results.json"
FIGURES_DIR = ROOT / "docs" / "studies" / "figures"
RECORDING_FILE = ROOT / "recordings" / "s1_hand_neuron_lap.npz"

START_FRACTIONS = (0.0, 0.2, 0.4, 0.6, 0.8)
SEEDS = tuple(range(10))
WEIGHTS = (1, 2, 4, 8, 12, 16, 24)  # |w|: the neuron is (-w, +w), so opposite signs
THROTTLES = tuple(round(0.15 + 0.01 * i, 2) for i in range(0, 16))  # 0.15 .. 0.30
FINE_THROTTLES = tuple(round(0.200 + 0.005 * i, 3) for i in range(0, 17))  # 0.200 .. 0.280
BASELINE_THROTTLE = 0.24
STATUS_NAMES = {s.value: s.name.lower() for s in Status}


# ── helpers ─────────────────────────────────────────────────────────────────────────────────


def exam_starts(length: float) -> tuple[list[float], list[float]]:
    """The ten exam-like starts: arc lengths and sideways offsets."""
    s0 = [f * length for _ in (0.0, 1.5) for f in START_FRACTIONS]
    lateral = [lat for lat in (0.0, 1.5) for _ in START_FRACTIONS]
    return s0, lateral


def quartiles(values: np.ndarray) -> tuple[float, float, float]:
    """Median and the 25th and 75th percentiles (the IQR is the gap between the last two)."""
    q1, med, q3 = np.percentile(values, [25, 50, 75])
    return float(med), float(q1), float(q3)


def summarise(track: Track, world: World, rec: EpisodeRecording) -> dict:
    """Outcome numbers of one episode (any number of cars) as plain Python values."""
    status = rec.status
    frac = np.minimum(rec.progress[-1] / (world.config.target_laps * track.length), 1.0)
    median, q1, q3 = quartiles(frac)
    times = []
    for i in np.flatnonzero(status == Status.FINISHED):
        last_alive = int(np.flatnonzero(rec.alive[:, i])[-1])  # the car stops after this frame
        times.append((last_alive + 1) * world.dt)
    return {
        "n": int(len(status)),
        "finished": int((status == Status.FINISHED).sum()),
        "crashed": int((status == Status.CRASHED).sum()),
        "stalled": int((status == Status.STALLED).sum()),
        "timeout": int((status == Status.TIMEOUT).sum()),
        "progress_frac_median": round(median, 4),
        "progress_frac_q1": round(q1, 4),
        "progress_frac_q3": round(q3, 4),
        "median_finish_time_s": round(float(np.median(times)), 2) if times else None,
        "median_crash_time_s": _median_time(rec, world, status == Status.CRASHED),
        "min_steer": round(float(rec.action[:, :, 0].min()), 4) if rec.n_steps else 0.0,
    }


def _median_time(rec: EpisodeRecording, world: World, mask: np.ndarray) -> float | None:
    times = []
    for i in np.flatnonzero(mask):
        last_alive = int(np.flatnonzero(rec.alive[:, i])[-1])
        times.append((last_alive + 1) * world.dt)
    return round(float(np.median(times)), 2) if times else None


def exam_run(track: Track, policy: NeuronPolicy) -> tuple[dict, EpisodeRecording, World]:
    """Drive the ten exam-like starts at once (a batch of 10 cars) with one neuron."""
    world = World(track, WorldConfig())
    s0, lateral = exam_starts(track.length)
    world.reset(s0=s0, lateral=lateral)
    rec = run_episode(world, policy)
    return summarise(track, world, rec), rec, world


def single_run(track: Track, policy: NeuronPolicy) -> tuple[dict, EpisodeRecording, World]:
    """One car from the track's own start line."""
    world = World(track, WorldConfig())
    world.reset()
    rec = run_episode(world, policy)
    return summarise(track, world, rec), rec, world


def neuron(
    w: float, bias: float = 0.0, activation: str = "tanh", throttle: float = BASELINE_THROTTLE
):
    """The hand-wired neuron with opposite-sign weights (-w on the left ray, +w on the right)."""
    return NeuronPolicy((-w, w), bias, activation, throttle)


def load_baseline() -> NeuronPolicy:
    """The committed baseline (configs/baselines/hand_neuron.yaml) as a NeuronPolicy."""
    spec = yaml.safe_load(BASELINE_FILE.read_text(encoding="utf-8"))["neuron"]
    return NeuronPolicy(
        (spec["w_left"], spec["w_right"]), spec["bias"], spec["activation"], spec["throttle"]
    )


# ── the experiments ─────────────────────────────────────────────────────────────────────────


def experiment_grid(track: Track) -> list[dict]:
    """E1: opposite-sign weights, tanh, bias 0: |w| against throttle, ten exam-like starts."""
    rows = []
    for throttle in THROTTLES:
        for w in WEIGHTS:
            result, _, _ = exam_run(track, neuron(w, throttle=throttle))
            rows.append({"throttle": throttle, "w": w, **result})
    return rows


def experiment_throttle_window(track: Track) -> list[dict]:
    """E2: a finer throttle sweep for the baseline weights (|w| = 12)."""
    rows = []
    for throttle in FINE_THROTTLES:
        result, _, _ = exam_run(track, neuron(12, throttle=throttle))
        rows.append({"throttle": throttle, **result})
    return rows


def experiment_controls(track: Track) -> dict:
    """E3: negative controls at the baseline throttle: zero weights, flipped signs, sigmoid."""
    out = {}
    cases = {
        "baseline_tanh_-12_+12": neuron(12),
        "zero_weights": NeuronPolicy((0.0, 0.0), 0.0, "tanh", BASELINE_THROTTLE),
        "flipped_signs_+12_-12": NeuronPolicy((12.0, -12.0), 0.0, "tanh", BASELINE_THROTTLE),
        "only_left_weight": NeuronPolicy((-12.0, 0.0), 0.0, "tanh", BASELINE_THROTTLE),
        "only_right_weight": NeuronPolicy((0.0, 12.0), 0.0, "tanh", BASELINE_THROTTLE),
        "sigmoid_-12_+12_bias0": neuron(12, 0.0, "sigmoid"),
        "sigmoid_-12_+12_bias-6": neuron(12, -6.0, "sigmoid"),
    }
    for name, policy in cases.items():
        exam, _, _ = exam_run(track, policy)
        start, _, _ = single_run(track, policy)
        out[name] = {"exam_like": exam, "from_start_line": start}
    return out


def experiment_sigmoid(track: Track) -> dict:
    """E4: can any sigmoid weights or bias drive? A grid, and the steering ever commanded."""
    rows = []
    best = None
    left_steer_min = 0.0
    for bias in (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0):
        for w in (1, 4, 8, 16, 24, 30):
            for throttle in (0.15, 0.20, BASELINE_THROTTLE):
                result, rec, _ = single_run(track, neuron(w, bias, "sigmoid", throttle))
                left_steer_min = min(left_steer_min, result["min_steer"])
                row = {"bias": bias, "w": w, "throttle": throttle, **result}
                rows.append(row)
                progress = float(rec.progress[-1, 0])
                if best is None or progress > best["progress_m"]:
                    best = {
                        "progress_m": round(progress, 1),
                        "bias": bias,
                        "w": w,
                        "throttle": throttle,
                    }
    return {
        "runs": len(rows),
        "any_finished": any(r["finished"] for r in rows),
        "best_from_start_line": best,
        "most_negative_steer_ever": left_steer_min,
        "median_progress_m_ratio_note": "see rows",
        "rows": rows,
    }


def experiment_bias(track: Track) -> list[dict]:
    """E5: how sensitive is the baseline to its bias?"""
    rows = []
    for bias in (-0.4, -0.2, -0.1, 0.0, 0.1, 0.2, 0.4):
        result, _, _ = exam_run(track, neuron(12, bias))
        rows.append({"bias": bias, **result})
    return rows


def experiment_seeds(track: Track) -> dict:
    """E6: the baseline from ten seeded random starts (place, sideways offset, heading)."""
    fractions, outcomes, laps_time = [], [], []
    for seed in SEEDS:
        world = World(track, WorldConfig())
        world.reset(rng=make_rng(seed), random_s0=True, lateral_jitter=1.5, heading_jitter=0.1)
        rec = run_episode(world, load_baseline(), seed=seed)
        summary = summarise(track, world, rec)
        fractions.append(summary["progress_frac_median"])
        outcomes.append(STATUS_NAMES[int(rec.status[0])])
        if summary["median_finish_time_s"] is not None:
            laps_time.append(summary["median_finish_time_s"])
    median, q1, q3 = quartiles(np.array(fractions))
    return {
        "seeds": list(SEEDS),
        "outcomes": outcomes,
        "finished": outcomes.count("finished"),
        "progress_frac_median": round(median, 4),
        "progress_frac_q1": round(q1, 4),
        "progress_frac_q3": round(q3, 4),
        "median_finish_time_s": round(float(np.median(laps_time)), 2) if laps_time else None,
    }


def experiment_failures(track: Track) -> dict:
    """E7: look inside three failures: what the neuron saw and said at the last step."""
    out = {}
    cases = {
        "flipped_signs": NeuronPolicy((12.0, -12.0), 0.0, "tanh", BASELINE_THROTTLE),
        "sigmoid": neuron(12, 0.0, "sigmoid"),
        "zero_weights": NeuronPolicy((0.0, 0.0), 0.0, "tanh", BASELINE_THROTTLE),
    }
    for name, policy in cases.items():
        _, rec, world = single_run(track, policy)
        last = rec.n_steps  # the final frame
        readout = policy.readout(rec.observation[last, 0])
        out[name] = {
            "run": "from the start line, default world rules",
            "track": track.name,
            "step": int(last),
            "time_s": round(last * world.dt, 2),
            "progress_m": round(float(rec.progress[-1, 0]), 1),
            "status": STATUS_NAMES[int(rec.status[0])],
            "x_y": [round(float(rec.x[last, 0]), 1), round(float(rec.y[last, 0]), 1)],
            "speed_m_s": round(float(rec.v[last, 0]), 1),
            "inputs": [round(v, 3) for v in readout.inputs],
            "z": round(readout.z, 3),
            "steer": round(readout.output, 3),
            "steer_over_run_min_max": [
                round(float(rec.action[:, 0, 0].min()), 3),
                round(float(rec.action[:, 0, 0].max()), 3),
            ],
        }
    return out


def record_baseline_lap(track: Track) -> dict:
    """Save one baseline episode from the start line, reload it, and replay it exactly."""
    result, rec, world = single_run(track, load_baseline())
    RECORDING_FILE.parent.mkdir(exist_ok=True)
    path = rec.save(RECORDING_FILE)
    reloaded = EpisodeRecording.load(path)
    return {
        **result,
        "file": "recordings/s1_hand_neuron_lap.npz (git-ignored)",
        "steps": rec.n_steps,
        "saved_identical_after_reload": reloaded.identical_to(rec),
        "replay_identical": replay(reloaded, track).identical_to(rec),
        "laps": int(world.laps[0]),
    }


# ── figures (need matplotlib) ───────────────────────────────────────────────────────────────


def make_figures(track: Track, results: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # Figure 1: finished episodes (of ten) for each throttle and weight size.
    grid = results["grid"]
    throttles = sorted({r["throttle"] for r in grid})
    weights = sorted({r["w"] for r in grid})
    finished = np.zeros((len(weights), len(throttles)))
    for r in grid:
        finished[weights.index(r["w"]), throttles.index(r["throttle"])] = r["finished"]
    fig, ax = plt.subplots(figsize=(8, 4))
    image = ax.imshow(finished, origin="lower", cmap="viridis", vmin=0, vmax=10, aspect="auto")
    ax.set_xticks(range(len(throttles)), [f"{t:.2f}" for t in throttles], rotation=60)
    ax.set_yticks(range(len(weights)), [str(w) for w in weights])
    ax.set_xlabel("fixed throttle")
    ax.set_ylabel("weight size |w| (left = −|w|, right = +|w|)")
    ax.set_title("S1: of 10 exam-like starts, how many finish 2 laps in 90 s (tanh, bias 0)")
    fig.colorbar(image, ax=ax, label="finished (of 10)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s1_grid.png", dpi=110)
    plt.close(fig)

    # Figure 2: progress against throttle for the baseline weights.
    window = results["throttle_window"]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot([r["throttle"] for r in window], [r["progress_frac_median"] for r in window], "o-")
    ax2 = ax.twinx()
    ax2.bar(
        [r["throttle"] for r in window],
        [r["crashed"] for r in window],
        width=0.003,
        color="tab:red",
        alpha=0.4,
    )
    ax2.set_ylabel("crashed (of 10)", color="tab:red")
    ax.set_xlabel("fixed throttle")
    ax.set_ylabel("median progress (fraction of 2 laps)")
    ax.set_title("S1: the usable throttle window is narrow (|w| = 12, tanh)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s1_throttle_window.png", dpi=110)
    plt.close(fig)

    # Figure 3: what each activation can say. Steering command against the weighted sum z.
    z = np.linspace(-6, 6, 241)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.axhspan(-1.1, 0, color="tab:orange", alpha=0.15)
    ax.text(-5.8, -0.55, "steer LEFT\n(never reached by sigmoid)", va="center", fontsize=9)
    ax.plot(z, np.tanh(z), label="tanh: −1 … +1")
    ax.plot(z, 1 / (1 + np.exp(-z)), label="sigmoid: 0 … 1")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlabel("weighted sum z")
    ax.set_ylabel("steering command (positive = right)")
    ax.set_title("S1: sigmoid can only steer straight or right")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s1_activation_range.png", dpi=110)
    plt.close(fig)

    # Figure 4: where each controller drives on the track.
    cases = {
        "baseline tanh (−12, +12): finishes": (load_baseline(), "tab:green"),
        "flipped signs (+12, −12): crashes": (
            NeuronPolicy((12.0, -12.0), 0.0, "tanh", BASELINE_THROTTLE),
            "tab:red",
        ),
        "sigmoid (−12, +12): crashes": (neuron(12, 0.0, "sigmoid"), "tab:purple"),
        "zero weights: crashes": (
            NeuronPolicy((0.0, 0.0), 0.0, "tanh", BASELINE_THROTTLE),
            "tab:orange",
        ),
    }
    fig, ax = plt.subplots(figsize=(9, 6.2))
    closed = lambda a: np.vstack([a, a[:1]])  # noqa: E731
    ax.fill(
        np.r_[closed(track.left)[:, 0], closed(track.right)[::-1, 0]],
        np.r_[closed(track.left)[:, 1], closed(track.right)[::-1, 1]],
        color="0.88",
    )
    ax.plot(*closed(track.left).T, "k-", lw=0.8)
    ax.plot(*closed(track.right).T, "k-", lw=0.8)
    for label, (policy, colour) in cases.items():
        _, rec, _ = single_run(track, policy)
        ax.plot(rec.x[:, 0], rec.y[:, 0], color=colour, lw=1.6, label=label)
        ax.plot(rec.x[-1, 0], rec.y[-1, 0], "x", color=colour, ms=9, mew=2)
    ax.set_aspect("equal")
    ax.legend(loc="upper center", fontsize=8)
    ax.set_title("S1: paths from the start line (× marks where an episode ended)")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s1_paths.png", dpi=110)
    plt.close(fig)


# ── printing ────────────────────────────────────────────────────────────────────────────────


def print_tables(results: dict) -> None:
    print("\nE1 grid: finished/crashed of 10 exam-like starts (tanh, bias 0)")
    weights = sorted({r["w"] for r in results["grid"]})
    print("thr   " + " ".join(f"|w|={w:<3d}" for w in weights))
    for throttle in sorted({r["throttle"] for r in results["grid"]}):
        cells = []
        for w in weights:
            r = next(x for x in results["grid"] if x["throttle"] == throttle and x["w"] == w)
            cells.append(f"{r['finished']:2d}/{r['crashed']:<2d}  ")
        print(f"{throttle:.2f}  " + "  ".join(cells))
    print("\nE2 throttle window (|w| = 12): finished / crashed / timeout, median progress fraction")
    for r in results["throttle_window"]:
        print(
            f"  {r['throttle']:.3f}: {r['finished']:2d}/{r['crashed']:2d}/{r['timeout']:2d}"
            f"  progress {r['progress_frac_median']:.3f}  finish time {r['median_finish_time_s']}"
        )
    print("\nE3 controls (throttle 0.24): exam-like finished/crashed, start-line progress")
    for name, r in results["controls"].items():
        e, s = r["exam_like"], r["from_start_line"]
        print(
            f"  {name:26s} {e['finished']:2d}/{e['crashed']:2d}"
            f"  frac {e['progress_frac_median']:.3f}"
            f"  | start line: frac {s['progress_frac_median']:.3f}"
            f" crash t={s['median_crash_time_s']}"
            f" min steer {s['min_steer']}"
        )
    sig = results["sigmoid"]
    print(
        f"\nE4 sigmoid grid: {sig['runs']} runs, any finished: {sig['any_finished']}, "
        f"best {sig['best_from_start_line']}, "
        f"most negative steer ever: {sig['most_negative_steer_ever']}"
    )
    print("\nE5 bias sensitivity (|w| = 12, throttle 0.24)")
    for r in results["bias"]:
        print(
            f"  bias {r['bias']:+.1f}: finished {r['finished']:2d}  crashed {r['crashed']:2d}"
            f"  frac {r['progress_frac_median']:.3f}"
        )
    print("\nE6 seeds:", json.dumps(results["seeds"]))
    print("\nE7 failures:")
    for name, r in results["failures"].items():
        print(f"  {name}: {json.dumps(r)}")
    print("\nrecording:", json.dumps(results["recording"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--figures", action="store_true", help="also write the figures (needs matplotlib)"
    )
    args = parser.parse_args()

    track = Track.from_json(TRACK_FILE)
    baseline = load_baseline()
    print(
        f"track {track.name}: {track.length:.1f} m; baseline {baseline.neuron.weights.tolist()} "
        f"bias {baseline.neuron.bias} {baseline.neuron.activation} throttle {baseline.throttle}"
    )
    results = {
        "track": track.name,
        "track_length_m": round(track.length, 2),
        "world_rules": {"time_limit_s": 90, "target_laps": 2},
        "grid": experiment_grid(track),
        "throttle_window": experiment_throttle_window(track),
        "controls": experiment_controls(track),
        "sigmoid": experiment_sigmoid(track),
        "bias": experiment_bias(track),
        "seeds": experiment_seeds(track),
        "failures": experiment_failures(track),
        "recording": record_baseline_lap(track),
    }
    results["sigmoid"].pop("rows_note", None)
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print_tables(results)
    if args.figures:
        make_figures(track, results)
        print(f"\nfigures written to {FIGURES_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
