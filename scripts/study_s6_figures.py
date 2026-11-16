"""The figures of Study S6 (docs/studies/S6.md), drawn from S6_results.json by
``scripts/study_s6.py --figures`` (matplotlib is fetched for that run only:
``uv run --with matplotlib python scripts/study_s6.py --figures``).

Colours: the reference categorical slots in fixed order (blue, orange, aqua), grey for the
expert and the baselines; one y-axis per panel; text in ink colours, never a series colour.
"""

from pathlib import Path
from typing import Any

import numpy as np

from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.policies import MLPPolicy
from neurodrive.env.track import Track
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, exam_track
from neurodrive.experiment.racing_line import bends, drive_offsets
from neurodrive.experiment.store import RunStore

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GREY, INK, INK_2, GRID, SURFACE = "#8a8984", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
ARM_LABELS = {
    "ga": "GA default",
    "random_search": "random search",
    "n20": "N = 20",
    "n100": "N = 100",
    "p005": "p = 0.005",
    "p01": "p = 0.1",
    "s002": "σ = 0.02",
    "s02": "σ = 0.2",
    "finish_bonus": "finish bonus",
    "progress": "progress only",
}


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 9,
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def _per_seed(made: dict[str, Any], arm: str, key: str, seeds=None) -> list[tuple[int, Any]]:
    recs = made["records"][arm]
    return [
        (int(s), r[key])
        for s, r in sorted(recs.items(), key=lambda kv: int(kv[0]))
        if seeds is None or int(s) in seeds
    ]


def ga_vs_random_search(made: dict[str, Any], plt, out: Path) -> None:
    """T and P of the latest examined checkpoint against environment steps, median and IQR."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, key, title in (
        (axes[0], "T", "Training loop: completion rate T"),
        (axes[1], "P", "Held-out tracks: median progress P"),
    ):
        for arm, colour in (("ga", BLUE), ("random_search", ORANGE)):
            pts = made["curves"][arm][key]["points"]
            x = np.array([p["env_steps"] for p in pts]) / 1e6
            med = np.array([np.nan if p["median"] is None else p["median"] for p in pts])
            lo = np.array([np.nan if p["q1"] is None else p["q1"] for p in pts])
            hi = np.array([np.nan if p["q3"] is None else p["q3"] for p in pts])
            ax.fill_between(x, lo, hi, color=colour, alpha=0.18, linewidth=0)
            ax.plot(x, med, color=colour, label=ARM_LABELS[arm])
        ax.set_xlabel("environment steps (millions)")
        ax.set_title(title, loc="left")
        ax.set_ylim(-0.03, 1.03)
    axes[0].legend(loc="lower right")
    axes[1].legend(loc="upper right")
    fig.suptitle(
        "S6 Part A: median and IQR over seeds 0-9 of the latest examined checkpoint",
        color=INK,
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "S6_ga_vs_random_search.png", dpi=130)
    plt.close(fig)


def _dots(ax, made, arms, key, seeds=None) -> None:
    for i, arm in enumerate(arms):
        vals = [v for _, v in _per_seed(made, arm, key, seeds) if v is not None]
        if not vals:
            continue
        jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0.0]
        ax.scatter(
            i + np.asarray(jitter), vals, s=26, color=BLUE, edgecolor=SURFACE, linewidth=1, zorder=3
        )
        ax.hlines(np.median(vals), i - 0.28, i + 0.28, color=INK, linewidth=2, zorder=4)
    ax.set_xticks(range(len(arms)), [ARM_LABELS[a] for a in arms], rotation=25, ha="right")


def arms(made: dict[str, Any], plt, out: Path) -> None:
    """Every arm's chosen checkpoint: L and P (exam; T is 1 for all 60) and, for the
    time-bonus arms, the best fitness and final diversity; one dot per seed, the bar is the
    median."""
    every = ["ga", "random_search", "n20", "n100", "p005", "p01", "s002", "s02"]
    every += ["finish_bonus", "progress"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    _dots(axes[0, 0], made, every, "L")
    axes[0, 0].set_title("Median best lap L on the training loop (s); T = 1 for all", loc="left")
    _dots(axes[0, 1], made, every, "P")
    axes[0, 1].set_title("Held-out median progress P", loc="left")
    sweep = ["ga", "n20", "n100", "p005", "p01", "s002", "s02"]
    _dots(axes[1, 0], made, sweep, "best_fitness", seeds=range(5))
    axes[1, 0].set_title("Best training fitness (time bonus arms, seeds 0-4)", loc="left")
    _dots(axes[1, 1], made, sweep, "final_diversity", seeds=range(5))
    axes[1, 1].set_title("Diversity of the last generation (seeds 0-4)", loc="left")
    fig.suptitle(
        "S6: one dot per seed, the bar is the median; every run 40 M environment steps",
        color=INK,
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out / "S6_arms.png", dpi=130)
    plt.close(fig)


def lap_times(made: dict[str, Any], plt, out: Path) -> None:
    """L per fitness formula, seeds 0-4, a line joining each seed's three values."""
    arms_ = ["ga", "finish_bonus", "progress"]
    fig, ax = plt.subplots(figsize=(6, 4))
    for seed in range(5):
        ys = [made["records"][a][str(seed)]["L"] for a in arms_]
        xs = [i for i, y in enumerate(ys) if y is not None]
        ax.plot(xs, [ys[i] for i in xs], color=GRID, linewidth=1.2, zorder=1)
    for i, (arm, colour) in enumerate(zip(arms_, (BLUE, ORANGE, AQUA), strict=True)):
        ys = [y for _, y in _per_seed(made, arm, "L", range(5)) if y is not None]
        ax.scatter([i] * len(ys), ys, s=34, color=colour, edgecolor=SURFACE, zorder=3)
        ax.annotate(
            f"{len(ys)} of 5 seeds finish",
            (i, max(ys) if ys else 0),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=INK_2,
            fontsize=8,
        )
    ax.set_xticks(range(3), ["time bonus\n(default)", "finish bonus", "progress only"])
    ax.set_ylabel("median best lap on the training loop (s)")
    ax.set_title(
        "S6 Part C: lap time by fitness formula (seeds 0-4)\n"
        "finish bonus and progress only chose the same networks",
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(out / "S6_lap_times.png", dpi=130)
    plt.close(fig)


def _offset_along(track: Track, policy_for, start: int) -> tuple[np.ndarray, np.ndarray]:
    _, rec = exam_track(track, policy_for)
    n = int(rec.alive[:, start].sum()) + 1
    return drive_offsets(
        track, rec.x[:n, start], rec.y[:n, start], rec.progress[:n, start], rec.start_s0[start]
    )


def racing_line(made: dict[str, Any], plt, out: Path, runs_dir: Path) -> None:
    """Left: e_in per bend, the expert against every GA default and finish-bonus seed with
    finished drives. Right: the sideways offset along the second lap from start 1 (the start
    of the lap, centred): the expert and the GA default seeds with the largest and the smallest
    median offset (post hoc: the side of the road they keep)."""
    track = Track.from_json(DEFAULT_TRACKS_DIR / "train_loop_01.json")
    found = bends(track)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1, 1.6]})
    ax = axes[0]
    exp = made["expert"]["racing_line"]["per_bend"]
    ax.scatter(
        range(len(exp)),
        [b["e_in"] for b in exp],
        marker="D",
        s=40,
        color=GREY,
        label="expert",
        zorder=4,
    )
    for arm, colour, dx in (("ga", BLUE, -0.15), ("finish_bonus", ORANGE, 0.15)):
        first = True
        for _, line in _per_seed(made, arm, "racing_line"):
            if line is None:
                continue
            ys = [b["e_in"] for b in line["per_bend"]]
            ax.scatter(
                np.arange(len(ys)) + dx,
                ys,
                s=16,
                color=colour,
                alpha=0.8,
                label=ARM_LABELS[arm] if first else None,
                zorder=3,
            )
            first = False
    ax.axhline(0, color=INK_2, linewidth=1)
    ax.set_xticks(
        range(len(found)),
        [f"{b.index + 1}\n{'L' if b.sign > 0 else 'R'} {b.radius_m:.0f} m" for b in found],
    )
    ax.set_ylabel("inside offset at the apex e_in (m)")
    ax.set_title("Inside offset per bend (positive = cutting)", loc="left")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)

    ax = axes[1]
    ga = [(s, r) for s, r in made["records"]["ga"].items() if r["median_offset_m"] is not None]
    if ga:
        ga.sort(key=lambda kv: (kv[1]["median_offset_m"], int(kv[0])))
        lap = track.length
        traces = [("expert", ScriptedExpert, GREY)]
        for (seed, rec), colour, side in (
            (ga[-1], BLUE, "keeps left"),
            (ga[0], ORANGE, "keeps right"),
        ):
            network = RunStore.open(runs_dir / rec["run_id"]).load_checkpoint(rec["unit"])
            traces.append(
                (
                    f"GA seed {seed}, {side}, L {rec['L']:.1f} s",
                    lambda world, net=network: MLPPolicy(net),
                    colour,
                )
            )
        for label, policy_for, colour in traces:
            d, e = _offset_along(track, policy_for, 0)
            keep = (d >= lap) & (d < 2 * lap)
            ax.plot(d[keep] - lap, e[keep], color=colour, label=label)
        for b in found:
            lo, hi = b.start_s, b.end_s if b.end_s >= b.start_s else b.end_s + lap
            ax.axvspan(lo, min(hi, lap), color=GRID, alpha=0.6, linewidth=0, zorder=0)
            ax.annotate(
                f"{b.index + 1}{'L' if b.sign > 0 else 'R'}",
                (b.apex_s, 1),
                xycoords=("data", "axes fraction"),
                xytext=(0, -10),
                textcoords="offset points",
                ha="center",
                color=INK_2,
                fontsize=8,
            )
        ax.axhline(0, color=INK_2, linewidth=1)
        ax.set_xlim(0, lap)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    ax.set_xlabel("distance along the lap (m); shaded: bends, radius under 50 m")
    ax.set_ylabel("sideways offset e (m, + = left)")
    ax.set_title("Offset along lap 2 from start 1", loc="left")
    fig.tight_layout()
    fig.savefig(out / "S6_racing_line.png", dpi=130)
    plt.close(fig)


def draw(made: dict[str, Any], out: Path, runs_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    out.mkdir(parents=True, exist_ok=True)
    ga_vs_random_search(made, plt, out)
    arms(made, plt, out)
    lap_times(made, plt, out)
    racing_line(made, plt, out, runs_dir)
    print("figures in", out)
