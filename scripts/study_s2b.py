"""Study S2b: how does the activation choice affect gradient-based learning (vanishing gradients)?

Run from the repository root:

    uv run python scripts/study_s2b.py                               # numbers only
    uv run --with matplotlib python scripts/study_s2b.py --figures   # also draw the figures
    uv run python scripts/study_s2b.py --save-cases                  # also write the failure
                                                                     # cases as run folders in runs/

The toy networks of task 3.7 (full-batch gradient descent, MSE, Xavier initialisation) learn sine
and XOR. Only the hidden activation (sigmoid, tanh, ReLU; linear as a control) and the depth (1,
3 or 5 hidden layers) change; each condition runs at four learning rates and ten seeds. A seed
fixes the data and the starting weights, and the same seeds are used for every activation, so
comparisons are paired. Everything is deterministic: the runs are spread over worker processes,
but every number depends only on its own seed, and results are put back in a fixed order.

Measured per run (definitions in docs/studies/S2b.md):
* final loss L_T after the budget (+inf if the run diverged), solved = L_T < 0.01, the first
  epoch with a loss below 0.01;
* rho = |g_1| / |g_last|: the first weight layer's gradient size over the output layer's, at
  epoch 1 (before any step, so it does not depend on the learning rate);
* the share of saturated hidden (neuron, example) pairs for sigmoid/tanh (slope below 10 % of
  the activation's maximum) and of dead hidden ReLU neurons (z <= 0 for every example), at the
  start and at the end.

The numbers used in docs/studies/S2b.md are printed and written to docs/studies/S2b_results.json;
with --figures the pictures go to docs/studies/figures/ (matplotlib is needed only for that).
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from neurodrive.core.activations import get_derivative
from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.network import Network
from neurodrive.core.rng import make_rng
from neurodrive.learn.supervised.train_toy import ToyConfig, ToyLearner

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "docs" / "studies" / "S2b_results.json"
FIGURES_DIR = ROOT / "docs" / "studies" / "figures"

# Fixed per task (written in the Brief before any run). XOR is 8 wide (not the default 4) so that
# deep ReLU networks are not starved of neurons.
TASKS = {
    "sine": {"n_in": 1, "width": 16, "output": "linear", "epochs": 3000,
             "lrs": (0.01, 0.03, 0.1, 0.3), "common_lr": 0.1},
    "xor": {"n_in": 2, "width": 8, "output": "sigmoid", "epochs": 3000,
            "lrs": (0.1, 0.3, 1.0, 3.0), "common_lr": 1.0},
}  # fmt: skip
DEPTHS = (1, 3, 5)
ACTIVATIONS = ("sigmoid", "tanh", "relu", "linear")
SEEDS = tuple(range(10))
THRESHOLD = 0.01
SATURATION = 0.1  # a slope below 10 % of the activation's maximum counts as saturated
MAX_SLOPE = {"sigmoid": 0.25, "tanh": 1.0}
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026
# Loss-curve sample points: epoch 0 (untrained), then about 12 per factor of ten.
CURVE_EPOCHS = tuple(sorted({0, *(int(e) for e in np.round(np.logspace(0, math.log10(3000), 45)))}))


def layer_sizes(task: str, depth: int) -> tuple[int, ...]:
    t = TASKS[task]
    return (t["n_in"], *([t["width"]] * depth), 1)


def make_config(task: str, depth: int, activation: str, lr: float, seed: int) -> ToyConfig:
    t = TASKS[task]
    return ToyConfig(
        task,
        layer_sizes(task, depth),
        hidden_activation=activation,
        output_activation=t["output"],
        learning_rate=lr,
        epochs=t["epochs"],
        seed=seed,
        tag=f"s2b_{activation}_d{depth}_s{seed}",
    )


def hidden_stats(network: Network, x: np.ndarray) -> dict:
    """Saturated share (sigmoid/tanh) and dead share (ReLU) of the hidden layers on inputs x.

    Saturated: the share of (hidden neuron, example) pairs whose slope f'(z) is below SATURATION
    times the activation's maximum slope. Dead: the share of hidden ReLU neurons with z <= 0 for
    every example (they pass back no gradient at all). None when it does not apply.
    """
    spec = network.spec
    trace = forward(network, x)
    hidden = [trace.weighted_sums[layer] for layer in range(spec.n_layers - 1)]
    act = spec.hidden_activation
    saturated = dead = None
    if hidden and act in MAX_SLOPE:
        slopes = np.concatenate([get_derivative(act)(z).ravel() for z in hidden])
        saturated = float(np.mean(slopes < SATURATION * MAX_SLOPE[act]))
    if hidden and act == "relu":
        dead = float(np.mean(np.concatenate([np.all(z <= 0, axis=0) for z in hidden])))
    return {"saturated": saturated, "dead": dead}


def run_one(key: tuple[str, int, str, float, int]) -> dict:
    """Train one condition and seed in memory and measure it (see the module docstring)."""
    task, depth, activation, lr, seed = key
    learner = ToyLearner(make_config(task, depth, activation, lr, seed), make_rng(seed))
    x = learner.dataset.x
    start = hidden_stats(learner.network, x)
    losses = [learner.loss()]  # epoch 0
    rho = grad_norm_start = None
    diverged_at = None
    for epoch in range(1, TASKS[task]["epochs"] + 1):
        result = learner.train_unit()
        if epoch == 1 and result.metrics.get("grad_norm"):
            grad_norm_start = result.metrics["grad_norm"]
        if epoch == 1 and grad_norm_start is None:  # diverged in the very first step
            grad_norm_start = []
        if result.diverged:
            diverged_at = epoch
            break
        losses.append(result.metrics["loss"])
    if grad_norm_start:
        last = grad_norm_start[-1]
        rho = grad_norm_start[0] / last if last > 0 else math.inf
    final = math.inf if diverged_at else losses[-1]
    below = [e for e, value in enumerate(losses) if value < THRESHOLD]
    curve = [losses[e] if e < len(losses) else math.inf for e in CURVE_EPOCHS]
    return {
        "task": task, "depth": depth, "activation": activation, "lr": lr, "seed": seed,
        "final_loss": final,
        "solved": final < THRESHOLD,
        "epochs_to_threshold": below[0] if below else None,
        "diverged_at": diverged_at,
        "rho": rho,
        "grad_norm_start": grad_norm_start,
        "saturated_start": start["saturated"], "dead_start": start["dead"],
        "saturated_end": hidden_stats(learner.network, x)["saturated"],
        "dead_end": hidden_stats(learner.network, x)["dead"],
        "curve": curve,
    }  # fmt: skip


def all_keys() -> list[tuple[str, int, str, float, int]]:
    return [
        (task, depth, act, lr, seed)
        for task in TASKS
        for depth in DEPTHS
        for act in ACTIVATIONS
        for lr in TASKS[task]["lrs"]
        for seed in SEEDS
    ]


# ── statistics ──────────────────────────────────────────────────────────────────────────────


def quartiles(values) -> tuple[float, float, float]:
    """Median, 25th and 75th percentile (inf values allowed: they sort to the top)."""
    v = np.asarray(values, dtype=float)
    q1, med, q3 = (
        np.percentile(v, [25, 50, 75], method="linear")
        if np.all(np.isfinite(v))
        else (_inf_percentile(v, 25), _inf_percentile(v, 50), _inf_percentile(v, 75))
    )
    return float(med), float(q1), float(q3)


def _inf_percentile(v: np.ndarray, q: float) -> float:
    """Percentile by position in the sorted values (so +inf can be a percentile, never nan)."""
    s = np.sort(v)
    pos = (len(s) - 1) * q / 100
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    if lo == hi or s[lo] == s[hi]:
        return float(s[lo])
    if not np.isfinite(s[hi]):
        return math.inf
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def log10_or_inf(x: float | None) -> float:
    if x is None:
        return math.nan
    return math.inf if x == math.inf else (math.log10(x) if x > 0 else -math.inf)


def bootstrap_median_difference(a, b, seed: int = BOOTSTRAP_SEED) -> dict:
    """Median(a) - median(b) and its 95 % bootstrap interval, resampling seeds jointly (paired).

    ``a`` and ``b`` are per-seed values in the same seed order. Returns None bounds when infinite
    values make the difference undefined.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = make_rng(seed)
    idx = rng.integers(0, len(a), size=(BOOTSTRAP_RESAMPLES, len(a)))
    with np.errstate(invalid="ignore"):
        diffs = np.median(a[idx], axis=1) - np.median(b[idx], axis=1)
        point = float(np.median(a) - np.median(b))
    finite = diffs[np.isfinite(diffs)]
    if not np.isfinite(point) or len(finite) < 0.95 * len(diffs):
        return {"difference": point if np.isfinite(point) else None, "ci95": None}
    lo, hi = np.percentile(finite, [2.5, 97.5])
    return {"difference": point, "ci95": [float(lo), float(hi)]}


def sig(x, digits: int = 4):
    """JSON-friendly rounding to significant digits (inf -> "inf", None stays None)."""
    if x is None or isinstance(x, str):
        return x
    if isinstance(x, (list, tuple)):
        return [sig(v, digits) for v in x]
    if isinstance(x, bool):
        return x
    if isinstance(x, (int, np.integer)):
        return int(x)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    if math.isnan(x):
        return None
    return float(f"{x:.{digits}g}")


# ── analysis ────────────────────────────────────────────────────────────────────────────────


def group(runs: list[dict]) -> dict:
    out: dict = {}
    for r in runs:
        out.setdefault((r["task"], r["depth"], r["activation"], r["lr"]), []).append(r)
    for rows in out.values():
        rows.sort(key=lambda r: r["seed"])
    return out


def summarise_condition(rows: list[dict]) -> dict:
    finals = [r["final_loss"] for r in rows]
    med, q1, q3 = quartiles(finals)
    epochs = [
        r["epochs_to_threshold"] if r["epochs_to_threshold"] is not None else math.inf for r in rows
    ]
    e_med, e_q1, e_q3 = quartiles(epochs)
    rho = [r["rho"] if r["rho"] is not None else math.nan for r in rows]
    out = {
        "final_loss": {"median": med, "q1": q1, "q3": q3},
        "solved": sum(r["solved"] for r in rows),
        "diverged": sum(r["diverged_at"] is not None for r in rows),
        "epochs_to_threshold": {"median": e_med, "q1": e_q1, "q3": e_q3},
        "rho": dict(
            zip(
                ("median", "q1", "q3"),
                quartiles([v for v in rho if not math.isnan(v)] or [math.nan]),
                strict=True,
            )
        ),
    }
    for name in ("saturated_start", "saturated_end", "dead_start", "dead_end"):
        values = [r[name] for r in rows if r[name] is not None]
        out[name] = (
            dict(zip(("median", "q1", "q3"), quartiles(values), strict=True)) if values else None
        )
    curves = np.array([r["curve"] for r in rows], dtype=float)
    out["curve"] = {
        "median": [_inf_percentile(curves[:, i], 50) for i in range(curves.shape[1])],
        "q1": [_inf_percentile(curves[:, i], 25) for i in range(curves.shape[1])],
        "q3": [_inf_percentile(curves[:, i], 75) for i in range(curves.shape[1])],
    }
    return out


def best_lr(groups: dict, task: str, depth: int, act: str) -> float:
    """The learning rate with the lowest median final loss (ties: the smaller rate)."""

    def score(lr):
        return (quartiles([r["final_loss"] for r in groups[(task, depth, act, lr)]])[0], lr)

    return min(TASKS[task]["lrs"], key=score)


def analyse(runs: list[dict]) -> dict:
    groups = group(runs)
    conditions = {}
    for (task, depth, act, lr), rows in sorted(groups.items()):
        conditions[f"{task}|{depth}|{act}|{lr}"] = summarise_condition(rows)
    best = {
        f"{task}|{depth}|{act}": best_lr(groups, task, depth, act)
        for task in TASKS for depth in DEPTHS for act in ACTIVATIONS
    }  # fmt: skip
    comparisons = {}
    for task in TASKS:
        for depth in DEPTHS:
            rows = {
                act: groups[(task, depth, act, best[f"{task}|{depth}|{act}"])]
                for act in ACTIVATIONS
            }
            for a, b in (("tanh", "sigmoid"), ("relu", "tanh"), ("relu", "sigmoid")):
                la = [log10_or_inf(r["final_loss"]) for r in rows[a]]
                lb = [log10_or_inf(r["final_loss"]) for r in rows[b]]
                ra = [log10_or_inf(r["rho"]) for r in rows[a]]
                rb = [log10_or_inf(r["rho"]) for r in rows[b]]
                comparisons[f"{task}|{depth}|{a}-{b}"] = {
                    "log10_final_loss": bootstrap_median_difference(la, lb),
                    "log10_rho": bootstrap_median_difference(ra, rb),
                }
    return {"conditions": conditions, "best_lr": best, "comparisons": comparisons}


# ── output ──────────────────────────────────────────────────────────────────────────────────


def fmt(x: float) -> str:
    if x == math.inf:
        return "inf"
    if x == 0:
        return "0"
    return f"{x:.3g}" if 1e-3 <= abs(x) < 1e4 else f"{x:.1e}"


def print_tables(analysis: dict) -> None:
    c, best = analysis["conditions"], analysis["best_lr"]
    for task in TASKS:
        print(f"\n=== {task}: at each activation's best learning rate (10 seeds) ===")
        print("depth  act      lr     final loss median [IQR]       solved  epochs to 0.01  "
              "rho median [IQR]          sat/dead start->end")  # fmt: skip
        for depth in DEPTHS:
            for act in ACTIVATIONS:
                lr = best[f"{task}|{depth}|{act}"]
                s = c[f"{task}|{depth}|{act}|{lr}"]
                fl, ep, rho = s["final_loss"], s["epochs_to_threshold"], s["rho"]
                extra = s["saturated_start"] or s["dead_start"]
                extra_end = s["saturated_end"] or s["dead_end"]
                tag = (
                    "sat"
                    if s["saturated_start"]
                    else ("dead" if s["dead_start"] is not None else "")
                )
                ends = f"{tag} {fmt(extra['median'])}->{fmt(extra_end['median'])}" if extra else ""
                loss = f"{fmt(fl['median'])} [{fmt(fl['q1'])}, {fmt(fl['q3'])}]"
                rho_text = f"{fmt(rho['median'])} [{fmt(rho['q1'])}, {fmt(rho['q3'])}]"
                print(
                    f"{depth:>5}  {act:<8} {lr:<6} {loss:<29} {s['solved']:>2}/10  "
                    f"{fmt(ep['median']):>14}  {rho_text:<25} {ends}"
                )
        print(f"--- {task}: median final loss at every learning rate ---")
        for depth in DEPTHS:
            for act in ACTIVATIONS:
                row = " ".join(
                    f"{lr}:{fmt(c[f'{task}|{depth}|{act}|{lr}']['final_loss']['median'])}"
                    f"({c[f'{task}|{depth}|{act}|{lr}']['solved']}s,{c[f'{task}|{depth}|{act}|{lr}']['diverged']}d)"
                    for lr in TASKS[task]["lrs"]
                )
                print(f"  d{depth} {act:>8}: {row}")
    print("\n=== paired bootstrap, difference in medians of log10 (A - B), best lr each ===")
    for k, v in analysis["comparisons"].items():
        fl, rh = v["log10_final_loss"], v["log10_rho"]

        def show(d):
            if d["difference"] is None:
                return "undefined"
            ci = "no CI" if d["ci95"] is None else f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
            return f"{d['difference']:+.2f} {ci}"

        print(f"  {k:<24} final loss {show(fl):<26} rho {show(rh)}")


def results_json(runs: list[dict], analysis: dict) -> dict:
    def order(r: dict) -> tuple:
        return (r["task"], r["depth"], ACTIVATIONS.index(r["activation"]), r["lr"], r["seed"])

    per_run = [{k: sig(v) for k, v in r.items() if k != "curve"} for r in sorted(runs, key=order)]

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        return sig(obj) if isinstance(obj, float) else obj

    return {
        "study": "S2b",
        "design": {
            "tasks": {k: {**v, "lrs": list(v["lrs"])} for k, v in TASKS.items()},
            "depths": list(DEPTHS), "activations": list(ACTIVATIONS), "seeds": list(SEEDS),
            "threshold": THRESHOLD, "saturation": SATURATION, "curve_epochs": list(CURVE_EPOCHS),
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        },
        "analysis": clean(analysis),
        "runs": per_run,
    }  # fmt: skip


# ── failure cases as run folders (open them in the toy lab) ─────────────────────────────────


def save_case(task: str, depth: int, act: str, lr: float, seed: int, runs_dir: Path) -> str:
    """Train one study run again through the runner, so it becomes a run folder (same numbers)."""
    from neurodrive.experiment.runner import run
    from neurodrive.experiment.store import RunStore

    config = make_config(task, depth, act, lr, seed)
    learner = ToyLearner(config, make_rng(seed))
    store = RunStore.create(runs_dir, learner.name, config.tag, config.to_dict())
    store.save_extra("dataset", x=learner.dataset.x, y=learner.dataset.y)
    run(learner, store, config.epochs)
    return store.path.name


# ── figures ─────────────────────────────────────────────────────────────────────────────────

# Categorical slots 1-4 of the dataviz reference palette (light), checked with its validator.
COLOURS = {"sigmoid": "#2a78d6", "tanh": "#eb6834", "relu": "#1baf7a", "linear": "#eda100"}


def figures(analysis: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    c, best = analysis["conditions"], analysis["best_lr"]
    x = np.array(CURVE_EPOCHS, dtype=float) + 1  # epoch + 1 on a log axis, so epoch 0 shows

    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
    for row, task in enumerate(TASKS):
        for col, depth in enumerate(DEPTHS):
            ax = axes[row, col]
            for act in ACTIVATIONS:
                s = c[f"{task}|{depth}|{act}|{best[f'{task}|{depth}|{act}']}"]["curve"]
                med, q1, q3 = (np.array(s[k], dtype=float) for k in ("median", "q1", "q3"))
                ax.fill_between(x, q1, q3, color=COLOURS[act], alpha=0.15, lw=0)
                ax.plot(
                    x,
                    med,
                    color=COLOURS[act],
                    lw=2,
                    label=f"{act} (lr {best[f'{task}|{depth}|{act}']})",
                )
            ax.axhline(THRESHOLD, color="#888", lw=1, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{task}, {depth} hidden layer{'s' if depth > 1 else ''}")
            ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
            if col == 0:
                ax.set_ylabel("loss (MSE, log scale)")
            if row == 1:
                ax.set_xlabel("epoch + 1 (log scale)")
            ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(
        "S2b: median loss (line), IQR over 10 seeds (band), each activation at its best rate"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s2b_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, task in zip(axes, TASKS, strict=True):
        for i, act in enumerate(ACTIVATIONS):
            lr = TASKS[task]["lrs"][0]  # rho is measured before the first step: any lr gives it
            stats = [c[f"{task}|{d}|{act}|{lr}"]["rho"] for d in DEPTHS]
            med = np.array([s["median"] for s in stats])
            err = np.array(
                [[s["median"] - s["q1"] for s in stats], [s["q3"] - s["median"] for s in stats]]
            )
            ax.errorbar(np.array(DEPTHS) + (i - 1.5) * 0.08, med, yerr=err, color=COLOURS[act],
                        marker="o", ms=6, lw=2, capsize=3, label=act)  # fmt: skip
        ax.axhline(1, color="#888", lw=1, ls="--")
        ax.set_yscale("log")
        ax.set_xticks(DEPTHS)
        ax.set_xlabel("hidden layers")
        ax.set_title(f"{task}: first-layer / output-layer gradient size at epoch 1")
        ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
    axes[0].set_ylabel("rho (log scale; 1 = no vanishing)")
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s2b_rho.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey="row")
    for row, task in enumerate(TASKS):
        lrs = TASKS[task]["lrs"]
        for col, depth in enumerate(DEPTHS):
            ax = axes[row, col]
            for act in ACTIVATIONS:
                med = [c[f"{task}|{depth}|{act}|{lr}"]["final_loss"]["median"] for lr in lrs]
                med = [m if np.isfinite(m) else np.nan for m in med]
                ax.plot(lrs, med, color=COLOURS[act], marker="o", lw=2, label=act)
            ax.axhline(THRESHOLD, color="#888", lw=1, ls="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_title(f"{task}, {depth} hidden layer{'s' if depth > 1 else ''}")
            ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
            if col == 0:
                ax.set_ylabel("median final loss (log)")
            if row == 1:
                ax.set_xlabel("learning rate (log); missing point = diverged")
            if row == 0 and col == 0:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle(
        "S2b: final loss after 3,000 epochs against the learning rate (median of 10 seeds)"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s2b_learning_rates.png", dpi=130)
    plt.close(fig)
    print(f"figures written to {FIGURES_DIR.relative_to(ROOT).as_posix()}/s2b_*.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures", action="store_true")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument("--save-cases", nargs="*", metavar="TASK:DEPTH:ACT:LR:SEED",
                        help="also write these runs as run folders in runs/")  # fmt: skip
    args = parser.parse_args()

    keys = all_keys()
    print(f"{len(keys)} training runs on {args.workers} worker processes ...")
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        runs = list(pool.map(run_one, keys, chunksize=4))
    analysis = analyse(runs)
    print_tables(analysis)
    RESULTS_FILE.write_text(
        json.dumps(results_json(runs, analysis), indent=1) + "\n", encoding="utf-8"
    )
    print(f"\nresults written to {RESULTS_FILE.relative_to(ROOT).as_posix()}")
    if args.figures:
        figures(analysis)
    for spec in args.save_cases or []:
        task, depth, act, lr, seed = spec.split(":")
        name = save_case(task, int(depth), act, float(lr), int(seed), ROOT / "runs")
        print(f"case {spec} saved as runs/{name}")


if __name__ == "__main__":
    main()
