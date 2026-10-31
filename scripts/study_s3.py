"""Study S3: how do learning rate and optimizer choice affect convergence and stability?

Run from the repository root:

    uv run python scripts/study_s3.py                               # numbers only
    uv run --with matplotlib python scripts/study_s3.py --figures   # also draw the figures
    uv run python scripts/study_s3.py --save-cases                  # also write representative
                                                                    # runs as run folders in runs/

Three parts (design and hypotheses in docs/studies/S3.md, written before any run):

* A. Learning-rate sweep: SGD, Momentum, RMSProp and Adam on sine ([1, 16, 1] tanh, full batch,
  MSE, 3000 epochs) at eight learning rates from 0.001 to 3, on the tuning seeds 0-4.
* B. Fair comparison: each optimizer at its best learning rate from part A (lowest median final
  loss), on the fresh test seeds 5-14, so the choice of rate cannot flatter the result.
* C. Overfitting: noisy_sine (64 points, noise std 0.2, a quarter held back for validation), a
  big network [1, 64, 64, 1] against a small one [1, 4, 1], Adam 0.01, 5000 epochs, seeds 0-9.

Measured per run: the final loss L_T (+inf if the run diverged), the epoch it diverged, the first
epoch with a loss below 0.01; in part C the training and validation curves, the best validation
epoch e*, and the overfitting ratio (final validation loss / lowest validation loss).

Statistics come from experiment/stats.py (median, IQR, curve bands, paired bootstrap of
log10 final losses, 10 000 resamples, seed 2026). Runs are spread over worker processes, but
every number depends only on its own seed and results are put back in a fixed order, so the
results file is identical on every run. Numbers go to docs/studies/S3_results.json; with --figures
the pictures go to docs/studies/figures/ (matplotlib is needed only for that).
"""

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from neurodrive.core.rng import make_rng
from neurodrive.experiment.stats import bootstrap_median_difference, curve_bands, summarise
from neurodrive.learn.supervised.train_toy import ToyConfig, ToyLearner

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "docs" / "studies" / "S3_results.json"
FIGURES_DIR = ROOT / "docs" / "studies" / "figures"

# Fixed in the Brief before any run.
OPTIMIZERS = ("sgd", "momentum", "rmsprop", "adam")
LEARNING_RATES = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
TUNE_SEEDS = tuple(range(5))
TEST_SEEDS = tuple(range(5, 15))
EPOCHS = 3000
THRESHOLD = 0.01
OVERFIT_NETWORKS = {"big": (1, 64, 64, 1), "small": (1, 4, 1)}
OVERFIT_SEEDS = tuple(range(10))
OVERFIT_EPOCHS = 5000
OVERFIT_LR = 0.01
NOISE_LEVEL = 0.2**2  # the loss of a perfect sine curve on the noisy points
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 2026


def sampled_epochs(last: int) -> tuple[int, ...]:
    """Epoch 0, then about 12 epochs per factor of ten up to ``last`` (for curves in the JSON)."""
    return tuple(sorted({0, *(int(e) for e in np.round(np.logspace(0, math.log10(last), 45)))}))


CURVE_EPOCHS = sampled_epochs(EPOCHS)
OVERFIT_CURVE_EPOCHS = sampled_epochs(OVERFIT_EPOCHS)


# ── single runs ─────────────────────────────────────────────────────────────────────────────


def sine_config(optimizer: str, lr: float, seed: int) -> ToyConfig:
    return ToyConfig.for_task(
        "sine", optimizer=optimizer, learning_rate=lr, seed=seed, epochs=EPOCHS
    )


def overfit_config(network: str, seed: int) -> ToyConfig:
    return ToyConfig.for_task(
        "noisy_sine", layer_sizes=OVERFIT_NETWORKS[network], seed=seed, epochs=OVERFIT_EPOCHS
    )


def run_sine(key: tuple[str, float, int]) -> dict:
    """One part A/B run: final loss (+inf if diverged), divergence epoch, first epoch below 0.01."""
    optimizer, lr, seed = key
    config = sine_config(optimizer, lr, seed)
    learner = ToyLearner(config, make_rng(seed))
    curve = [learner.loss()]  # epoch 0: before any step
    diverged_at = None
    for epoch in range(1, EPOCHS + 1):
        result = learner.train_unit()
        if result.diverged:
            diverged_at = epoch
            break
        curve.append(result.metrics["loss"])
    below = next((e for e, loss in enumerate(curve) if loss < THRESHOLD), None)
    return {
        "optimizer": optimizer,
        "lr": lr,
        "seed": seed,
        "final": math.inf if diverged_at is not None else curve[-1],
        "diverged_at": diverged_at,
        "first_below": below,
        "curve": curve,
    }


def run_overfit(key: tuple[str, int]) -> dict:
    """One part C run: training and validation curves, best validation epoch, overfitting ratio."""
    network, seed = key
    learner = ToyLearner(overfit_config(network, seed), make_rng(seed))
    train, val = [learner.loss()], [learner.val_loss()]
    for _ in range(OVERFIT_EPOCHS):
        metrics = learner.train_unit().metrics
        train.append(metrics["loss"])
        val.append(metrics["val_loss"])
    best = int(np.argmin(val))
    return {
        "network": network,
        "seed": seed,
        "best_val_epoch": best,
        "best_val": val[best],
        "final_val": val[-1],
        "final_train": train[-1],
        "ratio": val[-1] / val[best],
        "train": train,
        "val": val,
    }


def run_all(workers: int) -> tuple[list[dict], list[dict], list[dict]]:
    sweep_keys = [(o, lr, s) for o in OPTIMIZERS for lr in LEARNING_RATES for s in TUNE_SEEDS]
    overfit_keys = [(n, s) for n in OVERFIT_NETWORKS for s in OVERFIT_SEEDS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        sweep = list(pool.map(run_sine, sweep_keys))  # map keeps the order of the keys
        best = best_learning_rates(analyse_sweep(sweep))
        test_keys = [(o, best[o], s) for o in OPTIMIZERS for s in TEST_SEEDS]
        test = list(pool.map(run_sine, test_keys))
        overfit = list(pool.map(run_overfit, overfit_keys))
    return sweep, test, overfit


# ── analysis ────────────────────────────────────────────────────────────────────────────────


def summary_dict(values) -> dict:
    s = summarise(values)
    return {"median": s.median, "q1": s.q1, "q3": s.q3, "n": s.n}


def analyse_sweep(sweep: list[dict]) -> dict:
    """Part A: per optimizer and learning rate, the final loss summary and the divergence count."""
    out = {}
    for optimizer in OPTIMIZERS:
        for lr in LEARNING_RATES:
            rows = [r for r in sweep if r["optimizer"] == optimizer and r["lr"] == lr]
            out[f"{optimizer}|{lr}"] = {
                "final": summary_dict([r["final"] for r in rows]),
                "diverged": sum(r["diverged_at"] is not None for r in rows),
                "solved": sum(r["final"] < THRESHOLD for r in rows),
                "first_below": summary_dict(
                    [math.inf if r["first_below"] is None else r["first_below"] for r in rows]
                ),
            }
    return out


def best_learning_rates(cells: dict) -> dict:
    """Each optimizer's learning rate with the lowest median final loss (ties: the smaller rate)."""
    return {
        o: min(LEARNING_RATES, key=lambda lr: (cells[f"{o}|{lr}"]["final"]["median"], lr))
        for o in OPTIMIZERS
    }


def edges(cells: dict) -> dict:
    """Per optimizer: the smallest rate where any seed diverged and where all did; usable rates."""
    out = {}
    for o in OPTIMIZERS:
        any_div = [lr for lr in LEARNING_RATES if cells[f"{o}|{lr}"]["diverged"] > 0]
        all_div = [lr for lr in LEARNING_RATES if cells[f"{o}|{lr}"]["diverged"] == len(TUNE_SEEDS)]
        usable = [lr for lr in LEARNING_RATES if cells[f"{o}|{lr}"]["final"]["median"] < THRESHOLD]
        out[o] = {
            "first_any_diverged": any_div[0] if any_div else None,
            "first_all_diverged": all_div[0] if all_div else None,
            "usable": usable,
        }
    return out


def bands(curves: list[list[float]], diverged: list[bool], epochs: tuple[int, ...]) -> dict:
    b = curve_bands(curves, diverged)
    return {k: [getattr(b, k)[e] for e in epochs] for k in ("median", "q1", "q3")}


def analyse_test(test: list[dict], best: dict) -> dict:
    """Part B: each optimizer at its best rate on the fresh seeds; paired bootstrap against SGD."""
    rows = {o: [r for r in test if r["optimizer"] == o] for o in OPTIMIZERS}
    out = {"best_lr": best, "optimizers": {}, "versus_sgd": {}}
    for o in OPTIMIZERS:
        r = rows[o]
        out["optimizers"][o] = {
            "lr": best[o],
            "final": summary_dict([x["final"] for x in r]),
            "solved": sum(x["final"] < THRESHOLD for x in r),
            "diverged": sum(x["diverged_at"] is not None for x in r),
            "first_below": summary_dict(
                [math.inf if x["first_below"] is None else x["first_below"] for x in r]
            ),
            "curve": bands(
                [x["curve"] for x in r], [x["diverged_at"] is not None for x in r], CURVE_EPOCHS
            ),
        }
    log_sgd = [math.log10(x["final"]) for x in rows["sgd"]]
    for o in OPTIMIZERS[1:]:
        log_o = [math.log10(x["final"]) for x in rows[o]]
        interval = bootstrap_median_difference(
            log_o, log_sgd, make_rng(BOOTSTRAP_SEED), BOOTSTRAP_RESAMPLES, paired=True
        )
        out["versus_sgd"][o] = {
            "difference_log10": interval.estimate,
            "ci95": [interval.low, interval.high],
            "contains_zero": interval.contains_zero,
        }
    return out


def analyse_overfit(overfit: list[dict]) -> dict:
    """Part C: best validation epoch, lowest and final validation loss, ratio, final train loss."""
    out = {}
    for network in OVERFIT_NETWORKS:
        r = [x for x in overfit if x["network"] == network]
        out[network] = {
            "layer_sizes": list(OVERFIT_NETWORKS[network]),
            **{
                k: summary_dict([x[k] for x in r])
                for k in ("best_val_epoch", "best_val", "final_val", "final_train", "ratio")
            },
            "ratio_at_least_1_5": sum(x["ratio"] >= 1.5 for x in r),
            "ratio_within_1_2": sum(x["ratio"] <= 1.2 for x in r),
            "best_before_1000": sum(x["best_val_epoch"] < 1000 for x in r),
            "train_below_noise": sum(x["final_train"] < NOISE_LEVEL for x in r),
            "train_curve": bands([x["train"] for x in r], [False] * len(r), OVERFIT_CURVE_EPOCHS),
            "val_curve": bands([x["val"] for x in r], [False] * len(r), OVERFIT_CURVE_EPOCHS),
        }
    return out


def analyse(sweep: list[dict], test: list[dict], overfit: list[dict]) -> dict:
    cells = analyse_sweep(sweep)
    best = best_learning_rates(cells)
    return {
        "sweep": cells,
        "edges": edges(cells),
        "test": analyse_test(test, best),
        "overfit": analyse_overfit(overfit),
    }


# ── output ──────────────────────────────────────────────────────────────────────────────────


def sig(x, digits: int = 4):
    """JSON-friendly rounding to significant digits (inf -> "inf", None stays None)."""
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, dict):
        return {k: sig(v, digits) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [sig(v, digits) for v in x]
    if isinstance(x, (int, np.integer)):
        return int(x)
    if math.isinf(x):
        return "inf" if x > 0 else "-inf"
    return float(f"{x:.{digits}g}")


def fmt(x) -> str:
    if x is None:
        return "-"
    if isinstance(x, float) and math.isinf(x):
        return "inf"
    return f"{x:.3g}"


def print_tables(a: dict) -> None:
    print("A. Learning-rate sweep on sine, seeds 0-4: median final loss [IQR], diverged/5")
    print("   lr     " + "".join(f"{o:>28}" for o in OPTIMIZERS))
    for lr in LEARNING_RATES:
        cells = [a["sweep"][f"{o}|{lr}"] for o in OPTIMIZERS]
        text = [
            f"{fmt(c['final']['median'])} [{fmt(c['final']['q1'])}, {fmt(c['final']['q3'])}] "
            f"{c['diverged']}/5"
            for c in cells
        ]
        print(f"   {lr:<6} " + "".join(f"{t:>28}" for t in text))
    for o in OPTIMIZERS:
        e = a["edges"][o]
        print(
            f"   {o:<9} first any diverged {e['first_any_diverged']}, all diverged "
            f"{e['first_all_diverged']}, usable (median < 0.01): {e['usable']}"
        )
    t = a["test"]
    print("\nB. Each at its best rate, fresh seeds 5-14: median final loss [IQR], solved/10")
    for o in OPTIMIZERS:
        r = t["optimizers"][o]
        f = r["final"]
        print(
            f"   {o:<9} lr {r['lr']:<6} {fmt(f['median'])} [{fmt(f['q1'])}, {fmt(f['q3'])}] "
            f"solved {r['solved']}/10, first below 0.01 median {fmt(r['first_below']['median'])}"
        )
    for o, d in t["versus_sgd"].items():
        print(
            f"   {o} - sgd (log10 final): {d['difference_log10']:+.2f} "
            f"[{d['ci95'][0]:+.2f}, {d['ci95'][1]:+.2f}]"
        )
    print("\nC. Overfitting on noisy_sine (48 train, 16 validation), Adam 0.01, seeds 0-9")
    for n, r in a["overfit"].items():
        print(
            f"   {n:<6} {r['layer_sizes']}: best val epoch {fmt(r['best_val_epoch']['median'])} "
            f"[{fmt(r['best_val_epoch']['q1'])}, {fmt(r['best_val_epoch']['q3'])}], "
            f"best val {fmt(r['best_val']['median'])}, final val {fmt(r['final_val']['median'])}, "
            f"final train {fmt(r['final_train']['median'])}, ratio {fmt(r['ratio']['median'])} "
            f"[{fmt(r['ratio']['q1'])}, {fmt(r['ratio']['q3'])}]; ratio>=1.5: "
            f"{r['ratio_at_least_1_5']}/10, <=1.2: {r['ratio_within_1_2']}/10, best<1000: "
            f"{r['best_before_1000']}/10, train<0.04: {r['train_below_noise']}/10"
        )


def results_json(sweep: list[dict], test: list[dict], overfit: list[dict], a: dict) -> dict:
    def per_run(rows: list[dict], drop: tuple[str, ...]) -> list[dict]:
        return [{k: sig(v) for k, v in r.items() if k not in drop} for r in rows]

    return {
        "study": "S3",
        "design": {
            "optimizers": list(OPTIMIZERS),
            "learning_rates": list(LEARNING_RATES),
            "tune_seeds": list(TUNE_SEEDS),
            "test_seeds": list(TEST_SEEDS),
            "epochs": EPOCHS,
            "threshold": THRESHOLD,
            "curve_epochs": list(CURVE_EPOCHS),
            "overfit": {
                "networks": {k: list(v) for k, v in OVERFIT_NETWORKS.items()},
                "seeds": list(OVERFIT_SEEDS),
                "epochs": OVERFIT_EPOCHS,
                "learning_rate": OVERFIT_LR,
                "noise_level": NOISE_LEVEL,
                "curve_epochs": list(OVERFIT_CURVE_EPOCHS),
            },
            "bootstrap": {"resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        },
        "analysis": sig(a),
        "runs": {
            "sweep": per_run(sweep, ("curve",)),
            "test": per_run(test, ("curve",)),
            "overfit": per_run(overfit, ("train", "val")),
        },
    }


# ── representative runs as run folders (open them in the toy lab and Compare runs) ──────────


def save_cases(a: dict, runs_dir: Path) -> list[str]:
    """Train representative study runs again through `nd sweep` / `nd train` code (same numbers)."""
    import yaml

    from neurodrive.experiment.config import parse_config
    from neurodrive.experiment.runner import sweep_seeds, train_toy_run

    sine = yaml.safe_load((ROOT / "configs" / "toy_sine.yaml").read_text("utf-8"))
    noisy = yaml.safe_load((ROOT / "configs" / "toy_noisy_sine.yaml").read_text("utf-8"))
    made = []
    for o, lr in a["test"]["best_lr"].items():
        config = parse_config(
            {**sine, "tag": f"s3-{o}", "optimizer": {"name": o, "learning_rate": lr}}
        )
        sweep_id, _ = sweep_seeds(config, list(TEST_SEEDS), runs_dir)
        made.append(sweep_id)
    for network, sizes in OVERFIT_NETWORKS.items():
        config = parse_config(
            {
                **noisy,
                "tag": f"s3-overfit-{network}",
                "network": {**noisy["network"], "layer_sizes": list(sizes)},
            }
        )
        sweep_id, _ = sweep_seeds(config, list(OVERFIT_SEEDS), runs_dir)
        made.append(sweep_id)
    # SGD at the smallest rate where a tuning seed diverged, seed 0 (failure analysis).
    edge = a["edges"]["sgd"]["first_any_diverged"]
    if edge is not None:
        config = parse_config(
            {
                **sine,
                "seed": 0,
                "tag": "s3-sgd-diverged",
                "optimizer": {"name": "sgd", "learning_rate": edge},
            }
        )
        made.append(train_toy_run(config, runs_dir).store.path.name)
    return made


# ── figures ─────────────────────────────────────────────────────────────────────────────────

# Categorical slots 1-4 of the dataviz reference palette (light), as in the app (task 4.7b).
COLOURS = {"sgd": "#2a78d6", "momentum": "#eb6834", "rmsprop": "#1baf7a", "adam": "#eda100"}
NETWORK_COLOURS = {"big": "#2a78d6", "small": "#eb6834"}


def figures(a: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Final loss against learning rate (part A); diverged medians drawn at the top as crosses.
    fig, ax = plt.subplots(figsize=(8, 4.8))
    top = 10.0
    for o in OPTIMIZERS:
        med, q1, q3 = (
            np.array([a["sweep"][f"{o}|{lr}"]["final"][k] for lr in LEARNING_RATES])
            for k in ("median", "q1", "q3")
        )
        shown = np.minimum(med, top)
        ax.fill_between(
            LEARNING_RATES,
            np.minimum(q1, top),
            np.minimum(q3, top),
            color=COLOURS[o],
            alpha=0.15,
            lw=0,
        )
        ax.plot(LEARNING_RATES, shown, color=COLOURS[o], lw=2, marker="o", ms=4, label=o)
        blown = [lr for lr, m in zip(LEARNING_RATES, med, strict=True) if math.isinf(m)]
        ax.plot(blown, [top] * len(blown), ls="none", marker="x", ms=9, mew=2, color=COLOURS[o])
    ax.axhline(THRESHOLD, color="#888", lw=1, ls="--")
    ax.text(LEARNING_RATES[0], THRESHOLD * 1.15, "0.01 (solved)", color="#666", fontsize=8)
    ax.text(
        LEARNING_RATES[0],
        top * 0.62,
        "x = median diverged (counted as infinity)",
        color="#666",
        fontsize=8,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("learning rate (log scale)")
    ax.set_ylabel("final loss after 3000 epochs (log scale)")
    ax.set_title("S3 A: final loss vs learning rate on sine (median, IQR band, seeds 0-4)")
    ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s3_lr_sweep.png", dpi=130)
    plt.close(fig)

    # 2. Learning curves at each optimizer's best rate on the fresh seeds (part B).
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.array(CURVE_EPOCHS, dtype=float) + 1
    for o in OPTIMIZERS:
        r = a["test"]["optimizers"][o]
        med, q1, q3 = (np.array(r["curve"][k], dtype=float) for k in ("median", "q1", "q3"))
        ax.fill_between(x, q1, q3, color=COLOURS[o], alpha=0.15, lw=0)
        ax.plot(x, med, color=COLOURS[o], lw=2, label=f"{o} (lr {r['lr']})")
    ax.axhline(THRESHOLD, color="#888", lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch + 1 (log scale)")
    ax.set_ylabel("training loss (log scale)")
    ax.set_title("S3 B: each optimizer at its best rate, fresh seeds 5-14 (median, IQR)")
    ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s3_curves.png", dpi=130)
    plt.close(fig)

    # 3. Overfitting: training (dashed) and validation (solid) loss of both networks (part C).
    fig, ax = plt.subplots(figsize=(8, 4.8))
    x = np.array(OVERFIT_CURVE_EPOCHS, dtype=float) + 1
    for n, r in a["overfit"].items():
        for kind, style in (("train_curve", "--"), ("val_curve", "-")):
            med, q1, q3 = (np.array(r[kind][k], dtype=float) for k in ("median", "q1", "q3"))
            ax.fill_between(x, q1, q3, color=NETWORK_COLOURS[n], alpha=0.12, lw=0)
            label = (
                f"{n} {r['layer_sizes']}: {'training' if kind == 'train_curve' else 'validation'}"
            )
            ax.plot(x, med, color=NETWORK_COLOURS[n], lw=2, ls=style, label=label)
        e = r["best_val_epoch"]["median"]
        ax.axvline(e + 1, color=NETWORK_COLOURS[n], lw=1, ls=":")
    ax.axhline(NOISE_LEVEL, color="#888", lw=1, ls="--")
    ax.text(1.2, NOISE_LEVEL * 1.08, "noise level 0.04", color="#666", fontsize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("epoch + 1 (log scale)")
    ax.set_ylabel("loss (log scale)")
    ax.set_title(
        "S3 C: overfitting on noisy_sine (median, IQR, seeds 0-9; dotted = median best epoch)"
    )
    ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "s3_overfit.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figures", action="store_true", help="draw the figures (matplotlib)")
    parser.add_argument("--save-cases", action="store_true", help="write run folders to runs/")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    sweep, test, overfit = run_all(args.workers)
    analysis = analyse(sweep, test, overfit)
    print_tables(analysis)
    RESULTS_FILE.write_text(
        json.dumps(results_json(sweep, test, overfit, analysis), indent=1) + "\n", encoding="utf-8"
    )
    print(f"\nwrote {RESULTS_FILE.relative_to(ROOT).as_posix()}")
    if args.figures:
        figures(analysis)
        print(f"wrote figures to {FIGURES_DIR.relative_to(ROOT).as_posix()}")
    if args.save_cases:
        for name in save_cases(analysis, ROOT / "runs"):
            print(f"run folder / sweep: {name}")


if __name__ == "__main__":
    main()
