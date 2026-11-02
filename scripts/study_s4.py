"""Study S4: does random search beat the baselines, and at what budget?

Run from the repository root:

    uv run python scripts/study_s4.py --train                       # make the runs (~20 min)
    uv run python scripts/study_s4.py                               # analyse them (numbers)
    uv run --with matplotlib python scripts/study_s4.py --figures   # also draw the figures

    uv run python scripts/study_s4.py --train-seeds 5-9             # more seeds, one at a time
    uv run --with matplotlib python scripts/study_s4.py --seeds 0-9 --figures   # analyse 0-9

Design and hypotheses: docs/studies/S4.md (written in the task 5.10 Brief, before any run).

* Random search (configs/random_search.yaml: [7, 16, 16, 2] tanh, 20 fresh Xavier networks per
  unit, fitness on the 10 exam starts of train_loop_01) for seeds 0-4, 200 units each, every
  checkpoint (the schedule and every new best) examined. One sweep; every run has study S4.
* The baselines, made again for S4 (ADR-043): random network seeds 0-9, hand-wired neuron,
  scripted expert. The exam is deterministic, so their cards must equal the task 5.6 cards.

Measured: S_k(B), the exam score of the best network seed k holds after a budget of B environment
steps. With u the last unit whose ``env_steps_total`` is at most B, it is the exam card of the
unit that found the best network held at u (its ``best_unit``; every new best is examined, task
5.9). On a log grid of budgets: median and IQR over the seeds; the budget B* where the median
first passes each baseline; bootstrap intervals at the common final budget (paired with the
random network of the same seed; one-sample against the deterministic neuron and expert), from
experiment/stats.py (10 000 resamples, seed 2026). The runs are found by their meta.json
``study`` field and cross-checked against the database's ``study`` column. Numbers go to
docs/studies/S4_results.json (the same bytes on every analysis of the same runs); with --figures
the pictures go to docs/studies/figures/ (matplotlib is needed only for that).

The extension to 10 seeds (task 5.10 follow-up): ``--train-seeds`` trains the given seeds one
after another, each alone on the machine, and prints each one's wall-clock time;
``--seeds 0-9`` analyses those seeds instead of the pre-registered 0-4, into files named
``..._seeds0-9`` so the pre-registered results stay as they are.
"""

import argparse
import bisect
import json
import os
import sqlite3
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from neurodrive.core.rng import make_rng
from neurodrive.experiment.baselines import make_baselines
from neurodrive.experiment.config import load_config, parse_config, resolved
from neurodrive.experiment.db import DB_NAME, connect, run_folders, runs_of_study
from neurodrive.experiment.runner import parse_seeds, train_random_search_run
from neurodrive.experiment.stats import (
    bootstrap_median_ci,
    bootstrap_median_difference,
    summarise,
)
from neurodrive.experiment.store import RunStore

ROOT = Path(__file__).resolve().parent.parent
RESULTS_FILE = ROOT / "docs" / "studies" / "S4_results.json"
FIGURES_DIR = ROOT / "docs" / "studies" / "figures"
CONFIG = ROOT / "configs" / "random_search.yaml"

# Fixed in the Brief before any run (D3A).
STUDY = "S4"
TAG = "s4"
SEEDS = tuple(range(5))
UNITS = 200
RANDOM_SEEDS = tuple(range(10))
GRID_POINTS = 30
TABLE_BUDGETS = (0, 25_000, 50_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000, 4_000_000)
BOOTSTRAP_SEED = 2026
HYPOTHESIS_UNIT = 10  # H1: from the budget at which every seed has finished unit 10
OVERFIT_DROP = 0.1  # H3: held-out progress at least this far below the seed's best earlier exam

# The measures read from an exam card; "higher is better" for all but the crash rate.
MAIN_MEASURES = (
    "heldout_progress",
    "heldout_completion",
    "heldout_crash",
    "training_progress",
    "training_completion",
)


# ── training ────────────────────────────────────────────────────────────────────────────────


def _train_seed(job: tuple[int, str, str]) -> str:
    seed, sweep_id, runs_dir = job
    config = load_config(CONFIG, tag=TAG, units=UNITS)
    one = parse_config(resolved(config), seed=seed, tag=f"{TAG}-s{seed}")
    meta = {"sweep": sweep_id, "study": STUDY}
    return train_random_search_run(one, runs_dir, extra_meta=meta).store.path.name


def _make_baselines(runs_dir: str) -> list[str]:
    return [r.store.path.name for r in make_baselines(runs_dir, seeds=RANDOM_SEEDS, study=STUDY)]


def train(runs_dir: Path, workers: int) -> list[str]:
    """Make S4's runs: the baselines in one worker, each random-search seed in another.

    Every run depends only on its own seed, so running them side by side changes no number
    (only the wall-clock times). The database mirror takes one writer at a time (WAL, 5 s wait).
    """
    sweep_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-random_search-{TAG}"
    with ProcessPoolExecutor(workers) as pool:
        baselines = pool.submit(_make_baselines, str(runs_dir))
        seeds = list(pool.map(_train_seed, [(s, sweep_id, str(runs_dir)) for s in SEEDS]))
        return [*seeds, *baselines.result()]


def train_one_at_a_time(runs_dir: Path, seeds: list[int]) -> list[tuple[str, float]]:
    """Train random-search seeds one after another (one sweep), so each has the machine to
    itself: its wall-clock time is then that of a run alone. Returns (run folder, seconds)."""
    sweep_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-random_search-{TAG}"
    done = []
    for seed in seeds:
        start = time.perf_counter()
        name = _train_seed((seed, sweep_id, str(runs_dir)))
        done.append((name, time.perf_counter() - start))
        print(f"seed {seed}: {name}  {done[-1][1]:.0f} s", flush=True)
    return done


# ── reading the runs ────────────────────────────────────────────────────────────────────────


def read_card(store: RunStore, unit: int) -> dict[str, Any]:
    return json.loads((store.path / "exams" / f"unit_{unit}.json").read_text(encoding="utf-8"))


def measures(card: dict[str, Any]) -> dict[str, float]:
    """The numbers S4 reads from one exam card (held-out = the 40 held-out episodes pooled)."""
    t, h = card["training"], card["heldout"]
    out = {
        "heldout_progress": h["progress_frac_median"],
        "heldout_completion": h["completion_rate"],
        "heldout_crash": h["crash_rate"],
        "training_progress": t["progress_frac_median"],
        "training_completion": t["completion_rate"],
    }
    for part in card["tracks"]:
        if part["split"] == "heldout":
            out[f"progress:{part['track']}"] = part["aggregate"]["progress_frac_median"]
            out[f"completion:{part['track']}"] = part["aggregate"]["completion_rate"]
    return out


def find_runs(runs_dir: Path, seeds: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    """S4's run folders, found by the ``study`` field of their meta.json (the master copy).

    Returns {"random_search": {seed: path}, "random_network": {seed: path}, "hand_neuron": path,
    "scripted_expert": path, "all": every S4 folder}; "random_search" holds only ``seeds`` (other
    S4 seeds are left out of the analysis). Stops with a message if a run is missing, unfinished
    or repeated.
    """
    found: dict[str, dict[int | None, list[Path]]] = {}
    every: list[Path] = []
    for path in run_folders(runs_dir):
        store = RunStore.open(path)
        meta = store.read_meta()
        if meta.get("study") != STUDY:
            continue
        every.append(path)
        if meta["status"] != "completed":
            raise SystemExit(f"{path.name} is {meta['status']}; finish it (nd resume) first")
        seed = store.read_config().get("seed")
        found.setdefault(meta["method"], {}).setdefault(seed, []).append(path)
    for method, by_seed in found.items():
        for seed, paths in by_seed.items():
            if len(paths) > 1:
                raise SystemExit(f"more than one S4 run of {method} seed {seed}: {paths}")
    expected = {
        "random_search": set(seeds),
        "random_network": set(RANDOM_SEEDS),
        "hand_neuron": {None},
        "scripted_expert": {None},
    }
    for method, wanted in expected.items():
        have = set(found.get(method, {}))
        if not wanted <= have or (method != "random_search" and have != wanted):
            raise SystemExit(f"S4 needs {method} runs for seeds {sorted(wanted, key=str)}")
    runs = {"random_search": {s: found["random_search"][s][0] for s in seeds},
            "random_network": {s: found["random_network"][s][0] for s in RANDOM_SEEDS}}  # fmt: skip
    return {**runs, "hand_neuron": found["hand_neuron"][None][0],
            "scripted_expert": found["scripted_expert"][None][0], "all": every}  # fmt: skip


def best_unit_at(steps: list[int], best_units: list[int], budget: float) -> int:
    """The unit that found the best network held after a budget of ``budget`` steps.

    ``steps[u]`` is the cumulative environment steps after unit u (``steps[0]`` = 0, before any
    unit) and ``best_units[u]`` the ``best_unit`` after unit u (0 before any unit). The last
    unit within budget is u = the last index with steps[u] <= budget.
    """
    if budget < 0:
        raise ValueError("a budget cannot be negative")
    u = bisect.bisect_right(steps, budget) - 1
    return best_units[u]


def seed_record(path: Path) -> dict[str, Any]:
    """What one random-search run did: steps, best units, fitness, every exam card's measures."""
    store = RunStore.open(path)
    lines = store.read_metrics()
    meta = store.read_meta()
    cards = {u: measures(read_card(store, u)) for u in store.exam_units()}
    return {
        "run_id": path.name,
        "steps": [0] + [line["env_steps_total"] for line in lines],
        "best_units": [0] + [line["best_unit"] for line in lines],
        "best_fitness": [line["best_fitness"] for line in lines],
        "new_best_units": [line["unit"] for line in lines if line["improved"]],
        "wall_clock_s": lines[-1]["wall_clock_s"],
        "exam_wall_clock_s": meta["exam_wall_clock_s"],
        "cards": cards,
    }


def score_at(record: dict[str, Any], budget: float) -> dict[str, float]:
    """S_k(B): the measures of the best network the run held after ``budget`` steps."""
    unit = best_unit_at(record["steps"], record["best_units"], budget)
    return record["cards"][unit]


def first_budget_above(budgets: list[int], values: list[float], level: float) -> int | None:
    """B*: the first budget above 0 whose value is above ``level`` (None if none is).

    Budget 0 is skipped: before any unit, random search holds its unit-0 network, which *is*
    the random-network baseline of its seed, so it says nothing about what searching bought.
    """
    pairs = zip(budgets, values, strict=True)
    return next((b for b, v in pairs if b > 0 and v > level), None)


def budget_grid(records: list[dict[str, Any]]) -> list[int]:
    """0, then GRID_POINTS budgets evenly spaced on a log scale from the smallest first-unit
    total to the smallest final total, so every seed has a value at every budget."""
    low = min(r["steps"][1] for r in records)
    high = min(r["steps"][-1] for r in records)
    return [0, *sorted({round(b) for b in np.geomspace(low, high, GRID_POINTS)})]


def baseline_values(paths: dict[str, Any]) -> dict[str, Any]:
    """Each baseline's measures: the random networks per seed, the neuron, the expert."""

    def card(path: Path) -> dict[str, float]:
        return measures(read_card(RunStore.open(path), 0))

    return {
        "random_network": {s: card(p) for s, p in paths["random_network"].items()},
        "hand_neuron": card(paths["hand_neuron"]),
        "scripted_expert": card(paths["scripted_expert"]),
    }


def same_as_task_5_6(runs_dir: Path, paths: dict[str, Any]) -> dict[str, Any]:
    """Compare every S4 baseline card with the card of the same baseline made in task 5.6 (no
    study): everything but the ``policy`` reference (which names the run) must be equal."""
    older: dict[tuple[str, int | None], Path] = {}
    for path in run_folders(runs_dir):
        store = RunStore.open(path)
        meta = store.read_meta()
        if meta.get("study") is None and meta["tag"].startswith("baseline"):
            older.setdefault((meta["method"], store.read_config().get("seed")), path)
    pairs = [(("random_network", s), p) for s, p in paths["random_network"].items()]
    pairs += [(("hand_neuron", None), paths["hand_neuron"])]
    pairs += [(("scripted_expert", None), paths["scripted_expert"])]
    compared, equal = 0, 0
    for key, path in pairs:
        if key not in older:
            continue
        a, b = read_card(RunStore.open(path), 0), read_card(RunStore.open(older[key]), 0)
        a.pop("policy"), b.pop("policy")
        compared += 1
        equal += a == b
    return {"compared": compared, "equal": equal}


def database_check(runs_dir: Path, ids: list[str]) -> dict[str, Any]:
    """Every S4 run folder (``ids``) through the database: its ``study`` column and its
    ``studies`` row."""
    try:
        with closing(connect(runs_dir / DB_NAME)) as conn:
            listed = runs_of_study(conn, STUDY)
            row = conn.execute(
                "SELECT n_runs, methods, sweeps FROM studies WHERE study_id = ?", (STUDY,)
            ).fetchone()
    except sqlite3.Error as exc:
        return {"error": str(exc)}
    return {
        "same_runs_as_the_folders": listed == sorted(ids),
        "n_runs": row[0] if row else None,
        "methods": json.loads(row[1]) if row else None,
        "n_sweeps": len(json.loads(row[2])) if row else None,
    }


# ── analysis ────────────────────────────────────────────────────────────────────────────────


def _band(values: list[float]) -> dict[str, float]:
    s = summarise(values)
    return {"median": s.median, "q1": s.q1, "q3": s.q3}


def _verdict(low: float, high: float, level: float) -> str:
    if low > level:
        return "random search clearly higher"
    if high < level:
        return "random search clearly lower"
    return "no clear difference"


def analyse(runs_dir: Path, seeds_used: tuple[int, ...] = SEEDS) -> dict[str, Any]:
    paths = find_runs(runs_dir, seeds_used)
    records = [seed_record(paths["random_search"][s]) for s in seeds_used]
    base = baseline_values(paths)
    names = sorted(records[0]["cards"][0])  # every measure, per-track ones included
    grid = budget_grid(records)
    final_budget = grid[-1]
    table_budgets = [b for b in TABLE_BUDGETS if b < final_budget] + [final_budget]

    def at(budgets: list[int]) -> dict[str, list[dict[str, float]]]:
        out: dict[str, list[dict[str, float]]] = {}
        for name in names:
            per_budget = [[score_at(r, b)[name] for r in records] for b in budgets]
            out[name] = [{**_band(v), "seeds": v} for v in per_budget]
        return out

    curves, table = at(grid), at(table_budgets)
    random_band = {n: _band([base["random_network"][s][n] for s in RANDOM_SEEDS]) for n in names}
    levels = {
        "random_network": {n: random_band[n]["median"] for n in names},
        "hand_neuron": base["hand_neuron"],
        "scripted_expert": base["scripted_expert"],
    }
    b_star = {
        baseline: {
            n: first_budget_above(grid, [c["median"] for c in curves[n]], level[n])
            for n in MAIN_MEASURES
            if n != "heldout_crash"
        }
        for baseline, level in levels.items()
    }

    final = {n: [score_at(r, final_budget)[n] for r in records] for n in names}
    comparisons: dict[str, Any] = {}
    for n in MAIN_MEASURES:
        paired = bootstrap_median_difference(
            final[n],
            [base["random_network"][s][n] for s in seeds_used],
            make_rng(BOOTSTRAP_SEED),
            paired=True,
        )
        ci = bootstrap_median_ci(final[n], make_rng(BOOTSTRAP_SEED))
        comparisons[n] = {
            "random_search_median": ci.estimate,
            "random_search_ci": [ci.low, ci.high],
            "minus_random_network_paired": {
                "estimate": paired.estimate,
                "ci": [paired.low, paired.high],
                "verdict": _verdict(paired.low, paired.high, 0.0),
            },
            "vs_hand_neuron": {
                "value": base["hand_neuron"][n],
                "verdict": _verdict(ci.low, ci.high, base["hand_neuron"][n]),
            },
            "vs_scripted_expert": {
                "value": base["scripted_expert"][n],
                "verdict": _verdict(ci.low, ci.high, base["scripted_expert"][n]),
            },
        }

    seeds = []
    for seed, r in zip(seeds_used, records, strict=True):
        final_unit = best_unit_at(r["steps"], r["best_units"], final_budget)
        earlier = {u: c["heldout_progress"] for u, c in r["cards"].items() if u < final_unit}
        best_earlier_unit = max(earlier, key=lambda u: (earlier[u], -u))
        drop = earlier[best_earlier_unit] - r["cards"][final_unit]["heldout_progress"]
        seeds.append(
            {
                "seed": seed,
                "run_id": r["run_id"],
                "units": len(r["steps"]) - 1,
                "env_steps_total": r["steps"][-1],
                "wall_clock_s": r["wall_clock_s"],
                "exam_wall_clock_s": r["exam_wall_clock_s"],
                "new_best_units": r["new_best_units"],
                "final_best_fitness": r["best_fitness"][-1],
                "best_unit_at_final_budget": final_unit,
                "at_final_budget": r["cards"][final_unit],
                "best_earlier_heldout_progress": {
                    "unit": best_earlier_unit,
                    "value": earlier[best_earlier_unit],
                },
                "heldout_progress_drop": drop,
                "fitness_never_fell": all(
                    b >= a
                    for a, b in zip(r["best_fitness"][:-1], r["best_fitness"][1:], strict=True)
                ),
                "examined": {
                    str(u): {
                        "env_steps": r["steps"][u],
                        "fitness": r["best_fitness"][u - 1] if u else None,
                        **c,
                    }
                    for u, c in sorted(r["cards"].items())
                },
                "fitness_curve": [
                    [s, f] for s, f in zip(r["steps"][1:], r["best_fitness"], strict=True)
                ],
            }
        )

    unit10 = max(r["steps"][min(HYPOTHESIS_UNIT, len(r["steps"]) - 1)] for r in records)
    rn_progress = random_band["heldout_progress"]["median"]
    after = [
        c["median"] > rn_progress
        for b, c in zip(grid, curves["heldout_progress"], strict=True)
        if b >= unit10
    ]
    hp, hc = comparisons["heldout_progress"], comparisons["heldout_completion"]
    neuron, expert = base["hand_neuron"], base["scripted_expert"]
    overfit = [s["seed"] for s in seeds if s["heldout_progress_drop"] >= OVERFIT_DROP]
    hypotheses = {
        "H1": {
            "from_budget": unit10,
            "median_above_random_network_at_every_budget": all(after),
            "paired_ci_excludes_0": hp["minus_random_network_paired"]["ci"][0] > 0,
        },
        "H2": {
            "progress_above_neuron": hp["random_search_median"] > neuron["heldout_progress"],
            "completion_below_neuron": hc["random_search_median"] < neuron["heldout_completion"],
        },
        "H3": {
            "seeds_with_drop": overfit,
            "fitness_never_fell": all(s["fitness_never_fell"] for s in seeds),
            "supported": len(overfit) >= 2 and all(s["fitness_never_fell"] for s in seeds),
        },
        "H4": {
            "no_seed_reaches_expert": max(final["heldout_completion"])
            < expert["heldout_completion"],
            "median_completion_below_half": hc["random_search_median"] < 0.5,
        },
    }
    for h in ("H1", "H2", "H4"):
        hypotheses[h]["supported"] = all(v for k, v in hypotheses[h].items() if k != "from_budget")

    ids = [p.name for p in paths["random_search"].values()]
    ids += [p.name for p in paths["random_network"].values()]
    ids += [paths["hand_neuron"].name, paths["scripted_expert"].name]
    metas = [RunStore.open(p).read_meta() for p in paths["random_search"].values()]

    def one_or_all(values: list[Any]) -> Any:
        distinct = sorted(set(values), key=str)
        return distinct[0] if len(distinct) == 1 else distinct

    return {
        "study": STUDY,
        "design": {
            "config": "configs/random_search.yaml",
            "seeds": list(seeds_used),
            "units": UNITS,
            "random_network_seeds": list(RANDOM_SEEDS),
            "bootstrap": {"resamples": 10_000, "seed": BOOTSTRAP_SEED},
            # The short form: a full 40-digit hash looks like a secret to the secret scanners.
            "git_commit": one_or_all([(m["git_commit"] or "")[:7] or None for m in metas]),
            "git_dirty": one_or_all([m["git_dirty"] for m in metas]),
            "sweep": one_or_all([m["sweep"] for m in metas]),
        },
        "run_ids": sorted(ids),
        "database": database_check(runs_dir, [p.name for p in paths["all"]]),
        "baselines_equal_task_5_6": same_as_task_5_6(runs_dir, paths),
        "baselines": {
            "random_network": {"band": random_band, "seeds": base["random_network"]},
            "hand_neuron": neuron,
            "scripted_expert": expert,
        },
        "final_budget": final_budget,
        "grid": grid,
        "curves": curves,
        "table_budgets": table_budgets,
        "table": table,
        "b_star": b_star,
        "comparisons": comparisons,
        "seeds": seeds,
        "hypotheses": hypotheses,
    }


def _rounded(obj: Any) -> Any:
    """Floats to 6 significant digits, recursively (keys sorted when written)."""
    if isinstance(obj, float):
        return float(f"{obj:.6g}")
    if isinstance(obj, dict):
        return {str(k): _rounded(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_rounded(v) for v in obj]
    return obj


def seeds_label(seeds: list[int] | tuple[int, ...]) -> str:
    """ "seeds0-9" for a range without gaps, else "seeds0_2_7"; used in file names and titles."""
    if list(seeds) == list(range(seeds[0], seeds[-1] + 1)):
        return f"seeds{seeds[0]}-{seeds[-1]}"
    return "seeds" + "_".join(map(str, seeds))


def print_tables(a: dict[str, Any]) -> None:
    print(f"S4 runs analysed: {len(a['run_ids'])}; database: {a['database']}")
    print(f"baseline cards equal to task 5.6: {a['baselines_equal_task_5_6']}")
    print(f"final (common) budget: {a['final_budget']:,} environment steps")
    label = seeds_label(a["design"]["seeds"])[5:]
    print(f"\nheld-out progress / completion by budget (median [IQR] over seeds {label}):")
    for i, b in enumerate(a["table_budgets"]):
        p, c = a["table"]["heldout_progress"][i], a["table"]["heldout_completion"][i]
        print(
            f"  {b:>10,}  progress {p['median']:.3f} [{p['q1']:.3f}, {p['q3']:.3f}]"
            f"  completion {c['median']:.3f} [{c['q1']:.3f}, {c['q3']:.3f}]"
        )
    rn = a["baselines"]["random_network"]["band"]
    for name in ("heldout_progress", "heldout_completion"):
        print(
            f"baselines {name}: random networks median {rn[name]['median']:.3f}, neuron "
            f"{a['baselines']['hand_neuron'][name]:.3f}, expert "
            f"{a['baselines']['scripted_expert'][name]:.3f}"
        )
    print(f"\nB* (first budget with the median above): {json.dumps(a['b_star'], indent=1)}")
    for name, c in a["comparisons"].items():
        d = c["minus_random_network_paired"]
        print(
            f"{name}: median {c['random_search_median']:.3f} CI {c['random_search_ci']}, minus "
            f"random net {d['estimate']:.3f} CI {d['ci']} ({d['verdict']}); vs neuron: "
            f"{c['vs_hand_neuron']['verdict']}; vs expert: {c['vs_scripted_expert']['verdict']}"
        )
    for s in a["seeds"]:
        print(
            f"seed {s['seed']}: {s['run_id']}  fitness {s['final_best_fitness']:.4f}  bests "
            f"{s['new_best_units']}  held-out progress at final "
            f"{s['at_final_budget']['heldout_progress']:.3f} (best earlier "
            f"{s['best_earlier_heldout_progress']['value']:.3f} at unit "
            f"{s['best_earlier_heldout_progress']['unit']})  train {s['wall_clock_s']:.0f} s,"
            f" exams {s['exam_wall_clock_s']:.0f} s"
        )
    for h, v in a["hypotheses"].items():
        print(f"{h}: {v}")


# ── figures ─────────────────────────────────────────────────────────────────────────────────

# Categorical slots of the dataviz reference palette (light), as in the app and in S3. Seeds 5-9
# reuse them dashed (colour + line style): more than 8 hues cannot be told apart reliably.
SEED_COLOURS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8b5cf6")


def seed_style(seed: int) -> dict[str, Any]:
    return {"color": SEED_COLOURS[seed % 5], "ls": "-" if seed < 5 else "--"}


BASELINE_STYLE = {
    "random_network": ("#888888", ":", "random networks (median of 10)"),
    "hand_neuron": ("#c2410c", "--", "hand-wired neuron"),
    "scripted_expert": ("#111111", "-.", "scripted expert"),
}


def _level(a: dict[str, Any], baseline: str, name: str) -> float:
    if baseline == "random_network":
        return a["baselines"]["random_network"]["band"][name]["median"]
    return a["baselines"][baseline][name]


def figures(a: dict[str, Any], suffix: str = "") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    grid = np.array(a["grid"][1:], dtype=float)  # the log axis has no place for budget 0

    # 1. Exam score against budget: held-out progress and completion, median + IQR + seeds.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, name, label in (
        (axes[0], "heldout_progress", "held-out median progress (share of 2 laps)"),
        (axes[1], "heldout_completion", "held-out completion rate"),
    ):
        c = a["curves"][name][1:]
        med, q1, q3 = (np.array([p[k] for p in c]) for k in ("median", "q1", "q3"))
        for i, s in enumerate(a["seeds"]):
            seed_line = [p["seeds"][i] for p in c]
            ax.step(grid, seed_line, where="post", lw=0.8, alpha=0.5, **seed_style(s["seed"]))
        ax.fill_between(grid, q1, q3, step="post", color="#2a78d6", alpha=0.15, lw=0)
        ax.step(grid, med, where="post", color="#2a78d6", lw=2.4, label="random search (median)")
        for baseline, (colour, style, text) in BASELINE_STYLE.items():
            ax.axhline(_level(a, baseline, name), color=colour, ls=style, lw=1.3, label=text)
        ax.set_xscale("log")
        ax.set_ylim(-0.03, 1.05)
        ax.set_xlabel("budget: environment steps (log scale)")
        ax.set_ylabel(label)
        ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
    # Below the expert's line at 1.0, where the curves have not risen yet.
    axes[0].legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 0.92))
    fig.suptitle(
        "S4: Driving Exam score of the best network found so far "
        f"(median, IQR band, thin = seeds {seeds_label(a['design']['seeds'])[5:]})"
    )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"s4_budget{suffix}.png", dpi=130)
    plt.close(fig)

    # 2. Training fitness (top) and held-out progress (bottom) per seed: separate panels,
    #    because fitness is random search's own score and never shares the exam's axis.
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for s in a["seeds"]:
        style = seed_style(s["seed"])
        curve = np.array(s["fitness_curve"], dtype=float)
        top.step(curve[:, 0], curve[:, 1], where="post", lw=1.6, label=f"seed {s['seed']}",
                 **style)  # fmt: skip
        examined = sorted(s["examined"].items(), key=lambda kv: int(kv[0]))
        xs = [max(e["env_steps"], grid[0]) for _, e in examined]
        ys = [e["heldout_progress"] for _, e in examined]
        bottom.step(xs + [curve[-1, 0]], ys + ys[-1:], where="post", lw=1.6, **style)
        bottom.plot(xs, ys, ls="none", marker="o", ms=3.5, color=style["color"])
    top.set_ylabel("best fitness so far\n(training track, 10 starts)")
    top.set_title("Random search's own score: best fitness so far (never falls)")
    top.legend(frameon=False, fontsize=8, ncol=5, loc="upper left")
    bottom.set_ylabel("held-out median progress\nof that network (exam)")
    bottom.set_title("The same networks on the Driving Exam's held-out tracks (dots = exams)")
    bottom.set_xscale("log")
    bottom.set_xlabel("environment steps (log scale)")
    for ax in (top, bottom):
        ax.grid(True, which="major", color="#e5e5e5", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"s4_fitness_vs_heldout{suffix}.png", dpi=130)
    plt.close(fig)

    # 3. Per held-out track at the final budget: random search (median, IQR) vs the baselines.
    tracks = sorted(k.split(":", 1)[1] for k in a["curves"] if k.startswith("progress:"))
    fig, ax = plt.subplots(figsize=(9, 4.4))
    x = np.arange(len(tracks))
    width = 0.2
    rs = [a["curves"][f"progress:{t}"][-1] for t in tracks]
    med = np.array([r["median"] for r in rs])
    err = np.array([[r["median"] - r["q1"] for r in rs], [r["q3"] - r["median"] for r in rs]])
    label = "random search (median, IQR)"
    ax.bar(x - 1.5 * width, med, width, yerr=err, color="#2a78d6", label=label)
    for j, baseline in enumerate(BASELINE_STYLE):
        colour, _, text = BASELINE_STYLE[baseline]
        values = [_level(a, baseline, f"progress:{t}") for t in tracks]
        if baseline == "random_network":
            text += ": about 0, no bar"
        ax.bar(x + (j - 0.5) * width, values, width, color=colour, alpha=0.85, label=text)
    ax.set_xticks(x, tracks)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("median progress (share of 2 laps)")
    ax.set_title(f"S4: held-out tracks at the final budget ({a['final_budget']:,} steps)")
    ax.legend(frameon=False, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    ax.grid(True, axis="y", color="#e5e5e5", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / f"s4_tracks{suffix}.png", dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", action="store_true", help="make S4's runs first (~20 min)")
    parser.add_argument("--figures", action="store_true", help="draw the figures (matplotlib)")
    parser.add_argument("--runs-dir", default=str(ROOT / "runs"), help="folder of run folders")
    parser.add_argument("--workers", type=int, default=min(len(SEEDS) + 1, os.cpu_count() or 2))
    parser.add_argument(
        "--train-seeds",
        metavar="SEEDS",
        help="only train these random-search seeds, one at a time (e.g. 5-9), then stop",
    )
    parser.add_argument("--seeds", default="0-4", help="the seeds to analyse (default 0-4)")
    args = parser.parse_args()
    runs_dir = Path(args.runs_dir)

    if args.train_seeds:
        train_one_at_a_time(runs_dir, parse_seeds(args.train_seeds))
        return
    if args.train:
        for name in train(runs_dir, args.workers):
            print(f"run folder: {name}")
    seeds = tuple(parse_seeds(args.seeds))
    suffix = "" if seeds == SEEDS else f"_{seeds_label(seeds)}"
    analysis = analyse(runs_dir, seeds)
    print_tables(analysis)
    text = json.dumps(_rounded(analysis), indent=1, sort_keys=True) + "\n"
    results = RESULTS_FILE.with_stem(RESULTS_FILE.stem + suffix)
    results.write_text(text, encoding="utf-8", newline="\n")
    print(f"\nwrote {results.relative_to(ROOT).as_posix()}")
    if args.figures:
        figures(analysis, suffix)
        print(f"wrote figures to {FIGURES_DIR.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
