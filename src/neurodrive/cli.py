"""The ``nd`` command-line tool.

Each subcommand is one thing you can ask NeuroDrive to do from a terminal. ``version``,
``serve``, ``train``, ``sweep``, ``resume``, ``exam``, ``baseline``, ``demo-record`` and
``db rebuild`` / ``db check`` exist so far; others are added by later tasks.
"""

from pathlib import Path
from typing import Annotated

import typer

from neurodrive import __version__

app = typer.Typer(
    name="nd",
    help="NeuroDrive: neural networks learning to drive.",
    no_args_is_help=True,
    add_completion=False,
)


@app.callback()
def main() -> None:
    """NeuroDrive: neural networks learning to drive."""
    # Having a callback keeps `nd` a group of subcommands even while it has only one.


@app.command()
def version() -> None:
    """Print the installed NeuroDrive version."""
    typer.echo(__version__)


@app.command()
def serve(
    host: str = typer.Option(
        "127.0.0.1", help="Address to listen on (127.0.0.1 = this machine only)."
    ),
    port: int = typer.Option(8000, help="Port to listen on."),
    runs_dir: str | None = typer.Option(
        None, help="Folder of run folders to show (default: runs/ in the repository)."
    ),
    datasets_dir: str | None = typer.Option(
        None,
        help="Folder of demonstration datasets to show (default: data/demos/ in the repository).",
    ),
) -> None:
    """Start the NeuroDrive server (REST and WebSocket) for the web viewer."""
    # Imported here so that `nd --help` stays fast and only this command touches the server.
    import uvicorn

    from neurodrive.server.app import create_app

    app_ = create_app(
        runs_dir=Path(runs_dir) if runs_dir else None,
        datasets_dir=Path(datasets_dir) if datasets_dir else None,
    )
    uvicorn.run(app_, host=host, port=port)


_STUDY_HELP = "Study id written into each run's meta.json, e.g. S4 (ADR-043)."
_EPOCHS_HELP = "Override training.epochs (toy and imitation configs)."
_UNITS_HELP = "Override budget.units (random_search) or dagger.rounds (dagger configs)."
_DATASETS_HELP = (
    "Folder of demonstration datasets for imitation and DAgger (default: data/demos/); "
    "DAgger also saves the frames it collected there."
)


def _datasets_dir(datasets_dir: str | None) -> Path:
    from neurodrive.learn.supervised.demos import DEFAULT_DATASETS_DIR

    return Path(datasets_dir) if datasets_dir else DEFAULT_DATASETS_DIR


def _loss_line(result) -> str:
    """One line on an imitation run's last training and validation loss."""
    if result.end_loss is None:
        return "loss: none (diverged at once)"
    val = f", validation {result.end_val_loss:.6g}" if result.end_val_loss is not None else ""
    return f"loss after the last finite epoch: training {result.end_loss:.6g}{val}"


def _dagger_lines(result) -> list[str]:
    """What a DAgger run ends with: its losses, the teacher queries and the dataset saved."""
    lines = [_loss_line(result).replace("epoch", "round")]
    if result.end_onpolicy_loss is not None:
        lines.append(
            f"on-policy error of the last collection drive: {result.end_onpolicy_loss:.6g}"
        )
    lines.append(f"teacher queries: {result.teacher_queries}")
    if result.dagger_dataset is not None:
        lines.append(f"collected frames saved as dataset: {result.dagger_dataset}")
    return lines


def _check_study_option(study: str | None) -> None:
    """Refuse an invalid --study before anything is created."""
    if study is not None:
        from neurodrive.experiment.store import check_study

        check_study(study)


@app.command()
def train(
    config: str = typer.Option(
        ..., "--config", help="YAML config file, e.g. configs/toy_xor.yaml."
    ),
    seed: int | None = typer.Option(None, help="Override the config's seed."),
    tag: str | None = typer.Option(None, help="Override the config's tag (run folder label)."),
    epochs: int | None = typer.Option(None, help=_EPOCHS_HELP),
    units: int | None = typer.Option(None, help=_UNITS_HELP),
    runs_dir: str = typer.Option("runs", help="Folder to create the run folder in."),
    study: str | None = typer.Option(None, help=_STUDY_HELP),
    datasets_dir: str | None = typer.Option(None, help=_DATASETS_HELP),
) -> None:
    """Train from a config file (toy, random_search, imitation or dagger), write a run folder."""
    import sys

    from neurodrive.experiment.config import (
        ConfigError,
        DaggerRunConfig,
        ImitationRunConfig,
        ToyRunConfig,
        load_config,
    )
    from neurodrive.experiment.runner import (
        pin_dataset,
        train_dagger_run,
        train_imitation_run,
        train_random_search_run,
        train_toy_run,
    )

    demos = _datasets_dir(datasets_dir)
    try:
        _check_study_option(study)
        run_config = load_config(Path(config), seed=seed, tag=tag, epochs=epochs, units=units)
        if isinstance(run_config, ImitationRunConfig | DaggerRunConfig):
            run_config, _ = pin_dataset(run_config, demos)  # refuse a missing dataset up front
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    meta = {"study": study} if study is not None else None

    def announce(store) -> None:
        # Printed as soon as the folder exists, not at the end: the server reads this line to
        # learn the id of a run it launched (task 5.8, ADR-041).
        typer.echo(f"run folder: {store.path.as_posix()}")
        sys.stdout.flush()

    if isinstance(run_config, ImitationRunConfig):
        typer.echo(f"dataset: {run_config.data.dataset}")
        cloned = train_imitation_run(
            run_config, runs_dir, meta, on_start=announce, datasets_dir=demos
        )
        typer.echo(f"status: {cloned.status} after {cloned.epochs_done} epochs")
        typer.echo(_loss_line(cloned))
        return
    if isinstance(run_config, DaggerRunConfig):
        typer.echo(f"dataset: {run_config.data.dataset}")
        aggregated = train_dagger_run(
            run_config, runs_dir, meta, on_start=announce, datasets_dir=demos
        )
        typer.echo(f"status: {aggregated.status} after {aggregated.rounds_done} rounds")
        for line in _dagger_lines(aggregated):
            typer.echo(line)
        return
    if not isinstance(run_config, ToyRunConfig):
        found = train_random_search_run(run_config, runs_dir, meta, on_start=announce)
        typer.echo(f"status: {found.status} after {found.units_done} units")
        typer.echo(_best_line(found))
        return
    result = train_toy_run(run_config, runs_dir, meta, on_start=announce)
    typer.echo(f"status: {result.status} after {result.epochs_done} epochs")
    end = f"{result.end_loss:.6g}" if result.end_loss is not None else "none (diverged at once)"
    typer.echo(f"loss: {result.start_loss:.6g} at the start -> {end} at the last finite epoch")


def _best_line(found) -> str:
    """One line on a random-search run's best network."""
    if found.best_fitness is None:
        return "best fitness: none (no unit finished)"
    return f"best fitness: {found.best_fitness:.4f} (found in unit {found.best_unit})"


@app.command()
def resume(
    run: str = typer.Option(..., "--run", help="Run folder to continue, e.g. runs/<run_id>."),
    datasets_dir: str | None = typer.Option(None, help=_DATASETS_HELP),
) -> None:
    """Continue a killed or failed run exactly where it stopped (resumable methods only)."""
    from neurodrive.experiment.config import ConfigError
    from neurodrive.experiment.runner import (
        dagger_result,
        learner_from_config,
        save_dagger_dataset,
    )
    from neurodrive.experiment.runner import resume as resume_run
    from neurodrive.experiment.store import RunStore
    from neurodrive.learn.base import Resumable
    from neurodrive.learn.supervised.dagger import DaggerLearner

    demos = _datasets_dir(datasets_dir)
    try:
        store = RunStore.open(run)
        learner = learner_from_config(store.read_config(), datasets_dir=demos)
        if not isinstance(learner, Resumable):
            raise ValueError(
                f"{learner.name} runs cannot be resumed (the learner saves no state); "
                "train them again instead"
            )
        units_before = len(store.read_metrics())
        status = resume_run(learner, store)
    except (FileNotFoundError, ConfigError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    metrics = store.read_metrics()
    typer.echo(f"run folder: {store.path.as_posix()}")
    typer.echo(f"status: {status} after {len(metrics)} units ({units_before} were kept)")
    last = metrics[-1] if metrics else {}
    if "best_fitness" in last:
        typer.echo(f"best fitness: {last['best_fitness']:.4f} (found in unit {last['best_unit']})")
    if isinstance(learner, DaggerLearner):  # the frames are saved once the run has ended
        for line in _dagger_lines(
            dagger_result(store, status, save_dagger_dataset(store, learner, demos))
        ):
            typer.echo(line)
    elif last.get("val_loss") is not None:
        typer.echo(f"loss: training {last['loss']:.6g}, validation {last['val_loss']:.6g}")


@app.command()
def sweep(
    config: str = typer.Option(
        ..., "--config", help="YAML config file, e.g. configs/toy_sine.yaml."
    ),
    seeds: str = typer.Option(..., help="Seeds to run, e.g. 0-4 or 0,2,7."),
    tag: str | None = typer.Option(None, help="Override the config's tag; runs get <tag>-s<seed>."),
    epochs: int | None = typer.Option(None, help=_EPOCHS_HELP),
    units: int | None = typer.Option(None, help=_UNITS_HELP),
    runs_dir: str = typer.Option("runs", help="Folder to create the run folders in."),
    study: str | None = typer.Option(None, help=_STUDY_HELP),
    datasets_dir: str | None = typer.Option(None, help=_DATASETS_HELP),
) -> None:
    """Train one config once per seed; summarise the final loss or best fitness (median, IQR)."""
    from neurodrive.experiment.config import (
        ConfigError,
        DaggerRunConfig,
        ImitationRunConfig,
        ToyRunConfig,
        load_config,
    )
    from neurodrive.experiment.runner import parse_seeds, pin_dataset, sweep_seeds
    from neurodrive.experiment.stats import summarise

    demos = _datasets_dir(datasets_dir)
    try:
        _check_study_option(study)
        seed_list = parse_seeds(seeds)
        run_config = load_config(Path(config), tag=tag, epochs=epochs, units=units)
        if isinstance(run_config, ImitationRunConfig | DaggerRunConfig):
            run_config, _ = pin_dataset(run_config, demos)
    except (ConfigError, FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    sweep_id, results = sweep_seeds(
        run_config, seed_list, runs_dir, study=study, datasets_dir=demos
    )
    typer.echo(f"sweep: {sweep_id} ({len(results)} runs)")
    if isinstance(run_config, DaggerRunConfig):
        val, onp = [], []
        for seed, aggregated in zip(seed_list, results, strict=True):
            done = aggregated.status == "completed"
            for values, value in (
                (val, aggregated.end_val_loss),
                (onp, aggregated.end_onpolicy_loss),
            ):
                values.append(value if done and value is not None else float("inf"))
            typer.echo(
                f"  seed {seed}: {aggregated.store.path.name}  {aggregated.status}  "
                f"{aggregated.teacher_queries} teacher queries"
            )
        for name, values in (("validation loss", val), ("on-policy error", onp)):
            s = summarise(values)
            typer.echo(
                f"final {name}: median {s.median:.6g} [IQR {s.q1:.6g} to {s.q3:.6g}] over "
                f"{s.n} seeds (a diverged run, or one without the value, counts as inf)"
            )
        return
    if isinstance(run_config, ImitationRunConfig):
        finals = []
        for seed, cloned in zip(seed_list, results, strict=True):
            final = cloned.end_val_loss if cloned.status == "completed" else None
            finals.append(float("inf") if final is None else final)
            typer.echo(
                f"  seed {seed}: {cloned.store.path.name}  {cloned.status}  {_loss_line(cloned)}"
            )
        s = summarise(finals)
        typer.echo(
            f"final validation loss: median {s.median:.6g} [IQR {s.q1:.6g} to {s.q3:.6g}] over "
            f"{s.n} seeds (a diverged run, or one without a validation set, counts as inf)"
        )
        return
    if not isinstance(run_config, ToyRunConfig):
        for seed, found in zip(seed_list, results, strict=True):
            typer.echo(
                f"  seed {seed}: {found.store.path.name}  {found.status}  {_best_line(found)}"
            )
        s = summarise([f.best_fitness or 0.0 for f in results])
        typer.echo(
            f"best fitness: median {s.median:.4f} [IQR {s.q1:.4f} to {s.q3:.4f}] over {s.n} seeds"
        )
        return
    finals = []
    for seed, result in zip(seed_list, results, strict=True):
        final = result.end_loss if result.status == "completed" else float("inf")
        finals.append(final)
        typer.echo(
            f"  seed {seed}: {result.store.path.name}  {result.status}  final loss {final:.6g}"
        )
    s = summarise(finals)
    typer.echo(
        f"final loss: median {s.median:.6g} [IQR {s.q1:.6g} to {s.q3:.6g}] over {s.n} seeds"
        " (a diverged run counts as inf)"
    )


@app.command()
def exam(
    run: str = typer.Option(..., "--run", help="Run folder, e.g. runs/<run_id>."),
    unit: int = typer.Option(..., "--unit", help="Checkpoint unit to examine."),
) -> None:
    """Take the Driving Exam with a checkpoint: write exams/unit_<n>.json and print the card."""
    from neurodrive.env.policies import MLPPolicy
    from neurodrive.experiment.db import DB_NAME, Mirror
    from neurodrive.experiment.exam import format_card, run_exam
    from neurodrive.experiment.store import RunStore

    try:
        store = RunStore.open(run)
        if unit not in store.checkpoint_units():
            raise ValueError(
                f"no checkpoint for unit {unit}; saved units: {store.checkpoint_units()}"
            )
        network = store.load_checkpoint(unit)
        MLPPolicy(network)  # refuses a network that is not 7 inputs -> 2 outputs
    except (FileNotFoundError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    ref = {"kind": "checkpoint", "run": store.path.name, "unit": unit}
    result = run_exam(lambda world: MLPPolicy(network), ref)
    path = result.save(store.path / "exams" / f"unit_{unit}.json")
    mirror = Mirror(store.path.parent / DB_NAME)  # copy the run again, new card included
    mirror.sync_run(store)
    mirror.close()
    typer.echo(format_card(result))
    typer.echo(f"exam card: {path.as_posix()}")


@app.command()
def baseline(
    name: str = typer.Option(
        "all", help="random_network, hand_neuron, scripted_expert, or all three."
    ),
    seeds: str | None = typer.Option(
        None, help="Seeds of the random network, e.g. 0-4 or 0,2,7 (default 0-9)."
    ),
    runs_dir: str = typer.Option("runs", help="Folder to create the baseline runs in."),
    study: str | None = typer.Option(None, help=_STUDY_HELP),
) -> None:
    """Examine the baselines and store each as a run folder with its exam card."""
    from neurodrive.experiment.baselines import BASELINES, DEFAULT_SEEDS, make_baselines
    from neurodrive.experiment.runner import parse_seeds
    from neurodrive.experiment.stats import summarise

    try:
        _check_study_option(study)
        names = BASELINES if name == "all" else (name,)
        if name != "all" and name not in BASELINES:
            raise ValueError(f"unknown baseline {name!r}; choose from {', '.join(BASELINES)}, all")
        if seeds is not None and "random_network" not in names:
            raise ValueError(f"--seeds is only for random_network; {name} has no seed")
        seed_list = tuple(parse_seeds(seeds)) if seeds is not None else DEFAULT_SEEDS
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    runs = make_baselines(runs_dir, names, seed_list, study=study)
    for run in runs:
        t, h = run.card.training, run.card.heldout
        typer.echo(
            f"{run.store.path.name}  completion: training {t['completion_rate']:.2f}, "
            f"held-out {h['completion_rate']:.2f}  held-out median progress "
            f"{h['progress_frac_median']:.3f}"
        )
    randoms = [r.card.heldout for r in runs if r.store.read_meta()["method"] == "random_network"]
    if randoms:
        for key, label in (("completion_rate", "completion"), ("progress_frac_median", "progress")):
            s = summarise([c[key] for c in randoms])
            typer.echo(
                f"random networks, held-out {label}: median {s.median:.3f} "
                f"[IQR {s.q1:.3f} to {s.q3:.3f}] over {s.n} seeds"
            )


@app.command("demo-record")
def demo_record(
    teacher: str = typer.Option("expert", help="Who drives: expert (keyboard: task 6.1b)."),
    track: str = typer.Option("train_loop_01", help="Training track to record on."),
    random_starts: int = typer.Option(
        10, help="Seeded random starts besides the exam's 10 starts (0: exam starts only)."
    ),
    seed: int = typer.Option(0, help="Seed of the random starts."),
    start_lateral: float = typer.Option(
        1.0, help="Random starts are moved sideways by up to this many metres (default 1)."
    ),
    start_heading: float = typer.Option(
        0.1, help="Random starts are turned by up to this many radians (default 0.1)."
    ),
    starts_from: str | None = typer.Option(
        None,
        help="A dataset id or <teacher>-<tag>: the expert drives from the starts that dataset "
        "used, one drive per episode (task 6.7; the exam and random starts are not used).",
    ),
    match_frames: bool = typer.Option(
        False,
        help="With --starts-from: keep only as many frames of each drive as the source's "
        "episode has.",
    ),
    tag: str = typer.Option("default", help="Label in the dataset folder's name."),
    out_dir: str = typer.Option("data/demos", help="Folder to create the dataset folder in."),
) -> None:
    """Record a teacher's driving as a demonstration dataset (dataset.npz + meta.json)."""
    from neurodrive.env.track import Track
    from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, EXAM_V1
    from neurodrive.experiment.store import _git_info
    from neurodrive.learn.supervised.demos import (
        DemoDataset,
        find_dataset,
        record_expert,
        record_expert_from_starts,
        save_dataset,
    )

    try:
        if teacher != "expert":
            raise ValueError(
                f"unknown teacher {teacher!r}; only expert so far (keyboard demos are recorded "
                "from the live session, task 6.1b)"
            )
        path = DEFAULT_TRACKS_DIR / f"{track}.json"
        if not path.is_file():
            raise ValueError(f"unknown track {track!r}")
        if match_frames and starts_from is None:
            raise ValueError("--match-frames needs --starts-from")
        if starts_from is not None:
            source = DemoDataset.load(find_dataset(out_dir, starts_from))
            dataset = record_expert_from_starts(
                Track.from_json(path),
                source,
                match_frames=match_frames,
                target_laps=EXAM_V1.target_laps,
            )
        else:
            dataset = record_expert(
                Track.from_json(path),
                start_fractions=EXAM_V1.start_fractions,
                lateral_offsets=EXAM_V1.lateral_offsets,
                random_starts=random_starts,
                seed=seed,
                target_laps=EXAM_V1.target_laps,
                random_lateral_m=start_lateral,
                random_heading_rad=start_heading,
            )
        folder = save_dataset(dataset, out_dir, tag, provenance=_git_info())
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    meta = dataset.meta
    outcomes: dict[str, int] = {}
    for episode in meta["episodes"]:
        outcomes[episode["status"]] = outcomes.get(episode["status"], 0) + 1
    size = sum(p.stat().st_size for p in folder.iterdir())
    typer.echo(f"dataset folder: {folder.as_posix()}")
    typer.echo(
        f"frames: {dataset.n_frames:,} from {len(meta['episodes'])} episodes "
        f"({', '.join(f'{n} {s}' for s, n in sorted(outcomes.items()))}); {size / 1e6:.1f} MB"
    )
    typer.echo(f"nearly straight (|steer| < 0.1): {meta['straight_share']:.3f} of the frames")


@app.command("demo-merge")
def demo_merge(
    datasets: Annotated[
        list[str],
        typer.Argument(
            help="Dataset ids, or <teacher>-<tag> for the newest such dataset, in this order."
        ),
    ],
    tag: str = typer.Option("merged", help="Label in the new dataset folder's name."),
    datasets_dir: str = typer.Option("data/demos", help="Folder holding the datasets."),
) -> None:
    """Join datasets of one teacher into one new dataset (the sources are left as they are)."""
    from neurodrive.experiment.store import _git_info
    from neurodrive.learn.supervised.demos import (
        DemoDataset,
        find_dataset,
        merge_datasets,
        save_dataset,
    )

    try:
        sources = [DemoDataset.load(find_dataset(datasets_dir, name)) for name in datasets]
        merged = merge_datasets(sources)
        folder = save_dataset(
            merged,
            datasets_dir,
            tag,
            provenance={**_git_info(), "merged_from": merged.meta["starts"]["sources"]},
        )
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    typer.echo(f"dataset folder: {folder.as_posix()}")
    typer.echo(
        f"frames: {merged.n_frames:,} from {len(merged.meta['episodes'])} episodes "
        f"of {len(sources)} datasets"
    )


db_app = typer.Typer(
    help="The SQLite mirror of the run folders (runs/neurodrive.db).", no_args_is_help=True
)
app.add_typer(db_app, name="db")


@db_app.command("rebuild")
def db_rebuild(
    runs_dir: str = typer.Option("runs", help="Folder of run folders (the database lives in it)."),
) -> None:
    """Empty the database and copy every run folder into it again."""
    from neurodrive.experiment.db import DB_NAME, rebuild

    if not Path(runs_dir).is_dir():
        typer.echo(f"no such folder: {runs_dir}", err=True)
        raise typer.Exit(code=2)
    summary = rebuild(runs_dir)
    rows = summary.rows
    typer.echo(f"rebuilt {(Path(runs_dir) / DB_NAME).as_posix()} from {rows['runs']} run folders")
    for table in ("studies", "metrics", "checkpoints", "exams", "exam_episodes"):
        typer.echo(f"  {table}: {rows[table]:,} rows")
    for name, reason in summary.skipped:
        typer.echo(f"  skipped {name}: {reason}")


@db_app.command("check")
def db_check(
    runs_dir: str = typer.Option("runs", help="Folder of run folders (the database lives in it)."),
) -> None:
    """Compare the database with the run folders; exit code 1 if they differ."""
    from neurodrive.experiment.db import check, run_folders

    problems = check(runs_dir)
    if not problems:
        typer.echo(f"the database matches the folders ({len(run_folders(runs_dir))} runs)")
        return
    for line in problems:
        typer.echo(line)
    typer.echo(
        f"{len(problems)} difference(s). The folders are the master copy: nd db rebuild "
        "makes the database match them."
    )
    raise typer.Exit(code=1)
