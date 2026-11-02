"""The ``nd`` command-line tool.

Each subcommand is one thing you can ask NeuroDrive to do from a terminal. ``version``,
``serve``, ``train``, ``sweep``, ``resume``, ``exam``, ``baseline`` and ``db rebuild`` /
``db check`` exist so far; others are added by later tasks.
"""

from pathlib import Path

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
) -> None:
    """Start the NeuroDrive server (REST and WebSocket) for the web viewer."""
    # Imported here so that `nd --help` stays fast and only this command touches the server.
    import uvicorn

    from neurodrive.server.app import create_app

    app_ = create_app(runs_dir=Path(runs_dir) if runs_dir else None)
    uvicorn.run(app_, host=host, port=port)


@app.command()
def train(
    config: str = typer.Option(
        ..., "--config", help="YAML config file, e.g. configs/toy_xor.yaml."
    ),
    seed: int | None = typer.Option(None, help="Override the config's seed."),
    tag: str | None = typer.Option(None, help="Override the config's tag (run folder label)."),
    epochs: int | None = typer.Option(None, help="Override training.epochs (toy configs)."),
    units: int | None = typer.Option(None, help="Override budget.units (random_search configs)."),
    runs_dir: str = typer.Option("runs", help="Folder to create the run folder in."),
) -> None:
    """Train from a config file (toy or random_search) and write a run folder."""
    from neurodrive.experiment.config import ConfigError, ToyRunConfig, load_config
    from neurodrive.experiment.runner import train_random_search_run, train_toy_run

    try:
        run_config = load_config(Path(config), seed=seed, tag=tag, epochs=epochs, units=units)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    if not isinstance(run_config, ToyRunConfig):
        found = train_random_search_run(run_config, runs_dir)
        typer.echo(f"run folder: {found.store.path.as_posix()}")
        typer.echo(f"status: {found.status} after {found.units_done} units")
        typer.echo(_best_line(found))
        return
    result = train_toy_run(run_config, runs_dir)
    typer.echo(f"run folder: {result.store.path.as_posix()}")
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
) -> None:
    """Continue a killed or failed run exactly where it stopped (resumable methods only)."""
    from neurodrive.experiment.config import ConfigError
    from neurodrive.experiment.runner import learner_from_config
    from neurodrive.experiment.runner import resume as resume_run
    from neurodrive.experiment.store import RunStore
    from neurodrive.learn.base import Resumable

    try:
        store = RunStore.open(run)
        learner = learner_from_config(store.read_config())
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


@app.command()
def sweep(
    config: str = typer.Option(
        ..., "--config", help="YAML config file, e.g. configs/toy_sine.yaml."
    ),
    seeds: str = typer.Option(..., help="Seeds to run, e.g. 0-4 or 0,2,7."),
    tag: str | None = typer.Option(None, help="Override the config's tag; runs get <tag>-s<seed>."),
    epochs: int | None = typer.Option(None, help="Override training.epochs (toy configs)."),
    units: int | None = typer.Option(None, help="Override budget.units (random_search configs)."),
    runs_dir: str = typer.Option("runs", help="Folder to create the run folders in."),
) -> None:
    """Train one config once per seed; summarise the final loss or best fitness (median, IQR)."""
    from neurodrive.experiment.config import ConfigError, ToyRunConfig, load_config
    from neurodrive.experiment.runner import parse_seeds, sweep_seeds
    from neurodrive.experiment.stats import summarise

    try:
        seed_list = parse_seeds(seeds)
        run_config = load_config(Path(config), tag=tag, epochs=epochs, units=units)
    except (ConfigError, ValueError) as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    sweep_id, results = sweep_seeds(run_config, seed_list, runs_dir)
    typer.echo(f"sweep: {sweep_id} ({len(results)} runs)")
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
) -> None:
    """Examine the baselines and store each as a run folder with its exam card."""
    from neurodrive.experiment.baselines import BASELINES, DEFAULT_SEEDS, make_baselines
    from neurodrive.experiment.runner import parse_seeds
    from neurodrive.experiment.stats import summarise

    try:
        names = BASELINES if name == "all" else (name,)
        if name != "all" and name not in BASELINES:
            raise ValueError(f"unknown baseline {name!r}; choose from {', '.join(BASELINES)}, all")
        if seeds is not None and "random_network" not in names:
            raise ValueError(f"--seeds is only for random_network; {name} has no seed")
        seed_list = tuple(parse_seeds(seeds)) if seeds is not None else DEFAULT_SEEDS
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    runs = make_baselines(runs_dir, names, seed_list)
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
    for table in ("metrics", "checkpoints", "exams", "exam_episodes"):
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
