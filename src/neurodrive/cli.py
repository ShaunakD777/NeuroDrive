"""The ``nd`` command-line tool.

Each subcommand is one thing you can ask NeuroDrive to do from a terminal. ``version``,
``serve`` and ``train`` exist so far; ``sweep``, ``exam`` and others are added by later tasks.
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
    epochs: int | None = typer.Option(None, help="Override training.epochs."),
    runs_dir: str = typer.Option("runs", help="Folder to create the run folder in."),
) -> None:
    """Train from a config file and write a run folder (open it in the toy lab)."""
    from neurodrive.experiment.config import ConfigError, load_config
    from neurodrive.experiment.runner import train_toy_run

    try:
        run_config = load_config(Path(config), seed=seed, tag=tag, epochs=epochs)
    except ConfigError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from None
    result = train_toy_run(run_config, runs_dir)
    typer.echo(f"run folder: {result.store.path.as_posix()}")
    typer.echo(f"status: {result.status} after {result.epochs_done} epochs")
    end = f"{result.end_loss:.6g}" if result.end_loss is not None else "none (diverged at once)"
    typer.echo(f"loss: {result.start_loss:.6g} at the start -> {end} at the last finite epoch")
