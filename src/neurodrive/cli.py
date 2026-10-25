"""The ``nd`` command-line tool.

Each subcommand is one thing you can ask NeuroDrive to do from a terminal. ``version`` and
``serve`` exist so far; ``train``, ``sweep``, ``exam`` and others are added by later tasks.
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
