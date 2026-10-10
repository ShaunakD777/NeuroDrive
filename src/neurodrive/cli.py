"""The ``nd`` command-line tool.

Each subcommand is one thing you can ask NeuroDrive to do from a terminal. Only ``version``
exists for now; ``serve``, ``train``, ``sweep``, ``exam`` and others are added by later tasks.
"""

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
