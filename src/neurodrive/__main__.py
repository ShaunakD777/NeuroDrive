"""``python -m neurodrive ...`` runs the ``nd`` command-line tool.

The same commands as ``nd``, started through Python. Useful where Windows Smart App Control blocks
the generated ``nd.exe`` launcher: ``uv run python -m neurodrive serve``.
"""

from neurodrive.cli import app

app()
