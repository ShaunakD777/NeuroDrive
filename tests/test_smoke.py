"""Smoke test: the package installs and imports.

A smoke test only checks that nothing is on fire. It proves the project is wired up, not that
any behaviour is correct.
"""

import neurodrive


def test_package_imports_and_has_version() -> None:
    assert isinstance(neurodrive.__version__, str)
    assert neurodrive.__version__ != ""
