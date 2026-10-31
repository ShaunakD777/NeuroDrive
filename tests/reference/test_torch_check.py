"""The PyTorch cross-check (reference/torch_check.py) agrees with our maths, and its controls fail.

Skipped unless PyTorch is installed, which it is not in the normal environment. Run it with:

    uv run --with torch python -m pytest tests/reference
"""

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def check():
    spec = importlib.util.spec_from_file_location(
        "torch_check", ROOT / "reference" / "torch_check.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_worked_example_matches(check):
    numbers = check.worked_example()
    assert {name: ours for name, (ours, _) in numbers.items()} == {
        "loss": 9.0,
        "dL/dw": -18.0,
        "dL/db": -6.0,
        "dL/dx": -12.0,
    }
    assert all(ours == theirs for ours, theirs in numbers.values())


@pytest.mark.parametrize("group", ["gradient_checks", "optimizer_checks", "training_checks"])
def test_every_case_agrees_with_pytorch(check, group):
    checks = getattr(check, group)()
    assert checks
    assert [c.case for c in checks if not c.passed] == []


def test_the_negative_controls_are_caught(check):
    controls = check.control_checks()
    assert len(controls) == 2
    assert not any(c.passed for c in controls)
    assert min(c.max_diff for c in controls) > 1e-3


def test_the_compare_rule(check):
    assert check.compare([(1.0 + 1e-13, 1.0)]) == (pytest.approx(1e-13), True)
    assert check.compare([(1.0 + 1e-8, 1.0)])[1] is False
    assert check.compare([([1.0, 2.0], [1.0])]) == (float("inf"), False)
