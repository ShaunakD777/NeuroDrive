"""Guard for ADR-002: nothing under src/neurodrive imports an ML framework or the reference folder.

NeuroDrive implements its networks, gradients and optimizers itself. PyTorch may be used only in
reference/ to cross-check results, and src/ must never import that folder. The test reads every
source file's import statements (without running them), so it also catches an import that sits
inside a function or behind an ``if``.
"""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "neurodrive"
FORBIDDEN = ("torch", "tensorflow", "jax", "sklearn", "autograd", "keras", "reference")


def imported_modules(path: Path) -> list[str]:
    """Top-level names of every module imported in ``path`` (``a`` for ``import a.b``)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.append(node.module.split(".")[0])
    return names


def test_the_source_folder_is_found():
    assert len(list(SRC.rglob("*.py"))) > 20


def test_no_ml_framework_or_reference_import_in_src():
    offenders = [
        f"{path.relative_to(SRC.parent).as_posix()}: {name}"
        for path in sorted(SRC.rglob("*.py"))
        for name in imported_modules(path)
        if name in FORBIDDEN
    ]
    assert offenders == []


def test_the_scan_sees_nested_and_from_imports(tmp_path):
    sample = tmp_path / "sample.py"
    sample.write_text("import numpy as np\ndef f():\n    from torch.nn import Linear\n")
    assert imported_modules(sample) == ["numpy", "torch"]
