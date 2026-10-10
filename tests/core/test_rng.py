"""Tests for core/rng.py: repeatable seeds, independent stable child streams, no global RNG."""

import re
from pathlib import Path

import numpy as np
import pytest

from neurodrive.core.rng import child_rng, make_rng

SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "neurodrive"
RNG_FILE = SRC_ROOT / "core" / "rng.py"

# Anything that reaches NumPy's global generator or the standard-library random module.
BANNED_PATTERNS = [
    r"\bnp\.random\b",
    r"\bnumpy\.random\b",
    r"from\s+numpy\s+import\s+.*\brandom\b",
    r"^\s*import\s+random\b",
    r"^\s*from\s+random\s+import\b",
]


def draws(rng: np.random.Generator, n: int = 5) -> list[float]:
    return rng.random(n).tolist()


def test_same_seed_gives_identical_draws() -> None:
    assert draws(make_rng(42)) == draws(make_rng(42))


def test_different_seeds_give_different_draws() -> None:
    assert draws(make_rng(42)) != draws(make_rng(43))


def test_draws_are_between_zero_and_one() -> None:
    values = draws(make_rng(0), 1000)
    assert all(0.0 <= v < 1.0 for v in values)


def test_child_is_repeatable() -> None:
    assert draws(child_rng(make_rng(42), "init")) == draws(child_rng(make_rng(42), "init"))


def test_children_with_different_names_differ() -> None:
    parent = make_rng(42)
    assert draws(child_rng(parent, "init")) != draws(child_rng(parent, "mutation"))


def test_child_differs_from_parent_stream() -> None:
    assert draws(child_rng(make_rng(42), "init")) != draws(make_rng(42))


def test_children_of_different_seeds_differ() -> None:
    assert draws(child_rng(make_rng(42), "init")) != draws(child_rng(make_rng(43), "init"))


def test_child_does_not_depend_on_parent_usage() -> None:
    fresh = make_rng(42)
    used = make_rng(42)
    used.random(1000)
    assert draws(child_rng(fresh, "init")) == draws(child_rng(used, "init"))


def test_deriving_a_child_does_not_use_up_the_parent() -> None:
    touched = make_rng(42)
    child_rng(touched, "init")
    child_rng(touched, "mutation")
    assert draws(touched) == draws(make_rng(42))


def test_child_does_not_depend_on_request_order() -> None:
    a = make_rng(42)
    init_first = draws(child_rng(a, "init"))
    mutation_second = draws(child_rng(a, "mutation"))

    b = make_rng(42)
    mutation_first = draws(child_rng(b, "mutation"))
    init_second = draws(child_rng(b, "init"))

    assert init_first == init_second
    assert mutation_second == mutation_first


def test_same_name_twice_gives_identical_streams() -> None:
    parent = make_rng(42)
    assert draws(child_rng(parent, "init")) == draws(child_rng(parent, "init"))


def test_child_of_child_is_stable_and_distinct() -> None:
    def grandchild() -> np.random.Generator:
        return child_rng(child_rng(make_rng(42), "ga"), "mutation")

    assert draws(grandchild()) == draws(grandchild())
    assert draws(grandchild()) != draws(child_rng(make_rng(42), "mutation"))
    assert draws(grandchild()) != draws(child_rng(make_rng(42), "ga"))


def test_name_boundaries_cannot_collide() -> None:
    # "ab" then "c" must not equal "a" then "bc" (the length prefix guarantees this).
    root = make_rng(42)
    one = child_rng(child_rng(root, "ab"), "c")
    two = child_rng(child_rng(root, "a"), "bc")
    assert draws(one) != draws(two)


def test_make_rng_rejects_bad_seeds() -> None:
    with pytest.raises(ValueError):
        make_rng(-1)
    with pytest.raises(TypeError):
        make_rng(1.5)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        make_rng(True)


def test_child_rng_rejects_bad_names() -> None:
    parent = make_rng(42)
    with pytest.raises(ValueError):
        child_rng(parent, "")
    with pytest.raises(ValueError):
        child_rng(parent, 7)  # type: ignore[arg-type]


def find_banned_usage(root: Path, skip: Path) -> list[str]:
    problems = []
    for path in sorted(root.rglob("*.py")):
        if path == skip:
            continue
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            for pattern in BANNED_PATTERNS:
                if re.search(pattern, line):
                    problems.append(f"{path.relative_to(root)}:{number}: {line.strip()}")
    return problems


def test_no_global_random_in_source() -> None:
    assert find_banned_usage(SRC_ROOT, RNG_FILE) == []


def test_source_scan_catches_a_planted_violation(tmp_path: Path) -> None:
    (tmp_path / "bad.py").write_text("import numpy as np\nx = np.random.rand(3)\n")
    (tmp_path / "bad2.py").write_text("import random\n")
    found = find_banned_usage(tmp_path, skip=tmp_path / "nothing.py")
    assert len(found) == 2
