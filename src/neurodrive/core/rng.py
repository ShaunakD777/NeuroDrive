"""Seeded random number generators: the only place NeuroDrive creates them.

A pseudo-random generator is a machine that produces numbers that look random but follow a fixed
recipe fixed by its starting number, the *seed*. Two generators built from the same seed produce
the same numbers in the same order, which is what makes every run reproducible.

Rules this module supports (CLAUDE.md §4):

* Every random number comes from an explicit generator object passed in as an argument.
* No code uses NumPy's global ``np.random.*`` functions or the ``random`` module.
* Each sub-component (weight initialisation, mutation, start positions, ...) gets its own
  *child stream*, so adding a random draw in one component never shifts the numbers another
  component sees.
"""

import numpy as np

# The type of a seeded generator. Other modules use this name in type hints, so that this file
# stays the only place in the code base that mentions NumPy's random module.
Rng = np.random.Generator


def make_rng(seed: int) -> np.random.Generator:
    """Make a generator from a master seed.

    ``seed`` (s) is a non-negative whole number such as 42. The same s always gives the same
    sequence of draws; a different s gives a different sequence. NumPy expands s into a
    well-mixed starting state through a *seed sequence*, which the generator keeps so that
    :func:`child_rng` can derive children from it later.
    """
    if isinstance(seed, bool) or not isinstance(seed, int | np.integer):
        raise TypeError(f"seed must be an int, got {type(seed).__name__}")
    if seed < 0:
        raise ValueError(f"seed must be non-negative, got {seed}")
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(int(seed))))


def child_rng(parent: np.random.Generator, name: str) -> np.random.Generator:
    """Derive an independent, named child generator from ``parent``.

    The child depends only on the parent's seed (and its place in the family tree) and on
    ``name``. It does not depend on how many numbers the parent has already produced, or on the
    order in which children are requested, and deriving it does not use up any of the parent's
    numbers. So ``child_rng(g, "init")`` always gives the same stream for the same master seed,
    and adding a new child called "crossover" never changes "init" or "mutation".

    Asking twice for the same name gives two generators that produce *identical* numbers. That
    is deliberate (it makes results independent of call order), so a component should keep and
    reuse its child rather than asking again and expecting fresh numbers. A child of a child
    works too: the family-tree key simply grows by one more name.

    ``parent`` must have been made by :func:`make_rng` or :func:`child_rng`.
    """
    if not isinstance(name, str) or name == "":
        raise ValueError("name must be a non-empty string")
    seed_seq = parent.bit_generator.seed_seq
    if not isinstance(seed_seq, np.random.SeedSequence):
        raise TypeError("parent must be a generator made by make_rng or child_rng")
    # The name becomes numbers: its length, then each byte. Putting the length first means two
    # different names can never produce the same key (for example "ab" + "c" versus "a" + "bc").
    raw = name.encode("utf-8")
    key = (len(raw), *raw)
    child_seq = np.random.SeedSequence(seed_seq.entropy, spawn_key=(*seed_seq.spawn_key, *key))
    return np.random.Generator(np.random.PCG64(child_seq))
