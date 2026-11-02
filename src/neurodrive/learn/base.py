"""The Learner contract, version 1 (ADR-030, ADR-037): what every learning method offers the runner.

A *learner* owns the learning method (its networks, data and rules); a *runner*
(``experiment/runner.py``) owns the loop around it: how many units to run, when to save
checkpoints, what to write to disk and when to stop. Keeping the two apart means every method,
from plain gradient descent to evolution, can be trained, saved and compared the same way.

Version 0 is the subset needed by the toy tasks (task 3.7). A learner provides:

* ``name``: a short identifier used in run-folder names, e.g. ``"toy_xor"``;
* ``feedback_kind``: what kind of feedback it learns from, one of ``labels`` (supervised targets),
  ``episode_score`` (one score per drive), ``step_reward`` (a reward every step) or ``none``;
* ``train_unit()``: advance by one natural unit (an epoch, a generation, a policy update) and
  return a :class:`UnitResult` with the unit's own metrics and its *budget increments* (how much
  driving, demonstration data and teacher help the unit used);
* ``current_networks()``: the network(s) to checkpoint, best first.

Version 1 (task 5.5, ADR-037) adds, optionally, saving and restoring the learner's whole state,
so a killed run can be resumed exactly (:class:`Resumable`):

* ``state()`` returns a :class:`LearnerState`: named NumPy arrays (weights, optimizer moments, a
  population) plus a small JSON-friendly ``info`` dict (counters, and the state of every random
  generator, from :func:`generator_state`);
* ``restore(state)`` puts the learner back exactly as it was, generators included, so the next
  unit draws the same random numbers as it would have without the interruption.

A learner without them still trains; it just cannot be resumed. Wall-clock time is measured by
the runner.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np

from neurodrive.core.nn.network import Network
from neurodrive.core.rng import Rng

LEARNER_VERSION = 1

FeedbackKind = Literal["labels", "episode_score", "step_reward", "none"]

Metric = float | list[float] | None


@dataclass(frozen=True)
class UnitResult:
    """What one unit of training reports.

    ``metrics``: the method's native measurements for this unit (e.g. ``loss``), JSON-friendly:
    numbers, lists of numbers, or ``None`` for a value that could not be computed.
    ``env_steps`` / ``demo_frames`` / ``teacher_queries``: budget used in this unit (0 for toys).
    ``diverged``: True when learning broke down (a loss or parameter became infinite or NaN);
    the runner then stops the run.
    """

    metrics: dict[str, Metric] = field(default_factory=dict)
    env_steps: int = 0
    demo_frames: int = 0
    teacher_queries: int = 0
    diverged: bool = False


@runtime_checkable
class Learner(Protocol):
    """Anything with these members can be trained by the runner (learner v0 and v1)."""

    name: str
    feedback_kind: FeedbackKind

    def train_unit(self) -> UnitResult:
        """Advance one natural unit of learning and report it."""
        ...

    def current_networks(self) -> list[Network]:
        """The network(s) to checkpoint, the best first."""
        ...


@dataclass
class LearnerState:
    """Everything a learner needs to continue exactly where it was.

    ``arrays``: named NumPy arrays (saved as they are, no pickling). ``info``: JSON-friendly values
    only (numbers, strings, lists, dicts, None), e.g. counters and :func:`generator_state` of each
    random generator.
    """

    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Resumable(Protocol):
    """A learner that can hand over and take back its whole state (learner v1)."""

    def state(self) -> LearnerState:
        """A snapshot of the learner after its latest unit."""
        ...

    def restore(self, state: LearnerState) -> None:
        """Become exactly the learner that produced ``state``."""
        ...


def generator_state(rng: Rng) -> dict[str, Any]:
    """The full state of a NumPy generator as plain JSON values (a few large integers)."""
    return rng.bit_generator.state


def set_generator_state(rng: Rng, state: dict[str, Any]) -> None:
    """Set ``rng`` to ``state``: its next draws are the ones that would have followed."""
    rng.bit_generator.state = state
