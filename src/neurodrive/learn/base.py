"""The Learner contract, version 0 (ADR-030): what every learning method offers the runner.

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

Version 1 (task 5.5) adds saving and restoring the learner's own state, for resuming. Wall-clock
time is measured by the runner in v0.
"""

from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from neurodrive.core.nn.network import Network

LEARNER_VERSION = 0

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
    """Anything with these members can be trained by the runner (learner v0)."""

    name: str
    feedback_kind: FeedbackKind

    def train_unit(self) -> UnitResult:
        """Advance one natural unit of learning and report it."""
        ...

    def current_networks(self) -> list[Network]:
        """The network(s) to checkpoint, the best first."""
        ...
