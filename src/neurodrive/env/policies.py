"""Policies: functions from what a car sees to what it does (ARCHITECTURE §6.10).

A *policy* maps the observations of N cars, shape (N, 7), to their actions, shape (N, 2):
column 0 is ``steer`` and column 1 is ``throttle``, both in [-1, 1] (positive steer turns right,
positive throttle accelerates). The world clips anything outside that range.

Here live the simple hand-made policies of Phase 1. Network-driven policies (a neuron, an MLP, a
population) are added in later tasks next to them; this is the one place in ``env`` allowed to
adapt ``core`` networks into policies.
"""

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from neurodrive.env.sensors import OBSERVATION_SIZE

Array = NDArray[np.float64]


@runtime_checkable
class Policy(Protocol):
    """Anything with an ``act`` method from (N, 7) observations to (N, 2) actions."""

    def act(self, observations: Array) -> Array:
        """Return the actions (steer, throttle) for every car, shape (N, 2)."""
        ...


def _check_observations(observations: Array) -> Array:
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim != 2 or obs.shape[1] != OBSERVATION_SIZE:
        raise ValueError(f"observations must have shape (N, {OBSERVATION_SIZE}), got {obs.shape}")
    return obs


def _clip_action(steer: float, throttle: float) -> tuple[float, float]:
    if not (np.isfinite(steer) and np.isfinite(throttle)):
        raise ValueError("steer and throttle must be finite numbers")
    return float(np.clip(steer, -1.0, 1.0)), float(np.clip(throttle, -1.0, 1.0))


class ConstantPolicy:
    """Gives every car the same fixed action, whatever it sees."""

    def __init__(self, steer: float = 0.0, throttle: float = 0.0) -> None:
        self.steer, self.throttle = _clip_action(steer, throttle)

    def act(self, observations: Array) -> Array:
        obs = _check_observations(observations)
        return np.tile([self.steer, self.throttle], (len(obs), 1))


class KeyboardPolicy:
    """A human driver: holds the current key-driven steer and throttle and gives it to every car.

    A caller (later the live session, when keyboard commands arrive from the browser) sets the
    values with :meth:`set_input`; the policy just reports whatever was set last. It starts with
    no keys pressed: steer 0 and throttle 0.
    """

    def __init__(self) -> None:
        self.steer = 0.0
        self.throttle = 0.0

    def set_input(self, steer: float, throttle: float) -> None:
        """Set the current steer and throttle, each clipped to [-1, 1]."""
        self.steer, self.throttle = _clip_action(steer, throttle)

    def act(self, observations: Array) -> Array:
        obs = _check_observations(observations)
        return np.tile([self.steer, self.throttle], (len(obs), 1))
