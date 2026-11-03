"""A tiny seeded learner for the runner v1 tests (task 5.5), and a way to kill it on purpose.

``WalkLearner`` takes a random walk on the weights of a [7, 16, 16, 2] driving network: each unit
it adds Gaussian noise (from its own generator) to the flat weight vector and keeps the result
if it is closer to a fixed random target vector. It reports a fixed number of fake environment
steps per unit, so budgets can be tested. Its state is the weights, the target and the
generator: everything it needs to continue exactly.

``main`` is run in a separate Python process by the kill tests: it trains a 12-unit run with
exams and ends the process with ``os._exit`` at a chosen point (``KILLS``; "before_mirror",
between the folder and the database, since task 5.5b). ``os._exit`` stops at once, without
``finally`` blocks or any clean-up, like pulling the plug.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.experiment.db import Mirror
from neurodrive.experiment.runner import Budget, run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.base import LearnerState, UnitResult, generator_state, set_generator_state

UNITS = 12
SEED = 5
KILLED = 3  # the exit code of a killed run
KILLS = {  # name -> (unit, where in that unit)
    "during_unit": (7, "train"),  # its metrics line is never written
    "before_state": (7, "state"),  # after the metrics line, checkpoint and exam; before its state
    "mid_state_write": (7, "half_state"),  # a half-written temporary state file is left behind
    "first_unit": (1, "train"),  # before any state was ever saved
    "before_mirror": (7, "mirror"),  # unit 7 complete in the folder, not yet in the database
}


class WalkLearner:
    name, feedback_kind = "walk", "none"

    def __init__(
        self, seed: int, steps_per_unit: int = 300, sleep_s: float = 0.0, kill_unit: int = 0
    ):
        self.rng = make_rng(seed)
        self.spec = NetworkSpec()
        self.network = init_network(self.spec, self.rng)
        self.target = self.rng.normal(size=self.network.n_parameters)
        self.units = 0
        self.steps_per_unit, self.sleep_s, self.kill_unit = steps_per_unit, sleep_s, kill_unit

    def fitness(self, genome: np.ndarray) -> float:
        return -float(np.linalg.norm(genome - self.target))

    def train_unit(self) -> UnitResult:
        self.units += 1
        if self.units == self.kill_unit:
            os._exit(KILLED)
        if self.sleep_s:
            time.sleep(self.sleep_s)
        genome = self.network.flatten()
        candidate = genome + self.rng.normal(scale=0.05, size=genome.shape)
        better = self.fitness(candidate) > self.fitness(genome)
        if better:
            self.network = Network.unflatten(self.spec, candidate)
        fitness = self.fitness(self.network.flatten())
        return UnitResult(
            metrics={"fitness": fitness, "accepted": float(better)}, env_steps=self.steps_per_unit
        )

    def current_networks(self) -> list[Network]:
        return [self.network]

    def state(self) -> LearnerState:
        return LearnerState(
            arrays={"genome": self.network.flatten(), "target": self.target},
            info={"units": self.units, "rng": generator_state(self.rng)},
        )

    def restore(self, state: LearnerState) -> None:
        self.network = Network.unflatten(self.spec, state.arrays["genome"])
        self.target = state.arrays["target"]
        self.units = state.info["units"]
        set_generator_state(self.rng, state.info["rng"])


def main(runs_dir: str, kill: str) -> None:
    """Start the 12-unit walk run with exams under ``runs_dir`` and kill it as ``KILLS[kill]``."""
    unit, where = KILLS[kill]
    store = RunStore.create(runs_dir, "walk", "kill", {"seed": SEED})
    learner = WalkLearner(SEED, kill_unit=unit if where == "train" else 0)
    original = RunStore.save_state

    def save_state(self, state, runner):
        if runner["unit"] == unit and where == "state":
            os._exit(KILLED)
        if runner["unit"] == unit and where == "half_state":
            (self.path / "state" / "latest.tmp.npz").write_bytes(b"half a file")
            os._exit(KILLED)
        original(self, state, runner)

    original_add = Mirror.add_unit

    def add_unit(self, store, record, checkpoints, exams):
        if record is not None and record["unit"] == unit and where == "mirror":
            os._exit(KILLED)
        original_add(self, store, record, checkpoints, exams)

    RunStore.save_state = save_state
    Mirror.add_unit = add_unit
    run(learner, store, Budget(units=UNITS), exams=True)


def run_killed(runs_dir: Path, kill: str) -> RunStore:
    """Run :func:`main` in a separate process and return the run folder it left behind."""
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(runs_dir), kill],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if result.returncode != KILLED:
        raise RuntimeError(f"the run was not killed (exit {result.returncode}): {result.stderr}")
    (folder,) = [p for p in runs_dir.iterdir() if p.is_dir()]
    return RunStore.open(folder)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
