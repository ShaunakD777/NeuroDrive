"""Start a small random-search run and kill it on purpose (task 5.7, for tests/test_cli_resume.py).

Run as a separate Python process: ``python random_search_kill.py <config> <runs_dir> <kill>``.
It trains the config with ``train_random_search_run`` (as ``nd train`` does) and ends the process
with ``os._exit`` at the chosen moment of unit ``KILL_UNIT``, like pulling the plug:

* ``during_unit``: in the middle of the unit's training (its metrics line is never written);
* ``before_state``: after the unit's metrics line (and checkpoint, if any), before its state.
"""

import os
import sys

from neurodrive.experiment.config import load_config
from neurodrive.experiment.runner import train_random_search_run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.evolution.random_search import RandomSearchLearner

KILLED = 3  # the exit code of a killed run
KILL_UNIT = 4
KILLS = ("during_unit", "before_state")


def main(config_path: str, runs_dir: str, kill: str) -> None:
    original_train, original_save = RandomSearchLearner.train_unit, RunStore.save_state

    def train_unit(self):
        if self.units + 1 == KILL_UNIT and kill == "during_unit":
            os._exit(KILLED)
        return original_train(self)

    def save_state(self, state, runner):
        if runner["unit"] == KILL_UNIT and kill == "before_state":
            os._exit(KILLED)
        original_save(self, state, runner)

    RandomSearchLearner.train_unit = train_unit
    RunStore.save_state = save_state
    train_random_search_run(load_config(config_path), runs_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
