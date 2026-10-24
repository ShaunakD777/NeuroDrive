"""Train a network on a toy task and write a run folder (task 3.7; `nd train` replaces it in 4.6).

Run from the repository root, for example:

    uv run python scripts/train_toy.py --task xor
    uv run python scripts/train_toy.py --task sine --seed 3 --learning-rate 0.1 --epochs 5000
    uv run python scripts/train_toy.py --task sine --learning-rate 50 --tag too_fast

Tasks: line, sine, xor, two_spirals. Unset options take the task's defaults
(``ToyConfig.for_task``). The run folder appears under ``runs/`` (git-ignored); the script prints
its name, the final status and the loss at the start and at the end.
"""

import argparse

from neurodrive.core.rng import make_rng
from neurodrive.experiment.runner import run
from neurodrive.experiment.store import RunStore
from neurodrive.learn.supervised.toy_tasks import TOY_TASKS
from neurodrive.learn.supervised.train_toy import ToyConfig, ToyLearner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", required=True, choices=sorted(TOY_TASKS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--hidden-activation")
    parser.add_argument("--output-activation")
    parser.add_argument("--tag", default="default")
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    overrides = {"seed": args.seed, "tag": args.tag}
    for name in ("learning_rate", "epochs", "hidden_activation", "output_activation"):
        if getattr(args, name) is not None:
            overrides[name] = getattr(args, name)
    config = ToyConfig.for_task(args.task, **overrides)
    learner = ToyLearner(config, make_rng(config.seed))
    store = RunStore.create(args.runs_dir, learner.name, config.tag, config.to_dict())
    store.save_extra("dataset", x=learner.dataset.x, y=learner.dataset.y)
    start_loss = learner.loss()
    status = run(learner, store, config.epochs)
    metrics = store.read_metrics()
    print(f"run folder: {store.path.as_posix()}")
    print(f"status: {status} after {len(metrics)} epochs")
    finite = [m["loss"] for m in metrics if m["loss"] is not None]
    end = f"{finite[-1]:.6g}" if finite else "none (diverged in the first epoch)"
    print(f"loss: {start_loss:.6g} at the start -> {end} at the last finite epoch")


if __name__ == "__main__":
    main()
