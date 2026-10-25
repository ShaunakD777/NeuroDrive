"""Task 3.7 follow-up: plain gradient descent teaches the driving network to copy a teacher.

Run from the repository root:

    uv run python scripts/copy_the_neuron.py                 # about 10 minutes
    uv run python scripts/copy_the_neuron.py --epochs 100    # 2 to 3 minutes (most of it driving)

A preview of Phase 6 (imitation learning), with the only driver we have so far as the teacher:
the hand-wired neuron of study S1 (weights -12 / +12 on the far-left / far-right rays, tanh,
throttle 0.24), which finishes 2 laps of train_loop_01 from every start.

1. The teacher drives 20 seeded random-start episodes (seeds 0-19); every (observation, action)
   pair of a running car becomes a training example. 5 more episodes (seeds 1000-1004) give
   held-out examples the student never trains on.
2. The default driving network [7, 16, 16, 2] (tanh everywhere, Xavier, seed 0) is trained by
   full-batch gradient descent on the mean squared error: the same step as task 3.7.
3. At epochs 1, 10, 100, 500, 1000, 2000 (and the last) the student drives the 10 exam-like
   starts of study S1 and 10 new random starts (seeds 500-509), next to the teacher.

Everything is seeded, so the printed numbers are the same on every run.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from neurodrive.core.losses import mse, mse_grad
from neurodrive.core.nn.backward import backward
from neurodrive.core.nn.forward import forward, predict
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.env.episode import run_episode
from neurodrive.env.policies import MLPPolicy, NeuronPolicy, Policy
from neurodrive.env.track import Track
from neurodrive.env.world import Status, World, WorldConfig

ROOT = Path(__file__).resolve().parent.parent
TRACK = Track.from_json(ROOT / "tracks" / "train_loop_01.json")
TEACHER = NeuronPolicy((-12.0, 12.0), 0.0, "tanh", 0.24)
OUTCOMES = ("finished", "crashed", "stalled", "timeout")
REPORT_EPOCHS = (1, 10, 100, 500, 1000, 2000, 5000, 10000)


def drive(policy: Policy, seeds: range | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Drive ``policy``: the 10 exam-like starts (``seeds`` None) or one episode per seed.

    Returns the observations and actions of every step a car was running, and each car's outcome.
    """
    obs, act, outcomes = [], [], []
    for seed in [None] if seeds is None else seeds:
        world = World(TRACK, WorldConfig())
        if seed is None:
            s0 = [f * TRACK.length for _ in (0.0, 1.5) for f in (0.0, 0.2, 0.4, 0.6, 0.8)]
            world.reset(s0=s0, lateral=[lat for lat in (0.0, 1.5) for _ in range(5)])
        else:
            world.reset(rng=make_rng(seed), random_s0=True, lateral_jitter=1.5, heading_jitter=0.1)
        rec = run_episode(world, policy)
        n_steps = len(rec.action)  # frames have one more entry (the final state)
        alive = rec.alive[:n_steps]
        obs.append(rec.observation[:n_steps][alive])
        act.append(rec.action[alive])
        outcomes.extend(Status(int(s)).name.lower() for s in rec.status)
    return np.concatenate(obs), np.concatenate(act), outcomes


def tally(outcomes: list[str]) -> str:
    return ", ".join(f"{outcomes.count(o)} {o}" for o in OUTCOMES if outcomes.count(o))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.5)
    args = parser.parse_args()

    x_train, y_train, _ = drive(TEACHER, range(20))
    x_test, y_test, _ = drive(TEACHER, range(1000, 1005))
    print(f"teacher data: {len(x_train)} training and {len(x_test)} held-out examples")
    print(
        f"  steer {y_train[:, 0].min():+.3f} .. {y_train[:, 0].max():+.3f}, "
        f"throttle always {np.unique(y_train[:, 1])}"
    )

    spec = NetworkSpec()  # [7, 16, 16, 2], tanh everywhere: the driving network
    net = init_network(spec, make_rng(0))
    print(f"student {spec.layer_sizes}: {spec.n_parameters} parameters, lr {args.learning_rate}")
    start = time.perf_counter()
    snapshots: dict[int, Network] = {}
    for epoch in range(1, args.epochs + 1):
        trace = forward(net, x_train)
        grads = backward(net, trace, mse_grad(trace.output, y_train))
        lr = args.learning_rate
        net = Network(
            spec,
            [w - lr * g for w, g in zip(net.weights, grads.weights, strict=True)],
            [b - lr * g for b, g in zip(net.biases, grads.biases, strict=True)],
        )
        if epoch in REPORT_EPOCHS or epoch == args.epochs:
            held_out = predict(net, x_test)
            print(
                f"  epoch {epoch:>5}: train loss {mse(predict(net, x_train), y_train):.6f}, "
                f"held-out {mse(held_out, y_test):.6f}, "
                f"mean |steer error| {np.abs(held_out[:, 0] - y_test[:, 0]).mean():.4f}"
            )
            snapshots[epoch] = net
    print(f"  training took {time.perf_counter() - start:.0f} s")

    print("driving train_loop_01 (2 laps, 90 s): 10 exam-like starts | 10 new random starts")
    drivers = [("teacher (hand neuron)", TEACHER)]
    drivers += [(f"student, epoch {e}", MLPPolicy(n)) for e, n in snapshots.items()]
    for label, policy in drivers:
        exam = tally(drive(policy)[2])
        new = tally(drive(policy, range(500, 510))[2])
        print(f"  {label:<22} {exam:<22} | {new}")


if __name__ == "__main__":
    main()
