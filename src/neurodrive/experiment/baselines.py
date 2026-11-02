"""Baseline runs (task 5.6, ADR-039): the reference drivers every learned method is compared with.

Three baselines, each examined with the Driving Exam and stored as a run folder with no training
(EXPERIMENT_PROTOCOL rule 6):

- ``random_network``: the untrained [7, 16, 16, 2] tanh network with Xavier weights drawn from
  ``make_rng(seed)``, the network a learner with that seed starts from. The floor: what you get
  with no learning. One seed is luck, so it is examined for several seeds that share a sweep id.
- ``hand_neuron``: Study S1's hand-wired neuron (``configs/baselines/hand_neuron.yaml``):
  steer = tanh(w_left · d_left + w_right · d_right + bias) with d_left, d_right the far-left
  and far-right rays, throttle fixed. The cheap human solution.
- ``scripted_expert``: the pure-pursuit expert of task 5.3. It reads the track itself, which no
  learner can, so it is the ceiling within these physics.

A baseline run (run folder v1) has budget 0 units, an empty metrics.jsonl, status completed and
one exam card ``exams/unit_0.json``. Its ``config.yaml`` describes the policy completely, so
:func:`baseline_policy` rebuilds it exactly. Only the random network is a network, so only it
has ``checkpoints/unit_0.npz`` and a card that refers to that checkpoint; the neuron's and the
expert's cards refer to the baseline itself: ``{"kind": "baseline", "name": ..., "run": ...}``.
"""

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.env.experts import ExpertParams, ScriptedExpert
from neurodrive.env.policies import MLPPolicy, NeuronPolicy, Policy
from neurodrive.env.world import World
from neurodrive.experiment.db import DB_NAME, Mirror
from neurodrive.experiment.exam import DEFAULT_TRACKS_DIR, ExamResult, run_exam
from neurodrive.experiment.store import RunStore

BASELINES = ("random_network", "hand_neuron", "scripted_expert")
DEFAULT_SEEDS = tuple(range(10))  # random networks: 10 seeds, the protocol's headline number
HAND_NEURON_FILE = (
    Path(__file__).resolve().parents[3] / "configs" / "baselines" / "hand_neuron.yaml"
)
RANDOM_INIT = "xavier"
_NEURON_FIELDS = ("w_left", "w_right", "bias", "activation", "throttle")
_NEURON_INPUTS = ["far_left_ray", "far_right_ray"]  # observation entries 0 and 4

PolicyFactory = Callable[[World], Policy]


def load_hand_neuron(path: str | Path = HAND_NEURON_FILE) -> dict[str, Any]:
    """The hand-wired neuron's parameters from its YAML file, checked.

    Returns ``{"w_left", "w_right", "bias", "activation", "throttle"}``. Raises ``ValueError``
    if the file is not a neuron baseline, reads other inputs, or misses a field.
    """
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("policy") != "neuron":
        raise ValueError(f"{Path(path).name}: not a neuron baseline (policy: neuron)")
    if data.get("inputs") != _NEURON_INPUTS:
        raise ValueError(f"{Path(path).name}: inputs must be {_NEURON_INPUTS}")
    neuron = data.get("neuron") or {}
    missing = [f for f in _NEURON_FIELDS if f not in neuron]
    if missing:
        raise ValueError(f"{Path(path).name}: neuron is missing {', '.join(missing)}")
    params = {f: neuron[f] for f in _NEURON_FIELDS}
    for f in ("w_left", "w_right", "bias", "throttle"):
        params[f] = float(params[f])
    return params


def baseline_config(
    name: str, seed: int | None = None, neuron_file: str | Path = HAND_NEURON_FILE
) -> dict[str, Any]:
    """The complete description of a baseline, written as the run's config.yaml."""
    if name not in BASELINES:
        raise ValueError(f"unknown baseline {name!r}; choose from {', '.join(BASELINES)}")
    if (seed is None) == (name == "random_network"):
        raise ValueError(
            "a random network needs a seed" if seed is None else f"{name} takes no seed"
        )
    if name == "random_network":
        spec = NetworkSpec()
        network = {**spec.to_dict(), "init_scheme": RANDOM_INIT}
        return {"method": name, "seed": seed, "network": network}
    if name == "hand_neuron":
        return {
            "method": name,
            "source": f"configs/baselines/{Path(neuron_file).name}",
            "inputs": _NEURON_INPUTS,
            "neuron": load_hand_neuron(neuron_file),
        }
    return {"method": name, "expert": asdict(ExpertParams())}


def random_network(config: dict[str, Any]) -> Network:
    """The random network a ``random_network`` config describes: drawn from make_rng(seed)."""
    net = dict(config["network"])
    scheme = net.pop("init_scheme")
    return init_network(NetworkSpec.from_dict(net), make_rng(config["seed"]), scheme)


def baseline_policy(config: dict[str, Any]) -> PolicyFactory:
    """Rebuild a baseline's policy from its config: ``world -> policy``, as the exam takes it."""
    name = config["method"]
    if name == "random_network":
        network = random_network(config)
        return lambda world: MLPPolicy(network)
    if name == "hand_neuron":
        n = config["neuron"]
        weights = (n["w_left"], n["w_right"])
        return lambda world: NeuronPolicy(weights, n["bias"], n["activation"], n["throttle"])
    if name == "scripted_expert":
        params = ExpertParams(**config["expert"])
        return lambda world: ScriptedExpert(world, params)
    raise ValueError(f"unknown baseline {name!r}")


@dataclass(frozen=True)
class BaselineRun:
    """One baseline run: its folder and its exam card."""

    store: RunStore
    card: ExamResult


def make_baseline_run(
    name: str,
    runs_dir: str | Path,
    seed: int | None = None,
    sweep: str | None = None,
    tracks_dir: str | Path = DEFAULT_TRACKS_DIR,
    neuron_file: str | Path = HAND_NEURON_FILE,
) -> BaselineRun:
    """Examine one baseline and store it as a run folder under ``runs_dir``.

    Steps: write the config; record a budget of 0 units (nothing is trained); save the random
    network as checkpoint 0; take the Driving Exam and save the card as exams/unit_0.json; mark
    the run completed; copy it into the database mirror ``<runs_dir>/neurodrive.db``.
    """
    config = baseline_config(name, seed, neuron_file)
    tag = "baseline" if seed is None else f"baseline-s{seed}"
    store = RunStore.create(
        runs_dir, name, tag, config, extra_meta={"sweep": sweep} if sweep else None
    )
    budget = {"units": 0, "env_steps": None, "wall_clock_s": None}
    store.update_meta(budget=budget, exams=True, stop_reason=None, resumed_at=[])
    if name == "random_network":
        store.save_checkpoint(0, random_network(config))
        ref = {"kind": "checkpoint", "run": store.path.name, "unit": 0}
    else:
        ref = {"kind": "baseline", "name": name, "run": store.path.name}
    started = time.perf_counter()
    card = run_exam(baseline_policy(config), ref, tracks_dir)
    store.save_exam(0, card)
    store.update_meta(exam_wall_clock_s=time.perf_counter() - started)
    store.finish("completed", "units")
    mirror = Mirror(Path(runs_dir) / DB_NAME)
    mirror.sync_run(store)
    mirror.close()
    return BaselineRun(store, card)


def make_baselines(
    runs_dir: str | Path,
    names: tuple[str, ...] = BASELINES,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    when: datetime | None = None,
    tracks_dir: str | Path = DEFAULT_TRACKS_DIR,
) -> list[BaselineRun]:
    """Make every baseline in ``names``: one run per seed for the random network (sharing the
    sweep id ``<YYYYMMDD-HHMMSS>-random_network-baseline``), one run for each other baseline."""
    when = when or datetime.now(UTC)
    runs = []
    for name in names:
        if name == "random_network":
            sweep = f"{when:%Y%m%d-%H%M%S}-random_network-baseline"
            runs += [
                make_baseline_run(name, runs_dir, s, sweep, tracks_dir=tracks_dir) for s in seeds
            ]
        else:
            runs.append(make_baseline_run(name, runs_dir, tracks_dir=tracks_dir))
    return runs
