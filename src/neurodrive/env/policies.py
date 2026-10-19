"""Policies: functions from what a car sees to what it does (ARCHITECTURE §6.10).

A *policy* maps the observations of N cars, shape (N, 7), to their actions, shape (N, 2):
column 0 is ``steer`` and column 1 is ``throttle``, both in [-1, 1] (positive steer turns right,
positive throttle accelerates). The world clips anything outside that range.

Here live the simple hand-made policies of Phase 1 and the network-driven ones (a neuron, an MLP
shared by all cars, a population with one network per car); this is the one place in ``env`` allowed
to adapt ``core`` networks into policies.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from neurodrive.core.activations import ACTIVATIONS
from neurodrive.core.neuron import Neuron
from neurodrive.core.nn.forward import ForwardTrace, forward
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.population import Population
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.env.car import ACTION_SIZE
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


PARAM_LIMIT = 30.0  # weights and bias are limited to +-30 (a sanity bound, far above useful values)
NEURON_INPUTS = (0, 4)  # observation entries the neuron reads: far-left and far-right rays
PARAM_NAMES = ("w_left", "w_right", "bias", "throttle", "activation")


@dataclass(frozen=True)
class NeuronReadout:
    """What one car's neuron saw and computed in one step (for display).

    ``inputs`` are the readings x_left and x_right in [0, 1]; ``contributions`` are weight times
    input for each; ``z`` is their total plus the bias; ``output`` is the activation of ``z``.
    """

    inputs: tuple[float, float]
    input_rays: tuple[int, int]
    weights: tuple[float, float]
    bias: float
    contributions: tuple[float, float]
    z: float
    output: float
    activation: str
    throttle: float


class NeuronPolicy:
    """The Phase 1 driver: one neuron steers, the throttle is fixed.

    For every car, with d_left and d_right the far-left (+60 degrees) and far-right (-60 degrees)
    ray readings from the observation (entries 0 and 4):

        z     = w_left * d_left + w_right * d_right + bias      (the weighted sum)
        steer = f(z)                                              (f is the activation)
        throttle = a fixed number

    Positive steer turns right. All cars share the one neuron. It starts with every weight and
    the bias at 0, so the car drives straight until the weights are changed. A wall close on the
    left (small d_left) calls for a right turn, so a useful w_left is negative and w_right positive.
    """

    def __init__(
        self,
        weights: tuple[float, float] = (0.0, 0.0),
        bias: float = 0.0,
        activation: str = "tanh",
        throttle: float = 0.15,
    ) -> None:
        self.neuron = Neuron(weights, bias, activation)
        self.throttle = _clip_action(0.0, throttle)[1]

    def act(self, observations: Array) -> Array:
        obs = _check_observations(observations)
        steer = self.neuron.forward(obs[:, NEURON_INPUTS])
        return np.column_stack([steer, np.full(len(obs), self.throttle)])

    def set_param(self, name: str, value: float | str) -> None:
        """Change one parameter: ``w_left``, ``w_right``, ``bias``, ``throttle`` or ``activation``.

        Numbers must be finite; weights and bias are limited to +-30 and the throttle to [-1, 1].
        ``activation`` takes the name of one of the four activation functions. Raises
        ``ValueError`` for anything else and then changes nothing.
        """
        if name == "activation":
            if not isinstance(value, str) or value not in ACTIVATIONS:
                raise ValueError(f"activation must be one of {', '.join(sorted(ACTIVATIONS))}")
            self.neuron = Neuron(self.neuron.weights, self.neuron.bias, value)
            return
        if name not in PARAM_NAMES:
            raise ValueError(f"unknown parameter {name!r}; valid: {', '.join(PARAM_NAMES)}")
        if isinstance(value, str | bool) or not np.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        number = float(value)
        if name == "throttle":
            if not -1.0 <= number <= 1.0:
                raise ValueError("throttle must be between -1 and 1")
            self.throttle = number
            return
        if abs(number) > PARAM_LIMIT:
            raise ValueError(f"{name} must be between -{PARAM_LIMIT:g} and {PARAM_LIMIT:g}")
        weights = self.neuron.weights.copy()
        bias = self.neuron.bias
        if name == "w_left":
            weights[0] = number
        elif name == "w_right":
            weights[1] = number
        else:
            bias = number
        self.neuron = Neuron(weights, bias, self.neuron.activation)

    def readout(self, observation: Array) -> NeuronReadout:
        """The neuron's inputs, contributions, weighted sum and output for one car's observation.

        ``observation`` is that car's 7 numbers.
        """
        obs = np.asarray(observation, dtype=np.float64)
        if obs.shape != (OBSERVATION_SIZE,):
            raise ValueError(f"observation must have shape ({OBSERVATION_SIZE},), got {obs.shape}")
        x = obs[list(NEURON_INPUTS)]
        z, a = self.neuron.forward_with_sum(x)
        w = self.neuron.weights
        return NeuronReadout(
            inputs=(float(x[0]), float(x[1])),
            input_rays=NEURON_INPUTS,
            weights=(float(w[0]), float(w[1])),
            bias=self.neuron.bias,
            contributions=(float(w[0] * x[0]), float(w[1] * x[1])),
            z=float(z),
            output=float(a),
            activation=self.neuron.activation,
            throttle=self.throttle,
        )


def _check_spec(spec: NetworkSpec) -> None:
    """A driving network reads the 7-number observation and writes the 2-number action."""
    if spec.n_inputs != OBSERVATION_SIZE or spec.n_outputs != ACTION_SIZE:
        raise ValueError(
            f"a driving network needs {OBSERVATION_SIZE} inputs and {ACTION_SIZE} outputs, "
            f"got layer sizes {spec.layer_sizes}"
        )


def _car_trace(trace: ForwardTrace, index: int) -> ForwardTrace:
    """One car's slice (row ``index``) of a trace that holds every car."""
    n_cars = trace.activations[0].shape[0]
    if not 0 <= index < n_cars:
        raise IndexError(f"car {index} out of range for {n_cars} cars")
    return ForwardTrace(
        [a[index].copy() for a in trace.activations],
        [z[index].copy() for z in trace.weighted_sums],
    )


class MLPPolicy:
    """One multilayer network that drives every car (all cars share the same brain).

    For observations ``o`` of shape (N, 7), the actions are the network's output, shape (N, 2):
    column 0 is steer and column 1 is throttle. The network is the one passed in (not copied), so
    changing its weights changes the driving. ``last_trace`` is the full :class:`ForwardTrace` of
    the latest :meth:`act` call (every car, shape (N, width) per entry), and :meth:`trace_for`
    cuts out one car's trace for the Network view. Both are ``None`` until the first call.
    """

    def __init__(self, network: Network) -> None:
        _check_spec(network.spec)
        self.network = network
        self.last_trace: ForwardTrace | None = None

    def act(self, observations: Array) -> Array:
        obs = _check_observations(observations)
        self.last_trace = forward(self.network, obs)
        return self.last_trace.output

    def trace_for(self, car: int) -> ForwardTrace:
        """The latest forward trace of car ``car`` (shape (width,) per entry)."""
        if self.last_trace is None:
            raise RuntimeError("no trace yet: call act() first")
        return _car_trace(self.last_trace, car)


class PopulationPolicy:
    """A different network for every car: car i is driven by member i of the population.

    ``observations`` must have exactly one row per member, so the number of cars equals the
    population size. Row i goes through network i only. Traces work as in :class:`MLPPolicy`.
    """

    def __init__(self, population: Population) -> None:
        _check_spec(population.spec)
        self.population = population
        self.last_trace: ForwardTrace | None = None

    def act(self, observations: Array) -> Array:
        obs = _check_observations(observations)
        if len(obs) != self.population.size:
            raise ValueError(
                f"the population has {self.population.size} networks but got {len(obs)} cars"
            )
        self.last_trace = self.population.forward(obs)
        return self.last_trace.output

    def trace_for(self, car: int) -> ForwardTrace:
        """The latest forward trace of car ``car`` (shape (width,) per entry)."""
        if self.last_trace is None:
            raise RuntimeError("no trace yet: call act() first")
        return _car_trace(self.last_trace, car)
