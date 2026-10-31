"""Reference cross-check: do our from-scratch gradients and optimizer steps match PyTorch's?

Run from the repository root (PyTorch is fetched for this run only; it is not a dependency):

    uv run --with torch python reference/torch_check.py

NeuroDrive computes gradients with hand-written backpropagation (core/nn/backward.py) and updates
weights with hand-written optimizers (core/optim). PyTorch computes the same gradients with
*automatic differentiation* (it records the forward pass and applies the chain rule itself) and has
its own SGD, Momentum, RMSprop and Adam. If both agree on many random cases, our maths and our
conventions (MSE without a 1/2, Huber with one, PyTorch-style momentum, epsilon after the square
root) are right. Four groups of checks:

1. Gradients: every hidden activation x every output activation x three losses x two batch shapes
   (96 random networks): the loss, the output, dL/dW, dL/db and dL/d(inputs).
2. Optimizer steps: each optimizer at two settings is fed the same 50 gradients (sizes from 1e-6
   to 100, so epsilon matters) as torch.optim; the parameters are compared after every step.
3. Training: a ToyLearner on sine and xor (full batch, 200 epochs; RMSProp 20) against a PyTorch
   loop that starts from the same weights and data; the loss of every epoch and the final weights.
   Over 200 epochs RMSProp amplifies rounding differences (see TRAIN_EPOCHS_RMSPROP); that drift
   is printed for information next to the drift between two orderings of our own RMSProp.
4. Negative controls: two deliberately wrong optimizers (Adam without bias correction, RMSProp
   with epsilon inside the square root) must FAIL check 2, or the check could not see a mistake.

Pass rule (NumPy's allclose): |ours - torch| <= ATOL + RTOL * |torch| for every number, in
float64 (about 16 significant digits), so only rounding differences are allowed. The exit code is
0 only when every check passes and every control fails.

This folder may import PyTorch; nothing under src/ may import it or this folder (ADR-002,
tests/test_from_scratch.py).
"""

import sys
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from neurodrive.core.losses import huber, huber_grad, mse, mse_grad
from neurodrive.core.nn.backward import backward
from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.optim import make_optimizer
from neurodrive.core.optim.adam import Adam
from neurodrive.core.optim.base import params_of
from neurodrive.core.optim.rmsprop import RMSProp
from neurodrive.core.rng import child_rng, make_rng
from neurodrive.learn.supervised.train_toy import ToyConfig, ToyLearner

RTOL = 1e-10  # allowed gap relative to PyTorch's number
ATOL = 1e-12  # allowed gap for numbers close to 0
SEED = 2026

ACTIVATIONS = ("tanh", "sigmoid", "relu", "linear")
TORCH_ACTIVATIONS = {
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "relu": torch.nn.ReLU,
    "linear": torch.nn.Identity,
}
# (name, Huber delta or None): the same loss function on both sides.
LOSSES = (("mse", None), ("huber", 1.0), ("huber", 0.5))
GRAD_LAYERS = (3, 5, 4, 2)
GRAD_BATCH = 8

# (optimizer, learning rate, settings): defaults and a non-default set for each.
OPTIMIZER_CASES = (
    ("sgd", 0.1, {}),
    ("sgd", 0.003, {}),
    ("momentum", 0.05, {"beta": 0.9}),
    ("momentum", 0.01, {"beta": 0.5}),
    ("rmsprop", 0.01, {"rho": 0.9, "eps": 1e-8}),
    ("rmsprop", 0.001, {"rho": 0.99, "eps": 1e-6}),
    ("adam", 0.01, {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}),
    ("adam", 0.1, {"beta1": 0.8, "beta2": 0.99, "eps": 1e-6}),
)
OPTIMIZER_SHAPES = ((4, 3), (4,), (2, 4), (2,))
OPTIMIZER_STEPS = 50

TRAIN_TASKS = ("sine", "xor")
TRAIN_RATES = {"sgd": 0.1, "momentum": 0.05, "rmsprop": 0.003, "adam": 0.01}
TRAIN_EPOCHS = 200
# RMSProp is compared strictly for its first 20 epochs only. Later its parameters jitter by steps
# of about lr · sign(g), and which way a step goes depends on the last digit of a gradient close to
# 0, so two runs of the *same* maths in a different rounding order drift apart by up to about lr.
# rounding_sensitivity() measures that drift; it is information, not a check.
TRAIN_EPOCHS_RMSPROP = 20
TRAIN_SEED = 7


@dataclass(frozen=True)
class Check:
    """One compared case: its group and name, the largest |ours - torch|, and whether it passed."""

    group: str
    case: str
    max_diff: float
    passed: bool


def compare(pairs: list[tuple[np.ndarray, np.ndarray]]) -> tuple[float, bool]:
    """The largest absolute difference over all (ours, torch) pairs, and whether all are close."""
    max_diff, passed = 0.0, True
    for ours, theirs in pairs:
        ours, theirs = np.asarray(ours, dtype=np.float64), np.asarray(theirs, dtype=np.float64)
        if ours.shape != theirs.shape:
            return float("inf"), False
        if ours.size:
            max_diff = max(max_diff, float(np.max(np.abs(ours - theirs))))
        passed = passed and bool(np.allclose(ours, theirs, rtol=RTOL, atol=ATOL))
    return max_diff, passed


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().numpy().copy()


def torch_model(network: Network) -> torch.nn.Sequential:
    """A PyTorch copy of ``network``: nn.Linear layers (weights (n_out, n_in), like ours)."""
    spec = network.spec
    layers: list[torch.nn.Module] = []
    for layer, (w, b) in enumerate(zip(network.weights, network.biases, strict=True)):
        n_out, n_in = w.shape
        linear = torch.nn.Linear(n_in, n_out, dtype=torch.float64)
        with torch.no_grad():
            linear.weight.copy_(torch.from_numpy(w))
            linear.bias.copy_(torch.from_numpy(b))
        layers += [linear, TORCH_ACTIVATIONS[spec.activation_name(layer)]()]
    return torch.nn.Sequential(*layers)


def linear_layers(model: torch.nn.Sequential) -> list[torch.nn.Linear]:
    return [m for m in model if isinstance(m, torch.nn.Linear)]


def our_loss(name: str, delta: float | None) -> tuple[Callable, Callable]:
    if name == "mse":
        return mse, mse_grad
    return (lambda p, t: huber(p, t, delta)), (lambda p, t: huber_grad(p, t, delta))


def torch_loss(name: str, delta: float | None, output: torch.Tensor, target: torch.Tensor):
    if name == "mse":
        return F.mse_loss(output, target)
    return F.huber_loss(output, target, delta=delta)


# --- 1. Gradients ---------------------------------------------------------------------------


def gradient_case(
    x: np.ndarray, y: np.ndarray, network: Network, loss: str, delta: float | None
) -> tuple[float, bool]:
    """Loss, output and every gradient of ``network`` on (x, y), ours against PyTorch's."""
    loss_fn, loss_grad = our_loss(loss, delta)
    trace = forward(network, x)
    ours_loss = loss_fn(trace.output, y)
    grads = backward(network, trace, loss_grad(trace.output, y))

    model = torch_model(network)
    x_t = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    out = model(x_t)
    torch_value = torch_loss(loss, delta, out, torch.from_numpy(y))
    torch_value.backward()
    layers = linear_layers(model)
    pairs = [(np.array(ours_loss), to_numpy(torch_value)), (trace.output, to_numpy(out))]
    for gw, gb, layer in zip(grads.weights, grads.biases, layers, strict=True):
        pairs += [(gw, to_numpy(layer.weight.grad)), (gb, to_numpy(layer.bias.grad))]
    pairs.append((grads.inputs, to_numpy(x_t.grad)))
    return compare(pairs)


def gradient_checks() -> list[Check]:
    """Group 1: 4 hidden x 4 output activations x 3 losses x (one example, a batch of 8)."""
    rng = make_rng(SEED)
    checks = []
    for hidden in ACTIVATIONS:
        for output in ACTIVATIONS:
            spec = NetworkSpec(GRAD_LAYERS, hidden_activation=hidden, output_activation=output)
            for loss, delta in LOSSES:
                for batch in (None, GRAD_BATCH):
                    loss_name = loss if delta is None else f"huber(d={delta})"
                    shape = "1 example" if batch is None else f"batch {batch}"
                    name = f"{hidden:>7} -> {output:<7} {loss_name:<12} {shape}"
                    case_rng = child_rng(rng, name)
                    network = init_network(spec, case_rng)
                    network = Network(
                        spec,
                        network.weights,
                        [case_rng.uniform(-0.5, 0.5, size=b.shape) for b in network.biases],
                    )
                    lead = () if batch is None else (batch,)
                    x = case_rng.normal(size=(*lead, spec.n_inputs))
                    y = case_rng.normal(size=(*lead, spec.n_outputs))
                    diff, ok = gradient_case(x, y, network, loss, delta)
                    checks.append(Check("gradients", name, diff, ok))
    return checks


def worked_example() -> dict[str, tuple[float, float]]:
    """The Brief's neuron: w = 2, b = 1, x = 3, y = 10, MSE. Returns {name: (ours, torch)}."""
    spec = NetworkSpec((1, 1), hidden_activation="linear", output_activation="linear")
    network = Network(spec, [np.array([[2.0]])], [np.array([1.0])])
    x, y = np.array([3.0]), np.array([10.0])
    trace = forward(network, x)
    grads = backward(network, trace, mse_grad(trace.output, y))
    model = torch_model(network)
    x_t = torch.tensor(x, requires_grad=True)
    out = model(x_t)
    value = F.mse_loss(out, torch.from_numpy(y))
    value.backward()
    layer = linear_layers(model)[0]
    return {
        "loss": (mse(trace.output, y), value.item()),
        "dL/dw": (float(grads.weights[0][0, 0]), layer.weight.grad.item()),
        "dL/db": (float(grads.biases[0][0]), layer.bias.grad.item()),
        "dL/dx": (float(grads.inputs[0]), x_t.grad.item()),
    }


# --- 2. Optimizer steps ---------------------------------------------------------------------


def torch_optimizer(name: str, params: list[torch.Tensor], lr: float, settings: dict):
    """The torch.optim counterpart of our optimizer ``name`` with the same settings."""
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if name == "momentum":  # dampening 0, no Nesterov: v = beta·v + g, theta -= lr·v
        return torch.optim.SGD(params, lr=lr, momentum=settings.get("beta", 0.9))
    if name == "rmsprop":  # PyTorch calls the decay rho "alpha"; epsilon after the square root
        return torch.optim.RMSprop(
            params, lr=lr, alpha=settings.get("rho", 0.9), eps=settings.get("eps", 1e-8)
        )
    if name == "adam":
        betas = (settings.get("beta1", 0.9), settings.get("beta2", 0.999))
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=settings.get("eps", 1e-8))
    raise ValueError(f"no PyTorch counterpart for {name!r}")


def step_case(optimizer, name: str, lr: float, settings: dict, rng) -> tuple[float, bool]:
    """Feed the same gradient sequence to ``optimizer`` and to torch.optim; compare every step."""
    params = [rng.normal(size=shape) for shape in OPTIMIZER_SHAPES]
    tensors = [torch.tensor(p, requires_grad=True) for p in params]
    reference = torch_optimizer(name, tensors, lr, settings)
    max_diff, passed = 0.0, True
    for _ in range(OPTIMIZER_STEPS):
        # Entry sizes spread over 1e-6 ... 1e2, so epsilon's placement and RMSProp's / Adam's
        # per-parameter scaling are both exercised.
        grads = [
            rng.normal(size=shape) * 10.0 ** rng.uniform(-6.0, 2.0, size=shape)
            for shape in OPTIMIZER_SHAPES
        ]
        params = optimizer.step(params, grads)
        for tensor, g in zip(tensors, grads, strict=True):
            tensor.grad = torch.from_numpy(g.copy())
        reference.step()
        diff, ok = compare([(p, to_numpy(t)) for p, t in zip(params, tensors, strict=True)])
        max_diff, passed = max(max_diff, diff), passed and ok
    return max_diff, passed


def settings_label(settings: dict) -> str:
    return ", ".join(f"{k} {v:g}" for k, v in settings.items()) or "no settings"


def optimizer_checks() -> list[Check]:
    """Group 2: every optimizer case, 50 shared gradients each."""
    rng = make_rng(SEED)
    checks = []
    for name, lr, settings in OPTIMIZER_CASES:
        label = f"{name:<8} lr {lr:g}, {settings_label(settings)}"
        optimizer = make_optimizer(name, lr, settings)
        diff, ok = step_case(optimizer, name, lr, settings, child_rng(rng, label))
        checks.append(Check("optimizer steps", label, diff, ok))
    return checks


# --- 3. Training ----------------------------------------------------------------------------


def learner_for(task: str, optimizer: str) -> ToyLearner:
    """A fresh full-batch ToyLearner on ``task`` with ``optimizer`` at its TRAIN_RATES rate."""
    lr = TRAIN_RATES[optimizer]
    config = ToyConfig.for_task(task, optimizer=optimizer, learning_rate=lr, epochs=TRAIN_EPOCHS)
    return ToyLearner(config, make_rng(TRAIN_SEED))


def train_pair(task: str, optimizer: str, epochs: int) -> tuple[ToyLearner, list[np.ndarray], list]:
    """Train a ToyLearner and a PyTorch loop from the same weights and data for ``epochs``.

    Returns the trained learner, PyTorch's final parameters (W0, b0, W1, b1, ...) and the pairs
    to compare: the loss before every epoch, then every parameter array.
    """
    learner = learner_for(task, optimizer)
    model = torch_model(learner.network)
    x, y = torch.from_numpy(learner.x_train), torch.from_numpy(learner.y_train)
    reference = torch_optimizer(optimizer, list(model.parameters()), TRAIN_RATES[optimizer], {})
    ours_losses, torch_losses = [], []
    for _ in range(epochs):
        ours_losses.append(learner.train_unit().metrics["loss_before"])
        reference.zero_grad()
        value = F.mse_loss(model(x), y)
        value.backward()
        reference.step()
        torch_losses.append(value.item())
    torch_params = [to_numpy(p) for p in model.parameters()]
    pairs = [(np.array(ours_losses), np.array(torch_losses))]
    pairs += list(zip(params_of(learner.network), torch_params, strict=True))
    return learner, torch_params, pairs


def training_epochs(optimizer: str) -> int:
    return TRAIN_EPOCHS_RMSPROP if optimizer == "rmsprop" else TRAIN_EPOCHS


def training_checks() -> list[Check]:
    """Group 3: sine and xor, each with all four optimizers."""
    checks = []
    for task in TRAIN_TASKS:
        for optimizer in TRAIN_RATES:
            epochs = training_epochs(optimizer)
            diff, ok = compare(train_pair(task, optimizer, epochs)[2])
            label = f"{task:<5} {optimizer:<8} lr {TRAIN_RATES[optimizer]:g}, {epochs} epochs"
            checks.append(Check("training", label, diff, ok))
    return checks


class RMSPropRearranged(RMSProp):
    """The same maths as RMSProp in another rounding order: theta - (lr / (sqrt(s) + eps)) * g."""

    def step(self, params, grads):
        super().step(params, grads)  # only to update s
        return [
            p - (self.learning_rate / (np.sqrt(s) + self.eps)) * g
            for p, g, s in zip(params, grads, self.square_averages, strict=True)
        ]


def rounding_sensitivity(task: str) -> tuple[float, float]:
    """RMSProp over TRAIN_EPOCHS: the largest parameter gap to PyTorch, and to our own RMSProp
    with the same maths in another rounding order. Similar sizes mean the gap is rounding that
    RMSProp amplifies, not a difference in the maths."""
    learner, torch_params, _ = train_pair(task, "rmsprop", TRAIN_EPOCHS)
    twin = learner_for(task, "rmsprop")
    twin.optimizer = RMSPropRearranged(TRAIN_RATES["rmsprop"])
    for _ in range(TRAIN_EPOCHS):
        twin.train_unit()
    ours = params_of(learner.network)
    to_torch = compare(list(zip(ours, torch_params, strict=True)))[0]
    to_twin = compare(list(zip(ours, params_of(twin.network), strict=True)))[0]
    return to_torch, to_twin


# --- 4. Negative controls -------------------------------------------------------------------


class AdamWithoutBiasCorrection(Adam):
    """Wrong on purpose: uses m and v as they are, without dividing by 1 - beta1^t, 1 - beta2^t."""

    def step(self, params, grads):
        super().step(params, grads)  # only to update m, v and t
        return [
            p - self.learning_rate * m / (np.sqrt(v) + self.eps)
            for p, m, v in zip(params, self.first_moments, self.second_moments, strict=True)
        ]


class RMSPropEpsInside(RMSProp):
    """Wrong on purpose: divides by sqrt(s + eps) instead of sqrt(s) + eps."""

    def step(self, params, grads):
        super().step(params, grads)  # only to update s
        return [
            p - self.learning_rate * g / np.sqrt(s + self.eps)
            for p, g, s in zip(params, grads, self.square_averages, strict=True)
        ]


def control_checks() -> list[Check]:
    """Group 4: the wrong variants against the correct PyTorch optimizers (must not pass)."""
    rng = make_rng(SEED)
    adam = {"beta1": 0.9, "beta2": 0.999, "eps": 1e-8}
    rmsprop = {"rho": 0.9, "eps": 1e-8}
    cases = (
        ("Adam without bias correction", AdamWithoutBiasCorrection(0.01, **adam), "adam", adam),
        ("RMSProp with eps inside the sqrt", RMSPropEpsInside(0.01, **rmsprop), "rmsprop", rmsprop),
    )
    checks = []
    for label, optimizer, name, settings in cases:
        diff, ok = step_case(optimizer, name, 0.01, settings, child_rng(rng, label))
        checks.append(Check("controls", label, diff, ok))
    return checks


# --- Running --------------------------------------------------------------------------------


def run_all() -> list[Check]:
    torch.manual_seed(0)  # nn.Linear draws starting weights; ours overwrite them, but stay tidy
    return gradient_checks() + optimizer_checks() + training_checks() + control_checks()


def verdict(check: Check) -> str:
    if check.group == "controls":
        return "FAIL (expected)" if not check.passed else "PASSED - control broken!"
    return "PASS" if check.passed else "FAIL"


def all_good(checks: list[Check]) -> bool:
    """Every real check passed and every control failed."""
    return all(c.passed != (c.group == "controls") for c in checks)


def main() -> int:
    print(f"PyTorch {torch.__version__}, NumPy {np.__version__}; float64; pass rule ", end="")
    print(f"|ours - torch| <= {ATOL:g} + {RTOL:g} * |torch|\n")
    print("Worked example (w = 2, b = 1, x = 3, y = 10, MSE):    ours        torch")
    for name, (ours, theirs) in worked_example().items():
        print(f"  {name:<6} {ours:>40.6g} {theirs:>12.6g}")

    checks = run_all()
    group = None
    for check in checks:
        if check.group != group:
            group = check.group
            print(f"\n{group}")
        print(f"  {check.case:<52} max |diff| {check.max_diff:9.2e}  {verdict(check)}")

    print(f"\nrounding sensitivity (information, not a check): RMSProp, {TRAIN_EPOCHS} epochs")
    for task in TRAIN_TASKS:
        to_torch, to_twin = rounding_sensitivity(task)
        print(
            f"  {task:<5} largest parameter gap: ours vs PyTorch {to_torch:.2e}, "
            f"ours vs ours in another rounding order {to_twin:.2e}"
        )

    print("\nSummary")
    for name in dict.fromkeys(c.group for c in checks):
        members = [c for c in checks if c.group == name]
        good = sum(c.passed != (name == "controls") for c in members)
        largest = max(c.max_diff for c in members)
        print(f"  {name:<16} {good}/{len(members)} as expected, largest |diff| {largest:.2e}")
    ok = all_good(checks)
    print("\nAll checks agree with PyTorch." if ok else "\nMISMATCH: see the lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
