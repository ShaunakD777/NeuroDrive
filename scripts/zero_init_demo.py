"""Demo for task 2.4: why a network cannot start with all-zero (or all-equal) weights.

Run from the repository root:

    uv run python scripts/zero_init_demo.py

It builds three default [7, 16, 16, 2] networks and feeds each the same five observations:

1. zero weights: every hidden neuron outputs 0 for every input, so the two outputs (steer,
   throttle) are 0 whatever the car sees. The car would sit still.
2. every weight 0.1: the hidden neurons are not zero, but they are all exactly equal to each
   other, so a layer of 16 neurons does the work of one.
3. Xavier random weights: the hidden neurons differ, so each can come to specialise.

The number printed for each layer is the *spread*: the standard deviation across the 16 neurons
of that layer, averaged over the five inputs. Spread 0 means the neurons are indistinguishable.
"""

import numpy as np

from neurodrive.core.nn.forward import forward
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import child_rng, make_rng


def spread(layer_activations: np.ndarray) -> float:
    """Mean over inputs of the standard deviation across the neurons of one layer."""
    return float(layer_activations.std(axis=1).mean())


def main() -> None:
    spec = NetworkSpec()
    rng = make_rng(0)
    observations = child_rng(rng, "observations").uniform(-1.0, 1.0, size=(5, spec.n_inputs))
    equal = Network(
        spec,
        [np.full(spec.weight_shape(i), 0.1) for i in range(spec.n_layers)],
        [np.zeros(spec.bias_shape(i)) for i in range(spec.n_layers)],
    )
    networks = {
        "zero weights": init_network(spec, rng, "zeros"),
        "all weights 0.1": equal,
        "Xavier random": init_network(spec, child_rng(rng, "init"), "xavier"),
    }
    header = "outputs for the 5 inputs (steer, throttle)"
    print(f"{'network':<18}{'spread L1':>11}{'spread L2':>11}  {header}")
    for name, net in networks.items():
        trace = forward(net, observations)
        s1, s2 = spread(trace.activations[1]), spread(trace.activations[2])
        outputs = ", ".join(f"({a:+.3f}, {b:+.3f})" for a, b in trace.output)
        print(f"{name:<18}{s1:>11.4f}{s2:>11.4f}  {outputs}")
    print()
    print("Spread 0 = all 16 neurons in the layer are identical. Zero weights also give the")
    print("same output (0, 0) for every input: the car would not even react to a wall.")


if __name__ == "__main__":
    main()
