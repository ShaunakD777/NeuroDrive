"""Saving and loading networks as ``.npz`` files (docs/ARCHITECTURE.md §6.8).

An ``.npz`` file is a zip of named NumPy arrays. A network file holds ``W0, b0, W1, b1, ...``
(layer by layer) plus ``spec``, the :class:`NetworkSpec` as a JSON string. Numbers are stored as
64-bit floats, so loading gives back exactly the numbers that were saved.
"""

from pathlib import Path

import numpy as np

from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec


def save_network(network: Network, path: str | Path) -> None:
    """Write ``network`` to ``path`` (the file name is used exactly as given)."""
    arrays: dict[str, np.ndarray] = {"spec": np.array(network.spec.to_json())}
    for layer, (w, b) in enumerate(zip(network.weights, network.biases, strict=True)):
        arrays[f"W{layer}"] = w
        arrays[f"b{layer}"] = b
    with open(path, "wb") as f:  # an open file stops NumPy from adding ".npz" to the name
        np.savez(f, **arrays)


def load_network(path: str | Path) -> Network:
    """Read a network written by :func:`save_network`.

    The spec comes from the file; every array must exist and match the spec's shapes, otherwise
    ``ValueError`` is raised. Pickled data is never loaded.
    """
    with np.load(path, allow_pickle=False) as data:
        if "spec" not in data.files:
            raise ValueError(f"{path} is not a network file: it has no 'spec' entry")
        spec = NetworkSpec.from_json(str(data["spec"]))
        weights, biases = [], []
        for layer in range(spec.n_layers):
            for name in (f"W{layer}", f"b{layer}"):
                if name not in data.files:
                    raise ValueError(f"{path} is missing array {name!r}")
            weights.append(data[f"W{layer}"])
            biases.append(data[f"b{layer}"])
    return Network(spec, weights, biases)  # checks the shapes against the spec
