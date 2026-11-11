"""The genome: how evolution sees a network, and the gene map that says what each number means.

Evolution does not see layers, neurons or matrices. Each individual is one *genome*, a flat list
of P numbers, and each position in it is a *gene*. Mutation nudges genes, crossover copies each
gene from one parent or the other, and only when a car must drive is the list turned back into a
network (``Network.unflatten``; a whole generation at once with ``Population.from_genomes``). The
genome is the *genotype*; the network it builds is the *phenotype*. Here the step between them is
exact and reversible: one gene is exactly one weight or one bias.

The gene order is the network contract's (docs/ARCHITECTURE.md §6.8): for each layer l in turn,
the weight matrix W_l row by row, then the bias vector b_l. Symbols, for layer l with n_in inputs
and n_out neurons:

* s_l, the gene where layer l begins: s_0 = 0 and s_{l+1} = s_l + n_out * n_in + n_out;
* the weight W_l[j, m] (neuron j, input m) is gene i = s_l + j * n_in + m;
* the bias b_l[j] is gene i = s_l + n_out * n_in + j.

Going back: find the layer whose genes include i and let k = i - s_l. If k < n_out * n_in, gene
i is the weight in row k // n_in (whole-number division) and column k % n_in (the remainder);
otherwise it is the bias b_l[k - n_out * n_in].

Example, the default [7, 16, 16, 2] network (P = 434): W0 is genes 0-111, b0 112-127, W1
128-383, b1 384-399, W2 400-431 and b2 432-433. Gene 37 is in layer 0 with k = 37 = 5 * 7 + 2,
so it is W0[5, 2], the weight from input 2 (the straight-ahead ray) into hidden neuron 5.

A genome is a plain 1-D NumPy vector and a generation a plain (N, P) matrix, one genome per row:
mutation and crossover are arithmetic on whole arrays, and the network shape (the spec) travels
alongside, as in ``Population.from_genomes``. Gene names are generic ("W0[5,2]") so the map
works for every spec, toy networks included.

Permutation symmetry, the reason crossover is awkward for networks: swapping two hidden neurons
(their rows of W_l, their biases b_l and their columns of W_{l+1}) moves 2 * (n_in + 1 + n_next)
genes but leaves every output unchanged, so two parents that drive identically can have very
different genomes, and mixing them gene by gene can make a child that drives like neither.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from neurodrive.core.nn.init import DEFAULT_SCALE, init_population
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import Rng

GeneKind = Literal["weight", "bias"]


@dataclass(frozen=True)
class GeneAddress:
    """Where one gene lives in the network: a weight W_layer[row, column] or a bias b_layer[row].

    ``row`` is the neuron (counting from 0 in its layer); ``column`` is the input the weight
    reads, and ``None`` for a bias.
    """

    layer: int
    kind: GeneKind
    row: int
    column: int | None = None

    def __post_init__(self) -> None:
        if self.kind not in ("weight", "bias"):
            raise ValueError(f"kind must be 'weight' or 'bias', got {self.kind!r}")
        if (self.kind == "weight") != (self.column is not None):
            raise ValueError("a weight needs a column and a bias must not have one")

    @property
    def name(self) -> str:
        """Short name: "W0[5,2]" for a weight, "b2[1]" for a bias."""
        if self.kind == "weight":
            return f"W{self.layer}[{self.row},{self.column}]"
        return f"b{self.layer}[{self.row}]"


@dataclass(frozen=True)
class LayerGenes:
    """The genes of one layer: ``weights`` holds W_l (row by row), ``biases`` holds b_l."""

    weights: slice
    biases: slice


def layer_slices(spec: NetworkSpec) -> list[LayerGenes]:
    """For every layer in order, the range of genes holding its weights and its biases.

    Walk the layers keeping the start s (0 at first): the weights are genes s to
    s + n_out * n_in - 1, the biases the next n_out genes, and the next layer starts after them.
    Together the ranges cover genes 0 to P - 1 exactly once.
    """
    out, start = [], 0
    for layer in range(spec.n_layers):
        n_out, n_in = spec.weight_shape(layer)
        end_w = start + n_out * n_in
        out.append(LayerGenes(slice(start, end_w), slice(end_w, end_w + n_out)))
        start = end_w + n_out
    return out


def gene_address(spec: NetworkSpec, index: int) -> GeneAddress:
    """What gene ``index`` is: k = index - s_l in its layer; a weight in row k // n_in and
    column k % n_in while k < n_out * n_in, otherwise the bias of neuron k - n_out * n_in.

    Raises ``IndexError`` unless 0 <= index < spec.n_parameters.
    """
    if isinstance(index, bool) or not isinstance(index, int | np.integer):
        raise TypeError(f"a gene index must be a whole number, got {index!r}")
    if not 0 <= index < spec.n_parameters:
        raise IndexError(f"gene {index} out of range for {spec.n_parameters} genes")
    for layer, genes in enumerate(layer_slices(spec)):
        if index < genes.biases.stop:
            n_in = spec.weight_shape(layer)[1]
            k = int(index) - genes.weights.start
            if index < genes.weights.stop:
                return GeneAddress(layer, "weight", k // n_in, k % n_in)
            return GeneAddress(layer, "bias", int(index) - genes.biases.start)
    raise AssertionError("unreachable: the layer slices cover every gene")  # pragma: no cover


def gene_index(spec: NetworkSpec, address: GeneAddress) -> int:
    """The gene number of ``address``: s_l + row * n_in + column for a weight, and
    s_l + n_out * n_in + row for a bias (the exact reverse of :func:`gene_address`).

    Raises ``IndexError`` when the layer, row or column does not exist in ``spec``.
    """
    if not 0 <= address.layer < spec.n_layers:
        raise IndexError(f"layer {address.layer} out of range for {spec.n_layers} layers")
    n_out, n_in = spec.weight_shape(address.layer)
    if not 0 <= address.row < n_out:
        raise IndexError(f"layer {address.layer} has {n_out} neurons, not row {address.row}")
    genes = layer_slices(spec)[address.layer]
    if address.kind == "bias":
        return genes.biases.start + address.row
    if address.column is None or not 0 <= address.column < n_in:
        raise IndexError(f"layer {address.layer} has {n_in} inputs, not column {address.column}")
    return genes.weights.start + address.row * n_in + address.column


def genomes_of(networks: Sequence[Network]) -> NDArray[np.float64]:
    """Stack the networks' genomes into a matrix (N, P): row i is ``networks[i].flatten()``.

    All networks must share one spec (raises ``ValueError`` otherwise, or when there are none).
    """
    if len(networks) == 0:
        raise ValueError("need at least one network")
    spec = networks[0].spec
    if any(net.spec != spec for net in networks):
        raise ValueError("all networks must share one spec")
    return np.stack([net.flatten() for net in networks])


def networks_of(spec: NetworkSpec, genomes: ArrayLike) -> list[Network]:
    """Rebuild one network per row of a genome matrix (N, P) with ``Network.unflatten``.

    Raises ``ValueError`` unless ``genomes`` is 2-D with exactly ``spec.n_parameters`` columns.
    """
    g = np.asarray(genomes, dtype=np.float64)
    if g.ndim != 2 or g.shape[1] != spec.n_parameters:
        raise ValueError(f"genomes must have shape (N, {spec.n_parameters}), got {g.shape}")
    return [Network.unflatten(spec, row) for row in g]


def random_genomes(
    spec: NetworkSpec,
    n: int,
    rng: Rng,
    scheme: str = "xavier",
    scale: float = DEFAULT_SCALE,
) -> NDArray[np.float64]:
    """A random starting generation as a genome matrix (n, P).

    Draws n networks with the project's initialisation (``init_population``: each layer's
    weights from [-a, a] by ``scheme``, biases 0) from ``rng`` and returns their genomes, so
    the same generator state always gives the same matrix.
    """
    return init_population(spec, n, rng, scheme, scale).genomes()
