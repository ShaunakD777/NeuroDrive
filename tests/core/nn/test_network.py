"""Tests for core/nn: the spec, parameter counts, flatten/unflatten and .npz save/load."""

import numpy as np
import pytest

from neurodrive.core.nn.io import load_network, save_network
from neurodrive.core.nn.network import Network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng

DEFAULT = NetworkSpec()
SMALL = NetworkSpec(layer_sizes=(2, 2, 1))


def random_network(spec: NetworkSpec, seed: int = 0) -> Network:
    rng = make_rng(seed)
    return Network.unflatten(spec, rng.standard_normal(spec.n_parameters))


def hand_network() -> Network:
    """The Brief's [2,2,1] example: genome order must read 1, 2, ..., 9."""
    return Network(SMALL, [[[1, 2], [3, 4]], [[7, 8]]], [[5, 6], [9]])


# --- spec ------------------------------------------------------------------------------------


def test_default_spec_matches_the_contract():
    assert DEFAULT.layer_sizes == (7, 16, 16, 2)
    assert (DEFAULT.hidden_activation, DEFAULT.output_activation) == ("tanh", "tanh")
    assert DEFAULT.version == "v1"
    assert (DEFAULT.n_layers, DEFAULT.n_inputs, DEFAULT.n_outputs) == (3, 7, 2)


def test_default_spec_has_434_parameters():
    assert DEFAULT.n_parameters == 128 + 272 + 34 == 434
    assert Network.zeros(DEFAULT).n_parameters == 434
    assert Network.zeros(DEFAULT).flatten().size == 434


def test_small_spec_count_and_shapes():
    assert SMALL.n_parameters == 9
    assert SMALL.weight_shape(0) == (2, 2)
    assert SMALL.weight_shape(1) == (1, 2)
    assert SMALL.bias_shape(1) == (1,)


def test_activation_per_layer():
    spec = NetworkSpec((3, 4, 4, 1), hidden_activation="relu", output_activation="linear")
    assert [spec.activation_name(i) for i in range(3)] == ["relu", "relu", "linear"]


def test_spec_accepts_a_list_and_is_immutable():
    spec = NetworkSpec(layer_sizes=[3, 2])  # type: ignore[arg-type]
    assert spec.layer_sizes == (3, 2)
    with pytest.raises(AttributeError):
        spec.hidden_activation = "relu"  # type: ignore[misc]


@pytest.mark.parametrize("sizes", [(7,), (), (7, 0, 2), (7, -1, 2), (7, 2.5, 2), (True, 2)])
def test_bad_layer_sizes_rejected(sizes):
    with pytest.raises(ValueError):
        NetworkSpec(layer_sizes=sizes)


def test_unknown_activation_and_version_rejected():
    with pytest.raises(ValueError, match="unknown activation"):
        NetworkSpec(hidden_activation="swish")
    with pytest.raises(ValueError, match="unknown activation"):
        NetworkSpec(output_activation="nope")
    with pytest.raises(ValueError, match="version"):
        NetworkSpec(version="v9")


def test_layer_index_out_of_range():
    with pytest.raises(IndexError):
        SMALL.weight_shape(2)
    with pytest.raises(IndexError):
        SMALL.activation_name(-1)


def test_spec_json_round_trip():
    spec = NetworkSpec((5, 3, 2), "relu", "sigmoid")
    assert NetworkSpec.from_json(spec.to_json()) == spec
    assert NetworkSpec.from_dict({"layer_sizes": [4, 2]}) == NetworkSpec((4, 2))


# --- network construction ---------------------------------------------------------------------


def test_zeros_network_has_right_shapes():
    net = Network.zeros(DEFAULT)
    assert [w.shape for w in net.weights] == [(16, 7), (16, 16), (2, 16)]
    assert [b.shape for b in net.biases] == [(16,), (16,), (2,)]
    assert not net.flatten().any()


def test_constructor_rejects_wrong_shapes_and_counts():
    with pytest.raises(ValueError, match="W0"):
        Network(SMALL, [np.zeros((2, 3)), np.zeros((1, 2))], [np.zeros(2), np.zeros(1)])
    with pytest.raises(ValueError, match="b1"):
        Network(SMALL, [np.zeros((2, 2)), np.zeros((1, 2))], [np.zeros(2), np.zeros(2)])
    with pytest.raises(ValueError, match="3 weight"):
        Network(SMALL, [np.zeros((2, 2))] * 3, [np.zeros(2)] * 3)


def test_constructor_copies_its_inputs():
    w0 = np.ones((2, 2))
    net = Network(SMALL, [w0, np.zeros((1, 2))], [np.zeros(2), np.zeros(1)])
    w0[0, 0] = 99.0
    assert net.weights[0][0, 0] == 1.0


# --- flatten / unflatten ----------------------------------------------------------------------


def test_flatten_order_matches_hand_example():
    np.testing.assert_array_equal(hand_network().flatten(), np.arange(1.0, 10.0))


def test_unflatten_hand_example():
    net = Network.unflatten(SMALL, np.arange(1.0, 10.0))
    np.testing.assert_array_equal(net.weights[0], [[1, 2], [3, 4]])
    np.testing.assert_array_equal(net.biases[0], [5, 6])
    np.testing.assert_array_equal(net.weights[1], [[7, 8]])
    np.testing.assert_array_equal(net.biases[1], [9])


@pytest.mark.parametrize("spec", [DEFAULT, SMALL, NetworkSpec((3, 5, 4, 2, 1), "relu", "linear")])
def test_flatten_then_unflatten_gives_the_same_network(spec):
    net = random_network(spec, seed=3)
    again = Network.unflatten(spec, net.flatten())
    assert again.equals(net)
    for a, b in zip(again.weights + again.biases, net.weights + net.biases, strict=True):
        np.testing.assert_array_equal(a, b)


def test_unflatten_then_flatten_gives_the_same_vector():
    v = make_rng(5).standard_normal(DEFAULT.n_parameters)
    np.testing.assert_array_equal(Network.unflatten(DEFAULT, v).flatten(), v)


def test_unflatten_rejects_wrong_length_and_shape():
    with pytest.raises(ValueError, match="434"):
        Network.unflatten(DEFAULT, np.zeros(433))
    with pytest.raises(ValueError):
        Network.unflatten(SMALL, np.zeros((3, 3)))


def test_flatten_and_unflatten_do_not_share_memory():
    net = random_network(SMALL)
    v = net.flatten()
    v[:] = 0.0
    assert net.flatten().any()
    v2 = np.arange(9.0)
    rebuilt = Network.unflatten(SMALL, v2)
    v2[:] = -1.0
    np.testing.assert_array_equal(rebuilt.flatten(), np.arange(9.0))


def test_clone_is_independent_and_equal():
    net = random_network(DEFAULT, seed=1)
    twin = net.clone()
    assert twin.equals(net)
    twin.weights[0][0, 0] += 1.0
    assert not twin.equals(net)


def test_equals_checks_the_spec_too():
    a = Network.zeros(NetworkSpec((2, 2, 1)))
    b = Network.zeros(NetworkSpec((2, 2, 1), hidden_activation="relu"))
    assert not a.equals(b)


# --- save / load ------------------------------------------------------------------------------


def test_save_load_round_trip_is_exact(tmp_path):
    net = random_network(DEFAULT, seed=7)
    path = tmp_path / "net.npz"
    save_network(net, path)
    loaded = load_network(path)
    assert loaded.spec == net.spec
    assert loaded.equals(net)
    assert loaded.flatten().tobytes() == net.flatten().tobytes()


def test_save_keeps_the_file_name_and_stores_contract_arrays(tmp_path):
    path = tmp_path / "unit_3.npz"
    save_network(random_network(SMALL), path)
    assert [p.name for p in tmp_path.iterdir()] == ["unit_3.npz"]
    with np.load(path, allow_pickle=False) as data:
        assert sorted(data.files) == ["W0", "W1", "b0", "b1", "spec"]
        assert NetworkSpec.from_json(str(data["spec"])) == SMALL


def test_round_trip_keeps_non_default_spec(tmp_path):
    spec = NetworkSpec((4, 3, 2), "relu", "sigmoid")
    net = random_network(spec)
    save_network(net, tmp_path / "x.npz")
    assert load_network(tmp_path / "x.npz").spec == spec


def test_load_rejects_a_file_without_spec(tmp_path):
    np.savez(tmp_path / "bad.npz", W0=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="spec"):
        load_network(tmp_path / "bad.npz")


def test_load_rejects_missing_or_misshapen_arrays(tmp_path):
    spec_json = np.array(SMALL.to_json())
    np.savez(tmp_path / "missing.npz", spec=spec_json, W0=np.zeros((2, 2)), b0=np.zeros(2))
    with pytest.raises(ValueError, match="W1"):
        load_network(tmp_path / "missing.npz")
    np.savez(
        tmp_path / "shape.npz",
        spec=spec_json,
        W0=np.zeros((3, 3)),
        b0=np.zeros(2),
        W1=np.zeros((1, 2)),
        b1=np.zeros(1),
    )
    with pytest.raises(ValueError, match="W0"):
        load_network(tmp_path / "shape.npz")
