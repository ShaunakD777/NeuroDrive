"""Tests for `nd exam` (cli.py), task 5.4."""

from typer.testing import CliRunner

from neurodrive.cli import app
from neurodrive.core.nn.init import init_network
from neurodrive.core.nn.spec import NetworkSpec
from neurodrive.core.rng import make_rng
from neurodrive.env.policies import MLPPolicy
from neurodrive.experiment.exam import ExamResult, run_exam
from neurodrive.experiment.store import RunStore

runner = CliRunner()


def run_with(tmp_path, spec: NetworkSpec) -> RunStore:
    store = RunStore.create(tmp_path, "random", "exam", {"seed": 3})
    store.save_checkpoint(0, init_network(spec, make_rng(3)))
    return store


def test_exam_writes_the_card_into_the_run_folder_and_prints_it(tmp_path):
    store = run_with(tmp_path, NetworkSpec())  # a random [7, 16, 16, 2] driving network
    result = runner.invoke(app, ["exam", "--run", str(store.path), "--unit", "0"])
    assert result.exit_code == 0, result.output
    assert "HELD-OUT (pooled)" in result.stdout and "exams/unit_0.json" in result.stdout
    card = ExamResult.load(store.path / "exams" / "unit_0.json")
    assert card.policy == {"kind": "checkpoint", "run": store.path.name, "unit": 0}
    network = store.load_checkpoint(0)
    direct = run_exam(lambda world: MLPPolicy(network), card.policy)
    assert card.to_dict() == direct.to_dict()


def test_exam_refuses_a_toy_network_a_missing_unit_and_a_missing_folder(tmp_path):
    store = run_with(tmp_path, NetworkSpec(layer_sizes=[1, 4, 1]))  # a toy-task network
    toy = runner.invoke(app, ["exam", "--run", str(store.path), "--unit", "0"])
    assert toy.exit_code == 2 and "7 inputs and 2 outputs" in toy.output
    missing = runner.invoke(app, ["exam", "--run", str(store.path), "--unit", "5"])
    assert missing.exit_code == 2 and "no checkpoint for unit 5" in missing.output
    nowhere = runner.invoke(app, ["exam", "--run", str(tmp_path / "nope"), "--unit", "0"])
    assert nowhere.exit_code == 2 and "not a run folder" in nowhere.output
    assert not (store.path / "exams").exists()
