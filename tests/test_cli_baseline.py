"""Tests for `nd baseline` (cli.py), task 5.6. Only random networks are examined here (their
episodes end fast); the neuron and the expert are covered in tests/experiment/test_baselines.py."""

from typer.testing import CliRunner

from neurodrive.cli import app
from neurodrive.experiment.db import check
from neurodrive.experiment.store import RunStore

runner = CliRunner()


def test_random_network_baselines_for_two_seeds(tmp_path):
    result = runner.invoke(
        app, ["baseline", "--name", "random_network", "--seeds", "0-1", "--runs-dir", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    folders = sorted(p for p in tmp_path.iterdir() if p.is_dir())
    assert [f.name.rsplit("-", 1)[1] for f in folders] == ["s0", "s1"]
    for folder in folders:
        assert f"{folder.name}  completion: training 0.00, held-out 0.00" in result.stdout
        assert RunStore.open(folder).exam_units() == [0]
    assert "random networks, held-out completion: median 0.000" in result.stdout
    assert "over 2 seeds" in result.stdout
    assert check(tmp_path) == []


def test_a_study_tags_every_baseline_run(tmp_path):
    args = ["baseline", "--name", "random_network", "--seeds", "3,5", "--study", "S4"]
    result = runner.invoke(app, [*args, "--runs-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    metas = [RunStore.open(p).read_meta() for p in tmp_path.iterdir() if p.is_dir()]
    assert [m["study"] for m in metas] == ["S4", "S4"]
    assert len({m["sweep"] for m in metas}) == 1  # the seeds still share one sweep
    assert check(tmp_path) == []


def test_bad_names_and_seeds_are_refused(tmp_path):
    for args, message in (
        (["--study", "S 4"], "study must be letters"),
        (["--name", "nope"], "unknown baseline 'nope'"),
        (["--name", "hand_neuron", "--seeds", "3"], "--seeds is only for random_network"),
        (["--name", "random_network", "--seeds", "2-1"], "bad seed range"),
    ):
        result = runner.invoke(app, ["baseline", *args, "--runs-dir", str(tmp_path)])
        assert result.exit_code == 2 and message in result.output, result.output
    assert list(tmp_path.iterdir()) == []
