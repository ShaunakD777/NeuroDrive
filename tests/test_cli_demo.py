"""Tests for `nd demo-record` (cli.py), task 6.1."""

from typer.testing import CliRunner

from neurodrive.cli import app
from neurodrive.learn.supervised.demos import DemoDataset

runner = CliRunner()


def test_records_the_expert_into_a_dataset_folder(tmp_path):
    args = ["demo-record", "--random-starts", "1", "--seed", "2", "--tag", "t"]
    result = runner.invoke(app, [*args, "--out-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    (folder,) = list(tmp_path.iterdir())
    assert folder.name.endswith("-expert-t")
    assert f"dataset folder: {folder.as_posix()}" in result.stdout
    dataset = DemoDataset.load(folder)
    meta = dataset.meta
    assert len(meta["episodes"]) == 11 and meta["starts"]["seed"] == 2
    assert f"frames: {dataset.n_frames:,} from 11 episodes (11 finished)" in result.stdout
    assert f"{meta['straight_share']:.3f} of the frames" in result.stdout
    assert meta["track"] == "train_loop_01" and meta["teacher"]["kind"] == "expert"
    assert {"git_commit", "git_dirty", "created_at", "neurodrive_version"} <= meta.keys()


def test_refusals_create_nothing(tmp_path):
    for args, message in (
        (["--track", "hairpins_01"], "held-out track"),
        (["--track", "nowhere"], "unknown track"),
        (["--teacher", "keyboard"], "task 6.1b"),
        (["--random-starts", "-1"], "random_starts must be 0 or more"),
    ):
        result = runner.invoke(app, ["demo-record", *args, "--out-dir", str(tmp_path)])
        assert result.exit_code == 2, args
        assert message in result.output
    assert not tmp_path.exists() or list(tmp_path.iterdir()) == []
