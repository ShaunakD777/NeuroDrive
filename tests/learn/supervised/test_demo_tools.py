"""Tests for merging datasets and recording the expert from another dataset's starts
(task 6.7, ADR-052)."""

from pathlib import Path

import numpy as np
import pytest

from neurodrive.env.experts import ScriptedExpert
from neurodrive.env.sensors import edge_grid, observe
from neurodrive.env.track import Track
from neurodrive.env.world import World, WorldConfig
from neurodrive.learn.supervised.demos import (
    DemoDataset,
    DemoRecorder,
    merge_datasets,
    record_expert_from_starts,
    save_dataset,
)

TRACKS = Path(__file__).resolve().parents[3] / "tracks"


def track_of(name: str = "train_loop_01") -> Track:
    return Track.from_json(TRACKS / f"{name}.json")


def keyboard_dataset(frames: list[int], dataset_id: str, lateral: float = 0.0) -> DemoDataset:
    """A hand-made keyboard dataset: one episode per entry of ``frames``."""
    track = track_of()
    world = World(track, WorldConfig.for_track(track))
    rec = DemoRecorder(track, world.config, world.dt, world.params, seed=None)
    for i, n in enumerate(frames):
        rec.begin_episode(
            {
                "s0": np.array([100.0 * (i + 1)]),
                "lateral": np.array([lateral]),
                "heading_offset": np.array([0.01 * i]),
                "speed": np.array([0.0]),
            }
        )
        for t in range(n):
            rec.add(np.full(7, 0.1 * t + i), np.array([float(t % 3 - 1), 1.0]), t, i, 0.0, 1.0)
        rec.end_episode("stopped")
    dataset = rec.to_dataset()
    dataset.meta["dataset_id"] = dataset_id
    return dataset


# ── merging ─────────────────────────────────────────────────────────────────────────────────


def test_worked_example_three_plus_five_and_four_frames_merge_to_twelve():
    a = keyboard_dataset([3, 5], "20261111-000001-keyboard-a")
    b = keyboard_dataset([4], "20261111-000002-keyboard-b", lateral=0.5)
    merged = merge_datasets([a, b])
    assert merged.n_frames == 12 == merged.meta["demo_frames"] == merged.meta["env_steps"]
    assert [e["frames"] for e in merged.meta["episodes"]] == [3, 5, 4]
    assert [e["index"] for e in merged.meta["episodes"]] == [0, 1, 2]
    assert [e["source_index"] for e in merged.meta["episodes"]] == [0, 1, 0]
    ids = [a.meta["dataset_id"], a.meta["dataset_id"], b.meta["dataset_id"]]
    assert [e["source_dataset"] for e in merged.meta["episodes"]] == ids
    assert list(merged.episode) == [0] * 3 + [1] * 5 + [2] * 4
    assert list(merged.step) == [0, 1, 2, 0, 1, 2, 3, 4, 0, 1, 2, 3]
    sources = [a.meta["dataset_id"], b.meta["dataset_id"]]
    assert merged.meta["starts"] == {"kind": "merged", "sources": sources}
    assert merged.meta["episodes"][2]["start_lateral_m"] == 0.5  # each start is kept


def test_merged_rows_are_the_sources_rows_unchanged_and_order_is_the_given_order():
    a = keyboard_dataset([3, 5], "20261111-000001-keyboard-a")
    b = keyboard_dataset([4], "20261111-000002-keyboard-b")
    ab, ba = merge_datasets([a, b]), merge_datasets([b, a])
    for name in ("observation", "label", "x", "y", "psi", "v", "step"):
        assert np.array_equal(
            getattr(ab, name), np.concatenate([getattr(a, name), getattr(b, name)])
        )
        assert np.array_equal(
            getattr(ba, name), np.concatenate([getattr(b, name), getattr(a, name)])
        )
    assert ab.meta["straight_share"] == pytest.approx(np.mean(np.abs(ab.label[:, 0]) < 0.1))


def test_unlike_datasets_are_refused():
    a = keyboard_dataset([3], "a")
    b = keyboard_dataset([3], "b")
    b.meta["teacher"] = {"kind": "expert", "params": {}}
    with pytest.raises(ValueError, match="different teachers"):
        merge_datasets([a, b])
    c = keyboard_dataset([3], "c")
    c.meta["track"] = "somewhere_else"
    with pytest.raises(ValueError, match="'track'"):
        merge_datasets([a, c])
    d = keyboard_dataset([3], "d")
    d.meta["physics_version"] = 99
    with pytest.raises(ValueError, match="physics_version"):
        merge_datasets([a, d])
    with pytest.raises(ValueError, match="nothing to merge"):
        merge_datasets([])


def test_a_merged_dataset_saves_and_loads(tmp_path):
    merged = merge_datasets([keyboard_dataset([3, 5], "a"), keyboard_dataset([4], "b")])
    folder = save_dataset(merged, tmp_path, "m")
    assert folder.name.endswith("-keyboard-m")
    assert DemoDataset.load(folder).identical_to(merged)


# ── the expert from another dataset's starts ────────────────────────────────────────────────


def source_with_starts(frames: list[int]) -> DemoDataset:
    d = keyboard_dataset(frames, "20261111-000003-keyboard-src")
    d.meta["episodes"][0].update(
        start_s0_m=100.0, start_lateral_m=0.5, start_heading_offset_rad=0.02
    )
    d.meta["episodes"][1].update(
        start_s0_m=500.0, start_lateral_m=0.0, start_heading_offset_rad=0.0
    )
    d.meta["episodes"][2].update(
        start_s0_m=300.0, start_lateral_m=-1.0, start_heading_offset_rad=0.05
    )
    return d


def test_matched_frames_are_the_experts_first_frames_from_each_start():
    source = source_with_starts([3, 5, 40])
    matched = record_expert_from_starts(track_of(), source, match_frames=True)
    assert [e["frames"] for e in matched.meta["episodes"]] == [3, 5, 40]
    assert matched.n_frames == 48 == matched.meta["demo_frames"]
    assert [e["status"] for e in matched.meta["episodes"]] == ["truncated"] * 3
    assert matched.meta["starts"]["source_dataset"] == source.meta["dataset_id"]
    assert matched.meta["teacher"]["kind"] == "expert"
    # Each episode alone, in a fresh world from the stored start: those rows are its first frames.
    track = track_of()
    segments = edge_grid(track)
    for i, n in ((0, 3), (1, 5), (2, 40)):
        e = source.meta["episodes"][i]
        world = World(track, WorldConfig.for_track(track))
        world.reset(
            s0=[e["start_s0_m"]],
            lateral=[e["start_lateral_m"]],
            heading_offset=[e["start_heading_offset_rad"]],
            speed=[e["start_speed_mps"]],
        )
        expert = ScriptedExpert(world)
        rows = matched.episode == i
        assert rows.sum() == n
        for obs, label in zip(matched.observation[rows], matched.label[rows], strict=True):
            assert np.allclose(observe(world, segments)[0], obs, rtol=0, atol=1e-12)
            assert np.allclose(expert.label()[0], label, rtol=0, atol=1e-12)
            world.step(label[:1], label[1:])


def test_unmatched_drives_run_to_their_own_end():
    source = source_with_starts([3, 5, 40])
    full = record_expert_from_starts(track_of(), source, match_frames=False)
    assert [e["status"] for e in full.meta["episodes"]] == ["finished"] * 3
    assert all(e["frames"] > 1000 for e in full.meta["episodes"])
    assert full.meta["starts"]["match_frames"] is False
    matched = record_expert_from_starts(track_of(), source, match_frames=True)
    # the matched dataset is the first rows of the full one, episode by episode
    for i, n in enumerate((3, 5, 40)):
        first = full.observation[full.episode == i][:n]
        assert np.array_equal(first, matched.observation[matched.episode == i])


def test_the_experts_starts_are_the_sources_starts():
    source = source_with_starts([3, 5, 40])
    matched = record_expert_from_starts(track_of(), source, match_frames=True)
    keys = ("start_s0_m", "start_lateral_m", "start_heading_offset_rad", "start_speed_mps")
    for mine, theirs in zip(matched.meta["episodes"], source.meta["episodes"], strict=True):
        for key in keys:
            assert mine[key] == theirs[key]


def test_held_out_and_foreign_tracks_are_refused():
    source = source_with_starts([3, 5, 40])
    with pytest.raises(ValueError, match="held-out"):
        record_expert_from_starts(track_of("hairpins_01"), source, match_frames=True)
    source.meta["track"] = "monza_cw"
    with pytest.raises(ValueError, match="recorded on"):
        record_expert_from_starts(track_of(), source, match_frames=True)
