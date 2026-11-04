"""Tests for the dataset inspector's numbers (task 6.2, ADR-045): histograms, covered ranges, the
thinned map rows."""

from pathlib import Path

import numpy as np
import pytest

from neurodrive.env.track import Track
from neurodrive.experiment.dataset_view import (
    BINS,
    LABEL_RANGES,
    OBSERVATION_RANGES,
    covered_range,
    dataset_view,
    histogram,
    thin,
)
from neurodrive.learn.supervised.demos import DemoDataset, record_expert, straight_share

TRACKS_DIR = Path(__file__).resolve().parents[2] / "tracks"


def test_worked_example_histogram():
    # Brief 4a: five steers on [-1, 1] in 20 bins of width 0.1 (no value on a bin edge).
    steers = np.array([0.02, -0.05, 0.43, 0.01, -0.62])
    edges, counts = histogram(steers, -1.0, 1.0)
    assert len(edges) == BINS + 1 and edges[0] == -1.0 and edges[-1] == 1.0
    assert np.allclose(np.diff(edges), 0.1)
    expected = np.zeros(BINS, dtype=np.int64)
    expected[[3, 9, 10, 14]] = [1, 1, 2, 1]
    assert counts.tolist() == expected.tolist()
    assert (counts[[3, 9, 10, 14]] / len(steers)).tolist() == [0.2, 0.2, 0.4, 0.2]


def test_the_ends_of_the_range_are_counted():
    # lo goes in the first bin, hi in the last (the last bin is closed on both sides).
    edges, counts = histogram(np.array([-1.0, 1.0, 1.0]), -1.0, 1.0)
    assert counts[0] == 1 and counts[-1] == 2 and counts.sum() == 3


def test_a_value_outside_the_range_is_refused():
    with pytest.raises(ValueError, match="outside"):
        histogram(np.array([0.5, 1.2]), 0.0, 1.0)


def test_worked_example_covered_range():
    assert covered_range(np.array([0.2, 0.5, 0.3]), 0.0, 1.0) == pytest.approx(0.3)
    assert covered_range(np.array([-0.5, 0.5]), -1.0, 1.0) == pytest.approx(0.5)
    assert covered_range(np.array([]), 0.0, 1.0) == 0.0


def test_worked_example_thinning():
    stride, rows = thin(25, max_points=10)
    assert stride == 3 and rows.tolist() == [0, 3, 6, 9, 12, 15, 18, 21, 24]
    assert thin(39_338)[0] == 10 and len(thin(39_338)[1]) == 3934
    assert thin(4000)[0] == 1 and len(thin(4000)[1]) == 4000  # small datasets are kept whole
    assert thin(4001)[0] == 2
    assert thin(0)[1].tolist() == []


@pytest.fixture(scope="module")
def expert_demo() -> DemoDataset:
    """The expert from two exam-like starts on the training loop (about 3,900 frames)."""
    track = Track.from_json(TRACKS_DIR / "train_loop_01.json")
    return record_expert(
        track, start_fractions=(0.0, 0.5), lateral_offsets=(0.0,), random_starts=0, seed=0
    )


def test_view_of_a_real_dataset(expert_demo):
    f = expert_demo.n_frames
    view = dataset_view(expert_demo, max_points=1000)
    assert [h.name for h in view.labels] == list(LABEL_RANGES)
    assert [h.name for h in view.observations] == list(OBSERVATION_RANGES)
    for h, column in zip(
        view.labels + view.observations,
        [*expert_demo.label.T, *expert_demo.observation.T],
        strict=True,
    ):
        assert len(h.counts) == BINS and int(h.counts.sum()) == f  # every frame in some bin
        assert h.min == column.min() and h.max == column.max()
        assert h.covered == pytest.approx((h.max - h.min) / (h.hi - h.lo))
    # The steer histogram agrees with straight_share: bins 9 and 10 are [-0.1, 0.1).
    steer = view.labels[0]
    assert not np.any(expert_demo.label[:, 0] == -0.1)
    assert (steer.counts[9] + steer.counts[10]) / f == pytest.approx(
        straight_share(expert_demo.label)
    )
    # The expert drives at full throttle a lot: those frames (exactly 1.0) are in the last bin.
    throttle = view.labels[1]
    full = int(np.sum(expert_demo.label[:, 1] == 1.0))
    assert full > 0 and throttle.counts[-1] >= full
    # The map: every k-th row, and those are the dataset's own rows.
    assert view.stride == -(-f // 1000) and view.rows.tolist() == list(range(0, f, view.stride))
    assert len(view.rows) <= 1000


def test_the_view_is_deterministic(expert_demo):
    a, b = dataset_view(expert_demo), dataset_view(expert_demo)
    for ha, hb in zip(a.labels + a.observations, b.labels + b.observations, strict=True):
        assert np.array_equal(ha.counts, hb.counts) and np.array_equal(ha.edges, hb.edges)
    assert np.array_equal(a.rows, b.rows)
