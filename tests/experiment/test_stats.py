"""Tests for experiment/stats.py: percentiles, summaries, curve bands, bootstrap intervals."""

import math

import numpy as np
import pytest

from neurodrive.core.rng import make_rng
from neurodrive.experiment.stats import (
    bootstrap_median_ci,
    bootstrap_median_difference,
    curve_bands,
    percentile,
    summarise,
)

INF = math.inf


def test_brief_example_five_seeds():
    s = summarise([0.04, 0.01, 0.03, 0.10, 0.02])
    assert (s.median, s.q1, s.q3, s.n) == pytest.approx((0.03, 0.02, 0.04, 5))
    assert s.iqr == pytest.approx(0.02)
    assert np.mean([0.04, 0.01, 0.03, 0.10, 0.02]) == pytest.approx(0.04)  # the mean is pulled up


def test_brief_example_even_count():
    s = summarise([4, 1, 3, 2])
    assert (s.median, s.q1, s.q3) == pytest.approx((2.5, 1.75, 3.25))


@pytest.mark.parametrize("n", [1, 2, 5, 10, 33])
@pytest.mark.parametrize("p", [0, 10, 25, 50, 75, 90, 100])
def test_matches_numpy_for_finite_values(n, p):
    values = make_rng(n).normal(size=n)
    assert percentile(values, p) == pytest.approx(np.percentile(values, p), rel=1e-12, abs=1e-15)


def test_infinite_values_sort_to_the_top():
    assert summarise([1.0, 2.0, INF]).median == 2.0
    s = summarise([1.0, 2.0, INF, INF])
    assert s.median == INF and s.q1 == pytest.approx(1.75) and s.q3 == INF
    assert summarise([INF, INF]).median == INF  # never inf - inf = nan
    assert percentile([1.0, INF], 0) == 1.0


def test_bad_input():
    with pytest.raises(ValueError):
        summarise([])
    with pytest.raises(ValueError):
        summarise([1.0, math.nan])
    with pytest.raises(ValueError):
        percentile([1.0], 101)


def test_curve_bands_hand_example():
    curves = [
        [4.0, 2.0, 1.0, 0.5],
        [3.0, 1.5, 0.8, 0.4],
        [5.0, 3.0, 2.0, 1.0],
        [6.0, 9.0, None],  # diverged at position 2
        [2.0, 1.0],  # a shorter run that did not diverge
    ]
    bands = curve_bands(curves, [False, False, False, True, False])
    assert bands.n_present == [5, 5, 4, 4]
    assert bands.n_finite == [5, 5, 3, 3]
    assert bands.median[0] == 4.0 and bands.q1[0] == 3.0 and bands.q3[0] == 5.0
    assert bands.median[1] == 2.0
    # position 2: [1.0, 0.8, 2.0, inf] -> median halfway between 1.0 and 2.0, Q3 lands on inf
    assert bands.median[2] == pytest.approx(1.5) and bands.q3[2] == INF
    assert bands.q1[2] == pytest.approx(0.95)
    # position 3: [0.5, 0.4, 1.0, inf] (the diverged run stays inf, the short run is absent)
    assert bands.median[3] == pytest.approx(0.75)


def test_curve_bands_majority_diverged_shows_in_the_median():
    bands = curve_bands([[1.0, 0.5], [1.0], [1.0]], [False, True, True])
    assert bands.median[1] == INF and bands.n_finite[1] == 1


def test_curve_bands_bad_input():
    with pytest.raises(ValueError):
        curve_bands([[1.0]], [False, True])
    with pytest.raises(ValueError):
        curve_bands([], [])


def test_bootstrap_ci_is_reproducible_and_contains_the_median():
    values = make_rng(0).lognormal(size=7)
    a = bootstrap_median_ci(values, make_rng(1))
    b = bootstrap_median_ci(values, make_rng(1))
    assert a == b
    assert a.low <= a.estimate <= a.high
    assert a.estimate == pytest.approx(np.median(values))
    assert min(values) <= a.low and a.high <= max(values)
    # with an odd count every resampled median is one of the 7 values, so the bounds are too
    assert a.low in values and a.high in values


def test_bootstrap_difference_clear_and_no_difference():
    rng = make_rng(3)
    low = rng.normal(1.0, 0.1, size=10)
    high = rng.normal(5.0, 0.1, size=10)
    clear = bootstrap_median_difference(high, low, make_rng(4))
    assert not clear.contains_zero and clear.low > 3.0
    same = bootstrap_median_difference(low, low, make_rng(5))
    assert same.contains_zero
    paired = bootstrap_median_difference(low, low, make_rng(5), paired=True)
    assert (paired.low, paired.high) == (0.0, 0.0)  # same indices: every difference is 0


def test_bootstrap_with_diverged_runs():
    interval = bootstrap_median_difference([INF, INF, INF, 1.0], [0.1, 0.2, 0.3, 0.4], make_rng(6))
    assert interval.estimate == INF and interval.high == INF


def test_bootstrap_bad_input():
    with pytest.raises(ValueError, match="equal lengths"):
        bootstrap_median_difference([1.0, 2.0], [1.0], make_rng(0), paired=True)
    with pytest.raises(ValueError):
        bootstrap_median_ci([1.0], make_rng(0), level=1.0)
    with pytest.raises(ValueError):
        bootstrap_median_ci([1.0], make_rng(0), resamples=0)
