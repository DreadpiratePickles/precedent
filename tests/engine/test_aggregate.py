import numpy as np
import pytest

from precedent.engine.aggregate import (
    PrecedentSummary,
    effective_sample_size,
    summarize_responses,
    wilson_interval,
)

PLACEBO = np.linspace(-0.05, 0.05, 101)


def test_separate_windows_keep_full_sample_size_and_shrink_as_specified():
    s = summarize_responses(
        np.full(12, 0.02), np.ones(12), np.arange(12) * 10, horizon=5, placebo=PLACEBO
    )
    assert s.n_eff == pytest.approx(12.0) and s.refused is False
    assert s.centre_shrunk == pytest.approx(0.02 * 12 / 22)


def test_overlapping_windows_collapse_sample_size_and_refuse():
    s = summarize_responses(
        np.full(12, 0.02), np.ones(12), np.arange(12), horizon=5, placebo=PLACEBO
    )
    assert s.n_eff == pytest.approx(3.0) and s.refused is True and s.quantiles is None  # 0, 5, 10


def test_a_median_inside_the_placebo_range_is_flagged():
    s = summarize_responses(
        np.full(20, 0.01), np.ones(20), np.arange(20) * 10, 5, np.linspace(-0.05, 0.05, 101)
    )
    assert s.inside_ordinary_range is True


def test_wilson_hand_values():
    assert wilson_interval(14, 23) == pytest.approx((0.475243, 0.727658), abs=1e-6)
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.141078), abs=1e-6)
    assert wilson_interval(10, 10) == pytest.approx((0.858922, 1.0), abs=1e-6)


def test_binary_summary_rate_interval_and_placebo():
    values = np.array([1.0] * 9 + [0.0] * 3)
    s = summarize_responses(
        values, np.ones(12), np.arange(12) * 100, horizon=20, placebo=0.2, is_binary=True
    )
    assert (s.mode, s.k, s.n, s.rate) == ("binary", 9, 12, pytest.approx(0.75))
    assert s.interval == pytest.approx((0.566669, 0.873133), abs=1e-6)
    assert s.inside_ordinary_range is False and s.quantiles is None
    inside = summarize_responses(values, np.ones(12), np.arange(12) * 100, 20, 0.7, is_binary=True)
    assert inside.inside_ordinary_range is True


def test_binary_refusal_keeps_counts_and_drops_the_interval():
    s = summarize_responses(np.ones(5), np.ones(5), np.arange(5) * 100, 20, 0.5, is_binary=True)
    assert s.refused is True and (s.k, s.n) == (5, 5) and s.interval is None


def test_weighted_quantiles_hand_values_equal_weights():
    s = summarize_responses(
        np.array([4.0, 1.0, 3.0, 2.0]), np.ones(4), np.arange(4) * 10, 1, None, min_neff=1.0
    )
    # cumulative-weight midpoints 0.125, 0.375, 0.625, 0.875 over sorted 1, 2, 3, 4
    assert s.quantiles == {
        0.1: pytest.approx(1.0),
        0.25: pytest.approx(1.5),
        0.5: pytest.approx(2.5),
        0.75: pytest.approx(3.5),
        0.9: pytest.approx(4.0),
    }
    assert s.median == pytest.approx(2.5) and s.hit_rate == pytest.approx(1.0)
    assert s.inside_ordinary_range is None and s.placebo_rate is None


def test_weighted_quantiles_hand_value_unequal_weights():
    s = summarize_responses(
        np.array([-1.0, 2.0, 3.0]),
        np.array([1.0, 1.0, 2.0]),
        np.array([0, 100, 200]),
        1,
        None,
        min_neff=1.0,
    )
    # midpoints (0.5, 1.5, 3.0) / 4 = 0.125, 0.375, 0.75; the median is 2 + (0.125 / 0.375) * 1
    assert s.median == pytest.approx(2 + 1 / 3)
    assert s.hit_rate == pytest.approx(0.75)
    assert s.n_eff == pytest.approx(16 / 6)


def test_summary_is_frozen_with_plain_dict_quantiles():
    s = summarize_responses(np.full(12, 0.02), np.ones(12), np.arange(12) * 10, 5, PLACEBO)
    assert isinstance(s, PrecedentSummary) and type(s.quantiles) is dict
    assert s.mode == "continuous" and s.k is None and s.rate is None and s.interval is None
    assert s.placebo_rate is None
    with pytest.raises(AttributeError):
        s.n = 3  # type: ignore[misc]


def test_effective_sample_size_hand_values_and_kish_cap():
    assert effective_sample_size(np.ones(12), np.arange(12), 5) == pytest.approx(3.0)
    assert effective_sample_size(np.ones(12), np.arange(12) * 10, 5) == pytest.approx(12.0)
    uneven = np.array([1.0, 1.0, 1.0, 5.0])
    kish = uneven.sum() ** 2 / (uneven**2).sum()
    assert effective_sample_size(uneven, np.arange(4) * 10, 5) == pytest.approx(kish)
    assert effective_sample_size(np.ones(3), np.array([20, 0, 10]), 10) == pytest.approx(3.0)


def test_empty_input_gives_zero_and_refuses():
    assert effective_sample_size(np.array([]), np.array([]), 5) == 0.0
    s = summarize_responses(np.array([]), np.array([]), np.array([]), 5, PLACEBO)
    assert s.n == 0 and s.n_eff == 0.0 and s.refused is True and s.quantiles is None
    b = summarize_responses(np.array([]), np.array([]), np.array([]), 5, 0.5, is_binary=True)
    assert b.n == 0 and b.k == 0 and b.refused is True and b.interval is None


@pytest.mark.parametrize(
    "weights, starts, horizon",
    [
        (np.ones(3), np.arange(2), 5),
        (np.array([1.0, -1.0]), np.arange(2), 5),
        (np.array([1.0, np.nan]), np.arange(2), 5),
        (np.array([1.0, np.inf]), np.arange(2), 5),
        (np.ones(2), np.arange(2), 0),
    ],
)
def test_effective_sample_size_rejects_bad_input(weights, starts, horizon):
    with pytest.raises(ValueError):
        effective_sample_size(weights, starts, horizon)


@pytest.mark.parametrize("k, n", [(1, 0), (1, -3), (-1, 10), (11, 10)])
def test_wilson_rejects_bad_input(k, n):
    with pytest.raises(ValueError):
        wilson_interval(k, n)


def test_a_non_binary_value_in_binary_mode_raises():
    with pytest.raises(ValueError):
        summarize_responses(
            np.array([1.0, 0.0, 0.5]), np.ones(3), np.arange(3) * 100, 20, 0.5, is_binary=True
        )


def test_placebo_type_mismatch_raises():
    with pytest.raises(ValueError):
        summarize_responses(np.full(12, 0.02), np.ones(12), np.arange(12) * 10, 5, 0.2)
    with pytest.raises(ValueError):
        summarize_responses(
            np.ones(12), np.ones(12), np.arange(12) * 100, 20, PLACEBO, is_binary=True
        )


def test_empty_placebo_array_gives_no_range_flag():
    s = summarize_responses(np.full(12, 0.02), np.ones(12), np.arange(12) * 10, 5, np.array([]))
    assert s.inside_ordinary_range is None and s.placebo_rate is None


def test_binary_placebo_rate_is_reported_and_none_placebo_gives_none():
    values = np.array([1.0] * 9 + [0.0] * 3)
    s = summarize_responses(values, np.ones(12), np.arange(12) * 100, 20, 0.7, is_binary=True)
    assert s.placebo_rate == pytest.approx(0.7)
    none = summarize_responses(values, np.ones(12), np.arange(12) * 100, 20, None, is_binary=True)
    assert none.placebo_rate is None and none.inside_ordinary_range is None
    refused = summarize_responses(np.ones(5), np.ones(5), np.arange(5) * 100, 20, 0.5, is_binary=True)
    assert refused.placebo_rate is None and refused.inside_ordinary_range is None


def test_values_and_weights_length_mismatch_raises():
    with pytest.raises(ValueError):
        summarize_responses(np.ones(4), np.ones(3), np.arange(3) * 10, 5, None)
