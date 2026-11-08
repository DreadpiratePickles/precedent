"""Tests for the daily Spearman rank IC and its summary."""
import numpy as np
import pandas as pd
import pytest

from precedent.spine.rank_ic import daily_rank_ic, ic_summary


def panel(day_values: dict[str, tuple[list[float], list[float]]]) -> tuple[pd.Series, pd.Series]:
    keys, pred, label = [], [], []
    for day, (p_values, l_values) in day_values.items():
        for i, (p, lab) in enumerate(zip(p_values, l_values, strict=True)):
            keys.append((pd.Timestamp(day), f"S{i}"))
            pred.append(p)
            label.append(lab)
    index = pd.MultiIndex.from_tuples(keys, names=["datetime", "instrument"])
    return pd.Series(pred, index=index), pd.Series(label, index=index)


def test_perfect_and_inverse_rankings() -> None:
    pred, label = panel({"2024-01-02": ([1, 2, 3, 4], [10, 20, 30, 40]), "2024-01-03": ([1, 2, 3, 4], [9, 7, 5, 3])})
    ic = daily_rank_ic(pred, label)
    assert list(ic.index) == [pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    assert ic.tolist() == pytest.approx([1.0, -1.0])


def test_known_value_and_monotone_transform_invariance() -> None:
    # d = [0, -1, 1, -1, 1]; rho = 1 - 6 * 4 / (5 * 24) = 0.8
    pred, label = panel({"2024-01-02": ([1, 2, 3, 4, 5], [1, 3, 2, 5, 4])})
    assert daily_rank_ic(pred, label).iloc[0] == pytest.approx(0.8)
    assert daily_rank_ic(np.exp(pred), label.pow(3)).iloc[0] == pytest.approx(0.8)


def test_ties_use_average_ranks() -> None:
    pred, label = panel({"2024-01-02": ([1, 1, 2, 3], [1, 2, 3, 4])})
    expected = pd.Series([1, 1, 2, 3]).corr(pd.Series([1, 2, 3, 4]), method="spearman")
    assert daily_rank_ic(pred, label).iloc[0] == pytest.approx(expected)


def test_days_with_fewer_than_three_names_and_nan_pairs_are_dropped() -> None:
    pred, label = panel(
        {"2024-01-02": ([1, 2], [1, 2]), "2024-01-03": ([1, 2, 3, 4], [1, 2, 3, np.nan]), "2024-01-04": ([1, 2, 3], [3, 2, 1])}
    )
    ic = daily_rank_ic(pred, label)
    assert list(ic.index) == [pd.Timestamp("2024-01-03"), pd.Timestamp("2024-01-04")]
    assert ic.tolist() == pytest.approx([1.0, -1.0])


def test_constant_cross_section_has_no_defined_ic_and_is_dropped() -> None:
    pred, label = panel({"2024-01-02": ([5, 5, 5], [1, 2, 3]), "2024-01-03": ([1, 2, 3], [1, 2, 3])})
    assert list(daily_rank_ic(pred, label).index) == [pd.Timestamp("2024-01-03")]


def test_ic_summary_values() -> None:
    ic = pd.Series([0.1, 0.3, -0.1, 0.3])
    summary = ic_summary(ic)
    assert summary["n_days"] == 4
    assert summary["mean"] == pytest.approx(0.15)
    assert summary["std"] == pytest.approx(ic.std(ddof=1))
    assert summary["icir"] == pytest.approx(0.15 / ic.std(ddof=1))
    assert summary["hit_rate"] == 0.75


def test_ic_summary_of_nothing_raises_and_zero_std_gives_nan_icir() -> None:
    with pytest.raises(ValueError, match="no days"):
        ic_summary(pd.Series([], dtype=float))
    assert np.isnan(ic_summary(pd.Series([0.2, 0.2]))["icir"])


def test_missing_datetime_index_level_raises_a_clear_error() -> None:
    pred, label = panel({"2024-01-02": ([1, 2, 3], [1, 2, 3])})
    with pytest.raises(ValueError, match="datetime"):
        daily_rank_ic(pred.rename_axis(["date", "instrument"]), label.rename_axis(["date", "instrument"]))
