"""Tests for market-model abnormal returns and rolling betas."""
import math

import numpy as np
import pandas as pd
import pytest

from precedent.spine.abnormal_returns import (
    abnormal_return_at,
    cumulative_abnormal_return,
    estimate_betas,
)


def test_car_recovers_a_planted_abnormal_return():
    market = pd.Series(np.random.default_rng(0).normal(0, 0.01, 400))
    asset = 0.001 + 1.5 * market
    asset.iloc[301] += 0.03
    assert cumulative_abnormal_return(asset, market, event_pos=300, horizon=1) == pytest.approx(
        0.03, abs=1e-9
    )


def test_car_is_nan_without_enough_history():
    market = pd.Series(np.zeros(400))
    asset = market.copy()
    assert math.isnan(cumulative_abnormal_return(asset, market, event_pos=100, horizon=1))


def test_car_ignores_everything_after_its_window():
    market = pd.Series(np.random.default_rng(1).normal(0, 0.01, 400))
    asset = 1.2 * market
    before = cumulative_abnormal_return(asset, market, event_pos=300, horizon=5)
    asset.iloc[306:] += 0.5
    assert cumulative_abnormal_return(asset, market, event_pos=300, horizon=5) == before


def test_car_sums_abnormal_returns_over_the_horizon():
    market = pd.Series(np.random.default_rng(2).normal(0, 0.01, 400))
    asset = 0.0 + 1.0 * market
    asset.iloc[301] += 0.01
    asset.iloc[303] += 0.02
    assert cumulative_abnormal_return(asset, market, event_pos=300, horizon=3) == pytest.approx(
        0.03, abs=1e-9
    )


def test_car_is_nan_when_the_event_window_runs_past_the_data():
    market = pd.Series(np.random.default_rng(3).normal(0, 0.01, 400))
    asset = 1.1 * market
    assert math.isnan(cumulative_abnormal_return(asset, market, event_pos=398, horizon=5))


def test_car_is_nan_when_market_returns_have_no_variance():
    market = pd.Series(np.zeros(400))
    asset = pd.Series(np.random.default_rng(4).normal(0, 0.01, 400))
    assert math.isnan(cumulative_abnormal_return(asset, market, event_pos=300, horizon=1))


def test_car_skips_missing_estimation_rows_but_counts_only_observed_ones():
    market = pd.Series(np.random.default_rng(5).normal(0, 0.01, 400))
    asset = 0.001 + 1.5 * market
    asset.iloc[60:200] = np.nan  # the 50..279 window keeps 10 + 80 = 90 observed rows
    assert math.isnan(cumulative_abnormal_return(asset, market, event_pos=300, horizon=1))
    asset.iloc[60:100] = 0.001 + 1.5 * market.iloc[60:100]  # 130 observed rows
    assert not math.isnan(cumulative_abnormal_return(asset, market, event_pos=300, horizon=1))


def test_car_rejects_series_of_different_length_and_bad_arguments():
    market = pd.Series(np.zeros(400))
    with pytest.raises(ValueError, match="same length"):
        cumulative_abnormal_return(market.iloc[:300], market, event_pos=250, horizon=1)
    with pytest.raises(ValueError, match="horizon"):
        cumulative_abnormal_return(market, market, event_pos=300, horizon=0)
    with pytest.raises(ValueError, match="est_window"):
        cumulative_abnormal_return(market, market, event_pos=300, horizon=1, est_window=(-250, 0))


def test_abnormal_return_at_recovers_a_planted_abnormal_return():
    market = pd.Series(np.random.default_rng(6).normal(0, 0.01, 400))
    asset = 0.002 + 0.8 * market
    asset.iloc[320] -= 0.04
    assert abnormal_return_at(asset, market, pos=320) == pytest.approx(-0.04, abs=1e-9)
    assert abnormal_return_at(asset, market, pos=330) == pytest.approx(0.0, abs=1e-9)


def test_abnormal_return_at_is_nan_without_enough_history_or_out_of_range():
    market = pd.Series(np.random.default_rng(7).normal(0, 0.01, 400))
    asset = 0.8 * market
    assert math.isnan(abnormal_return_at(asset, market, pos=100))
    assert math.isnan(abnormal_return_at(asset, market, pos=400))
    assert math.isnan(abnormal_return_at(asset, market, pos=-1))


def _sessions(count: int) -> pd.DatetimeIndex:
    return pd.bdate_range("2020-01-01", periods=count)


def test_estimate_betas_recovers_a_constant_beta_after_warm_up():
    sessions = _sessions(500)
    market = pd.Series(np.random.default_rng(8).normal(0, 0.01, 500), index=sessions)
    assets = pd.DataFrame(
        {"HIGH": 0.0005 + 1.3 * market, "LOW": -0.0002 + 0.7 * market}, index=sessions
    )
    betas = estimate_betas(assets, market, sessions)
    assert list(betas.columns) == ["HIGH", "LOW"]
    assert betas.index.equals(sessions)
    # first usable session: window ends 21 rows back and needs 120 rows, so position 140
    assert betas.iloc[:140].isna().all().all()
    assert (betas["HIGH"].iloc[140:] - 1.3).abs().max() < 1e-6
    assert (betas["LOW"].iloc[140:] - 0.7).abs().max() < 1e-6


def test_estimate_betas_matches_the_single_session_fit_on_noisy_data():
    sessions = _sessions(400)
    rng = np.random.default_rng(9)
    market = pd.Series(rng.normal(0, 0.01, 400), index=sessions)
    asset = 1.1 * market + pd.Series(rng.normal(0, 0.005, 400), index=sessions)
    betas = estimate_betas(pd.DataFrame({"X": asset}), market, sessions)
    pos = 320
    window = slice(pos - 250, pos - 21 + 1)
    expected = np.polyfit(market.iloc[window], asset.iloc[window], 1)[0]
    assert betas["X"].iloc[pos] == pytest.approx(expected, abs=1e-9)


def test_estimate_betas_is_nan_where_the_asset_has_too_few_observations():
    sessions = _sessions(400)
    market = pd.Series(np.random.default_rng(10).normal(0, 0.01, 400), index=sessions)
    asset = 1.5 * market
    asset.iloc[:200] = np.nan
    betas = estimate_betas(pd.DataFrame({"X": asset}), market, sessions)
    # observed rows start at 200; 120 of them are in the window once position - 21 >= 319
    assert betas["X"].iloc[:340].isna().all()
    assert betas["X"].iloc[340:].notna().all()
