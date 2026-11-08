"""Market-model abnormal returns: the part of an asset's return the market does not explain.

Positions are integer row positions in return series that share one session axis. The model
`asset = alpha + beta * market` is fitted on an estimation window that ends before the event
window starts, so the event never leaks into its own baseline.
"""

import numpy as np
import pandas as pd

_MIN_MARKET_VARIANCE = 1e-18


def _check_inputs(
    asset_returns: pd.Series, market_returns: pd.Series, est_window: tuple[int, int], min_obs: int
) -> None:
    if len(asset_returns) != len(market_returns):
        raise ValueError("asset_returns and market_returns must have the same length")
    est_start, est_stop = est_window
    if not est_start <= est_stop < 0:
        raise ValueError(f"est_window must satisfy start <= stop < 0, got {est_window}")
    if min_obs < 2:
        raise ValueError(f"min_obs must be at least 2, got {min_obs}")


def _fit_market_model(
    asset: np.ndarray, market: np.ndarray, start: int, stop: int, min_obs: int
) -> tuple[float, float] | None:
    """OLS (alpha, beta) over positions start..stop inclusive; None when it cannot be fitted."""
    start = max(start, 0)
    stop = min(stop, len(asset) - 1)
    if stop < start:
        return None
    y = asset[start : stop + 1]
    x = market[start : stop + 1]
    observed = np.isfinite(x) & np.isfinite(y)
    if observed.sum() < min_obs:
        return None
    x, y = x[observed], y[observed]
    x_mean, y_mean = x.mean(), y.mean()
    variance = ((x - x_mean) ** 2).mean()
    if variance < _MIN_MARKET_VARIANCE:
        return None
    beta = ((x - x_mean) * (y - y_mean)).mean() / variance
    return float(y_mean - beta * x_mean), float(beta)


def cumulative_abnormal_return(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    event_pos: int,
    horizon: int,
    est_window: tuple[int, int] = (-250, -21),
    min_obs: int = 120,
) -> float:
    """Sum of abnormal returns over positions `event_pos + 1` to `event_pos + horizon`.

    The market model is fitted on positions `event_pos + est_window[0]` to
    `event_pos + est_window[1]`. Returns NaN when the estimation window has fewer than
    `min_obs` usable rows, the market has no variance there, or the event window is incomplete.
    """
    _check_inputs(asset_returns, market_returns, est_window, min_obs)
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    asset = asset_returns.to_numpy(dtype=float)
    market = market_returns.to_numpy(dtype=float)
    first, last = event_pos + 1, event_pos + horizon
    if event_pos < 0 or last >= len(asset):
        return float("nan")
    fit = _fit_market_model(
        asset, market, event_pos + est_window[0], event_pos + est_window[1], min_obs
    )
    if fit is None:
        return float("nan")
    alpha, beta = fit
    abnormal = asset[first : last + 1] - (alpha + beta * market[first : last + 1])
    if not np.isfinite(abnormal).all():
        return float("nan")
    return float(abnormal.sum())


def abnormal_return_at(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    pos: int,
    est_window: tuple[int, int] = (-250, -21),
    min_obs: int = 120,
) -> float:
    """Abnormal return of the single session at `pos`, fitted on `pos + est_window`."""
    _check_inputs(asset_returns, market_returns, est_window, min_obs)
    asset = asset_returns.to_numpy(dtype=float)
    market = market_returns.to_numpy(dtype=float)
    if pos < 0 or pos >= len(asset):
        return float("nan")
    fit = _fit_market_model(asset, market, pos + est_window[0], pos + est_window[1], min_obs)
    if fit is None:
        return float("nan")
    alpha, beta = fit
    return float(asset[pos] - (alpha + beta * market[pos]))


def estimate_betas(
    asset_returns: pd.DataFrame,
    market_returns: pd.Series,
    sessions: pd.DatetimeIndex,
    est_window: tuple[int, int] = (-250, -21),
    min_obs: int = 120,
) -> pd.DataFrame:
    """Market-model beta per session and asset, from the trailing estimation window.

    Positions are positions in `sessions`; both inputs are aligned to it (a missing session is
    a missing observation). The beta at a session uses sessions `pos + est_window[0]` to
    `pos + est_window[1]`, the same window `abnormal_return_at` uses. NaN where fewer than
    `min_obs` observations or the market has no variance.
    """
    est_start, est_stop = est_window
    if not est_start <= est_stop < 0:
        raise ValueError(f"est_window must satisfy start <= stop < 0, got {est_window}")
    if min_obs < 2:
        raise ValueError(f"min_obs must be at least 2, got {min_obs}")
    width = est_stop - est_start + 1
    market = market_returns.reindex(sessions).astype(float)
    assets = asset_returns.reindex(sessions).astype(float)
    result = {}
    for name in assets.columns:
        asset = assets[name]
        observed = market.notna() & asset.notna()
        x = market.where(observed, 0.0)
        y = asset.where(observed, 0.0)

        def window_sum(values: pd.Series) -> pd.Series:
            # Sum over the `width` rows ending at each row, then moved so row `pos` holds the
            # window ending at `pos + est_stop` (a window that starts before row 0 is clipped).
            return values.rolling(width, min_periods=1).sum().shift(-est_stop)

        count = window_sum(observed.astype(float))
        sum_x, sum_y = window_sum(x), window_sum(y)
        sum_xy, sum_xx = window_sum(x * y), window_sum(x * x)
        variance = sum_xx / count - (sum_x / count) ** 2
        covariance = sum_xy / count - (sum_x / count) * (sum_y / count)
        beta = covariance / variance
        usable = (count >= min_obs) & (variance >= _MIN_MARKET_VARIANCE)
        result[name] = beta.where(usable)
    return pd.DataFrame(result, index=sessions)
