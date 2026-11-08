"""The asset response library: cumulative abnormal returns after each first-report event.

Timing: an event is effective at session position `p`; the trade opens at `p + 1`; a horizon of
`h` sessions sums the abnormal returns of positions `p + 2` to `p + 1 + h`, so the spine's
`cumulative_abnormal_return` is called with `event_pos = p + 1`. The response is complete at
`sessions[p + 1 + h]`. An (event, asset, horizon) whose window runs past the last session, or
whose history is too short for the market model, produces no row; it is counted, never zeroed.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from precedent.libraries.common import (
    LIBRARY_COLUMNS,
    empty_library_rows,
    validate_library_rows,
)
from precedent.spine.abnormal_returns import cumulative_abnormal_return

__all__ = ["AssetLibraryReport", "build_asset_library", "sample_asset_placebo"]

_PLACEBO_COLUMNS = ("value", "response_complete_session")


@dataclass(frozen=True)
class AssetLibraryReport:
    rows: int
    skipped_incomplete_window: int
    skipped_short_history: int


def _check_axis(returns: pd.DataFrame, sessions: pd.DatetimeIndex) -> None:
    if len(returns) != len(sessions) or not returns.index.equals(sessions):
        raise ValueError(
            f"returns must be indexed by exactly sessions: returns has {len(returns)} rows "
            f"({_span(returns.index)}), sessions has {len(sessions)} ({_span(sessions)})"
        )


def _span(index: pd.Index) -> str:
    return f"{index[0]} to {index[-1]}" if len(index) else "empty"


def build_asset_library(
    events: pd.DataFrame,
    returns: pd.DataFrame,
    market: str,
    sessions: pd.DatetimeIndex,
    assets_for_event: Callable[[pd.Series], list[str]],
    horizons: tuple[int, ...] = (1, 5, 20),
) -> tuple[pd.DataFrame, AssetLibraryReport]:
    """One row per (first-report event, asset, horizon) with the CAR as `value`.

    `returns` must be indexed by exactly `sessions` and hold the `market` column; otherwise
    `ValueError`. A window that ends past the last session is counted in
    `skipped_incomplete_window` (checked first, without computing); any other missing CAR
    (a symbol absent from `returns`, too little history, a non-finite return) is counted in
    `skipped_short_history`. Rows pass `validate_library_rows`; zero rows give
    `empty_library_rows()`.
    """
    _check_axis(returns, sessions)
    if market not in returns.columns:
        raise ValueError(f"returns has no market column {market!r}")
    market_returns = returns[market]
    n_sessions = len(sessions)
    rows: list[dict[str, object]] = []
    incomplete = short_history = 0
    for _, event in events[events["is_first_report"]].iterrows():
        position = int(event["effective_position"])
        if not 0 <= position < n_sessions:
            raise ValueError(f"event {event['event_id']} has position {position} outside sessions")
        for symbol in dict.fromkeys(assets_for_event(event)):
            for horizon in horizons:
                complete = position + 1 + horizon
                if complete >= n_sessions:
                    incomplete += 1
                    continue
                value = _car(returns, symbol, market_returns, position, horizon)
                if not np.isfinite(value):
                    short_history += 1
                    continue
                rows.append(
                    {
                        "event_id": event["event_id"],
                        "outcome_type": "asset",
                        "outcome_key": symbol,
                        "horizon": horizon,
                        "value": value,
                        "is_binary": False,
                        "effective_session": sessions[position],
                        "response_complete_session": sessions[complete],
                        "source_ref": event["source_ref"],
                    }
                )
    report = AssetLibraryReport(len(rows), incomplete, short_history)
    if not rows:
        return empty_library_rows(), report
    return validate_library_rows(pd.DataFrame(rows, columns=list(LIBRARY_COLUMNS))), report


def _car(
    returns: pd.DataFrame, symbol: str, market_returns: pd.Series, position: int, horizon: int
) -> float:
    if symbol not in returns.columns:
        return float("nan")
    return cumulative_abnormal_return(
        returns[symbol], market_returns, event_pos=position + 1, horizon=horizon
    )


def sample_asset_placebo(
    asset_returns: pd.Series,
    market_returns: pd.Series,
    sessions: pd.DatetimeIndex,
    horizon: int,
    n_samples: int,
    rng: np.random.Generator,
    exclude_positions: set[int],
) -> pd.DataFrame:
    """CARs at random non-event positions, under the same timing rule as the library.

    Positions `p` are drawn uniformly with replacement from those with `p + 1 + horizon <
    len(sessions)` and `p` not in `exclude_positions`. A position whose CAR is not finite is
    dropped from the candidates and redrawn; when the candidates run out, fewer than `n_samples`
    rows come back. Columns: `value` (float64), `response_complete_session` (datetime64[ns]).
    """
    if n_samples < 0:
        raise ValueError(f"n_samples must not be negative, got {n_samples}")
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if not len(asset_returns) == len(market_returns) == len(sessions):
        raise ValueError("asset_returns, market_returns and sessions must have the same length")
    last_start = len(sessions) - 1 - horizon
    candidates = np.array(
        [p for p in range(max(last_start, 0)) if p not in exclude_positions], dtype=np.int64
    )
    cars: dict[int, float] = {}
    chosen: list[int] = []
    while len(chosen) < n_samples and len(candidates):
        position = int(rng.choice(candidates))
        if position not in cars:
            cars[position] = cumulative_abnormal_return(
                asset_returns, market_returns, event_pos=position + 1, horizon=horizon
            )
        if np.isfinite(cars[position]):
            chosen.append(position)
        else:
            candidates = candidates[candidates != position]
    return pd.DataFrame(
        {
            "value": pd.Series([cars[p] for p in chosen], dtype="float64"),
            "response_complete_session": pd.Series(
                sessions[[p + 1 + horizon for p in chosen]], dtype="datetime64[ns]"
            ),
        }
    )
