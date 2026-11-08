"""Shocks: the starting sizes that the ripple propagates along links.

A price shock is an instrument's market-model abnormal return on the session an event takes
effect; a text shock is the signed news factor where it is non-zero. Both are indexed
(`datetime`, `instrument`) with naive `datetime64[ns]` session labels, like the news panel.
"""

import pandas as pd

from precedent.spine.abnormal_returns import abnormal_return_at

SHOCK_NAME = "shock"
INDEX_NAMES = ["datetime", "instrument"]
_SIGNED_COLUMN = "news_signed_1d"


def _shock_series(
    sessions: pd.DatetimeIndex, instruments: list[str], values: list[float]
) -> pd.Series:
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex(sessions).as_unit("ns"), pd.Index(instruments, dtype=str)],
        names=INDEX_NAMES,
    )
    return pd.Series(values, index=index, name=SHOCK_NAME, dtype="float64").sort_index()


def _validate_price_inputs(
    returns: pd.DataFrame, market: str, sessions: pd.DatetimeIndex
) -> None:
    if sessions.tz is not None or not (sessions.is_unique and sessions.is_monotonic_increasing):
        raise ValueError("sessions must be a naive, strictly increasing DatetimeIndex")
    if not returns.index.equals(sessions):
        raise ValueError("returns.index must equal sessions exactly")
    if market not in returns.columns:
        raise ValueError(f"market column {market!r} is not in returns")


def price_shocks(
    event_frame: pd.DataFrame,
    returns: pd.DataFrame,
    market: str,
    sessions: pd.DatetimeIndex,
) -> tuple[pd.Series, int]:
    """Abnormal return of each instrument on each effective session on which it has an event.

    One shock per distinct (effective_session, instrument) pair, however many events share it.
    The abnormal return uses `abnormal_return_at` with the spine defaults, so the estimation
    window ends before the event session. A pair is dropped when its instrument has no column
    in `returns` or its abnormal return is NaN (too little history, no market variance); the
    second return value counts the dropped pairs, both reasons together.
    """
    _validate_price_inputs(returns, market, sessions)
    pairs = event_frame[["effective_session", "instrument"]].drop_duplicates()
    positions = sessions.get_indexer(pd.DatetimeIndex(pairs["effective_session"]))
    if (positions < 0).any():
        missing = pairs["effective_session"][positions < 0].iloc[0]
        raise ValueError(f"effective_session {missing} is not in sessions")
    market_returns = returns[market]
    kept_sessions: list[pd.Timestamp] = []
    kept_instruments: list[str] = []
    values: list[float] = []
    dropped = 0
    for (session, instrument), pos in zip(pairs.itertuples(index=False), positions, strict=True):
        if instrument not in returns.columns:
            dropped += 1
            continue
        shock = abnormal_return_at(returns[instrument], market_returns, int(pos))
        if pd.isna(shock):
            dropped += 1
            continue
        kept_sessions.append(session)
        kept_instruments.append(instrument)
        values.append(shock)
    return _shock_series(pd.DatetimeIndex(kept_sessions), kept_instruments, values), dropped


def text_shocks(news_panel: pd.DataFrame) -> pd.Series:
    """The `news_signed_1d` column of a news factor panel where it is non-zero.

    NaN counts as zero and is dropped.
    """
    if list(news_panel.index.names) != INDEX_NAMES:
        raise ValueError(
            f"news_panel index names must be {INDEX_NAMES}, got {list(news_panel.index.names)}"
        )
    if _SIGNED_COLUMN not in news_panel.columns:
        raise ValueError(f"news_panel has no {_SIGNED_COLUMN!r} column")
    signed = news_panel[_SIGNED_COLUMN].astype("float64")
    shocks = signed[signed.notna() & (signed != 0.0)]
    return shocks.rename(SHOCK_NAME).sort_index()
