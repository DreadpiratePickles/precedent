"""Research: tools that catch look-ahead. Shift a panel, audit cutoffs, trace an IC lag curve.

A correctly aligned panel has no signal that a one-session-earlier shift of itself would add;
a panel that leaks does. `shift_panel` makes that test possible, `cutoff_audit` checks the
alignment rule on the events behind a panel, and `lag_curve` shows how quickly the signal dies
as the panel is delayed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

DEFAULT_LAGS = (1, 2, 5)


def shift_panel(panel: pd.DataFrame, sessions: pd.DatetimeIndex, k_sessions: int) -> pd.DataFrame:
    """Move every row `k_sessions` positions along `sessions`: positive is later, negative earlier.

    A row that lands outside the calendar is dropped, never extrapolated.
    """
    if "datetime" not in panel.index.names:
        raise ValueError(f"panel needs an index level named 'datetime', got {panel.index.names}")
    calendar = sessions.as_unit("ns")
    dates = panel.index.get_level_values("datetime")
    position = calendar.get_indexer(dates.as_unit("ns"))
    n_unknown = int((position < 0).sum())
    if n_unknown:
        raise ValueError(f"{n_unknown} panel rows have a datetime that is not in sessions")
    moved = position + k_sessions
    on_calendar = (moved >= 0) & (moved < len(calendar))
    kept = panel[on_calendar]
    new_dates = calendar[moved[on_calendar]]
    level = panel.index.names.index("datetime")
    arrays = [kept.index.get_level_values(i) for i in range(panel.index.nlevels)]
    arrays[level] = new_dates.as_unit(dates.unit)
    shifted = kept.copy()
    shifted.index = pd.MultiIndex.from_arrays(arrays, names=panel.index.names)
    return shifted.sort_index()


def lag_curve(
    fit_fn: Callable[[pd.DataFrame], float],
    panel: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    lags: Sequence[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Mean rank IC (as returned by `fit_fn`) of the panel delayed by each lag, in sessions."""
    values = [float(fit_fn(shift_panel(panel, sessions, lag))) for lag in lags]
    return pd.DataFrame({"lag_sessions": list(lags), "mean_rank_ic": values})


def cutoff_audit(events: pd.DataFrame, session_cutoffs_utc: pd.Series) -> pd.DataFrame:
    """Return the events that break `cutoff(effective_session) >= available_utc > previous cutoff`.

    `events` has a naive `effective_session` and a timezone-aware `available_utc`;
    `session_cutoffs_utc` is indexed by naive session date. The first session has no previous
    cutoff. An event whose session is missing or not in the calendar is a violation too. An empty
    frame means the alignment rule holds; the result adds `cutoff_utc` and `previous_cutoff_utc`.
    """
    if not isinstance(events["available_utc"].dtype, pd.DatetimeTZDtype):
        raise ValueError("events.available_utc must be timezone-aware")
    cutoffs = session_cutoffs_utc.dt.tz_convert("UTC").dt.as_unit("ns")
    cutoffs = cutoffs.set_axis(pd.DatetimeIndex(cutoffs.index).as_unit("ns")).sort_index()
    previous = cutoffs.shift(1)
    sessions = pd.DatetimeIndex(events["effective_session"]).as_unit("ns")
    available = events["available_utc"].dt.tz_convert("UTC").dt.as_unit("ns")
    cutoff = pd.Series(cutoffs.reindex(sessions).to_numpy(), index=events.index)
    previous_cutoff = pd.Series(previous.reindex(sessions).to_numpy(), index=events.index)
    cutoff = pd.to_datetime(cutoff, utc=True)
    previous_cutoff = pd.to_datetime(previous_cutoff, utc=True)
    first_session = pd.Series(sessions == cutoffs.index[0], index=events.index)
    in_window = (cutoff >= available) & (first_session | (available > previous_cutoff))
    audited = events.assign(cutoff_utc=cutoff, previous_cutoff_utc=previous_cutoff)
    return audited[~np.asarray(in_window, dtype=bool)]
