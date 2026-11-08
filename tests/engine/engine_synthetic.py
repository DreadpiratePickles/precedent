"""Shared synthetic frame builders for the engine tests (a helper module, not collected)."""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterable

import pandas as pd

from precedent.spine.pit_alignment import build_session_cutoffs

SESSIONS = pd.bdate_range("2020-01-02", periods=400)
CUTOFFS = build_session_cutoffs(SESSIONS, dt.time(16, 0))

_ONE_HOUR = pd.Timedelta(hours=1)


def events_frame(rows: Iterable[tuple[str, str, str, int]]) -> pd.DataFrame:
    """Build an event frame from (event_id, key, primary_actor, effective_position) tuples.

    `available_utc` is the cutoff of the row's session minus one hour, tz-aware UTC.
    """
    records = [
        {
            "event_id": event_id,
            "key": key,
            "primary_actor": primary_actor,
            "effective_position": position,
            "available_utc": CUTOFFS.iloc[position] - _ONE_HOUR,
        }
        for event_id, key, primary_actor, position in rows
    ]
    return pd.DataFrame(
        records,
        columns=["event_id", "key", "primary_actor", "effective_position", "available_utc"],
    )
