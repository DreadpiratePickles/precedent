"""Tensions and shifts to a sparse (datetime, instrument) factor panel for Qlib.

The panel is bi-temporal. For each session the builder uses, per tension or shift id, the
latest stored line whose `recorded_utc` is at or before that session's cutoff; a record, or a
close or re-fire, recorded after the cutoff does not exist for that session. That is what
makes a row at session d independent of anything learned later (truncation invariance).

A record contributes from the session it is placed at (its opened or detected session, moved
to the next session on the calendar if that date is not one; dropped when it falls after the
last session) for `max_age_sessions` sessions, decaying by `0.5 ** (age / half_life_sessions)`.
Subjects that are link ids become instrument rows too; they fall away at Qlib's left join.
"""

from bisect import bisect_left
from collections import defaultdict
from collections.abc import Iterator
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from precedent.dream.claim_schema import ShiftKind, TensionKind
from precedent.dream.claim_store import ShiftStore, TensionStore

PANEL_COLUMNS = [
    "dream_open_tensions",
    "dream_text_vs_price",
    "dream_rate_shift",
    "dream_direction_drift",
    "dream_days_since_tension",
]
_CLAIM_SIGNS = {"positive": 1.0, "negative": -1.0}
_SHIFT_COLUMNS = {
    ShiftKind.event_rate: "dream_rate_shift",
    ShiftKind.direction_drift: "dream_direction_drift",
}


def _validate_inputs(
    sessions: pd.DatetimeIndex,
    session_cutoffs_utc: pd.Series,
    max_age_sessions: int,
    half_life_sessions: float,
) -> list[pd.Timestamp]:
    """Check the arguments; return the cutoffs as a list of UTC timestamps, one per session."""
    if not half_life_sessions > 0:
        raise ValueError(f"half_life_sessions must be > 0, got {half_life_sessions}")
    if max_age_sessions < 0:
        raise ValueError(f"max_age_sessions must be >= 0, got {max_age_sessions}")
    if sessions.tz is not None or not (sessions.is_unique and sessions.is_monotonic_increasing):
        raise ValueError("sessions must be a naive, strictly increasing DatetimeIndex")
    cutoffs = session_cutoffs_utc
    if not pd.api.types.is_datetime64_any_dtype(cutoffs) or cutoffs.dt.tz is None:
        raise ValueError("session_cutoffs_utc must hold timezone-aware datetimes")
    if len(cutoffs) != len(sessions) or not (cutoffs.index == sessions).all():
        raise ValueError("session_cutoffs_utc must be indexed by exactly the given sessions")
    if cutoffs.isna().any() or not cutoffs.is_monotonic_increasing or not cutoffs.is_unique:
        raise ValueError("session_cutoffs_utc must be strictly increasing with no missing values")
    return list(cutoffs.dt.tz_convert("UTC"))


def _session_position(sessions: pd.DatetimeIndex, day: date) -> int:
    """Position of the first session at or after `day`; `len(sessions)` when there is none."""
    return int(sessions.as_unit("ns").searchsorted(pd.Timestamp(day).as_unit("ns"), side="left"))


def _visible_spans(
    versions: list[Any], cutoffs: list[pd.Timestamp]
) -> Iterator[tuple[Any, int, int]]:
    """Yield (record, first, stop): the sessions in [first, stop) at which it is the latest line.

    A line is seen from the first session whose cutoff is at or after its `recorded_utc`; a later
    line (in file order) that is already seen replaces it.
    """
    first_seen = [bisect_left(cutoffs, pd.Timestamp(record.recorded_utc)) for record in versions]
    stop = len(cutoffs)
    for record, first in reversed(list(zip(versions, first_seen, strict=True))):
        if first < stop:
            yield record, first, stop
        stop = min(stop, first)


def _by_id(records: list[Any], id_field: str) -> dict[str, list[Any]]:
    grouped: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        grouped[getattr(record, id_field)].append(record)
    return grouped


def _empty_panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]).as_unit("ns"), pd.Index([], dtype=str)],
        names=["datetime", "instrument"],
    )
    return pd.DataFrame({name: pd.Series(dtype="float64") for name in PANEL_COLUMNS}).set_axis(
        index
    )


def build_dream_factor_panel(
    tension_store: TensionStore,
    shift_store: ShiftStore,
    sessions: pd.DatetimeIndex,
    session_cutoffs_utc: pd.Series,
    max_age_sessions: int = 20,
    half_life_sessions: float = 5.0,
) -> pd.DataFrame:
    """Per-(session, instrument) Dream factors; only rows with a live tension or shift exist.

    - `dream_open_tensions`: tensions (any kind) not closed as of the cutoff, within max age.
    - `dream_text_vs_price`: text_vs_price tensions, signed by `details["claim_value"]`
      (positive +1, negative -1, otherwise 0), decayed.
    - `dream_rate_shift`, `dream_direction_drift`: event_rate / direction_drift shifts, their
      statistic decayed. link_churn shifts have no column and create no row.
    - `dream_days_since_tension`: sessions since the latest tension opened (status irrelevant),
      capped at `max_age_sessions`, which is also the value when none has opened.
    """
    cutoffs = _validate_inputs(sessions, session_cutoffs_utc, max_age_sessions, half_life_sessions)
    n_sessions = len(sessions)
    # per (position, instrument): [open, text_vs_price, rate, drift, min tension age]
    cells: dict[tuple[int, str], list[float]] = defaultdict(
        lambda: [0.0, 0.0, 0.0, 0.0, float(max_age_sessions)]
    )

    def live_sessions(anchor_day: date, first: int, stop: int) -> Iterator[tuple[int, int]]:
        """Yield (position, age) for the sessions in [first, stop) within the age window."""
        anchor = _session_position(sessions, anchor_day)
        low = max(first, anchor)
        high = min(stop - 1, anchor + max_age_sessions, n_sessions - 1)
        for position in range(low, high + 1):
            yield position, position - anchor

    tensions = _by_id(tension_store.read_all(), "tension_id")
    for versions in tensions.values():
        for tension, first, stop in _visible_spans(versions, cutoffs):
            sign = 0.0
            if tension.kind is TensionKind.text_vs_price:
                sign = _CLAIM_SIGNS.get(str(tension.details.get("claim_value")), 0.0)
            for position, age in live_sessions(tension.opened_session, first, stop):
                cell = cells[(position, tension.subject)]
                cell[0] += tension.status != "closed"
                cell[1] += sign * 0.5 ** (age / half_life_sessions)
                cell[4] = min(cell[4], float(age))

    shifts = _by_id(shift_store.read_all(), "shift_id")
    for versions in shifts.values():
        for shift, first, stop in _visible_spans(versions, cutoffs):
            column = _SHIFT_COLUMNS.get(shift.kind)
            if column is None:
                continue
            slot = PANEL_COLUMNS.index(column)
            for position, age in live_sessions(shift.detected_session, first, stop):
                cells[(position, shift.subject)][slot] += shift.statistic * 0.5 ** (
                    age / half_life_sessions
                )

    if not cells:
        return _empty_panel()
    keys = sorted(cells, key=lambda key: (key[0], key[1]))
    index = pd.MultiIndex.from_arrays(
        [
            sessions.as_unit("ns")[np.array([position for position, _ in keys])],
            pd.Index([instrument for _, instrument in keys], dtype=str),
        ],
        names=["datetime", "instrument"],
    )
    values = np.array([cells[key] for key in keys], dtype="float64")
    return pd.DataFrame(values, index=index, columns=PANEL_COLUMNS)
