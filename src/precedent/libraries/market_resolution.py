"""The market resolution library: how prediction markets open at an event resolved."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from precedent.libraries.common import empty_library_rows, validate_library_rows
from precedent.spine.market_record import MarketRecord

__all__ = [
    "MARKETS_FRAME_COLUMNS",
    "MarketLibraryReport",
    "build_market_resolution_library",
    "unconditional_market_rate",
]

MARKETS_FRAME_COLUMNS = (
    "market_id",
    "outcome_key",
    "created_utc",
    "resolved_utc",
    "value",
    "response_complete_session",
)

_OUTCOME_TYPE = "market_resolution"


@dataclass(frozen=True)
class MarketLibraryReport:
    markets_read: int
    markets_untyped: int
    markets_refused_invalid: int
    rows: int
    collapsed_duplicates: int
    skipped_no_session: int


def _utc_ns(values: pd.Series | pd.DatetimeIndex) -> np.ndarray:
    """Naive datetime64[ns] holding the UTC instants of tz-aware `values`."""
    series = pd.Series(values)
    return series.dt.tz_convert("UTC").dt.tz_localize(None).dt.as_unit("ns").to_numpy()


def _resolution_value(market: MarketRecord) -> float:
    return 1.0 if market.resolution == "yes" else 0.0


def _markets_frame(
    typed: list[MarketRecord], positions: np.ndarray, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    session_ns = sessions.as_unit("ns").to_numpy()
    response = np.full(len(typed), np.datetime64("NaT"), dtype="datetime64[ns]")
    has_session = positions < len(sessions)
    response[has_session] = session_ns[positions[has_session]]
    created = pd.to_datetime(pd.Series([m.created_utc for m in typed], dtype=object), utc=True)
    resolved = pd.to_datetime(pd.Series([m.resolved_utc for m in typed], dtype=object), utc=True)
    return pd.DataFrame(
        {
            "market_id": pd.Series([m.market_id for m in typed], dtype="str"),
            "outcome_key": pd.Series([m.typed.key.as_string() for m in typed], dtype="str"),
            "created_utc": created.astype("datetime64[ns, UTC]"),
            "resolved_utc": resolved.astype("datetime64[ns, UTC]"),
            "value": pd.Series([_resolution_value(m) for m in typed], dtype="float64"),
            "response_complete_session": pd.Series(response, dtype="datetime64[ns]"),
        }
    )[list(MARKETS_FRAME_COLUMNS)]


def build_market_resolution_library(
    events: pd.DataFrame,
    markets: list[MarketRecord],
    sessions: pd.DatetimeIndex,
    session_cutoffs_utc: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, MarketLibraryReport]:
    """Rows of resolved prediction markets that were open when each first-report event broke.

    For an event at `t = available_utc`, a typed market qualifies when
    `created_utc < t < resolved_utc`. Its row carries the outcome key of the market's type,
    value 1.0 for YES and 0.0 for NO, and the session of its resolution (the first session whose
    cutoff is at or after `resolved_utc`) as `response_complete_session`. When several markets
    qualify for one `(event, outcome_key)`, the earliest resolving (ties by `market_id`) is kept
    and the others are counted in `collapsed_duplicates`. A qualifying market resolved after the
    last cutoff has no session: it is counted in `skipped_no_session` and gives no row.

    Returns `(rows, markets_frame, report)`; `markets_frame` holds every typed market, with
    `response_complete_session` NaT for a resolution after the last cutoff.
    """
    cutoff_ns = _utc_ns(session_cutoffs_utc)
    if len(cutoff_ns) != len(sessions):
        raise ValueError("sessions and session_cutoffs_utc must have the same length")
    typed = sorted(
        (m for m in markets if m.typed.status == "typed"),
        key=lambda m: (m.resolved_utc, m.market_id),
    )
    untyped = sum(1 for m in markets if m.typed.status == "untyped")
    refused = sum(1 for m in markets if m.typed.status == "refused_invalid")
    resolved_ns = _utc_ns(pd.Series([m.resolved_utc for m in typed], dtype="datetime64[ns, UTC]"))
    created_ns = _utc_ns(pd.Series([m.created_utc for m in typed], dtype="datetime64[ns, UTC]"))
    positions = np.searchsorted(cutoff_ns, resolved_ns, side="left")
    markets_frame = _markets_frame(typed, positions, sessions)
    keys = [m.typed.key.as_string() for m in typed]

    rows: list[dict[str, object]] = []
    collapsed = 0
    skipped = 0
    first_reports = events[events["is_first_report"]]
    event_times = _utc_ns(first_reports["available_utc"])
    for event_id, effective_position, event_ns in zip(
        first_reports["event_id"], first_reports["effective_position"], event_times, strict=True
    ):
        open_now = np.flatnonzero((created_ns < event_ns) & (event_ns < resolved_ns))
        seen: set[str] = set()
        for index in open_now:
            if positions[index] >= len(sessions):
                skipped += 1
            elif keys[index] in seen:
                collapsed += 1
            else:
                seen.add(keys[index])
                rows.append(
                    _row(
                        str(event_id),
                        int(effective_position),
                        typed[index],
                        keys[index],
                        int(positions[index]),
                        sessions,
                    )
                )
    library_rows = validate_library_rows(pd.DataFrame(rows)) if rows else empty_library_rows()
    report = MarketLibraryReport(
        markets_read=len(markets),
        markets_untyped=untyped,
        markets_refused_invalid=refused,
        rows=len(library_rows),
        collapsed_duplicates=collapsed,
        skipped_no_session=skipped,
    )
    return library_rows, markets_frame, report


def _row(
    event_id: str,
    effective_position: int,
    market: MarketRecord,
    outcome_key: str,
    position: int,
    sessions: pd.DatetimeIndex,
) -> dict[str, object]:
    return {
        "event_id": event_id,
        "outcome_type": _OUTCOME_TYPE,
        "outcome_key": outcome_key,
        "horizon": position - effective_position,
        "value": _resolution_value(market),
        "is_binary": True,
        "effective_session": sessions[effective_position],
        "response_complete_session": sessions[position],
        "source_ref": market.market_id,
    }


def unconditional_market_rate(
    markets: pd.DataFrame,
    outcome_key: str,
    cutoff_utc: pd.Timestamp,
    session_cutoffs_utc: pd.Series,
) -> float | None:
    """The YES rate of markets with `outcome_key` whose resolution session closed by `cutoff_utc`.

    A market counts when its `response_complete_session` is known and that session's cutoff is
    at or before `cutoff_utc`. Returns None when no market counts.
    """
    known = markets[
        (markets["outcome_key"] == outcome_key) & markets["response_complete_session"].notna()
    ]
    if known.empty:
        return None
    resolution_cutoffs = session_cutoffs_utc.reindex(
        pd.DatetimeIndex(known["response_complete_session"])
    )
    closed = (resolution_cutoffs <= cutoff_utc).to_numpy()
    if not closed.any():
        return None
    return float(known["value"].to_numpy()[closed].mean())
