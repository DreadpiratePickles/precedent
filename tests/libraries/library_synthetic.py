"""Shared synthetic builders for the library tests (a helper module, not collected)."""

from __future__ import annotations

import datetime as dt
import hashlib
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd
from fakes import make_event_record

from precedent.libraries.common import LIBRARY_COLUMNS
from precedent.libraries.events_table import EVENTS_TABLE_COLUMNS
from precedent.spine.event_key import EventKey, country_group_of
from precedent.spine.event_record import EventRecord, ResolvedImpact
from precedent.spine.event_schema import NewsEvent
from precedent.spine.pit_alignment import build_session_cutoffs

SESSIONS = pd.bdate_range("2020-01-02", periods=400)
CUTOFFS = build_session_cutoffs(SESSIONS, dt.time(16, 0))

ROW_SEQ: dict[str, Any] = {
    "event_id": "e1",
    "outcome_type": "event_sequence",
    "outcome_key": "armed_conflict/government/none/EM/high/unscheduled/0",
    "horizon": 5,
    "value": 1.0,
    "is_binary": True,
    "effective_session": dt.date(2020, 1, 10),
    "response_complete_session": dt.date(2020, 1, 17),
    "source_ref": "gdelt:1",
}
ROW_ASSET: dict[str, Any] = {
    **ROW_SEQ,
    "outcome_type": "asset",
    "outcome_key": "ACME",
    "value": 0.012,
    "is_binary": False,
}

_ONE_HOUR = pd.Timedelta(hours=1)


def _sha256_of(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rows_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """A library-rows frame with the columns in `LIBRARY_COLUMNS` order, values as given."""
    return pd.DataFrame(rows, columns=list(LIBRARY_COLUMNS))


def _aligned_fields(position: int | None) -> dict[str, Any]:
    if position is None:
        return {"available_utc": None, "effective_session": None}
    return {
        "available_utc": (CUTOFFS.iloc[position] - _ONE_HOUR).to_pydatetime(),
        "effective_session": SESSIONS[position].date(),
    }


def _news_event(
    event_type: str,
    actor_name: str,
    actor_kind: str,
    actor_country_iso3: str | None,
    action: str,
    severity: int,
    is_scheduled: bool,
) -> NewsEvent:
    return NewsEvent.model_validate(
        {
            "schema_version": "1.0",
            "event_type": event_type,
            "actor": {"name": actor_name, "kind": actor_kind, "country_iso3": actor_country_iso3},
            "action": action,
            "impacted": [],
            "severity": severity,
            "macro_direction_step": 0,
            "confidence": "high",
            "horizon": "days",
            "novelty": "new",
            "is_scheduled": is_scheduled,
            "evidence_span": "synthetic evidence",
        }
    )


def gdelt_record(
    event_id: str,
    position: int | None = 10,
    actor: str = "MALI",
    action: str = "protest",
    linked: Sequence[str] | None = None,
    event_type: str = "armed_conflict",
    severity: int = 8,
) -> EventRecord:
    """A GDELT-sourced record aligned to `SESSIONS[position]` (None: not aligned).

    `article_id` is `event_id` as given, so the table's `event_id` is what the caller wrote.
    """
    linked_countries = list(linked) if linked is not None else ["MLI"]
    return make_event_record(
        article_id=event_id,
        body_sha256=_sha256_of(event_id),
        source="gdelt",
        event=_news_event(
            event_type,
            actor,
            "government",
            linked_countries[0] if linked_countries else None,
            action,
            severity,
            False,
        ),
        resolved_impacts=[],
        linked_countries=linked_countries,
        **_aligned_fields(position),
    )


def edgar_record(
    event_id: str,
    instrument: str,
    linked: Sequence[str],
    position: int | None = 10,
) -> EventRecord:
    """An EDGAR-sourced earnings record with one resolved impact on `instrument`."""
    return make_event_record(
        article_id=event_id,
        body_sha256=_sha256_of(event_id),
        source="edgar",
        event=_news_event(
            "earnings", f"{instrument} Corp", "company", "USA", "reported results", 5, True
        ),
        resolved_impacts=[
            ResolvedImpact(
                instrument=instrument, kind="company", direction=1.0, match_method="ticker"
            )
        ],
        linked_countries=list(linked),
        **_aligned_fields(position),
    )


def table(rows: Iterable[tuple[Any, ...]]) -> pd.DataFrame:
    """An events table from `(event_id, event_type, linked_countries, position[, first])` tuples.

    Every column `build_events_table` emits. The key is built from the event type with a
    government actor and no impacts, the country group of `linked[0]`, severity high, unscheduled.
    `primary_actor` is `linked[0]`; a row that is not a first report joins the cluster of the
    previous first-report row.
    """
    out: list[dict[str, Any]] = []
    cluster_id = ""
    for row in rows:
        event_id, event_type, linked, position = row[:4]
        is_first_report = bool(row[4]) if len(row) > 4 else True
        if is_first_report:
            cluster_id = event_id
        group = country_group_of(linked[0])
        key = EventKey(event_type, "government", (), group, "high", False, 0)
        out.append(
            {
                "event_id": event_id,
                "key": key.as_string(),
                "event_type": event_type,
                "country_group": group,
                "available_utc": CUTOFFS.iloc[position] - _ONE_HOUR,
                "effective_session": SESSIONS[position].date(),
                "effective_position": position,
                "primary_actor": linked[0],
                "linked_countries": list(linked),
                "story_cluster_id": cluster_id,
                "is_first_report": is_first_report,
                "one_line": f"{linked[0]}: {event_type}"[:60],
                "source": "gdelt",
                "source_ref": event_id,
            }
        )
    frame = pd.DataFrame(out, columns=list(EVENTS_TABLE_COLUMNS))
    frame["effective_position"] = frame["effective_position"].astype("int64")
    frame["available_utc"] = frame["available_utc"].astype("datetime64[ns, UTC]")
    frame["is_first_report"] = frame["is_first_report"].astype(bool)
    return frame
