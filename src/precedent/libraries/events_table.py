"""The events table: one row per aligned event record, with story clusters assigned."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from precedent.engine.story_clusters import assign_story_clusters
from precedent.spine.event_key import build_event_key
from precedent.spine.event_record import EventRecord

__all__ = ["EVENTS_TABLE_COLUMNS", "EventsTableReport", "build_events_table", "one_line"]

EVENTS_TABLE_COLUMNS = (
    "event_id",
    "key",
    "event_type",
    "country_group",
    "available_utc",
    "effective_session",
    "effective_position",
    "primary_actor",
    "linked_countries",
    "story_cluster_id",
    "is_first_report",
    "one_line",
    "source",
    "source_ref",
)

_ONE_LINE_MAX_CHARACTERS = 60


@dataclass(frozen=True)
class EventsTableReport:
    records: int
    kept: int
    unaligned: int


def one_line(record: EventRecord) -> str:
    """`actor: action` for a card, cut to 60 characters."""
    event = record.event
    return f"{event.actor.name}: {event.action}"[:_ONE_LINE_MAX_CHARACTERS]


def _primary_actor(record: EventRecord) -> str:
    if record.resolved_impacts:
        return record.resolved_impacts[0].instrument
    if record.linked_countries:
        return record.linked_countries[0]
    return record.event.actor.name


def build_events_table(
    records: list[EventRecord],
    sessions: pd.DatetimeIndex,
    window_sessions: int = 3,
) -> tuple[pd.DataFrame, EventsTableReport]:
    """Tabulate `records` that have an `effective_session` found in `sessions`.

    A record without a session, or whose session is not in `sessions`, is skipped and counted
    in `unaligned`. `effective_position` is the session's index in `sessions`; `effective_session`
    stays a `datetime.date`; `available_utc` is tz-aware UTC. Rows keep the order of `records`.
    Story clusters use `window_sessions` (see `assign_story_clusters`).
    """
    if sessions.tz is not None:
        raise ValueError("sessions must be timezone-naive dates")
    position_of = {session.date(): position for position, session in enumerate(sessions)}
    rows: list[dict[str, object]] = []
    for record in records:
        position = (
            None if record.effective_session is None else position_of.get(record.effective_session)
        )
        if position is None:
            continue
        if record.available_utc is None:
            raise ValueError(f"record {record.article_id} has a session but no available_utc")
        key = build_event_key(record)
        rows.append(
            {
                "event_id": record.article_id,
                "key": key.as_string(),
                "event_type": key.event_type,
                "country_group": key.country_group,
                "available_utc": record.available_utc,
                "effective_session": record.effective_session,
                "effective_position": position,
                "primary_actor": _primary_actor(record),
                "linked_countries": list(record.linked_countries),
                "one_line": one_line(record),
                "source": record.source,
                "source_ref": record.article_id,
            }
        )
    report = EventsTableReport(
        records=len(records), kept=len(rows), unaligned=len(records) - len(rows)
    )
    return _clustered(rows, window_sessions), report


def _clustered(rows: list[dict[str, object]], window_sessions: int) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "event_id": pd.Series([r["event_id"] for r in rows], dtype="str"),
            "key": pd.Series([r["key"] for r in rows], dtype="str"),
            "event_type": pd.Series([r["event_type"] for r in rows], dtype="str"),
            "country_group": pd.Series([r["country_group"] for r in rows], dtype="str"),
            "available_utc": pd.to_datetime(
                pd.Series([r["available_utc"] for r in rows], dtype=object), utc=True
            ).astype("datetime64[ns, UTC]"),
            "effective_session": pd.Series([r["effective_session"] for r in rows], dtype=object),
            "effective_position": pd.Series([r["effective_position"] for r in rows], dtype="int64"),
            "primary_actor": pd.Series([r["primary_actor"] for r in rows], dtype="str"),
            "linked_countries": pd.Series([r["linked_countries"] for r in rows], dtype=object),
            "one_line": pd.Series([r["one_line"] for r in rows], dtype="str"),
            "source": pd.Series([r["source"] for r in rows], dtype="str"),
            "source_ref": pd.Series([r["source_ref"] for r in rows], dtype="str"),
        }
    )
    return assign_story_clusters(frame, window_sessions)[list(EVENTS_TABLE_COLUMNS)]
