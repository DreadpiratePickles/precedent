"""The event sequence library: what followed a country's event, as dated binary outcomes."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

from precedent.libraries.common import (
    LIBRARY_COLUMNS,
    empty_library_rows,
    validate_library_rows,
)

__all__ = [
    "SequenceLibraryReport",
    "build_event_sequence_library",
    "unconditional_sequence_rate",
]

_OUTCOME_TYPE = "event_sequence"
_OUTCOME_SUFFIX = "linked_country"


@dataclass(frozen=True)
class SequenceLibraryReport:
    rows: int
    skipped_incomplete_window: int
    skipped_no_country: int


@dataclass(frozen=True)
class _TypedCountryEvents:
    """First-report positions of one event type in one country, sorted, with their stories."""

    positions: np.ndarray
    story_ids: np.ndarray


def build_event_sequence_library(
    events: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    event_types: tuple[str, ...],
    horizons: tuple[int, ...] = (5, 20, 60),
    feed_end_position: int | None = None,
) -> tuple[pd.DataFrame, SequenceLibraryReport]:
    """One binary row per first report, event type and horizon.

    For a first report `e` at position `p` with actor country `c = e.linked_countries[0]`,
    `value` is 1.0 when a first report of the type, from another story, linked to `c`, has
    `effective_position` in `p + 1 .. p + h`, else 0.0. `h` is counted from `p` itself (no
    trade-open shift). A triple with `p + h` past `feed_end_position` (the last session the
    feed covers; default the last session of `sessions`, which a padded calendar overstates,
    P-45) is skipped and counted in
    `skipped_incomplete_window`; a first report without a linked country is counted once in
    `skipped_no_country`. No rows gives `empty_library_rows()`.
    """
    if any(horizon < 1 for horizon in horizons):
        raise ValueError(f"horizons must be positive session counts, got {horizons}")
    last_covered = len(sessions) - 1 if feed_end_position is None else feed_end_position
    index = _first_reports_by_type_and_country(events, set(event_types))
    records: list[dict[str, object]] = []
    skipped_incomplete = 0
    skipped_no_country = 0
    firsts = events.loc[events["is_first_report"]]
    for event in firsts.itertuples(index=False):
        if len(event.linked_countries) == 0:
            skipped_no_country += 1
            continue
        country = event.linked_countries[0]
        position = int(event.effective_position)
        for event_type in event_types:
            later = index.get((event_type, country))
            for horizon in horizons:
                if position + horizon > last_covered:
                    skipped_incomplete += 1
                    continue
                records.append(
                    {
                        "event_id": event.event_id,
                        "outcome_type": _OUTCOME_TYPE,
                        "outcome_key": f"{event_type}/{_OUTCOME_SUFFIX}",
                        "horizon": horizon,
                        "value": _follow_on(later, event.story_cluster_id, position, horizon),
                        "is_binary": True,
                        "effective_session": sessions[position],
                        "response_complete_session": sessions[position + horizon],
                        "source_ref": event.source_ref,
                    }
                )
    report_counts = {
        "skipped_incomplete_window": skipped_incomplete,
        "skipped_no_country": skipped_no_country,
    }
    if not records:
        return empty_library_rows(), SequenceLibraryReport(rows=0, **report_counts)
    rows = validate_library_rows(pd.DataFrame(records, columns=list(LIBRARY_COLUMNS)))
    return rows, SequenceLibraryReport(rows=len(rows), **report_counts)


def unconditional_sequence_rate(
    events: pd.DataFrame,
    event_type: str,
    country_group: str,
    horizon: int,
    end_position: int,
    country_groups: dict[str, str],
) -> float | None:
    """The base rate of a follow-on of `event_type` for a country of `country_group`.

    The countries are those in `country_groups` with this group that appear in any row's
    `linked_countries`. Over each such country and each start `s` in `0 .. end_position -
    horizon`, a hit is a first report of `event_type` linked to the country at a position in
    `s + 1 .. s + horizon`. Returns hits / (countries x starts); `None` when the group has no
    such country or `end_position < horizon`.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be a positive session count, got {horizon}")
    if end_position - horizon < 0:
        return None
    countries = sorted(
        {
            country
            for linked in events["linked_countries"]
            for country in linked
            if country_groups.get(country) == country_group
        }
    )
    if not countries:
        return None
    positions_by_country = _first_report_positions(events, event_type)
    starts = np.arange(0, end_position - horizon + 1)
    hits = 0
    for country in countries:
        positions = positions_by_country.get(country)
        if positions is None:
            continue
        nearest = np.searchsorted(positions, starts + 1, side="left")
        found = nearest < len(positions)
        hits += int(np.count_nonzero(positions[nearest[found]] <= starts[found] + horizon))
    return hits / (len(countries) * len(starts))


def _first_report_positions(events: pd.DataFrame, event_type: str) -> dict[str, np.ndarray]:
    """Sorted unique first-report positions of `event_type`, per linked country."""
    collected: dict[str, list[int]] = defaultdict(list)
    selected = events.loc[events["is_first_report"] & (events["event_type"] == event_type)]
    for linked, position in zip(
        selected["linked_countries"], selected["effective_position"], strict=True
    ):
        for country in set(linked):
            collected[country].append(int(position))
    return {country: np.unique(np.array(found)) for country, found in collected.items()}


def _first_reports_by_type_and_country(
    events: pd.DataFrame, event_types: set[str]
) -> dict[tuple[str, str], _TypedCountryEvents]:
    collected: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    selected = events.loc[events["is_first_report"] & events["event_type"].isin(event_types)]
    for event in selected.itertuples(index=False):
        for country in set(event.linked_countries):
            collected[(event.event_type, country)].append(
                (int(event.effective_position), str(event.story_cluster_id))
            )
    index: dict[tuple[str, str], _TypedCountryEvents] = {}
    for key, found in collected.items():
        found.sort()
        index[key] = _TypedCountryEvents(
            positions=np.array([position for position, _ in found]),
            story_ids=np.array([story for _, story in found], dtype=object),
        )
    return index


def _follow_on(
    later: _TypedCountryEvents | None, story_cluster_id: object, position: int, horizon: int
) -> float:
    if later is None:
        return 0.0
    start = np.searchsorted(later.positions, position + 1, side="left")
    stop = np.searchsorted(later.positions, position + horizon, side="right")
    return 1.0 if (later.story_ids[start:stop] != story_cluster_id).any() else 0.0
