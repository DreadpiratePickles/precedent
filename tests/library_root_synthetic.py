"""Synthetic data dirs holding a written library bundle, for the engine tests (not collected).

`write_library_root` builds 20 independent first-report `central_bank / central_bank / US / high /
unscheduled` events (25 sessions apart, so independent at every horizon up to 20) with
`event_sequence` and `market_resolution` rows and no asset rows. `write_sparse_library_root` holds
only 3 `cyber_incident / company / US / high / unscheduled` events. The calendar is rebuilt here
so that this module does not depend on another test directory being on `sys.path`.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd

from precedent.libraries.bundle import PLACEBO_COLUMNS, write_bundle
from precedent.libraries.common import LIBRARY_COLUMNS, empty_library_rows, validate_library_rows
from precedent.libraries.events_table import EVENTS_TABLE_COLUMNS
from precedent.libraries.manifest import ManifestSource
from precedent.libraries.market_resolution import MARKETS_FRAME_COLUMNS
from precedent.spine.pit_alignment import build_session_cutoffs

ROOT_SESSIONS = pd.bdate_range("2020-01-02", periods=600)
ROOT_CUTOFFS = build_session_cutoffs(ROOT_SESSIONS, dt.time(16, 0))

CENTRAL_BANK_KEY = "central_bank/central_bank/none/US/high/unscheduled/0"
CYBER_KEY = "cyber_incident/company/none/US/high/unscheduled/0"
SEQUENCE_OUTCOME_KEY = "election_or_political/linked_country"
MARKET_OUTCOME_KEY = "central_bank/central_bank/none/US/high/scheduled/0"
THIN_MARKET_OUTCOME_KEY = "central_bank/central_bank/bond/US/high/scheduled/-1"
ASSET_SYMBOL = "SPY"

EVENT_COUNT = 20
EVENT_SPACING_SESSIONS = 25
FIRST_EVENT_POSITION = 10
THIN_MARKET_EVENT_COUNT = 3
PLACEBO_FIRST_POSITION = 300

_ONE_HOUR = pd.Timedelta(hours=1)
_ASSET_HORIZONS = (1, 5, 20)
_SEQUENCE_HORIZONS = (5, 20, 60)


def _events(key: str, event_type: str, count: int, spacing: int, label: str) -> pd.DataFrame:
    records = []
    for index in range(count):
        position = FIRST_EVENT_POSITION + index * spacing
        records.append(
            {
                "event_id": f"gdelt:{label}{index}",
                "key": key,
                "event_type": event_type,
                "country_group": "US",
                "available_utc": ROOT_CUTOFFS.iloc[position] - _ONE_HOUR,
                "effective_session": ROOT_SESSIONS[position].date(),
                "effective_position": position,
                "primary_actor": "USA",
                "linked_countries": ["USA"],
                "story_cluster_id": f"story:{label}{index}",
                "is_first_report": True,
                "one_line": f"{label} event {index}",
                "source": "gdelt",
                "source_ref": f"gdelt:{label}{index}",
            }
        )
    return pd.DataFrame(records, columns=list(EVENTS_TABLE_COLUMNS))


def _row(
    event: pd.Series, outcome_type: str, outcome_key: str, horizon: int, value: float
) -> dict[str, Any]:
    position = int(event["effective_position"])
    return {
        "event_id": event["event_id"],
        "outcome_type": outcome_type,
        "outcome_key": outcome_key,
        "horizon": horizon,
        "value": value,
        "is_binary": outcome_type != "asset",
        "effective_session": ROOT_SESSIONS[position].date(),
        "response_complete_session": ROOT_SESSIONS[position + horizon].date(),
        "source_ref": event["source_ref"],
    }


def _sequence_rows(events: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        _row(event, "event_sequence", SEQUENCE_OUTCOME_KEY, horizon, _sequence_value(index, horizon))
        for index, (_, event) in enumerate(events.iterrows())
        for horizon in _SEQUENCE_HORIZONS
    ]


def _sequence_value(index: int, horizon: int) -> float:
    return 0.0 if (index + horizon) % 7 == 0 else 1.0


def _market_rows(events: pd.DataFrame, key: str, count: int) -> list[dict[str, Any]]:
    return [
        _row(event, "market_resolution", key, 1 + index % 5, 1.0 if index % 3 else 0.0)
        for index, (_, event) in enumerate(events.head(count).iterrows())
    ]


def _asset_rows(events: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        _row(event, "asset", ASSET_SYMBOL, horizon, 0.004 * ((index % 5) - 1) * horizon**0.5)
        for index, (_, event) in enumerate(events.iterrows())
        for horizon in _ASSET_HORIZONS
    ]


def _empty_placebo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome_key": pd.Series(dtype="str"),
            "horizon": pd.Series(dtype="int64"),
            "value": pd.Series(dtype="float64"),
            "response_complete_session": pd.Series(dtype="datetime64[ns]"),
        }
    )[list(PLACEBO_COLUMNS)]


def _asset_placebo() -> pd.DataFrame:
    positions = list(range(PLACEBO_FIRST_POSITION, PLACEBO_FIRST_POSITION + 100))
    return pd.DataFrame(
        {
            "outcome_key": pd.Series([ASSET_SYMBOL] * len(positions), dtype="str"),
            "horizon": pd.Series([5] * len(positions), dtype="int64"),
            "value": pd.Series(
                [0.002 * ((position % 7) - 3) for position in positions], dtype="float64"
            ),
            "response_complete_session": pd.Series(
                ROOT_SESSIONS[positions].as_unit("ns"), dtype="datetime64[ns]"
            ),
        }
    )[list(PLACEBO_COLUMNS)]


def _markets(key: str) -> pd.DataFrame:
    positions = [30 + 40 * index for index in range(10)]
    resolved = [ROOT_CUTOFFS.iloc[position] - _ONE_HOUR for position in positions]
    created = [ROOT_CUTOFFS.iloc[0] - _ONE_HOUR] * len(positions)
    frame = pd.DataFrame(
        {
            "market_id": pd.Series([f"kalshi:KX-{n}" for n in range(10)], dtype="str"),
            "outcome_key": pd.Series([key] * len(positions), dtype="str"),
            "created_utc": pd.Series(pd.to_datetime(created, utc=True).as_unit("ns")),
            "resolved_utc": pd.Series(pd.to_datetime(resolved, utc=True).as_unit("ns")),
            "value": pd.Series([1.0 if n % 2 else 0.0 for n in range(10)], dtype="float64"),
            "response_complete_session": pd.Series(
                ROOT_SESSIONS[positions].as_unit("ns"), dtype="datetime64[ns]"
            ),
        }
    )
    return frame[list(MARKETS_FRAME_COLUMNS)]


def _sources(events: pd.DataFrame, markets: int) -> list[ManifestSource]:
    sources = [
        ManifestSource(
            name="gdelt2",
            first_date=events["effective_session"].min(),
            last_date=events["effective_session"].max(),
            rows=len(events),
            citation="synthetic gdelt citation",
        )
    ]
    if markets:
        sources.append(
            ManifestSource(
                name="kalshi",
                first_date=dt.date(2020, 3, 1),
                last_date=dt.date(2021, 8, 30),
                rows=markets,
                citation="synthetic kalshi citation",
            )
        )
    return sources


def _write(
    root: Path,
    events: pd.DataFrame,
    rows: list[dict[str, Any]],
    markets: pd.DataFrame,
    placebo: pd.DataFrame,
    synthetic: bool,
) -> Path:
    frame = (
        validate_library_rows(pd.DataFrame(rows, columns=list(LIBRARY_COLUMNS)))
        if rows
        else empty_library_rows()
    )
    write_bundle(
        root / "library",
        events=events,
        rows=frame,
        placebo_asset=placebo,
        markets=markets,
        sessions=ROOT_SESSIONS,
        session_cutoffs_utc=ROOT_CUTOFFS,
        build_report={"synthetic": synthetic},
        sources=_sources(events, len(markets)),
        commit="0" * 40,
    )
    return root


def write_library_root(
    root: Path,
    *,
    with_assets: bool = False,
    with_thin_market_key: bool = False,
    synthetic: bool = True,
) -> Path:
    """Write the 20-event central bank library under `root / "library"` and return `root`.

    `with_assets` adds asset rows and an asset placebo (the default has none, so `asset` is not
    built); `with_thin_market_key` adds a second market key with only 3 precedents;
    `synthetic=False` leaves the build report without the synthetic marker (a "live" library).
    """
    events = _events(CENTRAL_BANK_KEY, "central_bank", EVENT_COUNT, EVENT_SPACING_SESSIONS, "cb")
    rows = _sequence_rows(events) + _market_rows(events, MARKET_OUTCOME_KEY, EVENT_COUNT)
    if with_thin_market_key:
        rows += _market_rows(events, THIN_MARKET_OUTCOME_KEY, THIN_MARKET_EVENT_COUNT)
    placebo = _empty_placebo()
    if with_assets:
        rows += _asset_rows(events)
        placebo = _asset_placebo()
    return _write(root, events, rows, _markets(MARKET_OUTCOME_KEY), placebo, synthetic)


def write_sparse_library_root(root: Path) -> Path:
    """Write a library of 3 `cyber_incident / company / US / high / unscheduled` events."""
    events = _events(CYBER_KEY, "cyber_incident", 3, EVENT_SPACING_SESSIONS, "cy")
    rows = _sequence_rows(events)
    empty_markets = _markets(MARKET_OUTCOME_KEY).iloc[0:0]
    return _write(root, events, rows, empty_markets, _empty_placebo(), True)
