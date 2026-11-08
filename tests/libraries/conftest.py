"""Fixtures shared by the library tests.

The synthetic builders live in `library_synthetic` and are imported directly by the tests that
need them (`SESSIONS`, `CUTOFFS`, `table`, `rows_frame`, `gdelt_record`, `edgar_record`).

The bundle fixtures: `bundle_parts` (the keyword arguments of `write_bundle` after `dest`),
`bundle_root` (a data dir whose `library/` was written from those parts) and `populated_stores`
(a data dir with 30 GDELT-style event records and 6 resolved markets in the spine stores).
"""

from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd
import pytest
from fakes import make_event_record
from library_synthetic import CUTOFFS, SESSIONS, rows_frame, table

from precedent.libraries.bundle import write_bundle
from precedent.libraries.common import validate_library_rows
from precedent.libraries.manifest import ManifestSource
from precedent.libraries.market_resolution import MARKETS_FRAME_COLUMNS
from precedent.spine.event_key import EventKey
from precedent.spine.event_schema import NewsEvent
from precedent.spine.event_store import EventStore
from precedent.spine.gdelt_events import SOURCE_NAME as GDELT_SOURCE
from precedent.spine.gdelt_events import gdelt_parser_version
from precedent.spine.market_record import MarketRecord
from precedent.spine.market_store import MarketStore
from precedent.spine.session_calendar import calendar_cutoffs, load_nyse_calendar, weekday_sessions
from precedent.spine.typers.base import TypedText
from precedent.spine.typers.rules import RulesTyper

_ONE_HOUR = pd.Timedelta(hours=1)
_FIRST_YEAR_COUNT = 14  # events 0..13 sit in 2020, the rest from position 270 (January 2021)


def _bundle_events() -> pd.DataFrame:
    """28 first reports of two keys across 2020 and 2021, plus one follow-up report."""
    rows: list[tuple[Any, ...]] = []
    for n in range(28):
        position = 10 + n * 10 if n < _FIRST_YEAR_COUNT else 270 + (n - _FIRST_YEAR_COUNT) * 5
        even = n % 2 == 0
        rows.append(
            (
                f"gdelt:{n}",
                "armed_conflict" if even else "central_bank",
                ["MLI"] if even else ["USA", "CHN"],
                position,
            )
        )
    rows.append(("gdelt:100", "armed_conflict", ["MLI"], 12, False))
    return table(rows)


def _row(
    event_id: str, outcome_type: str, outcome_key: str, horizon: int, value: float, position: int
) -> dict[str, Any]:
    return {
        "event_id": event_id,
        "outcome_type": outcome_type,
        "outcome_key": outcome_key,
        "horizon": horizon,
        "value": value,
        "is_binary": outcome_type != "asset",
        "effective_session": SESSIONS[position].date(),
        "response_complete_session": SESSIONS[position + horizon].date(),
        "source_ref": event_id,
    }


def _bundle_rows() -> pd.DataFrame:
    rows = [
        _row(
            "gdelt:0",
            "event_sequence",
            "armed_conflict/government/none/EM/high/unscheduled/0",
            5,
            1.0,
            10,
        ),
        _row(
            "gdelt:0",
            "event_sequence",
            "armed_conflict/government/none/EM/high/unscheduled/0",
            20,
            0.0,
            10,
        ),
        _row(
            "gdelt:1",
            "event_sequence",
            "central_bank/government/none/US/high/unscheduled/0",
            5,
            0.0,
            20,
        ),
        _row(
            "gdelt:0",
            "market_resolution",
            "central_bank/central_bank/bond/US/high/scheduled/-1",
            7,
            1.0,
            10,
        ),
        _row(
            "gdelt:2",
            "market_resolution",
            "central_bank/central_bank/bond/US/high/scheduled/-1",
            3,
            0.0,
            30,
        ),
        _row("gdelt:0", "asset", "QQQ", 1, 0.012, 10),
        _row("gdelt:0", "asset", "QQQ", 5, -0.004, 10),
        _row("gdelt:2", "asset", "TLT", 5, 0.0031, 30),
    ]
    return validate_library_rows(rows_frame(rows))


def _bundle_placebo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome_key": pd.Series(["QQQ", "QQQ", "TLT"], dtype="str"),
            "horizon": pd.Series([1, 1, 5], dtype="int64"),
            "value": pd.Series([0.01, -0.02, 0.003], dtype="float64"),
            "response_complete_session": pd.Series(
                SESSIONS[[21, 31, 46]].as_unit("ns"), dtype="datetime64[ns]"
            ),
        }
    )


def _bundle_markets() -> pd.DataFrame:
    created = pd.to_datetime(["2020-01-02T10:00:00Z"] * 3, utc=True).as_unit("ns")
    resolved = pd.to_datetime(
        ["2020-02-10T15:00:00Z", "2020-03-02T15:00:00Z", "2021-08-30T15:00:00Z"], utc=True
    ).as_unit("ns")
    frame = pd.DataFrame(
        {
            "market_id": pd.Series(["polymarket:0xa", "kalshi:KX-1", "kalshi:KX-2"], dtype="str"),
            "outcome_key": pd.Series(
                ["central_bank/central_bank/bond/US/high/scheduled/-1"] * 3, dtype="str"
            ),
            "created_utc": pd.Series(created, dtype="datetime64[ns, UTC]"),
            "resolved_utc": pd.Series(resolved, dtype="datetime64[ns, UTC]"),
            "value": pd.Series([1.0, 0.0, 1.0], dtype="float64"),
            "response_complete_session": pd.Series(
                [SESSIONS[27].as_unit("ns"), SESSIONS[44].as_unit("ns"), pd.NaT],
                dtype="datetime64[ns]",
            ),
        }
    )
    return frame[list(MARKETS_FRAME_COLUMNS)]


@pytest.fixture
def bundle_parts() -> dict[str, Any]:
    """The keyword arguments `write_bundle` takes after `dest`."""
    return {
        "events": _bundle_events(),
        "rows": _bundle_rows(),
        "placebo_asset": _bundle_placebo(),
        "markets": _bundle_markets(),
        "sessions": SESSIONS,
        "session_cutoffs_utc": CUTOFFS,
        "build_report": {
            "events": {"records": 29, "kept": 29, "unaligned": 0},
            "note": "synthetic",
        },
        "sources": [
            ManifestSource(
                name="gdelt2",
                first_date=dt.date(2020, 1, 16),
                last_date=dt.date(2021, 4, 1),
                rows=29,
                citation="synthetic gdelt citation",
            ),
            ManifestSource(
                name="kalshi",
                first_date=dt.date(2020, 2, 10),
                last_date=dt.date(2021, 8, 30),
                rows=3,
                citation="synthetic kalshi citation",
            ),
        ],
        "commit": "0" * 40,
    }


@pytest.fixture
def bundle_root(tmp_path, bundle_parts):
    """A data dir whose `library/` was written by `write_bundle` from `bundle_parts`."""
    write_bundle(tmp_path / "library", **bundle_parts)
    return tmp_path


_GDELT_TYPES = ("armed_conflict", "central_bank", "trade_or_sanctions")
_GDELT_COUNTRIES = ("MLI", "USA")


def _gdelt_news_event(event_type: str, country: str) -> NewsEvent:
    return NewsEvent.model_validate(
        {
            "schema_version": "1.0",
            "event_type": event_type,
            "actor": {
                "name": f"{country} authority",
                "kind": "government",
                "country_iso3": country,
            },
            "action": "announced a measure",
            "impacted": [],
            "severity": 8,
            "macro_direction_step": 0,
            "confidence": "high",
            "horizon": "days",
            "novelty": "new",
            "is_scheduled": False,
            "evidence_span": "synthetic evidence",
        }
    )


def _gdelt_store_records() -> list[Any]:
    calendar = load_nyse_calendar()
    sessions = weekday_sessions(dt.date(2021, 2, 1), dt.date(2021, 4, 30), calendar)[:30]
    cutoffs = calendar_cutoffs(sessions, calendar)
    records = []
    for n, (session, cutoff) in enumerate(zip(sessions, cutoffs, strict=True)):
        country = _GDELT_COUNTRIES[n % 2]
        other = _GDELT_COUNTRIES[(n + 1) % 2]
        available = (cutoff - _ONE_HOUR).to_pydatetime()
        records.append(
            make_event_record(
                article_id=f"gdelt:{n}",
                body_sha256=f"{n:064x}",
                source=GDELT_SOURCE,
                first_seen_utc=available - dt.timedelta(minutes=5),
                available_utc=available,
                effective_session=session.date(),
                parser_version=gdelt_parser_version(),
                event=_gdelt_news_event(_GDELT_TYPES[n % 3], country),
                resolved_impacts=[],
                linked_countries=[country, other],
            )
        )
    return records


def _typed_market(
    market_id: str, venue: str, resolved_day: int, key: EventKey | None, version: str
) -> MarketRecord:
    typed = (
        TypedText(
            status="typed", key=key, reason="rule", evidence_span="synthetic", typer_version=version
        )
        if key is not None
        else TypedText(
            status="untyped",
            key=None,
            reason="no event_type pattern matched",
            evidence_span=None,
            typer_version=version,
        )
    )
    return MarketRecord(
        market_id=market_id,
        venue=venue,
        question="Will something happen?",
        created_utc=dt.datetime(2021, 1, 15, 9, tzinfo=dt.UTC),
        close_utc=None,
        resolved_utc=dt.datetime(2021, 3, resolved_day, 15, tzinfo=dt.UTC),
        resolution="yes" if resolved_day % 2 else "no",
        category=None,
        source_ref="synthetic",
        typed=typed,
    )


@pytest.fixture
def populated_stores(tmp_path):
    """A data dir holding 30 GDELT-style records and 6 markets (2 untyped) in the stores."""
    EventStore(tmp_path / "events").append(_gdelt_store_records())
    version = RulesTyper().version
    fed = EventKey("central_bank", "central_bank", ("bond",), "US", "high", True, -1)
    war = EventKey("armed_conflict", "government", (), "EM", "high", False, 0)
    store = MarketStore(tmp_path / "markets")
    store.append(
        [
            _typed_market("polymarket:0xa", "polymarket", 10, fed, version),
            _typed_market("polymarket:0xb", "polymarket", 17, war, version),
            _typed_market("polymarket:0xc", "polymarket", 12, None, version),
            _typed_market("kalshi:KX-1", "kalshi", 15, fed, version),
            _typed_market("kalshi:KX-2", "kalshi", 22, war, version),
            _typed_market("kalshi:KX-3", "kalshi", 24, None, version),
        ]
    )
    return tmp_path
