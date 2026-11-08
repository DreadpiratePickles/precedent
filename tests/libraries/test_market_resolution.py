"""Tests for the market resolution library."""

import pandas as pd
import pytest
from library_synthetic import CUTOFFS, SESSIONS, table

from precedent.engine.keys import key_from_flags
from precedent.libraries.common import LIBRARY_COLUMNS, validate_library_rows
from precedent.libraries.market_resolution import (
    MarketLibraryReport,
    build_market_resolution_library,
    unconditional_market_rate,
)
from precedent.spine.market_record import MarketRecord
from precedent.spine.typers.base import TypedText

K1 = key_from_flags("central_bank", "central_bank", "high", True, country_group="US")
K2 = key_from_flags("armed_conflict", "government", "high", False, country_group="EM")
T_E = CUTOFFS.iloc[10] - pd.Timedelta("2h")
EVENTS_E = table([("E", "central_bank", ["USA"], 10)])
EVENTS_E["available_utc"] = pd.Series([T_E], dtype="datetime64[ns, UTC]")

UNTYPED = TypedText(
    status="untyped",
    key=None,
    reason="no event_type pattern matched",
    evidence_span=None,
    typer_version="fake-v1",
)
REFUSED = TypedText(
    status="refused_invalid",
    key=None,
    reason="model output failed validation",
    evidence_span=None,
    typer_version="fake-v1",
)
MARKETS_COLUMNS = [
    "market_id",
    "outcome_key",
    "created_utc",
    "resolved_utc",
    "value",
    "response_complete_session",
]


def typed_as(key=K1):
    return TypedText(
        status="typed", key=key, reason="rule", evidence_span="Fed", typer_version="fake-v1"
    )


def market(market_id, created, resolved, resolution="yes", typed=None):
    venue = market_id.split(":")[0]
    return MarketRecord(
        market_id=market_id,
        venue=venue,
        question="q",
        created_utc=created.to_pydatetime(),
        close_utc=None,
        resolved_utc=resolved.to_pydatetime(),
        resolution=resolution,
        category=None,
        source_ref=market_id,
        typed=typed if typed is not None else typed_as(),
    )


MARKET_YES_AT_20 = market(
    "polymarket:yes",
    T_E - pd.Timedelta("1D"),
    CUTOFFS.iloc[20] - pd.Timedelta("1h"),
    resolution="yes",
)
MARKET_NO_AT_30 = market(
    "polymarket:no",
    T_E - pd.Timedelta("1D"),
    CUTOFFS.iloc[30] - pd.Timedelta("1h"),
    resolution="no",
)


def markets_frame(rows):
    """A markets frame from `(key_name, resolution, position)`; "K1" is the K1 outcome key."""
    names = {"K1": K1.as_string()}
    records = []
    for index, (key_name, resolution, position) in enumerate(rows):
        resolved = CUTOFFS.iloc[position] - pd.Timedelta("1h")
        records.append(
            {
                "market_id": f"polymarket:{index}",
                "outcome_key": names.get(key_name, key_name),
                "created_utc": resolved - pd.Timedelta("10D"),
                "resolved_utc": resolved,
                "value": 1.0 if resolution == "yes" else 0.0,
                "response_complete_session": SESSIONS[position],
            }
        )
    return pd.DataFrame(records, columns=MARKETS_COLUMNS)


def test_only_markets_open_at_the_event_qualify():
    rows, _, report = build_market_resolution_library(
        EVENTS_E,
        [
            market(
                "polymarket:yes",
                created=T_E - pd.Timedelta("1D"),
                resolved=CUTOFFS.iloc[20] - pd.Timedelta("1h"),
                resolution="yes",
            ),
            market("polymarket:late", created=T_E + pd.Timedelta("1h"), resolved=CUTOFFS.iloc[25]),
            market(
                "polymarket:early",
                created=T_E - pd.Timedelta("5D"),
                resolved=T_E - pd.Timedelta("1h"),
            ),
            market("polymarket:same", created=T_E, resolved=CUTOFFS.iloc[25]),
            market(
                "kalshi:untyped",
                created=T_E - pd.Timedelta("1D"),
                resolved=CUTOFFS.iloc[25],
                typed=UNTYPED,
            ),
        ],
        SESSIONS,
        CUTOFFS,
    )
    assert rows["source_ref"].tolist() == ["polymarket:yes"]
    row = rows.iloc[0]
    assert (row.value, row.is_binary, row.horizon) == (1.0, True, 10)
    assert row.response_complete_session == SESSIONS[20] and row.outcome_key == K1.as_string()
    assert report.markets_untyped == 1


def test_several_markets_with_one_key_collapse_to_the_first_resolving():
    rows, _, report = build_market_resolution_library(
        EVENTS_E, [MARKET_YES_AT_20, MARKET_NO_AT_30], SESSIONS, CUTOFFS
    )
    assert rows["source_ref"].tolist() == ["polymarket:yes"] and report.collapsed_duplicates == 1


def test_unconditional_rate_respects_the_cutoff():
    frame = markets_frame(
        [
            ("K1", "yes", 20),
            ("K1", "yes", 21),
            ("K1", "no", 22),
            ("K1", "yes", 23),
            ("K1", "yes", 90),
        ]
    )
    assert unconditional_market_rate(
        frame, K1.as_string(), CUTOFFS.iloc[23], CUTOFFS
    ) == pytest.approx(0.75)
    assert unconditional_market_rate(frame, K1.as_string(), CUTOFFS.iloc[19], CUTOFFS) is None


def test_a_market_resolving_after_the_last_session_is_skipped_and_counted():
    late = market(
        "polymarket:late",
        T_E - pd.Timedelta("1D"),
        CUTOFFS.iloc[-1] + pd.Timedelta("1D"),
    )
    rows, markets, report = build_market_resolution_library(EVENTS_E, [late], SESSIONS, CUTOFFS)
    assert rows.empty and report.rows == 0
    assert report.skipped_no_session == 1 and report.collapsed_duplicates == 0
    assert markets["response_complete_session"].isna().tolist() == [True]
    assert markets["market_id"].tolist() == ["polymarket:late"]


def test_the_markets_frame_has_exactly_the_six_columns_for_typed_markets():
    _, frame, report = build_market_resolution_library(
        EVENTS_E,
        [
            MARKET_YES_AT_20,
            MARKET_NO_AT_30,
            market("kalshi:u", T_E, CUTOFFS.iloc[25], typed=UNTYPED),
        ],
        SESSIONS,
        CUTOFFS,
    )
    assert list(frame.columns) == MARKETS_COLUMNS
    assert sorted(frame["market_id"]) == ["polymarket:no", "polymarket:yes"]
    assert sorted(frame["value"]) == [0.0, 1.0]
    assert str(frame["created_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["resolved_utc"].dtype) == "datetime64[ns, UTC]"
    assert str(frame["value"].dtype) == "float64"
    assert str(frame["response_complete_session"].dtype) == "datetime64[ns]"
    by_id = frame.set_index("market_id")
    assert by_id.loc["polymarket:yes", "response_complete_session"] == SESSIONS[20]
    assert by_id.loc["polymarket:yes", "outcome_key"] == K1.as_string()
    assert report.markets_read == 3


def test_a_refused_invalid_market_is_counted_and_never_qualifies():
    rows, frame, report = build_market_resolution_library(
        EVENTS_E,
        [
            market(
                "polymarket:refused",
                T_E - pd.Timedelta("1D"),
                CUTOFFS.iloc[25],
                typed=REFUSED,
            )
        ],
        SESSIONS,
        CUTOFFS,
    )
    assert rows.empty and frame.empty
    assert report == MarketLibraryReport(
        markets_read=1,
        markets_untyped=0,
        markets_refused_invalid=1,
        rows=0,
        collapsed_duplicates=0,
        skipped_no_session=0,
    )


def test_rows_pass_the_schema_and_different_keys_each_get_a_row():
    other = market(
        "kalshi:other",
        T_E - pd.Timedelta("1D"),
        CUTOFFS.iloc[15] - pd.Timedelta("1h"),
        resolution="no",
        typed=typed_as(K2),
    )
    rows, _, report = build_market_resolution_library(
        EVENTS_E, [MARKET_YES_AT_20, other], SESSIONS, CUTOFFS
    )
    assert list(rows.columns) == list(LIBRARY_COLUMNS)
    assert len(rows) == 2 and report.collapsed_duplicates == 0 and report.rows == 2
    assert set(rows["outcome_type"]) == {"market_resolution"}
    by_ref = rows.set_index("source_ref")
    assert by_ref.loc["kalshi:other", "value"] == 0.0 and by_ref.loc["kalshi:other", "horizon"] == 5
    validate_library_rows(rows)


def test_only_first_report_events_get_rows():
    events = table([("A", "central_bank", ["USA"], 10), ("B", "central_bank", ["USA"], 10, False)])
    events["available_utc"] = pd.Series([T_E, T_E], dtype="datetime64[ns, UTC]")
    rows, _, _ = build_market_resolution_library(events, [MARKET_YES_AT_20], SESSIONS, CUTOFFS)
    assert rows["event_id"].tolist() == ["A"]


def test_no_events_or_no_markets_gives_empty_rows_with_the_schema():
    rows, frame, report = build_market_resolution_library(EVENTS_E, [], SESSIONS, CUTOFFS)
    assert rows.empty and list(rows.columns) == list(LIBRARY_COLUMNS)
    assert frame.empty and list(frame.columns) == MARKETS_COLUMNS
    assert report.markets_read == 0 and report.rows == 0
