import pandas as pd
import pytest
from library_synthetic import SESSIONS, table

from precedent.libraries.common import LIBRARY_COLUMNS, validate_library_rows
from precedent.libraries.event_sequence import (
    SequenceLibraryReport,
    build_event_sequence_library,
    unconditional_sequence_rate,
)

TYPES = ("trade_or_sanctions", "election_or_political")


def test_follow_on_events_are_counted_by_horizon():
    events = table([
        ("p", "armed_conflict", ["MLI", "FRA"], 10),
        ("f1", "trade_or_sanctions", ["FRA", "MLI"], 13),
        ("f2", "trade_or_sanctions", ["USA"], 12),           # not linked to MLI
        ("f3", "trade_or_sanctions", ["MLI"], 10),           # same session, not after
        ("f4", "election_or_political", ["MLI"], 40),
    ])
    rows, _ = build_event_sequence_library(events, SESSIONS, ("trade_or_sanctions", "election_or_political"))
    got = rows[rows.event_id == "p"].set_index(["outcome_key", "horizon"])["value"].to_dict()
    assert got == {("trade_or_sanctions/linked_country", 5): 1.0, ("trade_or_sanctions/linked_country", 20): 1.0,
                   ("trade_or_sanctions/linked_country", 60): 1.0, ("election_or_political/linked_country", 5): 0.0,
                   ("election_or_political/linked_country", 20): 0.0, ("election_or_political/linked_country", 60): 1.0}
    assert rows[(rows.event_id == "p") & (rows.horizon == 20)]["response_complete_session"].iloc[0] == SESSIONS[30]


def test_windows_past_the_data_are_absent_and_counted():
    events = table([("p", "armed_conflict", ["MLI"], len(SESSIONS) - 30)])
    rows, report = build_event_sequence_library(events, SESSIONS, ("trade_or_sanctions", "election_or_political"))
    assert set(rows.horizon) == {5, 20} and report.skipped_incomplete_window == 2


def test_later_reports_of_a_story_are_not_follow_ons():
    rows, _ = build_event_sequence_library(table([("p", "armed_conflict", ["MLI"], 10), ("u", "trade_or_sanctions", ["MLI"], 12, False)]),
                                           SESSIONS, ("trade_or_sanctions",))
    assert rows[rows.event_id == "p"]["value"].max() == 0.0


def test_unconditional_rate_hand_value():
    events = table([("x", "trade_or_sanctions", ["MLI"], 5)])
    assert unconditional_sequence_rate(events, "trade_or_sanctions", "EM", 2, 9, {"MLI": "EM"}) == pytest.approx(0.25)   # s in 0..7; hits at s = 3, 4
    assert unconditional_sequence_rate(events, "trade_or_sanctions", "DM", 2, 9, {"MLI": "EM"}) is None


def test_event_without_a_country_is_counted_and_yields_no_rows():
    events = table([("p", "armed_conflict", ["MLI"], 10)])
    events.at[0, "linked_countries"] = []
    rows, report = build_event_sequence_library(events, SESSIONS, TYPES)
    assert report == SequenceLibraryReport(rows=0, skipped_incomplete_window=0, skipped_no_country=1)
    assert list(rows.columns) == list(LIBRARY_COLUMNS) and rows.empty


def test_every_type_and_horizon_gets_a_row_even_without_follow_ons():
    events = table([("p", "armed_conflict", ["MLI"], 10)])
    rows, report = build_event_sequence_library(events, SESSIONS, TYPES)
    assert len(rows) == 6 and report.rows == 6
    assert (rows["value"] == 0.0).all() and rows["is_binary"].all()
    assert set(rows["outcome_type"]) == {"event_sequence"}
    assert set(rows["source_ref"]) == {"p"}
    assert (rows["effective_session"] == SESSIONS[10]).all()
    validate_library_rows(rows)


def test_only_first_reports_are_precedents():
    events = table([("p", "armed_conflict", ["MLI"], 10), ("u", "armed_conflict", ["MLI"], 12, False)])
    rows, _ = build_event_sequence_library(events, SESSIONS, ("trade_or_sanctions",))
    assert set(rows.event_id) == {"p"}


def test_a_follow_on_at_the_window_end_counts_and_one_past_does_not():
    events = table([
        ("p", "armed_conflict", ["MLI"], 10),
        ("in", "trade_or_sanctions", ["MLI"], 15),
        ("out", "election_or_political", ["MLI"], 16),
    ])
    rows, _ = build_event_sequence_library(events, SESSIONS, TYPES, horizons=(5,))
    got = rows[rows.event_id == "p"].set_index("outcome_key")["value"].to_dict()
    assert got == {"trade_or_sanctions/linked_country": 1.0, "election_or_political/linked_country": 0.0}


def test_no_rows_returns_the_empty_schema_frame():
    rows, report = build_event_sequence_library(table([("p", "armed_conflict", ["MLI"], 10)]), SESSIONS, ())
    assert rows.empty and list(rows.columns) == list(LIBRARY_COLUMNS) and report.rows == 0


def test_unconditional_rate_is_none_when_the_end_is_before_the_horizon():
    events = table([("x", "trade_or_sanctions", ["MLI"], 5)])
    assert unconditional_sequence_rate(events, "trade_or_sanctions", "EM", 5, 4, {"MLI": "EM"}) is None


def test_unconditional_rate_counts_countries_from_any_row_and_ignores_later_reports():
    events = table([
        ("x", "trade_or_sanctions", ["MLI"], 5),
        ("y", "armed_conflict", ["NER"], 6),
        ("z", "trade_or_sanctions", ["NER"], 3, False),
    ])
    groups = {"MLI": "EM", "NER": "EM"}
    # two countries x 8 starts; only MLI has a first-report hit at s = 3, 4
    assert unconditional_sequence_rate(events, "trade_or_sanctions", "EM", 2, 9, groups) == pytest.approx(2 / 16)


def test_windows_past_the_feed_end_are_skipped_and_counted_not_zero():
    """P-45 (V-1): the padded calendar must not make windows past the last feed session look complete."""
    sessions = pd.DatetimeIndex(pd.bdate_range("2021-01-04", periods=200))
    feed_end = int(sessions.get_loc(pd.Timestamp("2021-06-30")))
    events = table([("last", "armed_conflict", ["USA"], feed_end)])
    rows, report = build_event_sequence_library(
        events, sessions, ("trade_or_sanctions",), (5, 20, 60), feed_end_position=feed_end
    )
    assert rows.empty and report.rows == 0 and report.skipped_incomplete_window == 3


def test_the_feed_end_defaults_to_the_last_session():
    events = table([("p", "armed_conflict", ["MLI"], len(SESSIONS) - 30)])
    rows, report = build_event_sequence_library(events, SESSIONS, TYPES, (5, 20))
    assert len(rows) == 4 and report.skipped_incomplete_window == 0
