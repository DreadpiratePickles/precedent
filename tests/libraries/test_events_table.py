from datetime import date

import pandas as pd
from library_synthetic import CUTOFFS, SESSIONS, edgar_record, gdelt_record

from precedent.libraries.events_table import (
    EVENTS_TABLE_COLUMNS,
    EventsTableReport,
    build_events_table,
    one_line,
)
from precedent.spine.event_key import build_event_key


def test_events_table_skips_and_counts_unaligned_records():
    table, report = build_events_table(
        [gdelt_record("g1", position=10), gdelt_record("g2", position=None)], SESSIONS
    )
    assert table["event_id"].tolist() == ["g1"] and (report.kept, report.unaligned) == (1, 1)


def test_primary_actor_prefers_the_instrument_then_the_country():
    table, _ = build_events_table(
        [
            edgar_record("x:2.02", instrument="ACME", linked=["USA"]),
            gdelt_record("g1", linked=["MLI", "FRA"]),
        ],
        SESSIONS,
    )
    assert table["primary_actor"].tolist() == ["ACME", "MLI"]


def test_primary_actor_falls_back_to_the_actor_name_without_links():
    table, _ = build_events_table([gdelt_record("g1", actor="MALI", linked=[])], SESSIONS)
    assert table["primary_actor"].tolist() == ["MALI"]
    assert table["linked_countries"].tolist() == [[]]


def test_one_line_is_actor_and_action_cut_to_60():
    assert one_line(gdelt_record("g1", actor="MALI", action="x" * 80)) == ("MALI: " + "x" * 80)[:60]


def test_empty_input_gives_an_empty_table_with_every_column():
    table, report = build_events_table([], SESSIONS)
    assert list(table.columns) == list(EVENTS_TABLE_COLUMNS) and table.empty
    assert report == EventsTableReport(0, 0, 0)
    assert str(table["effective_position"].dtype) == "int64"
    assert str(table["available_utc"].dtype) == "datetime64[ns, UTC]"


def test_a_session_outside_the_sessions_index_is_counted_unaligned():
    saturday = gdelt_record("g1").model_copy(update={"effective_session": date(2020, 1, 4)})
    table, report = build_events_table([saturday, gdelt_record("g2")], SESSIONS)
    assert table["event_id"].tolist() == ["g2"]
    assert report == EventsTableReport(records=2, kept=1, unaligned=1)


def test_all_unaligned_still_returns_every_column():
    table, report = build_events_table([gdelt_record("g1", position=None)], SESSIONS)
    assert list(table.columns) == list(EVENTS_TABLE_COLUMNS) and table.empty
    assert report == EventsTableReport(1, 0, 1)


def test_columns_values_and_dtypes_for_one_record():
    record = edgar_record("x:2.02", instrument="ACME", linked=["USA"], position=12)
    table, _ = build_events_table([record], SESSIONS)
    assert list(table.columns) == list(EVENTS_TABLE_COLUMNS)
    row = table.iloc[0]
    assert row["event_id"] == "x:2.02" and row["source_ref"] == "x:2.02"
    assert row["key"] == build_event_key(record).as_string()
    assert row["event_type"] == "earnings" and row["country_group"] == "US"
    assert row["effective_session"] == SESSIONS[12].date()
    assert isinstance(row["effective_session"], date)
    assert row["effective_position"] == 12
    assert str(table["effective_position"].dtype) == "int64"
    assert str(table["available_utc"].dtype) == "datetime64[ns, UTC]"
    assert row["available_utc"] == CUTOFFS.iloc[12] - pd.Timedelta(hours=1)
    assert row["linked_countries"] == ["USA"] and isinstance(row["linked_countries"], list)
    assert row["source"] == "edgar" and row["one_line"] == "ACME Corp: reported results"
    assert row["story_cluster_id"] == "x:2.02" and row["is_first_report"]


def test_row_order_follows_the_records():
    table, _ = build_events_table(
        [gdelt_record("b", position=30), gdelt_record("a", position=10)], SESSIONS
    )
    assert table["event_id"].tolist() == ["b", "a"]


def test_same_key_and_actor_two_sessions_apart_is_one_story_with_one_first_report():
    records = [
        gdelt_record("g1", position=10),
        gdelt_record("g2", position=12),
        gdelt_record("g3", position=40),
    ]
    table, _ = build_events_table(records, SESSIONS)
    assert table["story_cluster_id"].tolist() == ["g1", "g1", "g3"]
    assert table["is_first_report"].tolist() == [True, False, True]
