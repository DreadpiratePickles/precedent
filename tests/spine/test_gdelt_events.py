"""Tests for GDELT 2.0 rule-typed events (rules-gdelt-v1)."""

import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from fakes import make_event_record
from fixtures.gdelt_rows import frame_of, gdelt_row, tsv, zip_bytes

from precedent.spine.event_key import build_event_key
from precedent.spine.gdelt_events import (
    GDELT_COLUMNS,
    GdeltFormatError,
    GdeltParams,
    dedupe_gdelt_records,
    events_from_gdelt_rows,
    load_cameo_map,
    read_export_csv,
)
from precedent.spine.session_calendar import calendar_cutoffs, load_nyse_calendar, weekday_sessions

CAMEO = load_cameo_map()
PARAMS = GdeltParams()
INGESTED = datetime(2024, 1, 4, tzinfo=UTC)
REFERENCES = Path(__file__).resolve().parents[2] / "references"


def utc(text: str) -> datetime:
    return pd.Timestamp(text, tz="UTC").to_pydatetime()


def one_record(*rows: list[str], params: GdeltParams = PARAMS):
    return events_from_gdelt_rows(frame_of(*rows), CAMEO, params, ingested_utc=INGESTED)[0][0]


def test_gdelt_event_uses_dateadded_not_sqldate():
    records, _ = events_from_gdelt_rows(
        frame_of(gdelt_row(SQLDATE="20240101", DATEADDED="20240103141500")),
        CAMEO,
        PARAMS,
        ingested_utc=INGESTED,
    )
    assert records[0].first_seen_utc == utc("2024-01-03 14:15:00")
    assert records[0].published_utc is None
    assert records[0].available_utc == utc("2024-01-03 14:45:00")
    assert records[0].effective_session == date(2024, 1, 3)  # before Wednesday's 21:00 UTC cutoff


def test_a_row_added_after_the_cutoff_goes_to_the_next_session():
    record = one_record(gdelt_row(DATEADDED="20240103210000"))  # 21:30 UTC available, after 21:00
    assert record.effective_session == date(2024, 1, 4)


def test_the_last_rows_of_a_frame_still_get_a_session():
    record = one_record(gdelt_row(DATEADDED="20240105235900"))  # Friday night, next is Monday
    assert record.effective_session == date(2024, 1, 8)


def test_explicit_session_cutoffs_are_used_and_late_rows_stay_unaligned():
    calendar = load_nyse_calendar()
    cutoffs = calendar_cutoffs(weekday_sessions(date(2024, 1, 2), date(2024, 1, 3), calendar), calendar)
    records, report = events_from_gdelt_rows(
        frame_of(gdelt_row(DATEADDED="20240110120000")),
        CAMEO,
        PARAMS,
        ingested_utc=INGESTED,
        session_cutoffs_utc=cutoffs,
    )
    assert records[0].effective_session is None and report.kept == 1


def test_columns_are_read_by_position():
    frame = read_export_csv(
        zip_bytes(tsv(gdelt_row(DATEADDED="20240103141500", SOURCEURL="https://example.test/x")))
    )
    assert (frame.loc[0, "DATEADDED"], frame.loc[0, "SOURCEURL"]) == (
        "20240103141500",
        "https://example.test/x",
    )
    assert list(frame.columns) == list(GDELT_COLUMNS)
    assert frame.loc[0, "EventRootCode"] == "16" and frame.loc[0, "Actor2CountryCode"] == "RUS"


def test_zero_padded_codes_survive_reading():
    frame = frame_of(gdelt_row(EventCode="010", EventBaseCode="010", EventRootCode="01"))
    assert frame.loc[0, "EventRootCode"] == "01" and frame.loc[0, "EventCode"] == "010"


def test_a_plain_tsv_is_read_like_a_zip_and_quotes_are_data():
    frame = read_export_csv(tsv(gdelt_row(Actor1Name='THE "BIG" ONE')).encode())
    assert frame.loc[0, "Actor1Name"] == 'THE "BIG" ONE'


def test_a_file_with_the_wrong_column_count_is_refused():
    with pytest.raises(GdeltFormatError, match="line 1"):
        read_export_csv(tsv(gdelt_row()[:60]).encode())


def test_a_wrong_column_count_on_a_later_line_names_the_line():
    with pytest.raises(GdeltFormatError, match="line 2"):
        read_export_csv((tsv(gdelt_row()) + tsv(gdelt_row() + ["extra"])).encode())


def test_a_government_sanction_is_trade_or_sanctions():
    record = events_from_gdelt_rows(frame_of(gdelt_row()), CAMEO, PARAMS, ingested_utc=INGESTED)[0][0]
    assert record.event.event_type.value == "trade_or_sanctions"
    assert record.event.evidence_span == "CAMEO 163"
    assert (record.article_id, record.source, record.linked_countries) == (
        "gdelt:1001",
        "gdelt2",
        ["USA", "RUS"],
    )
    assert record.parser_version == "rules-gdelt-v1"


def test_a_sanction_by_a_non_government_actor_falls_back_to_the_root_rule():
    records, report = events_from_gdelt_rows(
        frame_of(gdelt_row(Actor1Type1Code="BUS")), CAMEO, PARAMS, ingested_utc=INGESTED
    )
    assert records == [] and report.dropped_untyped == 1  # root 16 is untyped


def test_event_fields_follow_the_mapping_table():
    record = one_record(gdelt_row(Actor1Name="", Actor1Code="USAGOV"))
    event = record.event
    assert event.actor.name == "USAGOV" and event.actor.kind.value == "government"
    assert event.actor.country_iso3 == "USA"
    assert event.action == "Impose embargo, boycott, or sanctions"
    assert [(i.name, i.kind.value, i.ticker_hint, i.direction_step) for i in event.impacted] == [
        ("RUSSIA", "country", None, -1)
    ]
    assert event.macro_direction_step == -1 and event.confidence.value == "medium"
    assert (event.horizon.value, event.novelty.value, event.is_scheduled) == ("weeks", "new", False)
    assert event.schema_version == "1.0" and record.resolved_impacts == []
    assert record.ingested_utc == INGESTED


def test_actor_name_falls_back_to_unknown_and_is_cut_to_80():
    assert one_record(gdelt_row(Actor1Name="", Actor1Code="")).event.actor.name == "unknown actor"
    assert len(one_record(gdelt_row(Actor1Name="X" * 200)).event.actor.name) == 80


@pytest.mark.parametrize(
    "type_code, kind",
    [("GOV", "government"), ("MIL", "government"), ("IGO", "international_org"),
     ("BUS", "company"), ("COP", "other"), ("", "other")],
)
def test_actor_kind_from_type_code(type_code, kind):
    record = one_record(gdelt_row(EventCode="190", EventBaseCode="190", EventRootCode="19",
                                  Actor1Type1Code=type_code))
    assert record.event.actor.kind.value == kind


def test_no_actor2_means_no_impacted_and_goldstein_sign_sets_the_step():
    record = one_record(gdelt_row(EventCode="190", EventBaseCode="190", EventRootCode="19",
                                  Actor2Name="", GoldsteinScale="0.0"))
    assert record.event.impacted == [] and record.event.macro_direction_step == 0
    positive = one_record(gdelt_row(EventCode="190", EventBaseCode="190", EventRootCode="19",
                                    GoldsteinScale="3.0"))
    assert positive.event.impacted[0].direction_step == 1


@pytest.mark.parametrize(
    "root, event_type",
    [("14", "election_or_political"), ("15", "armed_conflict"), ("18", "armed_conflict"),
     ("19", "armed_conflict"), ("20", "armed_conflict"), ("13", "other")],
)
def test_root_rules(root, event_type):
    record = one_record(gdelt_row(EventCode=f"{root}0", EventBaseCode=f"{root}0", EventRootCode=root))
    assert record.event.event_type.value == event_type


def test_filters_and_counts_add_up():
    rows = frame_of(
        gdelt_row(GLOBALEVENTID="1", NumSources="2"),  # below min_sources
        gdelt_row(GLOBALEVENTID="2", EventCode="051", EventBaseCode="051", EventRootCode="05"),  # untyped
        gdelt_row(GLOBALEVENTID="3", DATEADDED="20240103141500"),
        gdelt_row(GLOBALEVENTID="4", DATEADDED="20240103143000"),  # same key, country and session
    )
    records, report = events_from_gdelt_rows(rows, CAMEO, PARAMS, ingested_utc=INGESTED)
    assert [r.article_id for r in records] == ["gdelt:3"]
    assert (
        report.rows_read,
        report.kept,
        report.dropped_min_sources,
        report.dropped_untyped,
        report.dropped_duplicate,
    ) == (4, 1, 1, 1, 1)


def test_dedup_ties_go_to_the_smaller_event_id_and_input_order_does_not_matter():
    rows = frame_of(
        gdelt_row(GLOBALEVENTID="9"),
        gdelt_row(GLOBALEVENTID="10"),  # same DATEADDED; numeric 9 < 10 though "10" < "9" as text
    )
    records, _ = events_from_gdelt_rows(rows, CAMEO, PARAMS, ingested_utc=INGESTED)
    assert [r.article_id for r in records] == ["gdelt:9"]


def test_different_sessions_or_countries_are_not_duplicates():
    rows = frame_of(
        gdelt_row(GLOBALEVENTID="1", DATEADDED="20240103141500"),
        gdelt_row(GLOBALEVENTID="2", DATEADDED="20240104141500"),  # next session
        gdelt_row(GLOBALEVENTID="3", Actor1CountryCode="CHN"),  # other Actor1 country
    )
    records, report = events_from_gdelt_rows(rows, CAMEO, PARAMS, ingested_utc=INGESTED)
    assert len(records) == 3 and report.dropped_duplicate == 0


def test_rows_without_a_session_are_kept_not_deduplicated():
    calendar = load_nyse_calendar()
    cutoffs = calendar_cutoffs(weekday_sessions(date(2024, 1, 2), date(2024, 1, 3), calendar), calendar)
    records, report = events_from_gdelt_rows(
        frame_of(gdelt_row(GLOBALEVENTID="1", DATEADDED="20240110120000"),
                 gdelt_row(GLOBALEVENTID="2", DATEADDED="20240110120000")),
        CAMEO, PARAMS, ingested_utc=INGESTED, session_cutoffs_utc=cutoffs,
    )
    assert len(records) == 2 and report.dropped_duplicate == 0
    assert report.rows_read == report.kept + report.dropped_duplicate


@pytest.mark.parametrize(
    "goldstein, sources, severity",
    [("-8.0", "5", 8), ("2.0", "5", 2), ("2.0", "12", 5), ("5.0", "5", 5), ("5.0", "10", 8),
     ("-9.0", "40", 8), ("3.0", "5", 2), ("10.0", "5", 8)],
)
def test_severity_from_goldstein_and_sources(goldstein, sources, severity):
    record = one_record(gdelt_row(GoldsteinScale=goldstein, NumSources=sources))
    assert record.event.severity == severity


def test_a_bad_numeric_field_is_refused_not_defaulted():
    with pytest.raises(GdeltFormatError, match="GoldsteinScale"):
        events_from_gdelt_rows(frame_of(gdelt_row(GoldsteinScale="")), CAMEO, PARAMS,
                               ingested_utc=INGESTED)
    with pytest.raises(GdeltFormatError, match="DATEADDED"):
        events_from_gdelt_rows(frame_of(gdelt_row(DATEADDED="2024-01-03")), CAMEO, PARAMS,
                               ingested_utc=INGESTED)


def test_non_iso_actor_codes_are_left_out_and_counted():
    records, report = events_from_gdelt_rows(
        frame_of(gdelt_row(Actor2CountryCode="AFR")), CAMEO, PARAMS, ingested_utc=INGESTED
    )
    assert records[0].linked_countries == ["USA"] and report.non_iso_country_codes == 1


def test_same_country_on_both_sides_is_listed_once():
    record = one_record(gdelt_row(Actor2CountryCode="USA"))
    assert record.linked_countries == ["USA"]


def test_a_non_iso_actor1_country_leaves_the_actor_country_empty():
    record = one_record(gdelt_row(Actor1CountryCode="AFR"))
    assert record.event.actor.country_iso3 is None and record.linked_countries == ["RUS"]


def test_body_hash_covers_the_named_columns():
    first = one_record(gdelt_row())
    changed = one_record(gdelt_row(SOURCEURL="https://example.test/other"))
    assert first.body_sha256 != changed.body_sha256 and len(first.body_sha256) == 64
    assert first.body_sha256 == one_record(gdelt_row()).body_sha256


def test_empty_input_yields_no_records():
    records, report = events_from_gdelt_rows(frame_of(), CAMEO, PARAMS, ingested_utc=INGESTED)
    assert records == [] and report.rows_read == 0


def _one_run(*rows: list[str]):
    return events_from_gdelt_rows(frame_of(*rows), CAMEO, PARAMS, ingested_utc=INGESTED)[0]


def test_dedupe_gdelt_records_across_runs():
    later = _one_run(gdelt_row(GLOBALEVENTID="20", DATEADDED="20240103143000"))
    earlier = _one_run(gdelt_row(GLOBALEVENTID="21", DATEADDED="20240103141500"))
    elsewhere = _one_run(gdelt_row(GLOBALEVENTID="40", Actor1CountryCode="CHN"))
    assert build_event_key(later[0]).as_string() == build_event_key(earlier[0]).as_string()
    kept = dedupe_gdelt_records([*later, *earlier, *elsewhere])
    assert [r.article_id for r in kept] == ["gdelt:21", "gdelt:40"]


def test_dedupe_gdelt_records_ties_go_to_the_smaller_numeric_event_id():
    high = _one_run(gdelt_row(GLOBALEVENTID="31"))
    low = _one_run(gdelt_row(GLOBALEVENTID="9"))
    assert [r.article_id for r in dedupe_gdelt_records([*high, *low])] == ["gdelt:9"]


def test_dedupe_gdelt_records_keeps_records_without_a_session():
    unaligned = [make_event_record(article_id=f"gdelt:{n}") for n in (1, 2)]
    assert dedupe_gdelt_records(unaligned) == unaligned


def test_shipped_params_are_the_defaults():
    assert GdeltParams.from_file(REFERENCES / "precedent_params.json") == GdeltParams()
    assert GdeltParams.from_file(Path("references/precedent_params.json")) == GdeltParams()


def test_params_ignore_other_sections_and_refuse_bad_values(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"engine": {"k": 1}, "gdelt": {"min_sources": 4}}))
    assert GdeltParams.from_file(path).min_sources == 4
    path.write_text(json.dumps({"gdelt": {"min_sources": 0}}))
    with pytest.raises(ValueError):
        GdeltParams.from_file(path)
    path.write_text(json.dumps({"engine": {}}))
    with pytest.raises(ValueError, match="gdelt"):
        GdeltParams.from_file(path)
    path.write_text(json.dumps({"gdelt": {"goldstein_cuts": [6.0, 3.0]}}))
    with pytest.raises(ValueError):
        GdeltParams.from_file(path)


def test_cameo_tables_are_complete_and_proposed():
    mapping = json.loads((REFERENCES / "cameo_to_event_type.json").read_text())
    assert set(mapping["roots"]) == {f"{n:02d}" for n in range(1, 21)}
    assert all(rule["proposed"] is True for rule in [*mapping["roots"].values(), *mapping["subcodes"].values()])
    assert CAMEO["roots"]["14"]["event_type"] == "election_or_political"
    assert CAMEO["subcodes"]["163"]["requires_actor1_type"] == ("GOV",)
    assert CAMEO["descriptions"]["163"] == "Impose embargo, boycott, or sanctions"
    assert len(CAMEO["descriptions"]) > 250


def test_a_mapping_to_an_unknown_event_type_is_refused(tmp_path):
    bad = json.loads((REFERENCES / "cameo_to_event_type.json").read_text())
    bad["roots"]["14"]["event_type"] = "made_up"
    path = tmp_path / "map.json"
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        load_cameo_map(path)
