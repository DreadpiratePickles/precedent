import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from precedent.spine.edgar_8k_events import (
    EdgarIngestReport,
    ItemRule,
    edgar_parser_version,
    events_from_submissions,
    load_item_rules,
    load_us_state_codes,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE = json.loads((FIXTURES / "submissions_sample.json").read_text())
FOREIGN_SAMPLE = json.loads((FIXTURES / "submissions_foreign_sample.json").read_text())
INGESTED = datetime(2024, 3, 1, tzinfo=UTC)


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=UTC)


def records_of(submissions, instruments=("ACME",), **kwargs):
    return events_from_submissions(
        submissions, list(instruments), ingested_utc=INGESTED, **kwargs
    )


def test_multi_item_filing_yields_one_event_per_item():
    records = [r for r in records_of(SAMPLE)[0] if r.article_id.startswith("0001234567-24-000001")]
    assert [r.event.event_type.value for r in records] == ["earnings", "management_change"]
    assert [r.article_id for r in records] == [
        "0001234567-24-000001:2.02",
        "0001234567-24-000001:5.02",
    ]
    assert {r.first_seen_utc for r in records} == {utc("2024-02-01 21:30:05")}
    assert {r.published_utc for r in records} == {utc("2024-02-01 21:30:05")}


def test_amendment_is_an_update():
    amended = [r for r in records_of(SAMPLE)[0] if r.article_id.startswith("0001234567-24-000002")]
    assert amended
    assert all(r.event.novelty.value == "update" for r in amended)
    originals = [
        r for r in records_of(SAMPLE)[0] if r.article_id.startswith("0001234567-24-000001")
    ]
    assert all(r.event.novelty.value == "new" for r in originals)


def test_the_filer_country_is_linked():
    assert {tuple(r.linked_countries) for r in records_of(SAMPLE)[0]} == {("USA",)}
    foreign_records = records_of(FOREIGN_SAMPLE, ["XYZ"])[0]
    assert foreign_records
    assert all(r.linked_countries == [] for r in foreign_records)
    assert all(r.event.actor.country_iso3 is None for r in foreign_records)


def test_the_record_fields_follow_the_field_table():
    earnings = next(r for r in records_of(SAMPLE)[0] if r.article_id.endswith("000001:2.02"))
    assert earnings.source == "edgar"
    assert earnings.parser_version == edgar_parser_version()
    assert earnings.parser_version.startswith("rules-8k-v1:")
    assert earnings.ingested_utc == INGESTED
    assert len(earnings.body_sha256) == 64
    event = earnings.event
    assert event.actor.name == "Synthetic Devices Inc."
    assert event.actor.kind.value == "company"
    assert event.actor.country_iso3 == "USA"
    assert event.action == "Results of Operations and Financial Condition"
    assert event.evidence_span == "8-K item 2.02"
    assert event.is_scheduled is True
    assert event.severity == 5
    assert event.macro_direction_step == 0
    assert event.confidence.value == "high"
    assert event.horizon.value == "weeks"
    impact = event.impacted[0]
    assert (impact.name, impact.kind.value, impact.ticker_hint, impact.direction_step) == (
        "Synthetic Devices Inc.",
        "company",
        "ACME",
        0,
    )
    resolved = earnings.resolved_impacts[0]
    assert (resolved.instrument, resolved.kind, resolved.direction, resolved.match_method) == (
        "ACME",
        "company",
        0.0,
        "cik_map",
    )
    management = next(r for r in records_of(SAMPLE)[0] if r.article_id.endswith("000001:5.02"))
    assert management.event.is_scheduled is False


def test_changing_an_item_rule_byte_changes_the_parser_version(tmp_path, monkeypatch):
    from precedent.config import references_dir
    from precedent.spine import edgar_8k_events

    version = edgar_parser_version()
    assert version.startswith("rules-8k-v1:") and len(version.split(":")[1]) == 12
    rules = tmp_path / "item_rules_8k.json"
    rules.write_bytes((references_dir() / "item_rules_8k.json").read_bytes())
    monkeypatch.setattr(edgar_8k_events, "references_dir", lambda: tmp_path)
    assert edgar_parser_version() == version
    rules.write_bytes(rules.read_bytes() + b"\n")
    assert edgar_parser_version() != version


def test_share_classes_give_one_record_per_item_with_one_impact_per_instrument():
    single = records_of(SAMPLE, ["ACME"])[0]
    both = records_of(SAMPLE, ["ACME", "ACMEB"])[0]
    assert [r.article_id for r in both] == [r.article_id for r in single]
    assert len({r.article_id for r in both}) == len(both)
    assert all([i.instrument for i in r.resolved_impacts] == ["ACME", "ACMEB"] for r in both)
    assert all(r.event.impacted[0].ticker_hint == "ACME" for r in both)


def test_ticker_hint_is_the_first_instrument_that_fits_the_pattern():
    record = records_of(SAMPLE, ["synthetic-id 0042", "ACMEB"])[0][0]
    assert record.event.impacted[0].ticker_hint == "ACMEB"


@pytest.mark.parametrize("bad", [[], ["ACME", "ACME"]])
def test_instruments_must_be_non_empty_and_unique(bad):
    with pytest.raises(ValueError, match="instruments"):
        records_of(SAMPLE, bad)


def test_an_instrument_that_is_not_a_ticker_gets_no_ticker_hint():
    record = records_of(SAMPLE, ["synthetic-id 0042"])[0][0]
    assert record.event.impacted[0].ticker_hint is None
    assert record.resolved_impacts[0].instrument == "synthetic-id 0042"


def test_a_2009_filing_stays_unaligned_and_the_store_counts_it(tmp_path):
    from precedent.spine.event_store import EventStore

    document = json.loads(json.dumps(SAMPLE))
    document["filings"]["recent"]["acceptanceDateTime"][3] = "2009-12-15T21:05:00.000Z"
    records, _ = records_of(document)
    old = [r for r in records if r.first_seen_utc.year == 2009]
    assert old and all(r.effective_session is None and r.available_utc is not None for r in old)
    assert any(r.effective_session is not None for r in records)
    report = EventStore(tmp_path).append(records)
    _, frame_report = EventStore(tmp_path).to_frame(edgar_parser_version())
    assert report.added == len(records)
    assert frame_report.unaligned_records == len(old)


def test_records_are_aligned_to_the_next_session_cutoff():
    by_id = {r.article_id: r for r in records_of(SAMPLE)[0]}
    # Thu 2024-02-01 21:30 UTC is after the 16:00 New York cutoff (21:00 UTC): next session, Fri.
    assert by_id["0001234567-24-000001:2.02"].effective_session.isoformat() == "2024-02-02"
    assert by_id["0001234567-24-000001:2.02"].available_utc == utc("2024-02-01 21:35:05")
    # Thu 2024-02-08 14:00 UTC is before the cutoff that day.
    assert by_id["0001234567-24-000002:5.02"].effective_session.isoformat() == "2024-02-08"


def test_explicit_cutoffs_and_latency_are_used():
    cutoffs = pd.Series(
        pd.to_datetime(["2024-02-09 21:00", "2024-02-12 21:00"], utc=True),
        index=pd.DatetimeIndex(["2024-02-09", "2024-02-12"], name="session_date"),
    )
    records, _ = records_of(SAMPLE, session_cutoffs_utc=cutoffs, latency=pd.Timedelta(minutes=30))
    by_id = {r.article_id: r for r in records}
    assert by_id["0001234567-24-000001:2.02"].available_utc == utc("2024-02-01 22:00:05")
    assert by_id["0001234567-24-000001:2.02"].effective_session.isoformat() == "2024-02-09"


def test_the_report_counts_what_was_kept_and_skipped():
    records, report = records_of(SAMPLE)
    assert report == EdgarIngestReport(
        filings_read=4,
        records=len(records),
        skipped_items=2,
        skipped_forms=1,
        skipped_filings_without_items=0,
    )
    assert len(records) == 3  # 2.02, 5.02 and the amendment's 5.02; 9.01 and 7.01 are not rules


def test_only_8k_forms_produce_records():
    assert all(not r.article_id.startswith("0001234567-24-000003") for r in records_of(SAMPLE)[0])


def test_a_submissions_document_with_no_filings_gives_nothing():
    empty = json.loads(json.dumps(SAMPLE))
    for column in empty["filings"]["recent"].values():
        column.clear()
    records, report = records_of(empty)
    assert records == []
    assert report == EdgarIngestReport(0, 0, 0, 0, 0)


def test_records_are_keyed_uniquely():
    ids = [r.article_id for r in records_of(SAMPLE)[0]]
    assert len(ids) == len(set(ids))


def test_a_repeated_item_code_in_one_filing_gives_one_record():
    doubled = json.loads(json.dumps(SAMPLE))
    doubled["filings"]["recent"]["items"][3] = "2.02,2.02"
    ids = [r.article_id for r in records_of(doubled)[0] if "000001" in r.article_id]
    assert ids == ["0001234567-24-000001:2.02"]


@pytest.mark.parametrize("empty_items", ["", " , ", None], ids=["empty", "separators", "null"])
def test_an_8k_without_items_is_counted_not_dropped_or_failed(empty_items):
    document = json.loads(json.dumps(SAMPLE))
    document["filings"]["recent"]["items"][3] = empty_items  # the last filing, -000001: 8-K with 2.02,5.02,9.01
    records, report = records_of(document)
    assert [r.article_id for r in records] == ["0001234567-24-000002:5.02"]
    assert report == EdgarIngestReport(
        filings_read=4,
        records=1,
        skipped_items=1,  # 7.01 of the first filing in the arrays (-000004)
        skipped_forms=1,
        skipped_filings_without_items=1,
    )


def test_a_non_string_items_value_is_still_malformed():
    document = json.loads(json.dumps(SAMPLE))
    document["filings"]["recent"]["items"][3] = 202
    with pytest.raises(ValueError, match="submissions"):
        records_of(document)


@pytest.mark.parametrize(
    "damage",
    [
        lambda s: s["filings"]["recent"]["form"].pop(),
        lambda s: s["filings"]["recent"].__setitem__("acceptanceDateTime", ["2024-02-01"] * 4),
        lambda s: s["filings"]["recent"].pop("items"),
        lambda s: s.pop("name"),
        lambda s: s.pop("filings"),
    ],
    ids=["ragged-columns", "naive-timestamp", "missing-column", "missing-name", "missing-filings"],
)
def test_a_malformed_document_is_an_error_not_an_empty_result(damage):
    broken = json.loads(json.dumps(SAMPLE))
    damage(broken)
    with pytest.raises(ValueError, match="submissions"):
        records_of(broken)


def test_the_committed_item_rules_load_and_are_proposals():
    rules = load_item_rules()
    assert rules.proposed is True
    assert rules.items["2.02"].is_scheduled is True
    assert rules.items["1.03"].severity == 9
    assert all(not rule.is_scheduled for item, rule in rules.items.items() if item != "2.02")


def test_an_item_rule_entry_rejects_unknown_keys_and_event_types():
    good = {"event_type": "other", "description": "x", "severity": 4}
    assert ItemRule.model_validate(good).is_scheduled is False
    with pytest.raises(ValidationError):
        ItemRule.model_validate({**good, "surprise": 1})
    with pytest.raises(ValidationError):
        ItemRule.model_validate({**good, "event_type": "not_a_type"})


def test_a_malformed_item_rules_file_is_an_error(tmp_path):
    path = tmp_path / "rules.json"
    path.write_text('{"proposed": true, "source": "s", "items": {"1.01": {"event_type": "other"}}}')
    with pytest.raises(ValueError, match="item rules"):
        load_item_rules(path)


def test_us_state_codes_cover_states_dc_and_territories():
    codes = load_us_state_codes()
    assert len(codes) == 56
    assert {"CA", "NY", "DC", "PR", "GU"} <= codes
    assert "X0" not in codes
