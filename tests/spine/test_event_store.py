"""Tests for the append-only event store."""
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from precedent.spine.event_record import EventRecord, ResolvedImpact
from precedent.spine.event_schema import NewsEvent
from precedent.spine.event_store import EventConflictError, EventStore

EVENT = NewsEvent.model_validate(
    {
        "schema_version": "1.0",
        "event_type": "earnings",
        "actor": {"name": "Acme Corp", "kind": "company", "country_iso3": "USA"},
        "action": "reported revenue above guidance",
        "impacted": [
            {"name": "Acme Corp", "kind": "company", "ticker_hint": "ACME", "direction_step": 2}
        ],
        "severity": 5,
        "macro_direction_step": 0,
        "confidence": "high",
        "horizon": "days",
        "novelty": "new",
        "is_scheduled": True,
        "evidence_span": "Acme reported revenue of 12 billion dollars",
    }
)

RECORD = EventRecord(
    article_id="gdelt:123",
    body_sha256="0" * 64,
    source="gdelt",
    published_utc=datetime(2024, 1, 5, 13, 0, tzinfo=UTC),
    first_seen_utc=datetime(2024, 1, 5, 14, 0, tzinfo=UTC),
    ingested_utc=datetime(2024, 1, 5, 14, 5, tzinfo=UTC),
    available_utc=datetime(2024, 1, 5, 14, 0, tzinfo=UTC),
    effective_session=date(2024, 1, 8),
    parser_version="rules-8k-v1",
    event=EVENT,
    resolved_impacts=[
        ResolvedImpact(instrument="ACME", kind="company", direction=1.0, match_method="ticker")
    ],
    linked_countries=["USA"],
)
UNALIGNED_RECORD = RECORD.model_copy(update={"article_id": "gdelt:124", "effective_session": None})
UNRESOLVED_RECORD = RECORD.model_copy(update={"article_id": "gdelt:125", "resolved_impacts": []})


def test_append_is_idempotent_and_refuses_conflicts(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    assert store.append([RECORD]).added == 1
    assert store.append([RECORD]).identical_skipped == 1
    with pytest.raises(EventConflictError):
        store.append([RECORD.model_copy(update={"body_sha256": "f" * 64})])


def test_a_record_differing_only_in_ingested_utc_is_skipped_not_a_conflict(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.append([RECORD])
    later = RECORD.model_copy(update={"ingested_utc": datetime(2024, 2, 1, tzinfo=UTC)})
    report = store.append([later])
    assert (report.added, report.identical_skipped) == (0, 1)
    assert store.read("rules-8k-v1") == [RECORD]


def test_a_conflict_in_a_batch_writes_nothing(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.append([RECORD])
    changed = RECORD.model_copy(update={"body_sha256": "f" * 64})
    with pytest.raises(EventConflictError):
        store.append([UNALIGNED_RECORD, changed])
    assert [r.article_id for r in store.read("rules-8k-v1")] == ["gdelt:123"]


def test_read_round_trips_records_in_append_order(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.append([RECORD, UNALIGNED_RECORD])
    assert store.read("rules-8k-v1") == [RECORD, UNALIGNED_RECORD]


def test_read_is_scoped_to_one_parser_version(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.append([RECORD, RECORD.model_copy(update={"parser_version": "other-v2"})])
    assert [r.parser_version for r in store.read("rules-8k-v1")] == ["rules-8k-v1"]


def test_read_and_frame_of_a_missing_version_are_empty_not_errors(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    assert store.read("never-written") == []
    frame, report = store.to_frame("never-written")
    assert frame.empty
    assert list(frame.columns) == [
        "effective_session", "instrument", "severity", "direction", "confidence_weight",
        "novelty", "event_type", "article_id", "available_utc",
    ]
    assert (report.records, report.impact_rows) == (0, 0)
    assert (report.unresolved_impacts, report.unaligned_records) == (0, 0)


def test_the_file_name_is_a_slug_of_the_parser_version(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.append([RECORD.model_copy(update={"parser_version": "a/b c:v1"})])
    assert [p.name for p in tmp_path.iterdir()] == ["events_a_b_c_v1.jsonl"]


def test_two_versions_with_one_slug_raise(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.append([RECORD.model_copy(update={"parser_version": "a/b"})])
    with pytest.raises(ValueError, match="slug"):
        store.append([RECORD.model_copy(update={"parser_version": "a b"})])
    with pytest.raises(ValueError, match="slug"):
        store.append(
            [
                RECORD.model_copy(update={"parser_version": "x/y"}),
                RECORD.model_copy(update={"parser_version": "x y"}),
            ]
        )


def test_a_malformed_line_names_the_file_and_line(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.append([RECORD])
    path = store.path_for("rules-8k-v1")
    path.write_text(path.read_text() + "{not json}\n")
    with pytest.raises(ValueError, match=r"events_rules-8k-v1\.jsonl line 2"):
        store.read("rules-8k-v1")


def test_frame_counts_unaligned_and_unresolved_records(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.append([RECORD, UNALIGNED_RECORD, UNRESOLVED_RECORD])
    frame, report = store.to_frame("rules-8k-v1")
    assert len(frame) == 1
    assert (report.unaligned_records, report.unresolved_impacts) == (1, 1)
    assert (report.records, report.impact_rows) == (3, 1)


def test_frame_has_one_row_per_resolved_impact_with_typed_columns(tmp_path: Path) -> None:
    two = RECORD.model_copy(
        update={
            "resolved_impacts": [
                ResolvedImpact(instrument="ACME", kind="company", direction=1.0, match_method="t"),
                ResolvedImpact(instrument="SPY", kind="index", direction=-0.5, match_method="t"),
            ]
        }
    )
    store = EventStore(tmp_path)
    store.append([two])
    frame, _ = store.to_frame("rules-8k-v1")
    assert frame["instrument"].tolist() == ["ACME", "SPY"]
    assert frame["direction"].tolist() == [1.0, -0.5]
    row = frame.iloc[0]
    assert row["severity"] == 5
    assert row["confidence_weight"] == 0.9
    assert (row["novelty"], row["event_type"], row["article_id"]) == ("new", "earnings", "gdelt:123")
    assert str(frame["effective_session"].dtype) == "datetime64[ns]"
    assert frame["effective_session"].iloc[0] == pd.Timestamp("2024-01-08")
    assert str(frame["available_utc"].dtype) == "datetime64[ns, UTC]"
    assert frame["available_utc"].iloc[0] == pd.Timestamp("2024-01-05 14:00", tz="UTC")


def test_unresolved_impacts_counts_impacts_not_records(tmp_path: Path) -> None:
    two_named = EVENT.model_copy(
        update={
            "impacted": [
                *EVENT.impacted,
                EVENT.impacted[0].model_copy(update={"name": "Beta Inc", "ticker_hint": "BETA"}),
            ]
        }
    )
    partly_resolved = RECORD.model_copy(update={"event": two_named})
    store = EventStore(tmp_path)
    store.append([partly_resolved])
    frame, report = store.to_frame("rules-8k-v1")
    assert len(frame) == 1
    assert (report.unresolved_impacts, report.unaligned_records) == (1, 0)


def test_an_unaligned_record_adds_nothing_to_unresolved_impacts(tmp_path: Path) -> None:
    unaligned_unresolved = UNALIGNED_RECORD.model_copy(update={"resolved_impacts": []})
    store = EventStore(tmp_path)
    store.append([unaligned_unresolved])
    _, report = store.to_frame("rules-8k-v1")
    assert (report.unresolved_impacts, report.unaligned_records) == (0, 1)


def test_the_same_article_twice_in_one_append_is_skipped_once(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    report = store.append([RECORD, RECORD])
    assert (report.added, report.identical_skipped) == (1, 1)
    assert store.read("rules-8k-v1") == [RECORD]


def test_a_differing_repeat_in_one_append_conflicts_and_writes_nothing(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    changed = RECORD.model_copy(update={"body_sha256": "f" * 64})
    with pytest.raises(EventConflictError):
        store.append([RECORD, changed])
    assert list(tmp_path.iterdir()) == []


def test_a_missing_available_utc_becomes_nat_in_the_frame(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.append([RECORD.model_copy(update={"available_utc": None})])
    frame, _ = store.to_frame("rules-8k-v1")
    assert pd.isna(frame["available_utc"].iloc[0])
    assert str(frame["available_utc"].dtype) == "datetime64[ns, UTC]"
