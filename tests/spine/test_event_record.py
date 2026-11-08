"""Tests for the stored event record envelope."""
from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from precedent.spine.event_record import EventRecord, ResolvedImpact
from precedent.spine.event_schema import NewsEvent

EVENT = {
    "schema_version": "1.0",
    "event_type": "earnings",
    "actor": {"name": "Acme Corp", "kind": "company", "country_iso3": "USA"},
    "action": "reported revenue above guidance",
    "impacted": [{"name": "Acme Corp", "kind": "company", "ticker_hint": "ACME", "direction_step": 2}],
    "severity": 5,
    "macro_direction_step": 0,
    "confidence": "high",
    "horizon": "days",
    "novelty": "new",
    "is_scheduled": True,
    "evidence_span": "Acme reported revenue of 12 billion dollars",
}

VALID_RECORD = {
    "article_id": "gdelt:123",
    "body_sha256": "0" * 64,
    "source": "gdelt",
    "published_utc": datetime(2024, 1, 5, 13, 0, tzinfo=UTC),
    "first_seen_utc": datetime(2024, 1, 5, 14, 0, tzinfo=UTC),
    "ingested_utc": datetime(2024, 1, 5, 14, 5, tzinfo=UTC),
    "available_utc": None,
    "effective_session": None,
    "parser_version": "v1",
    "event": NewsEvent.model_validate(EVENT),
    "resolved_impacts": [],
    "linked_countries": ["USA"],
}


def test_a_valid_record_is_accepted_and_keeps_the_event() -> None:
    record = EventRecord(**VALID_RECORD)
    assert record.event.severity == 5
    assert record.effective_session is None


def test_effective_session_is_a_date() -> None:
    record = EventRecord(**{**VALID_RECORD, "effective_session": date(2024, 1, 8)})
    assert record.effective_session == date(2024, 1, 8)


def test_event_record_rejects_a_naive_first_seen() -> None:
    with pytest.raises(ValidationError):
        EventRecord(**{**VALID_RECORD, "first_seen_utc": datetime(2024, 1, 5, 14, 0)})


@pytest.mark.parametrize("field", ["published_utc", "ingested_utc", "available_utc"])
def test_every_timestamp_must_be_timezone_aware(field: str) -> None:
    with pytest.raises(ValidationError):
        EventRecord(**{**VALID_RECORD, field: datetime(2024, 1, 5, 14, 0)})


def test_first_seen_is_required() -> None:
    payload = {k: v for k, v in VALID_RECORD.items() if k != "first_seen_utc"}
    with pytest.raises(ValidationError):
        EventRecord(**payload)


def test_linked_countries_must_be_iso3() -> None:
    with pytest.raises(ValidationError):
        EventRecord(**{**VALID_RECORD, "linked_countries": ["usa"]})


def test_linked_countries_is_required_but_may_be_empty() -> None:
    payload = {k: v for k, v in VALID_RECORD.items() if k != "linked_countries"}
    with pytest.raises(ValidationError):
        EventRecord(**payload)
    assert EventRecord(**{**VALID_RECORD, "linked_countries": []}).linked_countries == []


def test_extra_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        EventRecord(**{**VALID_RECORD, "price_target": 1})


def test_resolved_impact_round_trips() -> None:
    impact = ResolvedImpact(instrument="AAPL", kind="company", direction=-0.5, match_method="ticker")
    record = EventRecord(**{**VALID_RECORD, "resolved_impacts": [impact]})
    assert EventRecord.model_validate_json(record.model_dump_json()) == record
