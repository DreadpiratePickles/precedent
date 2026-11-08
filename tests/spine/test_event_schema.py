"""Tests for the NewsEvent contract that the LLM parser may emit."""
import json

import pytest
from pydantic import ValidationError

from precedent.config import references_dir
from precedent.spine.event_schema import (
    CONFIDENCE_WEIGHTS,
    Confidence,
    EventType,
    NewsEvent,
    news_event_json_schema,
)


def valid_event_dict() -> dict:
    return {
        "schema_version": "1.0",
        "event_type": "earnings",
        "actor": {"name": "Acme Corp", "kind": "company", "country_iso3": "USA"},
        "action": "reported quarterly revenue above its own guidance",
        "impacted": [
            {"name": "Acme Corp", "kind": "company", "ticker_hint": "ACME", "direction_step": 2},
            {"name": "Widgets", "kind": "sector", "ticker_hint": None, "direction_step": -1},
        ],
        "severity": 5,
        "macro_direction_step": 1,
        "confidence": "high",
        "horizon": "days",
        "novelty": "new",
        "is_scheduled": True,
        "evidence_span": "Acme reported revenue of 12 billion dollars",
    }


def test_valid_example_round_trips_through_json() -> None:
    event = NewsEvent.model_validate(valid_event_dict())
    again = NewsEvent.model_validate_json(event.model_dump_json())
    assert again == event
    assert event.event_type is EventType.earnings


def test_numeric_properties() -> None:
    event = NewsEvent.model_validate(valid_event_dict())
    assert event.impacted[0].direction == 1.0
    assert event.impacted[1].direction == -0.5
    assert event.macro_direction == 0.5
    assert event.confidence_weight == CONFIDENCE_WEIGHTS[Confidence.high] == 0.9
    assert dict(CONFIDENCE_WEIGHTS) == {
        Confidence.low: 0.3,
        Confidence.medium: 0.6,
        Confidence.high: 0.9,
    }


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("severity", 11),
        ("severity", 0),
        ("macro_direction_step", 3),
        ("event_type", "alien_invasion"),
        ("confidence", "certain"),
        ("schema_version", "2.0"),
    ],
)
def test_out_of_contract_values_are_rejected(field: str, bad_value: object) -> None:
    payload = valid_event_dict()
    payload[field] = bad_value
    with pytest.raises(ValidationError):
        NewsEvent.model_validate(payload)


def test_impact_direction_step_3_is_rejected() -> None:
    payload = valid_event_dict()
    payload["impacted"][0]["direction_step"] = 3
    with pytest.raises(ValidationError):
        NewsEvent.model_validate(payload)


def test_extra_key_is_rejected_on_event_and_nested_models() -> None:
    top = valid_event_dict() | {"price_target": 123}
    with pytest.raises(ValidationError):
        NewsEvent.model_validate(top)
    nested = valid_event_dict()
    nested["actor"]["lei"] = "X"
    with pytest.raises(ValidationError):
        NewsEvent.model_validate(nested)


def test_ticker_and_country_patterns_are_enforced() -> None:
    payload = valid_event_dict()
    payload["impacted"][0]["ticker_hint"] = "lower"
    with pytest.raises(ValidationError):
        NewsEvent.model_validate(payload)
    payload = valid_event_dict()
    payload["actor"]["country_iso3"] = "US"
    with pytest.raises(ValidationError):
        NewsEvent.model_validate(payload)


def test_impacted_is_capped_at_eight() -> None:
    payload = valid_event_dict()
    payload["impacted"] = [payload["impacted"][0]] * 9
    with pytest.raises(ValidationError):
        NewsEvent.model_validate(payload)


def test_json_schema_closes_severity_and_forbids_extra_keys() -> None:
    schema = news_event_json_schema()
    assert schema["properties"]["severity"]["enum"] == list(range(1, 11))
    assert schema["properties"]["macro_direction_step"]["enum"] == [-2, -1, 0, 1, 2]
    assert schema["additionalProperties"] is False
    assert "evidence_span" in schema["required"]
    for nested in ("Entity", "Impact"):
        assert schema["$defs"][nested]["additionalProperties"] is False


def test_reference_schema_file_matches_the_code() -> None:
    reference = json.loads((references_dir() / "event_schema_v1.json").read_text())
    assert reference == news_event_json_schema()
