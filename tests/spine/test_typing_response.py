"""Tests for the typing response model: schema subset, evidence check, key building."""
import json

import pytest

from precedent.spine.typers.typing_response import (
    TypingResponse,
    to_typed_text,
    typing_response_json_schema,
)

TEXT = "Fed unexpectedly cuts rates in emergency move"
VALID = {
    "event_type": "central_bank",
    "actor_kind": "central_bank",
    "impacted_kinds": ["country"],
    "country_group": "US",
    "severity_band": "high",
    "is_scheduled": False,
    "direction_sign": 1,
    "evidence_span": "cuts rates",
}
VERSION = "test:v1"


def object_nodes(node):
    """Every dict in the schema tree that declares `type: object`."""
    found = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found.extend(object_nodes(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(object_nodes(value))
    return found


def typed(payload, text=TEXT):
    return to_typed_text(json.dumps(payload), text, VERSION)


def test_schema_fits_the_structured_output_subset():
    schema = typing_response_json_schema()
    dumped = json.dumps(schema)
    for keyword in ('"minLength"', '"maxLength"', '"minimum"', '"maximum"', '"maxItems"'):
        assert keyword not in dumped
    nodes = object_nodes(schema)
    assert nodes
    assert all(node.get("additionalProperties") is False for node in nodes)


def test_schema_is_a_pure_function_of_the_model():
    assert typing_response_json_schema() == typing_response_json_schema()
    assert set(typing_response_json_schema()["properties"]) == set(TypingResponse.model_fields)


def test_a_valid_response_is_typed_with_the_expected_key():
    result = typed(VALID)
    assert result.status == "typed"
    assert result.key.as_string() == "central_bank/central_bank/country/US/high/unscheduled/1"
    assert result.evidence_span == "cuts rates"
    assert result.typer_version == VERSION


def test_impacted_kinds_are_sorted_and_unique():
    result = typed({**VALID, "impacted_kinds": ["index", "country", "index", "currency"]})
    assert result.key.impacted_kinds == ("country", "currency", "index")


def test_evidence_check_is_whitespace_normalised_and_case_sensitive():
    spaced = "Fed unexpectedly cuts\n   rates in emergency move"
    assert typed(VALID, spaced).status == "typed"
    wrong_case = typed({**VALID, "evidence_span": "Cuts Rates"})
    assert wrong_case.status == "refused_invalid"
    assert wrong_case.reason == "evidence_span not in text"


def test_an_empty_or_missing_span_is_refused():
    assert typed({**VALID, "evidence_span": "  "}).reason == "evidence_span not in text"
    result = typed({k: v for k, v in VALID.items() if k != "evidence_span"})
    assert result.status == "refused_invalid" and result.reason.startswith("schema: ")


def test_no_market_event_is_untyped_with_no_key_or_span():
    result = typed({**VALID, "event_type": "no_market_event"})
    assert (result.status, result.reason) == ("untyped", "model: no_market_event")
    assert result.key is None and result.evidence_span is None


@pytest.mark.parametrize(
    "raw",
    [
        "not json",
        "[]",
        json.dumps({**VALID, "extra": 1}),
        json.dumps({**VALID, "event_type": "made_up"}),
        json.dumps({**VALID, "direction_sign": 2}),
        json.dumps({**VALID, "is_scheduled": "false"}),
        json.dumps({**VALID, "impacted_kinds": ["nonsense"]}),
    ],
)
def test_invalid_json_or_schema_is_refused_with_a_schema_reason(raw):
    result = to_typed_text(raw, TEXT, VERSION)
    assert result.status == "refused_invalid"
    assert result.reason.startswith("schema: ")
    assert result.key is None and result.evidence_span is None
