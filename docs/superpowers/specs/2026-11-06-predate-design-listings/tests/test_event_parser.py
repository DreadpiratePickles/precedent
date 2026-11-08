"""Tests for the article -> NewsEvent parsing step, using a canned fake instead of a model."""
import hashlib
import json

import pytest

from event_parser import (
    PROMPT_TEMPLATE,
    ParseStatus,
    build_prompt,
    cache_key,
    parse_article,
)
from event_schema import news_event_json_schema

TITLE = "Acme beats estimates"
BODY = "Acme reported revenue of 12 billion dollars on Tuesday.\nAnalysts had expected 11 billion."
EVIDENCE = "Acme reported revenue of 12 billion dollars"


def event_json(**overrides: object) -> str:
    payload: dict = {
        "schema_version": "1.0",
        "event_type": "earnings",
        "actor": {"name": "Acme", "kind": "company", "country_iso3": None},
        "action": "reported revenue above expectations",
        "impacted": [{"name": "Acme", "kind": "company", "ticker_hint": "ACME", "direction_step": 2}],
        "severity": 4,
        "macro_direction_step": 0,
        "confidence": "high",
        "horizon": "days",
        "novelty": "new",
        "is_scheduled": True,
        "evidence_span": EVIDENCE,
    }
    payload.update(overrides)
    return json.dumps(payload)


class FakeExtractor:
    """Returns one canned string and records every call."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, json_schema: dict) -> str:
        self.calls.append((prompt, json_schema))
        return self.reply


class RaisingExtractor:
    def generate(self, prompt: str, json_schema: dict) -> str:
        raise ConnectionError("transport down")


def run(reply: str, title: str = TITLE, body: str = BODY):
    extractor = FakeExtractor(reply)
    return parse_article(title, body, extractor, parser_version="v1"), extractor


def test_prompt_template_has_exactly_two_placeholders() -> None:
    assert PROMPT_TEMPLATE.count("{title}") == 1
    assert PROMPT_TEMPLATE.count("{body}") == 1
    assert PROMPT_TEMPLATE.startswith("You convert one news article into one JSON object")
    assert PROMPT_TEMPLATE.rstrip().endswith("Article:\n{body}")


def test_ok_status() -> None:
    result, _ = run(event_json())
    assert result.status is ParseStatus.ok
    assert result.event is not None and result.event.severity == 4
    assert result.error is None
    assert result.parser_version == "v1"
    assert result.body_sha256 == hashlib.sha256(BODY.encode("utf-8")).hexdigest()


def test_no_market_event_status_keeps_the_event() -> None:
    reply = event_json(event_type="no_market_event", severity=1, impacted=[])
    result, _ = run(reply)
    assert result.status is ParseStatus.no_market_event
    assert result.event is not None and result.event.impacted == []


def test_invalid_json_status() -> None:
    result, _ = run("Sure! Here is the JSON: {")
    assert result.status is ParseStatus.invalid_json
    assert result.event is None and result.error


def test_schema_violation_status_and_error_does_not_leak_the_article() -> None:
    secret = "ZEBRA-SECRET-PHRASE"
    result, _ = run(event_json(severity=11, evidence_span=secret * 20), body=f"{BODY} {secret}")
    assert result.status is ParseStatus.schema_violation
    assert result.event is None and result.error
    assert secret not in result.error and "ZEBRA" not in result.error


@pytest.mark.parametrize("reply", ["[]", '"text"', "null", event_json(severity="4")])
def test_non_object_or_loosely_typed_json_is_a_schema_violation(reply: str) -> None:
    result, _ = run(reply)
    assert result.status is ParseStatus.schema_violation


def test_evidence_not_in_article_status() -> None:
    result, _ = run(event_json(evidence_span="Acme announced a merger with Globex"))
    assert result.status is ParseStatus.evidence_not_in_article
    assert result.event is None and result.error


def test_evidence_matching_is_case_sensitive() -> None:
    result, _ = run(event_json(evidence_span=EVIDENCE.lower()))
    assert result.status is ParseStatus.evidence_not_in_article


def test_evidence_with_different_whitespace_still_matches() -> None:
    body = "Acme reported\n  revenue of   12 billion\tdollars on Tuesday."
    result, _ = run(event_json(), body=body)
    assert result.status is ParseStatus.ok


def test_blank_evidence_never_matches() -> None:
    result, _ = run(event_json(evidence_span="   \n  "))
    assert result.status is ParseStatus.evidence_not_in_article


def test_evidence_beyond_the_truncation_point_is_rejected() -> None:
    body = "x" * 6000 + " Acme reported revenue of 12 billion dollars"
    result, _ = run(event_json(), body=body)
    assert result.status is ParseStatus.evidence_not_in_article


def test_extractor_called_once_with_prompt_and_schema() -> None:
    _, extractor = run(event_json())
    assert len(extractor.calls) == 1
    prompt, schema = extractor.calls[0]
    assert prompt == build_prompt(TITLE, BODY)
    assert schema == news_event_json_schema()


def test_extractor_exceptions_propagate() -> None:
    with pytest.raises(ConnectionError):
        parse_article(TITLE, BODY, RaisingExtractor(), parser_version="v1")


def test_build_prompt_truncates_the_head_and_is_deterministic() -> None:
    body = "A" * 10 + "B" * 10
    prompt = build_prompt("T", body, max_body_chars=10)
    assert prompt.endswith("Article:\n" + "A" * 10)
    assert "B" not in prompt.split("Article:")[1]
    assert prompt == build_prompt("T", body, max_body_chars=10)


def test_build_prompt_survives_braces_in_the_article() -> None:
    prompt = build_prompt("{title} {0}", "{body} {} {x}")
    assert "Title: {title} {0}" in prompt and prompt.endswith("{body} {} {x}")


def test_build_prompt_rejects_nonpositive_limit() -> None:
    with pytest.raises(ValueError, match="max_body_chars"):
        build_prompt("T", "body", max_body_chars=0)


def test_cache_key_depends_on_parser_version_and_body_hash() -> None:
    sha_a = hashlib.sha256(b"a").hexdigest()
    sha_b = hashlib.sha256(b"b").hexdigest()
    assert cache_key(sha_a, "v1") == cache_key(sha_a, "v1")
    assert len({cache_key(sha_a, "v1"), cache_key(sha_a, "v2"), cache_key(sha_b, "v1")}) == 3
    with pytest.raises(ValueError, match="body_sha256"):
        cache_key("not-a-hash", "v1")
