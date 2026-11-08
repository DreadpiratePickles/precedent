"""Tests for the Anthropic typer; every transport is a fake, no socket is opened."""
import hashlib
import json

import pytest
from fakes import ExplodingTransport, FakeClock, FakeTransport, ok, status

from precedent import config
from precedent.config import ConfigError, references_dir
from precedent.spine.typers.anthropic_typer import AnthropicTyper
from precedent.spine.typers.typing_response import typing_response_json_schema

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


def message_with_text(text):
    return {"stop_reason": "end_turn", "content": [{"type": "text", "text": text}]}


def anthropic_typer_with(responses_or_transport, **kwargs):
    transport = (
        responses_or_transport
        if isinstance(responses_or_transport, FakeTransport)
        else FakeTransport(responses_or_transport)
    )
    return AnthropicTyper("test-key", config.ANTHROPIC_TYPER_MODEL, transport, FakeClock(), **kwargs)


def test_anthropic_request_matches_the_messages_api():
    transport = FakeTransport([ok(message_with_text(json.dumps(VALID)))])
    typer = AnthropicTyper("test-key", config.ANTHROPIC_TYPER_MODEL, transport, FakeClock())
    typed = typer.type_text(TEXT)
    request = transport.requests[0]
    assert (request.method, request.url) == ("POST", f"{config.ANTHROPIC_BASE_URL}/v1/messages")
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["anthropic-version"] == config.ANTHROPIC_VERSION
    assert request.headers["content-type"] == "application/json"
    body = request.json_body
    assert body["output_config"]["format"] == {
        "type": "json_schema",
        "schema": typing_response_json_schema(),
    }
    assert body["output_config"]["effort"] == "low"
    assert body["model"] == config.ANTHROPIC_TYPER_MODEL and body["max_tokens"] == 1024
    assert body["messages"] == [{"role": "user", "content": TEXT}]
    assert body["system"] == (references_dir() / "typing_prompt_v1.txt").read_text("utf-8")
    assert not {"temperature", "top_p", "top_k", "thinking"} & body.keys()
    assert request.timeout_s == 30.0 and typed.status == "typed"
    assert typed.key.as_string() == "central_bank/central_bank/country/US/high/unscheduled/1"


def test_version_names_the_model_and_the_prompt_hash():
    prompt = (references_dir() / "typing_prompt_v1.txt").read_bytes()
    digest = hashlib.sha256(prompt).hexdigest()[:12]
    typer = anthropic_typer_with([])
    assert typer.version == f"anthropic:{config.ANTHROPIC_TYPER_MODEL}:prompt-{digest}"


def test_text_is_read_from_text_blocks_not_by_position():
    reply = {
        "stop_reason": "end_turn",
        "content": [
            {"type": "thinking", "thinking": "", "signature": "s"},
            {"type": "text", "text": json.dumps(VALID)},
        ],
    }
    assert anthropic_typer_with([ok(reply)]).type_text(TEXT).status == "typed"


def test_several_text_blocks_are_concatenated():
    raw = json.dumps(VALID)
    reply = {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": raw[:10]}, {"type": "text", "text": raw[10:]}],
    }
    assert anthropic_typer_with([ok(reply)]).type_text(TEXT).status == "typed"


@pytest.mark.parametrize(
    "reply, reason",
    [
        (
            message_with_text(json.dumps({**VALID, "evidence_span": "not in the text"})),
            "evidence_span not in text",
        ),
        (message_with_text(json.dumps({**VALID, "extra": 1})), "schema"),
        ({"stop_reason": "refusal", "content": []}, "model refusal"),
        ({"stop_reason": "max_tokens", "content": [{"type": "text", "text": "{"}]}, "truncated"),
        ({"stop_reason": "end_turn", "content": "nope"}, "schema"),
        ({"stop_reason": "end_turn"}, "schema"),
    ],
)
def test_bad_model_output_is_refused_invalid(reply, reason):
    typed = anthropic_typer_with([ok(reply)]).type_text(TEXT)
    assert typed.status == "refused_invalid" and typed.key is None
    assert typed.evidence_span is None and typed.reason.startswith(reason)


def test_a_non_json_body_is_refused_invalid():
    from fakes import ok_text

    typed = anthropic_typer_with([ok_text("<html>")]).type_text(TEXT)
    assert typed.status == "refused_invalid" and typed.reason.startswith("schema")


def test_http_failure_is_refused_invalid_after_bounded_retries():
    transport = FakeTransport([status(529)] * 3)
    typer = anthropic_typer_with(transport, max_calls=10)
    typed = typer.type_text(TEXT)
    assert typed.status == "refused_invalid" and typed.reason == "http 529"
    assert len(transport.requests) == 3 and typer.calls_made == 3


def test_a_client_error_is_not_retried():
    transport = FakeTransport([status(400)])
    typed = anthropic_typer_with(transport).type_text(TEXT)
    assert typed.reason == "http 400" and len(transport.requests) == 1


def test_dry_run_opens_no_socket():
    typer = AnthropicTyper("k", "m", ExplodingTransport(), FakeClock(), dry_run=True)
    typed = typer.type_text(TEXT)
    assert (typed.status, typed.reason) == ("untyped", "dry-run")
    assert typed.key is None and typed.evidence_span is None and typer.calls_made == 0


def test_missing_key_is_a_config_error_before_any_request(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError):
        AnthropicTyper.from_env(transport=ExplodingTransport(), dry_run=True)


def test_from_env_reads_the_key_and_the_configured_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "env-key")
    transport = FakeTransport([ok(message_with_text(json.dumps(VALID)))])
    typer = AnthropicTyper.from_env(transport=transport, clock=FakeClock())
    typer.type_text(TEXT)
    assert transport.requests[0].headers["x-api-key"] == "env-key"
    assert transport.requests[0].json_body["model"] == config.ANTHROPIC_TYPER_MODEL


def test_max_calls_is_a_hard_limit():
    transport = FakeTransport([ok(message_with_text(json.dumps(VALID)))] * 3)
    typer = anthropic_typer_with(transport, max_calls=2)
    results = [typer.type_text(TEXT) for _ in range(3)]
    assert len(transport.requests) == 2 and typer.calls_made == 2
    assert results[2].status == "refused_invalid" and results[2].reason == "max_calls reached"


def test_retries_count_toward_max_calls():
    transport = FakeTransport([status(529)] * 3)
    typer = anthropic_typer_with(transport, max_calls=2)
    typed = typer.type_text(TEXT)
    assert typed.reason == "max_calls reached"
    assert len(transport.requests) == 2 and typer.calls_made == 2
