"""Tests for the Ollama typer; every transport is a fake, no socket is opened."""
import hashlib
import json

import pytest
from fakes import ExplodingTransport, FakeClock, FakeTransport, ok, status

from precedent.config import ConfigError, references_dir
from precedent.spine.typers.ollama_typer import OllamaTyper
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
BASE = "http://localhost:11434"


def chat_reply(content):
    return {"message": {"role": "assistant", "content": content}, "done": True}


def test_ollama_posts_to_chat_with_the_schema_as_format():
    transport = FakeTransport([ok(chat_reply(json.dumps(VALID)))])
    typer = OllamaTyper(BASE, "a-local-model", transport, FakeClock())
    typed = typer.type_text(TEXT)
    request = transport.requests[0]
    assert (request.method, request.url) == ("POST", f"{BASE}/api/chat")
    body = request.json_body
    assert body["format"] == typing_response_json_schema() and body["stream"] is False
    assert body["model"] == "a-local-model" and body["options"] == {"temperature": 0}
    assert [m["role"] for m in body["messages"]] == ["system", "user"]
    assert body["messages"][0]["content"] == (
        references_dir() / "typing_prompt_v1.txt"
    ).read_text("utf-8")
    assert body["messages"][1]["content"] == TEXT
    assert request.timeout_s == 60.0
    assert typed.status == "typed" and typer.calls_made == 1
    assert typed.key.as_string() == "central_bank/central_bank/country/US/high/unscheduled/1"


def test_a_trailing_slash_on_the_base_url_is_tolerated():
    transport = FakeTransport([ok(chat_reply(json.dumps(VALID)))])
    OllamaTyper(f"{BASE}/", "m", transport, FakeClock()).type_text(TEXT)
    assert transport.requests[0].url == f"{BASE}/api/chat"


def test_version_names_the_model_and_the_prompt_hash():
    prompt = (references_dir() / "typing_prompt_v1.txt").read_bytes()
    digest = hashlib.sha256(prompt).hexdigest()[:12]
    typer = OllamaTyper(BASE, "a-local-model", FakeTransport([]), FakeClock())
    assert typer.version == f"ollama:a-local-model:prompt-{digest}"


@pytest.mark.parametrize(
    "reply, reason",
    [
        (chat_reply(json.dumps({**VALID, "evidence_span": "absent"})), "evidence_span not in text"),
        (chat_reply("not json"), "schema"),
        ({"done": True}, "schema"),
        ({"message": {"content": 5}}, "schema"),
    ],
)
def test_bad_model_output_is_refused_invalid(reply, reason):
    typed = OllamaTyper(BASE, "m", FakeTransport([ok(reply)]), FakeClock()).type_text(TEXT)
    assert typed.status == "refused_invalid" and typed.reason.startswith(reason)
    assert typed.key is None and typed.evidence_span is None


def test_http_failure_is_refused_invalid_after_bounded_retries():
    transport = FakeTransport([status(503)] * 3)
    typer = OllamaTyper(BASE, "m", transport, FakeClock())
    typed = typer.type_text(TEXT)
    assert typed.reason == "http 503" and len(transport.requests) == 3


def test_max_calls_is_a_hard_limit():
    transport = FakeTransport([ok(chat_reply(json.dumps(VALID)))] * 3)
    typer = OllamaTyper(BASE, "m", transport, FakeClock(), max_calls=1)
    results = [typer.type_text(TEXT) for _ in range(2)]
    assert len(transport.requests) == 1 and results[1].reason == "max_calls reached"


def test_dry_run_opens_no_socket():
    typed = OllamaTyper(BASE, "m", ExplodingTransport(), FakeClock(), dry_run=True).type_text(TEXT)
    assert (typed.status, typed.reason) == ("untyped", "dry-run")


def test_ollama_needs_a_base_url_and_a_model(monkeypatch):
    monkeypatch.delenv("PRECEDENT_OLLAMA_BASE_URL", raising=False)
    monkeypatch.setenv("PRECEDENT_OLLAMA_MODEL", "a-local-model")
    with pytest.raises(ConfigError):
        OllamaTyper.from_env(transport=ExplodingTransport())
    monkeypatch.setenv("PRECEDENT_OLLAMA_BASE_URL", BASE)
    monkeypatch.delenv("PRECEDENT_OLLAMA_MODEL")
    with pytest.raises(ConfigError):
        OllamaTyper.from_env(transport=ExplodingTransport())


def test_from_env_builds_a_typer_from_both_variables(monkeypatch):
    monkeypatch.setenv("PRECEDENT_OLLAMA_BASE_URL", BASE)
    monkeypatch.setenv("PRECEDENT_OLLAMA_MODEL", "a-local-model")
    transport = FakeTransport([ok(chat_reply(json.dumps(VALID)))])
    typer = OllamaTyper.from_env(transport=transport, clock=FakeClock())
    assert typer.type_text(TEXT).status == "typed"
    assert transport.requests[0].json_body["model"] == "a-local-model"
