import json
from pathlib import Path

import pytest
from fakes import FakeClock, FakeTransport, ok, status

from precedent.config import SEC_SUBMISSIONS_BASE_URL, ConfigError
from precedent.spine.edgar_client import EdgarClient, EdgarError, EdgarRateLimitError

SAMPLE = json.loads(
    (Path(__file__).resolve().parents[1] / "fixtures" / "submissions_sample.json").read_text()
)


def test_client_declares_user_agent_and_throttles():
    transport, clock = FakeTransport([ok(SAMPLE)] * 3), FakeClock()
    client = EdgarClient(
        "precedent-research contact@example.org", transport, clock, max_requests_per_second=5
    )
    for _ in range(3):
        client.fetch_submissions(320193)
    assert all(r.headers["User-Agent"].startswith("precedent-research") for r in transport.requests)
    assert clock.slept_seconds >= 0.4


def test_the_cik_is_zero_padded_to_ten_digits_in_the_url():
    transport = FakeTransport([ok(SAMPLE)])
    EdgarClient("ua", transport, FakeClock()).fetch_submissions(320193)
    assert transport.requests[0].url == f"{SEC_SUBMISSIONS_BASE_URL}/CIK0000320193.json"
    assert transport.requests[0].method == "GET"


def test_client_raises_a_typed_error_after_bounded_retries():
    transport = FakeTransport([status(429)] * 4)
    client = EdgarClient("ua", transport, FakeClock(), max_retries=3)
    with pytest.raises(EdgarRateLimitError):
        client.fetch_submissions(320193)
    assert len(transport.requests) == 4


def test_a_rate_limit_error_is_an_edgar_error():
    assert issubclass(EdgarRateLimitError, EdgarError)


def test_other_failures_become_an_edgar_error_naming_the_cik():
    client = EdgarClient("ua", FakeTransport([status(404)]), FakeClock())
    with pytest.raises(EdgarError, match="CIK0000320193") as raised:
        client.fetch_submissions(320193)
    assert not isinstance(raised.value, EdgarRateLimitError)


def test_a_body_that_is_not_a_submissions_object_is_an_edgar_error():
    client = EdgarClient("ua", FakeTransport([ok([1, 2, 3])]), FakeClock())
    with pytest.raises(EdgarError, match="JSON object"):
        client.fetch_submissions(320193)


def test_a_body_that_is_not_json_is_an_edgar_error():
    from precedent.spine.http import HttpResponse

    garbled = HttpResponse(200, {}, b"<html>not json</html>")
    client = EdgarClient("ua", FakeTransport([garbled]), FakeClock())
    with pytest.raises(EdgarError, match="not valid JSON"):
        client.fetch_submissions(320193)


def test_missing_user_agent_is_a_config_error(monkeypatch):
    monkeypatch.delenv("PRECEDENT_SEC_USER_AGENT", raising=False)
    with pytest.raises(ConfigError):
        EdgarClient.from_env()


def test_from_env_reads_the_declared_user_agent(monkeypatch):
    monkeypatch.setenv("PRECEDENT_SEC_USER_AGENT", "precedent-research contact@example.org")
    transport = FakeTransport([ok(SAMPLE)])
    EdgarClient.from_env(transport=transport, clock=FakeClock()).fetch_submissions(1)
    assert transport.requests[0].headers["User-Agent"] == "precedent-research contact@example.org"


def test_a_non_positive_request_rate_is_rejected():
    with pytest.raises(ValueError, match="max_requests_per_second"):
        EdgarClient("ua", FakeTransport([]), FakeClock(), max_requests_per_second=0)
