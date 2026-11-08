import json
import random

import httpx
import pytest
from fakes import FakeClock, FakeTransport, ok, status

from precedent.spine.http import (
    HttpError,
    HttpRequest,
    HttpxTransport,
    RateLimiter,
    RateLimitError,
    SystemClock,
    send_with_retries,
)


def test_retries_429_and_5xx_then_succeeds():
    transport, clock = FakeTransport([status(429), status(503), ok({"a": 1})]), FakeClock()
    response = send_with_retries(
        transport,
        clock,
        HttpRequest("GET", "https://example.test/x", {}),
        max_retries=3,
        rng=random.Random(0),
    )
    assert response.json() == {"a": 1}
    assert len(transport.requests) == 3
    assert clock.slept_seconds > 0


def test_a_404_is_not_retried():
    transport = FakeTransport([status(404)])
    with pytest.raises(HttpError) as raised:
        send_with_retries(transport, FakeClock(), HttpRequest("GET", "https://example.test/x", {}))
    assert raised.value.status == 404
    assert len(transport.requests) == 1


def test_exhausted_429s_raise_a_rate_limit_error():
    with pytest.raises(RateLimitError):
        send_with_retries(
            FakeTransport([status(429)] * 4),
            FakeClock(),
            HttpRequest("GET", "https://example.test/x", {}),
            max_retries=3,
        )


def test_exhausted_5xx_raise_a_plain_http_error():
    with pytest.raises(HttpError) as raised:
        send_with_retries(
            FakeTransport([status(500)] * 4),
            FakeClock(),
            HttpRequest("GET", "https://example.test/x", {}),
            max_retries=3,
        )
    assert not isinstance(raised.value, RateLimitError)
    assert raised.value.status == 500


def test_backoff_is_bounded_by_the_exponential_cap():
    clock = FakeClock()
    send_with_retries(
        FakeTransport([status(503), status(503), ok({})]),
        clock,
        HttpRequest("GET", "https://example.test/x", {}),
        base_delay_s=0.5,
        rng=random.Random(1),
    )
    assert 0 < clock.slept_seconds <= 0.5 + 1.0


def test_rate_limiter_waits_for_the_window():
    clock = FakeClock()
    limiter = RateLimiter(max_requests=3, per_seconds=1.0, clock=clock)
    for _ in range(4):
        limiter.acquire()
    assert clock.slept_seconds == pytest.approx(1.0)


def test_rate_limiter_with_one_request_spaces_evenly():
    clock = FakeClock()
    limiter = RateLimiter(1, 0.5, clock)
    for _ in range(3):
        limiter.acquire()
    assert clock.slept_seconds == pytest.approx(1.0)


def test_response_decodes_text_and_json():
    response = ok({"k": "v"})
    assert response.json() == {"k": "v"}
    assert '"k"' in response.text()


def test_httpx_transport_sends_method_url_params_and_json():
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json={"echo": True})

    transport = HttpxTransport(httpx.Client(transport=httpx.MockTransport(handler)))
    response = transport.send(
        HttpRequest("POST", "https://example.test/p", {"x-k": "v"}, {"q": "1"}, {"a": 2})
    )
    assert response.status == 200
    assert response.json() == {"echo": True}
    assert seen[0].method == "POST"
    assert seen[0].url.params["q"] == "1"
    assert seen[0].headers["x-k"] == "v"
    assert json.loads(seen[0].content) == {"a": 2}


def test_system_clock_is_utc_aware_and_monotonic():
    clock = SystemClock()
    assert clock.now_utc().utcoffset().total_seconds() == 0
    first = clock.monotonic()
    clock.sleep(0)
    assert clock.monotonic() >= first


@pytest.mark.parametrize(("max_requests", "per_seconds", "acquisitions", "expected_sleep"), [
    (1, 0.1, 50, 4.9),
    (2, 0.7, 10, 0.7 * 4),
    (1, 0.3, 20, 0.3 * 19),
    (1, 1 / 3, 30, 29 / 3),
])
def test_rate_limiter_terminates_with_non_dyadic_periods(
    max_requests, per_seconds, acquisitions, expected_sleep
):
    clock = FakeClock()
    limiter = RateLimiter(max_requests, per_seconds, clock)
    for _ in range(acquisitions):
        limiter.acquire()
    assert clock.slept_seconds == pytest.approx(expected_sleep)


def test_httpx_transport_wraps_transport_errors_as_status_zero():
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        raise httpx.ConnectError("refused", request=request)

    transport = HttpxTransport(httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(HttpError) as raised:
        transport.send(HttpRequest("GET", "https://example.test/x", {}))
    assert raised.value.status == 0
    assert "https://example.test/x" in str(raised.value)
    assert "ConnectError" in str(raised.value)
    assert isinstance(raised.value.__cause__, httpx.ConnectError)


def test_status_zero_is_retried_like_a_5xx_then_succeeds():
    transport, clock = FakeTransport([status(0), status(0), ok({"a": 1})]), FakeClock()
    response = send_with_retries(
        transport, clock, HttpRequest("GET", "https://example.test/x", {}), rng=random.Random(0)
    )
    assert response.json() == {"a": 1}
    assert len(transport.requests) == 3
    assert clock.slept_seconds > 0


def test_exhausted_status_zero_raises_a_plain_http_error():
    transport = FakeTransport([status(0)] * 4)
    with pytest.raises(HttpError) as raised:
        send_with_retries(
            transport, FakeClock(), HttpRequest("GET", "https://example.test/x", {}), max_retries=3
        )
    assert not isinstance(raised.value, RateLimitError)
    assert raised.value.status == 0
    assert len(transport.requests) == 4


def test_a_real_transport_failure_is_retried_end_to_end():
    attempts = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) < 3:
            raise httpx.ReadTimeout("slow", request=request)
        return httpx.Response(200, json={"ok": True})

    transport = HttpxTransport(httpx.Client(transport=httpx.MockTransport(handler)))
    response = send_with_retries(
        transport, FakeClock(), HttpRequest("GET", "https://example.test/x", {}), max_retries=3
    )
    assert response.json() == {"ok": True}
    assert len(attempts) == 3
