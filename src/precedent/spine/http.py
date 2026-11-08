"""The shared HTTP seam: every client takes an injected Transport and Clock."""

from __future__ import annotations

import json
import random
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

import httpx

RETRYABLE_STATUS_MIN_5XX = 500
STATUS_TOO_MANY_REQUESTS = 429
STATUS_TRANSPORT_FAILURE = 0


@dataclass(frozen=True)
class HttpRequest:
    method: Literal["GET", "POST"]
    url: str
    headers: Mapping[str, str]
    params: Mapping[str, str] | None = None
    json_body: Any = None
    timeout_s: float = 30.0


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: Mapping[str, str]
    body: bytes

    def json(self) -> Any:
        return json.loads(self.body)

    def text(self) -> str:
        return self.body.decode("utf-8")


class Transport(Protocol):
    def send(self, request: HttpRequest) -> HttpResponse: ...


class Clock(Protocol):
    def now_utc(self) -> datetime: ...

    def monotonic(self) -> float: ...

    def sleep(self, seconds: float) -> None: ...


class HttpError(RuntimeError):
    """A non-2xx response. `status` is the HTTP status code, or 0 when no response arrived."""

    def __init__(self, message: str, status: int) -> None:
        super().__init__(message)
        self.status = status


class RateLimitError(HttpError):
    """The server kept answering 429 after the retry budget was spent."""


class HttpxTransport:
    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client if client is not None else httpx.Client()

    def send(self, request: HttpRequest) -> HttpResponse:
        try:
            response = self._client.request(
                request.method,
                request.url,
                headers=dict(request.headers),
                params=dict(request.params) if request.params is not None else None,
                json=request.json_body,
                timeout=request.timeout_s,
            )
        except httpx.TransportError as error:
            raise HttpError(
                f"{request.method} {request.url} failed: {type(error).__name__}: {error}",
                STATUS_TRANSPORT_FAILURE,
            ) from error
        return HttpResponse(response.status_code, dict(response.headers), response.content)


class SystemClock:
    def now_utc(self) -> datetime:
        return datetime.now(UTC)

    def monotonic(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def _is_retryable(status: int) -> bool:
    return (
        status in (STATUS_TOO_MANY_REQUESTS, STATUS_TRANSPORT_FAILURE)
        or status >= RETRYABLE_STATUS_MIN_5XX
    )


_Attempt = tuple[int, HttpResponse | None, HttpError | None]


def _send_once(transport: Transport, request: HttpRequest) -> _Attempt:
    """Return (status, response, transport_error). A transport that raises HttpError with
    status 0 is treated like a status-0 response, so both are retried the same way."""
    try:
        response = transport.send(request)
    except HttpError as error:
        if error.status != STATUS_TRANSPORT_FAILURE:
            raise
        return error.status, None, error
    return response.status, response, None


def send_with_retries(
    transport: Transport,
    clock: Clock,
    request: HttpRequest,
    max_retries: int = 3,
    base_delay_s: float = 0.5,
    rng: random.Random | None = None,
) -> HttpResponse:
    """Send, retrying 429, 5xx and transport failures (status 0) with exponential backoff
    and full jitter. Any other non-2xx raises at once."""
    jitter = rng if rng is not None else random.Random()
    for attempt in range(max_retries + 1):
        status, response, transport_error = _send_once(transport, request)
        if response is not None and 200 <= status < 300:
            return response
        if not _is_retryable(status):
            raise HttpError(f"{request.method} {request.url} returned {status}", status)
        if attempt == max_retries:
            error_type = RateLimitError if status == STATUS_TOO_MANY_REQUESTS else HttpError
            raise error_type(
                f"{request.method} {request.url} returned {status} after {max_retries} retries",
                status,
            ) from transport_error
        clock.sleep(jitter.uniform(0, base_delay_s * 2**attempt))
    raise AssertionError("unreachable: the loop returns or raises")  # pragma: no cover


class RateLimiter:
    """At most `max_requests` acquisitions in any window of `per_seconds`."""

    def __init__(self, max_requests: int, per_seconds: float, clock: Clock) -> None:
        if max_requests < 1 or per_seconds <= 0:
            raise ValueError("max_requests must be at least 1 and per_seconds positive")
        self._max_requests = max_requests
        self._per_seconds = per_seconds
        self._clock = clock
        self._stamps: deque[float] = deque()

    def acquire(self) -> None:
        while True:
            now = self._clock.monotonic()
            while self._stamps and self._stamps[0] + self._per_seconds <= now:
                self._stamps.popleft()
            if len(self._stamps) < self._max_requests:
                self._stamps.append(now)
                return
            self._clock.sleep(self._stamps[0] + self._per_seconds - now)  # always > 0 here
