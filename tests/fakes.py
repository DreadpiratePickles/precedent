"""Test doubles for the HTTP seam, plus the truncation-invariance helper."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
from pandas.testing import assert_frame_equal

from precedent.spine.http import HttpRequest, HttpResponse


def ok(json_value: Any) -> HttpResponse:
    return HttpResponse(200, {"content-type": "application/json"}, json.dumps(json_value).encode())


def ok_bytes(body: bytes) -> HttpResponse:
    return HttpResponse(200, {"content-type": "application/octet-stream"}, body)


def ok_text(text: str) -> HttpResponse:
    return HttpResponse(200, {"content-type": "text/plain"}, text.encode("utf-8"))


def status(code: int) -> HttpResponse:
    return HttpResponse(code, {}, b"")


def _url_without_query(url: str) -> str:
    return url.split("?", 1)[0]


class FakeTransport:
    """Replays canned responses. A sequence is consumed in order; a mapping is keyed by URL
    without the query string, each value one response (repeated) or a sequence (consumed)."""

    def __init__(
        self,
        responses: Sequence[HttpResponse] | Mapping[str, HttpResponse | Sequence[HttpResponse]],
    ) -> None:
        self.requests: list[HttpRequest] = []
        self._queue: list[HttpResponse] | None = None
        self._by_url: dict[str, HttpResponse | list[HttpResponse]] = {}
        if isinstance(responses, Mapping):
            self._by_url = {
                url: value if isinstance(value, HttpResponse) else list(value)
                for url, value in responses.items()
            }
        else:
            self._queue = list(responses)

    def send(self, request: HttpRequest) -> HttpResponse:
        self.requests.append(request)
        if self._queue is not None:
            if not self._queue:
                raise AssertionError(f"FakeTransport has no response left for {request.url}")
            return self._queue.pop(0)
        key = _url_without_query(request.url)
        if key not in self._by_url:
            raise AssertionError(f"FakeTransport has no response for {key}")
        entry = self._by_url[key]
        if isinstance(entry, HttpResponse):
            return entry
        if not entry:
            raise AssertionError(f"FakeTransport has no response left for {key}")
        return entry.pop(0)


class ExplodingTransport:
    """Fails the test if anything tries to use the network."""

    def send(self, request: HttpRequest) -> HttpResponse:
        raise AssertionError(f"unexpected network call: {request.method} {request.url}")


class FakeClock:
    def __init__(self, start_utc: datetime | None = None) -> None:
        self._now = start_utc if start_utc is not None else datetime(2026, 1, 1, tzinfo=UTC)
        self._monotonic = 0.0
        self.slept_seconds = 0.0

    def now_utc(self) -> datetime:
        return self._now

    def monotonic(self) -> float:
        return self._monotonic

    def sleep(self, seconds: float) -> None:
        self.slept_seconds += seconds
        self._monotonic += seconds
        self._now += timedelta(seconds=seconds)


def assert_truncation_invariant(
    builder: Callable[[pd.DataFrame], pd.DataFrame],
    events: pd.DataFrame,
    sessions: Sequence[Any],
    cut_positions: Sequence[int],
) -> None:
    """For each cut, building from the full events and restricting to sessions up to the cut
    must equal building from events effective by the cut, restricted the same way. The builder's
    output carries an `effective_session` column, or has the session as its first index level."""
    full = builder(events)
    for position in cut_positions:
        cut = sessions[position]
        truncated = builder(events[events["effective_session"] <= cut])
        assert_frame_equal(_up_to(full, cut), _up_to(truncated, cut))


def _up_to(frame: pd.DataFrame, cut: Any) -> pd.DataFrame:
    if "effective_session" in frame.columns:
        return frame[frame["effective_session"] <= cut].reset_index(drop=True)
    return frame[frame.index.get_level_values(0) <= cut]
