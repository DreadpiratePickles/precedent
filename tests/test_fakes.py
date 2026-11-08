import pandas as pd
import pytest
from fakes import (
    ExplodingTransport,
    FakeClock,
    FakeTransport,
    assert_truncation_invariant,
    ok_bytes,
    ok_text,
    status,
)

from precedent.spine.http import HttpRequest


def _events() -> pd.DataFrame:
    sessions = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    return pd.DataFrame(
        {"effective_session": sessions.repeat(2), "value": [1, 2, 3, 4, 5, 6, 7, 8]}
    ).reset_index(drop=True)


def _count_per_session(events: pd.DataFrame) -> pd.DataFrame:
    counts = events.groupby("effective_session").size().rename("count")
    return counts.reset_index()


def test_a_trivially_invariant_builder_passes():
    events = _events()
    sessions = sorted(events["effective_session"].unique())
    assert_truncation_invariant(_count_per_session, events, sessions, [0, 1, 2])


def test_a_builder_that_peeks_ahead_fails():
    def peeking(events: pd.DataFrame) -> pd.DataFrame:
        out = events[["effective_session"]].drop_duplicates().reset_index(drop=True)
        out["total"] = len(events)
        return out

    events = _events()
    sessions = sorted(events["effective_session"].unique())
    with pytest.raises(AssertionError):
        assert_truncation_invariant(peeking, events, sessions, [0, 1, 2])


def test_mapping_transport_keys_by_url_without_query():
    transport = FakeTransport({"https://example.test/a": [status(500), ok_text("fine")]})
    request = HttpRequest("GET", "https://example.test/a", {}, params={"q": "1"})
    assert transport.send(request).status == 500
    assert transport.send(request).text() == "fine"
    assert len(transport.requests) == 2


def test_ok_bytes_and_clock_record_sleep():
    assert ok_bytes(b"xyz").body == b"xyz"
    clock = FakeClock()
    clock.sleep(1.5)
    assert clock.slept_seconds == 1.5
    assert clock.now_utc().tzinfo is not None


def test_exploding_transport_fails_the_test_when_called():
    with pytest.raises(AssertionError):
        ExplodingTransport().send(HttpRequest("GET", "https://example.test/", {}))
