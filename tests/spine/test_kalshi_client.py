"""Tests for the Kalshi market mapping and the two-phase client."""

import copy
import json
from pathlib import Path

import pytest
from fakes import FakeClock, FakeTransport, ok, status
from market_fixtures import KALSHI_FIXTURE, kalshi_transport

from precedent.config import KALSHI_BASE_URL
from precedent.spine.http import HttpError
from precedent.spine.kalshi_client import KalshiClient, kalshi_market_from
from precedent.spine.market_record import MarketIngestReport

FIXTURE = KALSHI_FIXTURE["response"]


def raw_market(**overrides) -> dict:
    market = dict(FIXTURE["live"]["markets"][0])
    market.update(overrides)
    return {k: v for k, v in market.items() if v is not None}


def test_kalshi_reads_live_then_historical_and_joins_the_category(tmp_path: Path):
    report = MarketIngestReport()
    transport = kalshi_transport(KALSHI_FIXTURE)
    markets = KalshiClient(
        transport, FakeClock(), tmp_path / "kalshi.json", max_requests=10
    ).fetch_settled(report)
    assert [(m.market_id, m.resolution, m.category) for m in markets] == [
        ("kalshi:FED-26OCT-T4.25", "no", "Economics"),  # settled after the 2026-09-08 cutoff
        ("kalshi:FED-25MAR-T4.50", "yes", "Economics"),  # settled before it: /historical/markets
    ]
    assert report.skipped_unresolved == 1  # result "" is not a resolution
    urls = [r.url.removeprefix(KALSHI_BASE_URL) for r in transport.requests]
    assert urls == [
        "/historical/cutoff",
        "/markets",
        "/events/FED-26OCT",
        "/series/KXFED",
        "/historical/markets",
        "/events/FED-25MAR",
    ]
    assert report.requests == 6 and report.seen == 3


def test_a_live_market_before_the_cutoff_is_left_to_the_historical_phase(tmp_path: Path):
    report = MarketIngestReport()
    markets = KalshiClient(
        kalshi_transport(KALSHI_FIXTURE), FakeClock(), tmp_path / "s.json", max_requests=10
    ).fetch_settled(report)
    assert "kalshi:FED-26AUG-T4.25" not in [m.market_id for m in markets]


def test_a_ticker_in_both_phases_is_returned_once(tmp_path: Path):
    both = json.loads(json.dumps(KALSHI_FIXTURE))
    both["response"]["historical"]["markets"].append(dict(FIXTURE["live"]["markets"][0]))
    markets = KalshiClient(
        kalshi_transport(both), FakeClock(), tmp_path / "s.json", max_requests=10
    ).fetch_settled(MarketIngestReport())
    ids = [m.market_id for m in markets]
    assert sorted(ids) == sorted(set(ids)) and len(ids) == 2


def test_the_state_file_holds_the_documented_keys_and_caches_lookups(tmp_path: Path):
    state = tmp_path / "s.json"
    KalshiClient(
        kalshi_transport(KALSHI_FIXTURE), FakeClock(), state, max_requests=10
    ).fetch_settled(MarketIngestReport())
    saved = json.loads(state.read_text())
    assert set(saved) == {
        "cutoff",
        "phase",
        "cursor",
        "series_of_event",
        "category_of_series",
        "page_done",
    }
    assert saved["cutoff"] == "2026-09-08T00:00:00Z" and saved["phase"] == "done"
    assert saved["series_of_event"] == {"FED-26OCT": "KXFED", "FED-25MAR": "KXFED"}
    assert saved["category_of_series"] == {"KXFED": "Economics"}
    assert saved["page_done"] == []


def test_a_second_run_resumes_without_repeating_finished_work(tmp_path: Path):
    state = tmp_path / "s.json"
    first = MarketIngestReport()
    KalshiClient(
        kalshi_transport(KALSHI_FIXTURE), FakeClock(), state, max_requests=3
    ).fetch_settled(first)
    assert first.requests == 3  # cutoff, live page, one event lookup: the budget ends mid-page
    saved = json.loads(state.read_text())
    assert saved["phase"] == "live" and saved["series_of_event"] == {"FED-26OCT": "KXFED"}
    second_transport = kalshi_transport(KALSHI_FIXTURE)
    second = MarketIngestReport()
    markets = KalshiClient(
        second_transport, FakeClock(), state, max_requests=10
    ).fetch_settled(second)
    assert [m.market_id for m in markets] == ["kalshi:FED-26OCT-T4.25", "kalshi:FED-25MAR-T4.50"]
    urls = [r.url.removeprefix(KALSHI_BASE_URL) for r in second_transport.requests]
    assert "/historical/cutoff" not in urls and "/events/FED-26OCT" not in urls


def test_the_live_phase_pages_by_cursor(tmp_path: Path):
    live_one = dict(FIXTURE["live"], cursor="next-page")
    live_two = {"markets": [], "cursor": ""}
    transport = FakeTransport(
        {
            f"{KALSHI_BASE_URL}/historical/cutoff": ok(FIXTURE["cutoff"]),
            f"{KALSHI_BASE_URL}/markets": [ok(live_one), ok(live_two)],
            f"{KALSHI_BASE_URL}/historical/markets": ok({"markets": [], "cursor": ""}),
            f"{KALSHI_BASE_URL}/events/FED-26OCT": ok(FIXTURE["events"]["FED-26OCT"]),
            f"{KALSHI_BASE_URL}/series/KXFED": ok(FIXTURE["series"]["KXFED"]),
        }
    )
    KalshiClient(
        transport, FakeClock(), tmp_path / "s.json", max_requests=10, page_size=2
    ).fetch_settled(MarketIngestReport())
    live_calls = [r for r in transport.requests if r.url == f"{KALSHI_BASE_URL}/markets"]
    assert live_calls[0].params == {"status": "settled", "limit": "2"}
    assert live_calls[1].params["cursor"] == "next-page"


def test_a_404_on_an_event_or_series_gives_a_null_category(tmp_path: Path):
    transport = FakeTransport(
        {
            f"{KALSHI_BASE_URL}/historical/cutoff": ok(FIXTURE["cutoff"]),
            f"{KALSHI_BASE_URL}/markets": ok(
                {"markets": [FIXTURE["live"]["markets"][0]], "cursor": ""}
            ),
            f"{KALSHI_BASE_URL}/historical/markets": ok({"markets": [], "cursor": ""}),
            f"{KALSHI_BASE_URL}/events/FED-26OCT": status(404),
        }
    )
    markets = KalshiClient(
        transport, FakeClock(), tmp_path / "s.json", max_requests=10
    ).fetch_settled(MarketIngestReport())
    assert [m.category for m in markets] == [None]


def test_another_http_error_surfaces(tmp_path: Path):
    transport = FakeTransport([status(403)])
    with pytest.raises(HttpError):
        KalshiClient(
            transport, FakeClock(), tmp_path / "s.json", max_requests=10
        ).fetch_settled(MarketIngestReport())


def test_the_market_mapping_uses_the_full_title_and_a_composite_source_ref():
    market = kalshi_market_from(raw_market(), "Economics", MarketIngestReport())
    assert market is not None
    assert market.question == "Will the upper bound be above 4.25% after the October meeting?"
    assert market.source_ref == "FED-26OCT/FED-26OCT-T4.25"
    assert market.venue == "kalshi" and market.category == "Economics"
    assert market.resolved_utc.isoformat() == "2026-10-29T18:05:00.250000+00:00"
    assert market.created_utc.isoformat() == "2025-01-01T10:00:00.500000+00:00"
    assert market.close_utc is not None and market.close_utc.isoformat() == "2025-01-02T10:00:00+00:00"


def test_an_empty_title_falls_back_to_the_yes_subtitle():
    market = kalshi_market_from(raw_market(title=""), None, MarketIngestReport())
    assert market is not None and market.question == "Above 4.25%"


@pytest.mark.parametrize("result", ["", "scalar", None, "YES"])
def test_only_yes_and_no_results_are_resolutions(result):
    report = MarketIngestReport()
    assert kalshi_market_from(raw_market(result=result), None, report) is None
    assert report.skipped_unresolved == 1


@pytest.mark.parametrize("field", ["settlement_ts", "created_time"])
def test_a_missing_or_unparseable_time_skips_the_market(field: str):
    report = MarketIngestReport()
    assert kalshi_market_from(raw_market(**{field: None}), None, report) is None
    assert kalshi_market_from(raw_market(**{field: "later"}), None, report) is None
    assert report.skipped_missing_time == 2


def test_a_missing_close_time_is_null_and_a_market_without_a_ticker_is_malformed():
    report = MarketIngestReport()
    market = kalshi_market_from(raw_market(close_time=None), None, report)
    assert market is not None and market.close_utc is None
    assert kalshi_market_from(raw_market(ticker=None), None, report) is None
    assert kalshi_market_from(raw_market(title="", yes_sub_title=""), None, report) is None
    assert report.skipped_malformed == 2


def _three_market_fixture() -> dict:
    """One live page: an unresolved market, then three resolved ones in two events."""
    fixture = copy.deepcopy(KALSHI_FIXTURE)
    template = FIXTURE["live"]["markets"][0]
    unresolved = {**template, "ticker": "E1-U", "event_ticker": "E1", "result": ""}
    resolved = [
        {**template, "ticker": f"{event}-{n}", "event_ticker": event, "title": f"Question {n}?"}
        for n, event in enumerate(["E1", "E1", "E2"])
    ]
    fixture["response"]["live"] = {"markets": [unresolved, *resolved], "cursor": ""}
    fixture["response"]["historical"] = {"markets": [], "cursor": ""}
    for event in ("E1", "E2"):
        body = {"event": {"event_ticker": event, "series_ticker": "KXFED"}, "markets": []}
        fixture["response"]["events"][event] = body
    return fixture


def test_a_budget_stop_mid_page_keeps_the_partial_page_and_a_resume_finishes_it(tmp_path: Path):
    fixture = _three_market_fixture()
    state = tmp_path / "s.json"
    # cutoff, page, event E1, series KXFED: then market E2-2 needs an event lookup and cannot run
    first = MarketIngestReport()
    pages = list(
        KalshiClient(
            kalshi_transport(fixture), FakeClock(), state, max_requests=4
        ).iter_pages(first)
    )
    assert [[m.market_id for m in page] for page in pages] == [["kalshi:E1-0", "kalshi:E1-1"]]
    assert first.requests == 4 and first.stopped_by_budget is True
    # counted: the unresolved market and the two resolved ones; not the one that could not finish
    assert (first.seen, first.skipped_unresolved) == (3, 1)
    saved = json.loads(state.read_text())
    assert saved["phase"] == "live" and saved["cursor"] is None
    assert sorted(saved["page_done"]) == ["E1-0", "E1-1", "E1-U"]

    second = MarketIngestReport()
    resumed = KalshiClient(
        kalshi_transport(fixture), FakeClock(), state, max_requests=10
    ).fetch_settled(second)
    assert [m.market_id for m in resumed] == ["kalshi:E2-2"]
    assert (second.seen, second.skipped_unresolved, second.stopped_by_budget) == (1, 0, False)
    assert json.loads(state.read_text())["phase"] == "done"
    assert json.loads(state.read_text())["page_done"] == []


def test_a_non_dict_element_and_a_tickerless_market_are_counted_as_malformed(tmp_path: Path):
    fixture = copy.deepcopy(KALSHI_FIXTURE)
    tickerless = {k: v for k, v in FIXTURE["live"]["markets"][0].items() if k != "ticker"}
    fixture["response"]["live"] = {"markets": ["junk", tickerless], "cursor": ""}
    fixture["response"]["historical"] = {"markets": [], "cursor": ""}
    report = MarketIngestReport()
    markets = KalshiClient(
        kalshi_transport(fixture), FakeClock(), tmp_path / "s.json", max_requests=10
    ).fetch_settled(report)
    assert markets == [] and report.skipped_malformed == 2 and report.seen == 2
