"""Tests for the Polymarket market mapping and the keyset-paged client."""

import json
from pathlib import Path

import pytest
from fakes import FakeClock, FakeTransport, ok, status
from market_fixtures import POLYMARKET_FIXTURE

from precedent.config import POLYMARKET_GAMMA_BASE_URL
from precedent.spine.http import HttpError
from precedent.spine.market_record import MarketIngestReport
from precedent.spine.polymarket_client import PolymarketClient, polymarket_market_from

PM = POLYMARKET_FIXTURE
KEYSET_URL = f"{POLYMARKET_GAMMA_BASE_URL}/markets/keyset"


def raw_market(condition_id: str = "0x1", **overrides) -> dict:
    market = dict(PM["response"][0], conditionId=condition_id)
    market.update(overrides)
    return {k: v for k, v in market.items() if v is not None}


def page(count: int, next_cursor: str | None = None, start: int = 0) -> dict:
    body: dict = {"markets": [raw_market(f"0x{start + i}") for i in range(count)]}
    if next_cursor is not None:
        body["next_cursor"] = next_cursor
    return body


def test_winner_is_derived_from_outcome_prices():
    report = MarketIngestReport()
    kept = [m for m in (polymarket_market_from(raw, report) for raw in PM["response"]) if m]
    assert [(m.market_id, m.resolution) for m in kept] == [
        ("polymarket:0xaaa", "yes"),
        ("polymarket:0xbbb", "no"),
    ]
    assert (report.ambiguous, report.skipped_not_binary, report.skipped_unresolved) == (1, 1, 1)
    assert report.resolved_by_prices_only == 0


def test_fields_are_mapped_and_times_are_utc():
    market = polymarket_market_from(PM["response"][0], MarketIngestReport())
    assert market is not None
    assert market.question == "Will the Fed cut rates in January 2024?"
    assert market.venue == "polymarket" and market.category == "Politics"
    assert market.source_ref == "fed-cut-jan-2024"
    assert market.resolved_utc.isoformat() == "2024-01-03T14:15:00+00:00"
    assert market.created_utc.isoformat() == "2024-01-01T09:00:00.123000+00:00"
    assert market.close_utc is not None and market.close_utc.isoformat() == "2024-01-03T00:00:00+00:00"


def test_empty_category_is_null_and_a_missing_slug_falls_back_to_the_condition_id():
    market = polymarket_market_from(
        raw_market("0xzzz", category="", slug=None), MarketIngestReport()
    )
    assert market is not None and market.category is None and market.source_ref == "0xzzz"


def test_outcomes_and_prices_may_already_be_decoded_lists():
    market = polymarket_market_from(
        raw_market(outcomes=["Yes", "No"], outcomePrices=["0", "1"]), MarketIngestReport()
    )
    assert market is not None and market.resolution == "no"


def test_a_non_dict_element_in_a_page_is_counted_as_malformed(tmp_path: Path):
    body = {"markets": ["junk", raw_market("0x9")]}
    report = MarketIngestReport()
    markets = PolymarketClient(
        FakeTransport([ok(body)]), FakeClock(), tmp_path / "s.json", max_requests=1
    ).fetch_resolved(report)
    assert [m.market_id for m in markets] == ["polymarket:0x9"]
    assert report.skipped_malformed == 1 and report.seen == 2


def test_stopping_at_the_request_budget_is_flagged(tmp_path: Path):
    report = MarketIngestReport()
    PolymarketClient(
        FakeTransport([ok(page(1, "c1"))]), FakeClock(), tmp_path / "s.json", max_requests=1
    ).fetch_resolved(report)
    assert report.stopped_by_budget is True
    done = MarketIngestReport()
    PolymarketClient(
        FakeTransport([ok(page(1))]), FakeClock(), tmp_path / "t.json", max_requests=1
    ).fetch_resolved(done)
    assert done.stopped_by_budget is False


def test_the_winner_follows_the_label_not_the_position():
    market = polymarket_market_from(
        raw_market(outcomes='["no", " YES "]', outcomePrices='["0", "1"]'), MarketIngestReport()
    )
    assert market is not None and market.resolution == "yes"


def test_a_market_with_no_status_key_counts_by_prices_alone_and_is_counted():
    report = MarketIngestReport()
    market = polymarket_market_from(raw_market(umaResolutionStatus=None), report)
    assert market is not None and market.resolution == "yes"
    assert report.resolved_by_prices_only == 1


def test_a_present_status_other_than_resolved_is_unresolved_whatever_the_prices():
    report = MarketIngestReport()
    assert polymarket_market_from(raw_market(umaResolutionStatus="disputed"), report) is None
    assert polymarket_market_from(raw_market(closed=False), report) is None
    assert report.skipped_unresolved == 2


@pytest.mark.parametrize("prices", ['["0", "0"]', '["1", "1"]', '["0.99", "0.01"]', "oops", '["1"]'])
def test_prices_that_are_not_exactly_one_and_zero_are_ambiguous(prices: str):
    report = MarketIngestReport()
    assert polymarket_market_from(raw_market(outcomePrices=prices), report) is None
    assert report.ambiguous == 1


def test_a_status_less_market_with_zero_zero_prices_is_ambiguous():
    report = MarketIngestReport()
    raw = raw_market(umaResolutionStatus=None, outcomePrices='["0", "0"]')
    assert polymarket_market_from(raw, report) is None
    assert report.ambiguous == 1 and report.resolved_by_prices_only == 0


@pytest.mark.parametrize("outcomes", ['["Yes", "No", "Maybe"]', '["Up", "Down"]', "not json", None])
def test_only_two_outcome_yes_no_markets_are_kept(outcomes):
    report = MarketIngestReport()
    assert polymarket_market_from(raw_market(outcomes=outcomes), report) is None
    assert report.skipped_not_binary == 1


@pytest.mark.parametrize("field", ["createdAt", "closedTime"])
def test_a_missing_or_unparseable_time_skips_the_market(field: str):
    report = MarketIngestReport()
    assert polymarket_market_from(raw_market(**{field: None}), report) is None
    assert polymarket_market_from(raw_market(**{field: "soon"}), report) is None
    assert report.skipped_missing_time == 2


def test_a_missing_end_date_leaves_close_null_but_a_garbled_one_skips():
    report = MarketIngestReport()
    market = polymarket_market_from(raw_market(endDate=None), report)
    assert market is not None and market.close_utc is None
    assert polymarket_market_from(raw_market(endDate="garbled"), report) is None
    assert report.skipped_missing_time == 1


def test_a_market_without_a_question_or_condition_id_is_malformed():
    report = MarketIngestReport()
    assert polymarket_market_from(raw_market(question=None), report) is None
    assert polymarket_market_from(raw_market(conditionId=None), report) is None
    assert report.skipped_malformed == 2


def test_pagination_follows_the_cursor_stops_at_the_budget_and_resumes(tmp_path: Path):
    state = tmp_path / "polymarket.json"
    first = MarketIngestReport()
    transport = FakeTransport([ok(page(2, "c1")), ok(page(2, "c2", start=2))])
    PolymarketClient(transport, FakeClock(), state, max_requests=2, page_size=2).fetch_resolved(
        first
    )
    assert first.requests == 2 and first.seen == 4
    assert json.loads(state.read_text()) == {"cursor": "c2", "exhausted": False}
    first_call, second_call = transport.requests
    assert first_call.url == KEYSET_URL and "after_cursor" not in first_call.params
    assert first_call.params["closed"] == "true" and first_call.params["limit"] == "2"
    assert first_call.params["order"] == "closedTime" and first_call.params["ascending"] == "false"
    assert second_call.params["after_cursor"] == "c1"

    resumed = FakeTransport([ok(page(1, start=4))])
    second = MarketIngestReport()
    markets = PolymarketClient(
        resumed, FakeClock(), state, max_requests=5, page_size=2
    ).fetch_resolved(second)
    assert second.requests == 1 and len(markets) == 1
    assert resumed.requests[0].params["after_cursor"] == "c2"
    assert json.loads(state.read_text())["exhausted"] is True


def test_an_empty_next_cursor_or_empty_page_ends_the_feed(tmp_path: Path):
    state = tmp_path / "state.json"
    transport = FakeTransport([ok(page(2, "")), ok({"markets": []})])
    report = MarketIngestReport()
    markets = PolymarketClient(
        transport, FakeClock(), state, max_requests=5, page_size=2
    ).fetch_resolved(report)
    assert report.requests == 1 and len(markets) == 2
    assert json.loads(state.read_text())["exhausted"] is True
    again = FakeTransport([])
    PolymarketClient(again, FakeClock(), state, max_requests=5).fetch_resolved(report)
    assert again.requests == []


def test_a_5xx_is_retried_and_other_errors_surface(tmp_path: Path):
    state = tmp_path / "state.json"
    transport = FakeTransport([status(503), ok(page(1))])
    clock = FakeClock()
    report = MarketIngestReport()
    PolymarketClient(transport, clock, state, max_requests=5, page_size=2).fetch_resolved(report)
    assert len(transport.requests) == 2 and clock.slept_seconds > 0
    failing = FakeTransport([status(404)])
    with pytest.raises(HttpError):
        PolymarketClient(
            failing, FakeClock(), tmp_path / "other.json", max_requests=5
        ).fetch_resolved(MarketIngestReport())


def test_state_is_saved_only_after_the_caller_has_taken_each_page(tmp_path: Path):
    state = tmp_path / "state.json"
    client = PolymarketClient(
        FakeTransport([ok(page(2, "c1")), ok(page(2, "c2"))]),
        FakeClock(), state, max_requests=5, page_size=2,
    )
    pages = client.iter_pages(MarketIngestReport())
    next(pages)
    assert not state.exists()
    next(pages)
    assert json.loads(state.read_text())["cursor"] == "c1"


def test_a_malformed_response_is_a_value_error(tmp_path: Path):
    client = PolymarketClient(
        FakeTransport([ok({"unexpected": 1})]), FakeClock(), tmp_path / "s.json", max_requests=1
    )
    with pytest.raises(ValueError, match="markets"):
        client.fetch_resolved(MarketIngestReport())
