"""Polymarket Gamma: map a closed market to a ResolvedMarket, and page the closed markets."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from precedent import config
from precedent.spine.http import (
    Clock,
    HttpRequest,
    RateLimiter,
    Transport,
    send_with_retries,
)
from precedent.spine.jsonl_io import read_json_state, write_json_state
from precedent.spine.market_record import MarketIngestReport, ResolvedMarket, parse_utc_timestamp

RATE_LIMIT_REQUESTS = 300
RATE_LIMIT_WINDOW_S = 10.0
DEFAULT_PAGE_SIZE = 100  # the endpoint's own cap
RESOLVED_STATUS = "resolved"
_YES_NO = ("yes", "no")


def _decode_list(raw_value: object) -> list[Any] | None:
    """`outcomes` and `outcomePrices` arrive as JSON-encoded strings; a real list is accepted."""
    if isinstance(raw_value, list):
        return raw_value
    if not isinstance(raw_value, str):
        return None
    try:
        decoded = json.loads(raw_value)
    except json.JSONDecodeError:
        return None
    return decoded if isinstance(decoded, list) else None


def _winning_label(labels: list[str], prices_raw: object) -> str | None:
    """The outcome priced exactly 1.0 when the other is exactly 0.0, else None."""
    prices = _decode_list(prices_raw)
    if prices is None or len(prices) != len(labels):
        return None
    try:
        values = [float(price) for price in prices]
    except (TypeError, ValueError):
        return None
    if sorted(values) != [0.0, 1.0]:
        return None
    return labels[values.index(1.0)]


def _labels_if_yes_no(raw_outcomes: object) -> list[str] | None:
    outcomes = _decode_list(raw_outcomes)
    if outcomes is None or len(outcomes) != 2:
        return None
    labels = [str(outcome).strip().lower() for outcome in outcomes]
    return labels if sorted(labels) == sorted(_YES_NO) else None


def polymarket_market_from(raw: object, report: MarketIngestReport) -> ResolvedMarket | None:
    """A resolved Yes/No market, or None with exactly one skip counter incremented.

    A present `umaResolutionStatus` must be "resolved"; an old market with no such key counts as
    resolved on its prices alone (and is counted in `resolved_by_prices_only`).
    """
    if not isinstance(raw, dict):
        report.skipped_malformed += 1
        return None
    condition_id, question = raw.get("conditionId"), raw.get("question")
    if not (isinstance(condition_id, str) and condition_id and isinstance(question, str)):
        report.skipped_malformed += 1
        return None
    has_status = "umaResolutionStatus" in raw
    if raw.get("closed") is not True or (
        has_status and raw["umaResolutionStatus"] != RESOLVED_STATUS
    ):
        report.skipped_unresolved += 1
        return None
    labels = _labels_if_yes_no(raw.get("outcomes"))
    if labels is None:
        report.skipped_not_binary += 1
        return None
    winner = _winning_label(labels, raw.get("outcomePrices"))
    if winner is None:
        report.ambiguous += 1
        return None
    created = parse_utc_timestamp(raw.get("createdAt"))
    resolved = parse_utc_timestamp(raw.get("closedTime"))
    end_date = raw.get("endDate")
    close = parse_utc_timestamp(end_date)
    if created is None or resolved is None or (end_date is not None and close is None):
        report.skipped_missing_time += 1
        return None
    if not has_status:
        report.resolved_by_prices_only += 1
    category = raw.get("category")
    slug = raw.get("slug")
    return ResolvedMarket(
        market_id=f"polymarket:{condition_id}",
        venue="polymarket",
        question=question,
        created_utc=created,
        close_utc=close,
        resolved_utc=resolved,
        resolution=winner,
        category=category if isinstance(category, str) and category else None,
        source_ref=slug if isinstance(slug, str) and slug else condition_id,
    )


class PolymarketClient:
    """Pages closed markets newest first through the keyset endpoint, resumable from a state file.

    State is `{"cursor": str | None, "exhausted": bool}`, saved after the caller takes each page.
    """

    def __init__(
        self,
        transport: Transport,
        clock: Clock,
        state_path: Path,
        max_requests: int,
        page_size: int = DEFAULT_PAGE_SIZE,
    ) -> None:
        self._transport = transport
        self._clock = clock
        self._state_path = state_path
        self._max_requests = max_requests
        self._page_size = page_size
        self._limiter = RateLimiter(RATE_LIMIT_REQUESTS, RATE_LIMIT_WINDOW_S, clock)

    def fetch_resolved(self, report: MarketIngestReport) -> list[ResolvedMarket]:
        """Every resolved market of the pages this run fetches."""
        return [market for page in self.iter_pages(report) for market in page]

    def iter_pages(self, report: MarketIngestReport) -> Iterator[list[ResolvedMarket]]:
        """One list of resolved markets per fetched page; state is saved when the caller returns."""
        state = read_json_state(self._state_path)
        cursor: str | None = state.get("cursor")
        while not state.get("exhausted", False) and report.requests < self._max_requests:
            body = self._get_page(cursor, report)
            raw_markets = body["markets"]
            next_cursor = body.get("next_cursor") or None
            report.seen += len(raw_markets)
            maybe = (polymarket_market_from(raw, report) for raw in raw_markets)
            yield [market for market in maybe if market is not None]
            exhausted = next_cursor is None or not raw_markets
            cursor = None if exhausted else next_cursor
            state = {"cursor": cursor, "exhausted": exhausted}
            write_json_state(self._state_path, state)
        if not state.get("exhausted", False):
            report.stopped_by_budget = True

    def _get_page(self, cursor: str | None, report: MarketIngestReport) -> dict[str, Any]:
        params = {
            "closed": "true",
            "limit": str(self._page_size),
            "order": "closedTime",
            "ascending": "false",
        }
        if cursor:
            params["after_cursor"] = cursor
        request = HttpRequest(
            "GET",
            f"{config.POLYMARKET_GAMMA_BASE_URL}/markets/keyset",
            headers={"Accept": "application/json"},
            params=params,
        )
        self._limiter.acquire()
        report.requests += 1
        body = send_with_retries(self._transport, self._clock, request).json()
        if not isinstance(body, dict) or not isinstance(body.get("markets"), list):
            raise ValueError("Polymarket keyset response has no 'markets' list")
        return body
