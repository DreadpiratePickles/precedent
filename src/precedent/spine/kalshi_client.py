"""Kalshi: map a settled market to a ResolvedMarket, and read the live and historical phases."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

from precedent import config
from precedent.spine.http import (
    Clock,
    HttpError,
    HttpRequest,
    RateLimiter,
    Transport,
    send_with_retries,
)
from precedent.spine.jsonl_io import read_json_state, write_json_state
from precedent.spine.market_record import MarketIngestReport, ResolvedMarket, parse_utc_timestamp

RATE_LIMIT_REQUESTS = 20
RATE_LIMIT_WINDOW_S = 1.0
DEFAULT_PAGE_SIZE = 200
STATUS_NOT_FOUND = 404
PHASE_LIVE = "live"
PHASE_HISTORICAL = "historical"
PHASE_DONE = "done"
_RESULTS = ("yes", "no")


class _BudgetSpent(Exception):
    """The request budget ran out part-way through a page."""


def kalshi_market_from(
    raw: object, category: str | None, report: MarketIngestReport
) -> ResolvedMarket | None:
    """A market that resolved yes or no, or None with exactly one skip counter incremented."""
    if not isinstance(raw, dict):
        report.skipped_malformed += 1
        return None
    ticker, event_ticker = raw.get("ticker"), raw.get("event_ticker")
    title, subtitle = raw.get("title"), raw.get("yes_sub_title")
    question = title if isinstance(title, str) and title else subtitle
    if not (
        isinstance(ticker, str)
        and ticker
        and isinstance(event_ticker, str)
        and event_ticker
        and isinstance(question, str)
        and question
    ):
        report.skipped_malformed += 1
        return None
    result = raw.get("result")
    if result not in _RESULTS:
        report.skipped_unresolved += 1
        return None
    created = parse_utc_timestamp(raw.get("created_time"))
    resolved = parse_utc_timestamp(raw.get("settlement_ts"))
    close_raw = raw.get("close_time")
    close = parse_utc_timestamp(close_raw)
    if created is None or resolved is None or (close_raw is not None and close is None):
        report.skipped_missing_time += 1
        return None
    return ResolvedMarket(
        market_id=f"kalshi:{ticker}",
        venue="kalshi",
        question=question,
        created_utc=created,
        close_utc=close,
        resolved_utc=resolved,
        resolution=result,
        category=category,
        source_ref=f"{event_ticker}/{ticker}",
    )


class KalshiClient:
    """Reads settled markets: newer than the historical cutoff from `/markets`, older from
    `/historical/markets`; each market's category is joined through its event's series.

    State is `{"cutoff", "phase", "cursor", "series_of_event", "category_of_series",
    "page_done"}`; phase is "live", "historical" or "done", and `page_done` lists the tickers of
    the current page already handed over, so a page cut short by the request budget is finished,
    not recounted, on the next run. It is saved after the caller has taken each page.
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

    def fetch_settled(self, report: MarketIngestReport) -> list[ResolvedMarket]:
        """Every resolved market of the pages this run fetches."""
        return [market for page in self.iter_pages(report) for market in page]

    def iter_pages(self, report: MarketIngestReport) -> Iterator[list[ResolvedMarket]]:
        """One list of resolved markets per fetched page; state is saved when the caller returns."""
        state = self._load_state()
        taken: set[str] = set()
        try:
            if state["cutoff"] is None and state["phase"] != PHASE_DONE:
                state["cutoff"] = self._get_cutoff(report)
                write_json_state(self._state_path, state)
            while state["phase"] != PHASE_DONE:
                body = self._get_page(state, report)
                page, handled, complete = self._resolve_page(body["markets"], state, taken, report)
                yield page
                state["page_done"] = [*state["page_done"], *handled]
                if not complete:
                    report.stopped_by_budget = True
                    write_json_state(self._state_path, state)
                    return
                state["cursor"] = body.get("cursor") or None
                state["page_done"] = []
                if state["cursor"] is None:
                    live = state["phase"] == PHASE_LIVE
                    state["phase"] = PHASE_HISTORICAL if live else PHASE_DONE
                write_json_state(self._state_path, state)
        except _BudgetSpent:
            report.stopped_by_budget = True
            write_json_state(self._state_path, state)  # lookups cached so far are progress

    def _load_state(self) -> dict[str, Any]:
        saved = read_json_state(self._state_path)
        return {
            "cutoff": saved.get("cutoff"),
            "phase": saved.get("phase", PHASE_LIVE),
            "cursor": saved.get("cursor"),
            "series_of_event": dict(saved.get("series_of_event", {})),
            "category_of_series": dict(saved.get("category_of_series", {})),
            "page_done": list(saved.get("page_done", [])),
        }

    def _resolve_page(
        self,
        raw_markets: list[Any],
        state: dict[str, Any],
        taken: set[str],
        report: MarketIngestReport,
    ) -> tuple[list[ResolvedMarket], list[str], bool]:
        """(resolved markets, tickers handled, page complete).

        An item is counted in `report` only once it is handled: resolved, or bucketed as a skip.
        When the budget runs out part-way the markets resolved so far are returned with
        `complete` False, and the cursor stays put so the next run finishes the page. (A
        malformed item with no ticker cannot be remembered, so a resume counts it again.)
        """
        cutoff = parse_utc_timestamp(state["cutoff"])
        already = set(state["page_done"])
        page: list[ResolvedMarket] = []
        handled: list[str] = []
        for raw in raw_markets:
            ticker = raw.get("ticker") if isinstance(raw, dict) else None
            if isinstance(ticker, str) and (ticker in already or ticker in taken):
                continue
            is_live = state["phase"] == PHASE_LIVE
            if is_live and isinstance(raw, dict) and _settled_before(raw, cutoff):
                continue  # the historical phase provides it
            item = MarketIngestReport()
            market = kalshi_market_from(raw, None, item)
            if market is not None:
                try:
                    category = self._category_of(raw["event_ticker"], state, report)
                except _BudgetSpent:
                    return page, handled, False
                market = market.model_copy(update={"category": category})
                page.append(market)
            report.seen += 1
            report.absorb(item)
            if isinstance(ticker, str):
                taken.add(ticker)
                handled.append(ticker)
        return page, handled, True

    def _category_of(
        self, event_ticker: str, state: dict[str, Any], report: MarketIngestReport
    ) -> str | None:
        series_of_event: dict[str, str | None] = state["series_of_event"]
        if event_ticker not in series_of_event:
            body = self._get_optional(f"/events/{event_ticker}", report)
            event = body.get("event") if body is not None else None
            series = event.get("series_ticker") if isinstance(event, dict) else None
            series_of_event[event_ticker] = series if isinstance(series, str) and series else None
        series_ticker = series_of_event[event_ticker]
        if series_ticker is None:
            return None
        category_of_series: dict[str, str | None] = state["category_of_series"]
        if series_ticker not in category_of_series:
            body = self._get_optional(f"/series/{series_ticker}", report)
            series_body = body.get("series") if body is not None else None
            category = series_body.get("category") if isinstance(series_body, dict) else None
            category_of_series[series_ticker] = (
                category if isinstance(category, str) and category else None
            )
        return category_of_series[series_ticker]

    def _get_cutoff(self, report: MarketIngestReport) -> str:
        body = self._get("/historical/cutoff", None, report)
        cutoff = body.get("market_settled_ts") if isinstance(body, dict) else None
        if parse_utc_timestamp(cutoff) is None:
            raise ValueError("Kalshi /historical/cutoff has no usable 'market_settled_ts'")
        return cutoff

    def _get_page(self, state: dict[str, Any], report: MarketIngestReport) -> dict[str, Any]:
        params = {"limit": str(self._page_size)}
        if state["phase"] == PHASE_LIVE:
            path = "/markets"
            params = {"status": "settled", **params}
        else:
            path = "/historical/markets"
        if state["cursor"]:
            params["cursor"] = state["cursor"]
        body = self._get(path, params, report)
        if not isinstance(body, dict) or not isinstance(body.get("markets"), list):
            raise ValueError(f"Kalshi {path} response has no 'markets' list")
        return body

    def _get_optional(self, path: str, report: MarketIngestReport) -> dict[str, Any] | None:
        """A 404 means the event or series is gone: the category is unknown, not an error."""
        try:
            body = self._get(path, None, report)
        except HttpError as error:
            if error.status == STATUS_NOT_FOUND:
                return None
            raise
        return body if isinstance(body, dict) else None

    def _get(self, path: str, params: dict[str, str] | None, report: MarketIngestReport) -> Any:
        if report.requests >= self._max_requests:
            raise _BudgetSpent
        request = HttpRequest(
            "GET",
            f"{config.KALSHI_BASE_URL}{path}",
            headers={"Accept": "application/json"},
            params=params,
        )
        self._limiter.acquire()
        report.requests += 1
        return send_with_retries(self._transport, self._clock, request).json()


def _settled_before(raw: dict, cutoff: datetime | None) -> bool:
    settled = parse_utc_timestamp(raw.get("settlement_ts"))
    return settled is not None and cutoff is not None and settled < cutoff
