"""The shared body of `precedent ingest polymarket|kalshi`: page, filter, type, store, report."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Generator, Iterator
from dataclasses import asdict
from pathlib import Path

from precedent.config import ConfigError, data_dir
from precedent.spine.http import Clock, HttpError, HttpxTransport, SystemClock, Transport
from precedent.spine.jsonl_io import AppendReport
from precedent.spine.market_record import MarketIngestReport, ResolvedMarket, type_markets
from precedent.spine.market_store import MarketConflictError, MarketStore
from precedent.spine.typers.base import Typer
from precedent.spine.typers.select import typer_from_env

DEFAULT_MAX_REQUESTS = 50
EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_USAGE = 2

PageSource = Callable[
    [Transport, Clock, Path, int, MarketIngestReport], Iterator[list[ResolvedMarket]]
]


def _parser(venue: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"precedent ingest {venue}")
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    parser.add_argument("--category", action="append", default=[], help="repeatable")
    parser.add_argument("--max-calls", type=int, default=None, help="typer call budget")
    parser.add_argument(
        "--dry-run", action="store_true", help="plan only: no request is sent, nothing is written"
    )
    return parser


def _only_categories(
    markets: list[ResolvedMarket], wanted: set[str], report: MarketIngestReport
) -> list[ResolvedMarket]:
    if not wanted:
        return markets
    kept = [m for m in markets if m.category is not None and m.category.lower() in wanted]
    report.skipped_category += len(markets) - len(kept)
    return kept


TRANSIENT_STOP_MESSAGE = (
    "ingest stopped: {n} market(s) could not be typed (transient: budget or transport); "
    "nothing past this page was fetched; re-run to continue"
)


def _drop_already_stored(
    markets: list[ResolvedMarket], stored_ids: set[str], report: MarketIngestReport
) -> list[ResolvedMarket]:
    """Leave out markets the store already holds for this typer version; count them."""
    fresh = [m for m in markets if m.market_id not in stored_ids]
    report.already_stored += len(markets) - len(fresh)
    return fresh


def _print_plan(venue: str, args: argparse.Namespace, typer: Typer) -> int:
    """`--dry-run`: say what would happen, with an all-zero report. No request, no write."""
    categories = ", ".join(args.category) if args.category else "all"
    summary = {
        "venue": venue,
        "dry_run": True,
        "planned": (
            f"would fetch up to {args.max_requests} page(s) of resolved {venue} markets "
            f"(categories: {categories}), type each question with "
            f"{getattr(typer, 'version', 'the selected typer')}, and append them to the store; "
            "no request was sent and nothing was written"
        ),
        "report": asdict(MarketIngestReport()),
        "append": asdict(AppendReport(added=0, identical_skipped=0)),
        "error": None,
    }
    print(json.dumps(summary))
    return EXIT_OK


def run_market_ingest(
    venue: str,
    argv: list[str],
    pages_from: PageSource,
    transport: Transport | None = None,
    clock: Clock | None = None,
    typer: Typer | None = None,
) -> int:
    """Run one venue's ingest. Exit 0 ok, 1 partial failure (kept pages stay), 2 usage or config.

    A transient typing stop (`stopped_by_transient`) is a partial failure too: exit 1 with one
    stderr line. A paging budget stop (`stopped_by_budget`) stays exit 0.
    """
    try:
        args = _parser(venue).parse_args(argv)
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    if args.max_requests < 1:
        print("--max-requests must be at least 1", file=sys.stderr)
        return EXIT_USAGE
    try:
        typer = typer or typer_from_env(max_calls=args.max_calls, dry_run=args.dry_run)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return EXIT_USAGE

    if args.dry_run:
        return _print_plan(venue, args, typer)
    root = data_dir()
    store = MarketStore(root / "markets")
    report = MarketIngestReport()
    pages = pages_from(
        transport or HttpxTransport(),
        clock or SystemClock(),
        root / "state" / f"{venue}.json",
        args.max_requests,
        report,
    )
    wanted = {name.lower() for name in args.category}
    stored_ids = {r.market_id for r in store.read(venue, typer.version)}
    added = skipped = 0
    error: Exception | None = None
    try:
        for page in pages:
            transient_before = report.not_stored_transient
            wanted_markets = _only_categories(page, wanted, report)
            candidates = _drop_already_stored(wanted_markets, stored_ids, report)
            records = type_markets(candidates, typer, report)
            appended = store.append(records)
            added, skipped = added + appended.added, skipped + appended.identical_skipped
            if report.not_stored_transient > transient_before:
                # Stop without saving the cursor past this page, so a later run fetches it again
                # and types the markets that could not be typed now.
                report.stopped_by_transient = True
                if isinstance(pages, Generator):
                    pages.close()
                break
    except (HttpError, ValueError, MarketConflictError) as failure:
        error = failure
    summary = {
        "venue": venue,
        "report": asdict(report),
        "append": asdict(AppendReport(added=added, identical_skipped=skipped)),
        "error": None if error is None else f"{type(error).__name__}: {error}",
    }
    print(json.dumps(summary))
    if error is not None:
        print(f"ingest stopped: {type(error).__name__}: {error}", file=sys.stderr)
        return EXIT_PARTIAL_FAILURE
    if report.stopped_by_transient:
        print(TRANSIENT_STOP_MESSAGE.format(n=report.not_stored_transient), file=sys.stderr)
        return EXIT_PARTIAL_FAILURE
    return EXIT_OK
