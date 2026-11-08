"""The shared body of `precedent ingest polymarket|kalshi`: page, filter, type, store, report."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Iterator
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
    parser.add_argument("--dry-run", action="store_true", help="model typers: open no socket")
    return parser


def _only_categories(
    markets: list[ResolvedMarket], wanted: set[str], report: MarketIngestReport
) -> list[ResolvedMarket]:
    if not wanted:
        return markets
    kept = [m for m in markets if m.category is not None and m.category.lower() in wanted]
    report.skipped_category += len(markets) - len(kept)
    return kept


def run_market_ingest(
    venue: str,
    argv: list[str],
    pages_from: PageSource,
    transport: Transport | None = None,
    clock: Clock | None = None,
    typer: Typer | None = None,
) -> int:
    """Run one venue's ingest. Exit 0 ok, 1 partial failure (kept pages stay), 2 usage or config."""
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
    added = skipped = 0
    error: Exception | None = None
    try:
        for page in pages:
            records = type_markets(_only_categories(page, wanted, report), typer, report)
            appended = store.append(records)
            added, skipped = added + appended.added, skipped + appended.identical_skipped
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
    return EXIT_OK
