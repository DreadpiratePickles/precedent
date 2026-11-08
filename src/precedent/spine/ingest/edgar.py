"""`precedent ingest edgar --ciks <file> [--max-requests N]`: 8-K items into the event store."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from precedent.config import ConfigError, data_dir
from precedent.spine.edgar_8k_events import events_from_submissions
from precedent.spine.edgar_client import EdgarClient, EdgarError, EdgarRateLimitError
from precedent.spine.event_store import EventStore
from precedent.spine.http import Clock, SystemClock, Transport

DEFAULT_MAX_REQUESTS = 50
EXIT_OK = 0
EXIT_PARTIAL_FAILURE = 1
EXIT_USAGE_OR_CONFIG = 2


@dataclass(frozen=True)
class CikEntry:
    cik: int
    symbol: str


def read_ciks(path: Path) -> list[CikEntry]:
    """Parse `cik,symbol` lines; blank lines, `#` comments and a `cik,symbol` header are skipped."""
    entries: list[CikEntry] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#") or text.lower().replace(" ", "") == "cik,symbol":
            continue
        cik_text, _, symbol = (part.strip() for part in text.partition(","))
        if not cik_text.isdigit() or not symbol:
            raise ValueError(f"{path}:{number}: expected 'cik,symbol', got {text!r}")
        entries.append(CikEntry(int(cik_text), symbol))
    return entries


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="precedent ingest edgar")
    parser.add_argument("--ciks", type=Path, required=True, help="file of `cik,symbol` lines")
    parser.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    return parser.parse_args(argv)


def run(argv: list[str], *, transport: Transport | None = None, clock: Clock | None = None) -> int:
    """Fetch at most `--max-requests` submissions and append their 8-K events.

    Prints one JSON line of counts. Exit 0 when every fetch succeeded, 1 when some failed (the
    rest are kept), 2 for a usage or configuration error. `transport` and `clock` are for tests.
    """
    try:
        args = _parse_args(argv)
    except SystemExit as exit_request:
        return int(exit_request.code or EXIT_USAGE_OR_CONFIG)
    if args.max_requests < 1:
        print("--max-requests must be at least 1", file=sys.stderr)
        return EXIT_USAGE_OR_CONFIG
    try:
        entries = read_ciks(args.ciks)
        client = EdgarClient.from_env(transport=transport, clock=clock)
    except (ConfigError, OSError, ValueError) as error:
        print(f"precedent ingest edgar: {error}", file=sys.stderr)
        return EXIT_USAGE_OR_CONFIG
    return _ingest(client, entries, args.max_requests, clock)


def _ingest(
    client: EdgarClient, entries: list[CikEntry], max_requests: int, clock: Clock | None
) -> int:
    now = (clock if clock is not None else SystemClock()).now_utc()
    store = EventStore(data_dir() / "events")
    attempted = entries[:max_requests]
    totals = {"filings_read": 0, "records": 0, "skipped_items": 0, "skipped_forms": 0}
    failures: list[dict[str, str | int]] = []
    fetched = 0
    stopped_by_rate_limit = False
    for entry in attempted:
        try:
            records, report = events_from_submissions(
                client.fetch_submissions(entry.cik), entry.symbol, ingested_utc=now
            )
        except EdgarRateLimitError as error:
            failures.append({"cik": entry.cik, "error": str(error)})
            stopped_by_rate_limit = True
            break
        except (EdgarError, ValueError) as error:
            failures.append({"cik": entry.cik, "error": str(error)})
            continue
        store.append(records)
        fetched += 1
        totals["filings_read"] += report.filings_read
        totals["records"] += report.records
        totals["skipped_items"] += report.skipped_items
        totals["skipped_forms"] += report.skipped_forms
    print(
        json.dumps(
            {
                "ciks_listed": len(entries),
                "ciks_fetched": fetched,
                "ciks_failed": failures,
                "ciks_not_attempted": len(entries) - fetched - len(failures),
                "stopped_by_rate_limit": stopped_by_rate_limit,
                **totals,
            },
            sort_keys=True,
        )
    )
    return EXIT_PARTIAL_FAILURE if failures else EXIT_OK
