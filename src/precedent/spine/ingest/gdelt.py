"""`precedent ingest gdelt`: fetch the window's export files, type their rows, append the events."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import date

import pandas as pd

from precedent.config import data_dir, references_dir
from precedent.spine.event_store import EventStore
from precedent.spine.gdelt_events import (
    GdeltParams,
    events_from_gdelt_rows,
    load_cameo_map,
    read_export_csv,
)
from precedent.spine.gdelt_fetch import GdeltFetcher
from precedent.spine.http import Clock, HttpxTransport, SystemClock, Transport

DEFAULT_MAX_FILES = 96  # one day of 15-minute files
_STATE_FILE = "gdelt_fetch_state.json"
_PARAMS_FILE = "precedent_params.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precedent ingest gdelt")
    parser.add_argument("--since", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--until", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--max-files", type=int, default=DEFAULT_MAX_FILES)
    return parser


def run(argv: list[str], *, transport: Transport | None = None, clock: Clock | None = None) -> int:
    """Return 1 when any file failed its MD5 check, else 0; print both reports as one JSON line."""
    parser = _parser()
    args = parser.parse_args(argv)
    if args.since > args.until:
        parser.error(f"--since {args.since} is after --until {args.until}")
    if args.max_files < 1:
        parser.error("--max-files must be at least 1")
    clock = clock if clock is not None else SystemClock()
    root = data_dir()
    fetcher = GdeltFetcher(
        transport if transport is not None else HttpxTransport(),
        root / _STATE_FILE,
        root,
        args.max_files,
        clock,
    )
    fetch_report = fetcher.fetch(args.since, args.until)
    paths = fetcher.done_paths(args.since, args.until)
    frames = [read_export_csv(path.read_bytes()) for path in paths]
    rows = pd.concat(frames, ignore_index=True) if frames else read_export_csv(b"")
    records, ingest_report = events_from_gdelt_rows(
        rows,
        load_cameo_map(),
        GdeltParams.from_file(references_dir() / _PARAMS_FILE),
        ingested_utc=clock.now_utc(),
    )
    store_report = EventStore(root / "events").append(records)
    print(
        json.dumps(
            {
                "fetch": asdict(fetch_report),
                "ingest": asdict(ingest_report),
                "store": asdict(store_report),
            },
            sort_keys=True,
        )
    )
    return 1 if fetch_report.md5_failed > 0 else 0
