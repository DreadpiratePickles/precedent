"""Append-only store of MarketRecords: one JSON Lines file per venue."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

import pandas as pd

from precedent.spine.jsonl_io import AppendReport, atomic_write_lines, merge_keyed, read_models
from precedent.spine.market_record import MarketRecord

Venue = Literal["polymarket", "kalshi"]
VENUES: tuple[Venue, ...] = ("polymarket", "kalshi")
FRAME_COLUMNS = [
    "market_id",
    "venue",
    "created_utc",
    "close_utc",
    "resolved_utc",
    "resolution",
    "event_key",
    "typed_status",
    "category",
    "source_ref",
]
UNTYPED_KEY = "UNTYPED"


class MarketConflictError(Exception):
    """A record with the same (market_id, typer_version) key but different content."""


def _key_of(record: MarketRecord) -> tuple[str, str]:
    return record.market_id, record.typed.typer_version


def _conflict(record: MarketRecord) -> MarketConflictError:
    return MarketConflictError(
        f"market {record.market_id!r} already stored for typer version "
        f"{record.typed.typer_version!r} with different content"
    )


class MarketStore:
    """Markets keyed by (market_id, typer_version), in `markets_<venue>.jsonl` under root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def path_for(self, venue: Venue) -> Path:
        """The JSON Lines file that holds one venue's records, every typer version."""
        return self._root / f"markets_{venue}.jsonl"

    def append(self, records: Iterable[MarketRecord]) -> AppendReport:
        """Add new records; skip identical ones; refuse changed ones. All or nothing."""
        by_venue: dict[Venue, list[MarketRecord]] = defaultdict(list)
        for record in records:
            by_venue[record.venue].append(record)
        merged_by_venue: dict[Venue, list[MarketRecord]] = {}
        added = skipped = 0
        for venue, incoming in by_venue.items():
            merged, venue_added, venue_skipped = merge_keyed(
                self._read_file(venue), incoming, _key_of, _conflict
            )
            skipped += venue_skipped
            if venue_added:
                merged_by_venue[venue] = merged
                added += venue_added
        for venue, merged in merged_by_venue.items():
            atomic_write_lines(self.path_for(venue), (r.model_dump_json() for r in merged))
        return AppendReport(added=added, identical_skipped=skipped)

    def read(self, venue: Venue, typer_version: str) -> list[MarketRecord]:
        """Every record one typer version produced for one venue, in append order."""
        return [r for r in self._read_file(venue) if r.typed.typer_version == typer_version]

    def to_frame(self, typer_version: str) -> pd.DataFrame:
        """One row per market, both venues, for one typer version."""
        rows = [
            {
                "market_id": record.market_id,
                "venue": record.venue,
                "created_utc": record.created_utc,
                "close_utc": record.close_utc,
                "resolved_utc": record.resolved_utc,
                "resolution": record.resolution,
                "event_key": record.typed.key.as_string() if record.typed.key else UNTYPED_KEY,
                "typed_status": record.typed.status,
                "category": record.category,
                "source_ref": record.source_ref,
            }
            for venue in VENUES
            for record in self.read(venue, typer_version)
        ]
        frame = pd.DataFrame(rows, columns=FRAME_COLUMNS)
        for column in ("created_utc", "close_utc", "resolved_utc"):
            frame[column] = pd.to_datetime(frame[column], utc=True).astype("datetime64[ns, UTC]")
        return frame

    def _read_file(self, venue: Venue) -> list[MarketRecord]:
        return read_models(self.path_for(venue), MarketRecord)
