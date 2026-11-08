"""Append-only store of EventRecords: one JSON Lines file per parser version."""

import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from precedent.spine.event_record import EventRecord
from precedent.spine.jsonl_io import AppendReport, atomic_write_lines

FRAME_COLUMNS = [
    "effective_session",
    "instrument",
    "severity",
    "direction",
    "confidence_weight",
    "novelty",
    "event_type",
    "article_id",
    "available_utc",
]
_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_AUDIT_ONLY_FIELDS = {"ingested_utc"}


class EventConflictError(Exception):
    """A record with the same (article_id, parser_version) key but different content."""


@dataclass(frozen=True)
class FrameReport:
    """What `to_frame` kept and set aside.

    `unaligned_records` have no effective session, so they cannot be placed on a trading day.
    `unresolved_impacts` counts aligned records with no resolved instrument. The two are disjoint.
    """

    records: int
    impact_rows: int
    unresolved_impacts: int
    unaligned_records: int


def _slug(parser_version: str) -> str:
    return _SLUG_UNSAFE.sub("_", parser_version)


def _same_content(left: EventRecord, right: EventRecord) -> bool:
    """Equality that ignores audit-only fields such as when the record was ingested."""
    return left.model_dump(exclude=_AUDIT_ONLY_FIELDS) == right.model_dump(
        exclude=_AUDIT_ONLY_FIELDS
    )


class EventStore:
    """Events keyed by (article_id, parser_version), stored as `events_<slug>.jsonl` under root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def path_for(self, parser_version: str) -> Path:
        """The JSON Lines file that holds one parser version's records."""
        return self._root / f"events_{_slug(parser_version)}.jsonl"

    def append(self, records: Iterable[EventRecord]) -> AppendReport:
        """Add new records; skip identical ones; refuse changed ones. All or nothing."""
        by_version: dict[str, list[EventRecord]] = defaultdict(list)
        for record in records:
            by_version[record.parser_version].append(record)
        self._refuse_slug_collisions(by_version)

        merged_by_version: dict[str, list[EventRecord]] = {}
        added = skipped = 0
        for version, incoming in by_version.items():
            merged = self._read_file(version)
            index = {record.article_id: record for record in merged}
            version_added = 0
            for record in incoming:
                stored = index.get(record.article_id)
                if stored is None:
                    index[record.article_id] = record
                    merged.append(record)
                    version_added += 1
                elif _same_content(stored, record):
                    skipped += 1
                else:
                    raise EventConflictError(
                        f"article {record.article_id!r} already stored for parser version "
                        f"{version!r} with different content"
                    )
            if version_added:
                merged_by_version[version] = merged
                added += version_added
        for version, merged in merged_by_version.items():
            atomic_write_lines(
                self.path_for(version), (record.model_dump_json() for record in merged)
            )
        return AppendReport(added=added, identical_skipped=skipped)

    def read(self, parser_version: str) -> list[EventRecord]:
        """Every record stored for one parser version, in append order."""
        return self._read_file(parser_version)

    def to_frame(self, parser_version: str) -> tuple[pd.DataFrame, FrameReport]:
        """One row per resolved impact, and a report of what was set aside."""
        records = self._read_file(parser_version)
        rows: list[dict] = []
        unaligned = unresolved = 0
        for record in records:
            if record.effective_session is None:
                unaligned += 1
                continue
            if not record.resolved_impacts:
                unresolved += 1
                continue
            for impact in record.resolved_impacts:
                rows.append(
                    {
                        "effective_session": record.effective_session,
                        "instrument": impact.instrument,
                        "severity": record.event.severity,
                        "direction": impact.direction,
                        "confidence_weight": record.event.confidence_weight,
                        "novelty": record.event.novelty.value,
                        "event_type": record.event.event_type.value,
                        "article_id": record.article_id,
                        "available_utc": record.available_utc,
                    }
                )
        frame = pd.DataFrame(rows, columns=FRAME_COLUMNS)
        frame["effective_session"] = pd.to_datetime(frame["effective_session"]).astype(
            "datetime64[ns]"
        )
        frame["available_utc"] = pd.to_datetime(frame["available_utc"], utc=True).astype(
            "datetime64[ns, UTC]"
        )
        report = FrameReport(
            records=len(records),
            impact_rows=len(frame),
            unresolved_impacts=unresolved,
            unaligned_records=unaligned,
        )
        return frame, report

    def _refuse_slug_collisions(self, by_version: dict[str, list[EventRecord]]) -> None:
        seen: dict[str, str] = {}
        for version in by_version:
            other = seen.setdefault(_slug(version), version)
            if other != version:
                raise ValueError(
                    f"parser versions {other!r} and {version!r} share the file slug "
                    f"{_slug(version)!r}"
                )

    def _read_file(self, parser_version: str) -> list[EventRecord]:
        path = self.path_for(parser_version)
        if not path.exists():
            return []
        records: list[EventRecord] = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = EventRecord.model_validate_json(line)
                except ValidationError as error:
                    raise ValueError(f"{path.name} line {line_number}: {error}") from error
                if record.parser_version != parser_version:
                    raise ValueError(
                        f"{path.name} line {line_number}: parser version "
                        f"{record.parser_version!r} shares the file slug of {parser_version!r}"
                    )
                records.append(record)
        return records
