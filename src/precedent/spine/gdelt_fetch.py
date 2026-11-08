"""Resumable download of GDELT 2.0 export files, verified against the master list's MD5."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from precedent.config import GDELT_MASTERFILELIST_URL
from precedent.spine.gdelt_events import GdeltFormatError
from precedent.spine.http import Clock, HttpRequest, SystemClock, Transport, send_with_retries
from precedent.spine.jsonl_io import atomic_write_lines

_EXPORT_NAME = re.compile(r"^(\d{14})\.export\.CSV\.zip$")
_MD5_HEX = re.compile(r"^[0-9a-fA-F]{32}$")
_STAMP_FORMAT = "%Y%m%d%H%M%S"
_MASTER_TIMEOUT_S = 600.0
_FILE_TIMEOUT_S = 120.0
_RAW_SUBDIR = ("raw", "gdelt")


@dataclass(frozen=True)
class MasterEntry:
    """One export file named in the master list."""

    size_bytes: int
    md5: str
    url: str
    stamp_utc: datetime


@dataclass(frozen=True)
class FetchReport:
    """What a fetch did. `remaining` counts window files still not done (cap reached or failed)."""

    listed: int
    in_window: int
    downloaded: int
    already_done: int
    md5_failed: int
    remaining: int


def parse_masterfilelist(text: str) -> list[MasterEntry]:
    """Keep the export files of a master list (`<bytes> <md5> <url>` lines); refuse a bad line."""
    entries: list[MasterEntry] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 3 or not parts[0].isdigit() or not _MD5_HEX.match(parts[1]):
            raise GdeltFormatError(f"master list line {line_number} is malformed: {line[:120]!r}")
        size, md5, url = parts
        file_name = url.rsplit("/", 1)[-1]
        match = _EXPORT_NAME.match(file_name)
        if match is None:
            continue
        try:
            stamp = datetime.strptime(match.group(1), _STAMP_FORMAT).replace(tzinfo=UTC)
        except ValueError as error:
            raise GdeltFormatError(
                f"master list line {line_number} has an impossible stamp in {file_name!r}"
            ) from error
        entries.append(MasterEntry(int(size), md5.lower(), url, stamp))
    return entries


def _write_bytes_atomically(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            Path(tmp_name).unlink()
        raise


class GdeltFetcher:
    """Downloads the export files stamped in a date window, at most `max_files` per run.

    Progress lives in a JSON state file (`{"done": {file name: {...}}}`), rewritten atomically
    after each verified file, so an interrupted run resumes where it stopped.
    """

    def __init__(
        self,
        transport: Transport,
        state_path: Path,
        data_dir: Path,
        max_files: int,
        clock: Clock | None = None,
    ) -> None:
        if max_files < 1:
            raise ValueError(f"max_files must be at least 1, got {max_files}")
        self._transport = transport
        self._state_path = state_path
        self._raw_dir = data_dir.joinpath(*_RAW_SUBDIR)
        self._max_files = max_files
        self._clock = clock if clock is not None else SystemClock()

    def fetch(self, since: date, until: date) -> FetchReport:
        """Download and verify the window's files that are not yet done."""
        if since > until:
            raise ValueError(f"since {since} is after until {until}")
        master = send_with_retries(
            self._transport,
            self._clock,
            HttpRequest("GET", GDELT_MASTERFILELIST_URL, {}, timeout_s=_MASTER_TIMEOUT_S),
        )
        entries = parse_masterfilelist(master.text())
        window = sorted(
            (e for e in entries if since <= e.stamp_utc.date() <= until), key=lambda e: e.stamp_utc
        )
        state = self._read_state()
        already_done = downloaded = md5_failed = attempts = 0
        for entry in window:
            name = entry.url.rsplit("/", 1)[-1]
            if name in state["done"] and (self._raw_dir / name).exists():
                already_done += 1
                continue
            if attempts >= self._max_files:
                continue
            attempts += 1
            body = send_with_retries(
                self._transport, self._clock, HttpRequest("GET", entry.url, {}, _FILE_TIMEOUT_S)
            ).body
            if hashlib.md5(body).hexdigest() != entry.md5:  # noqa: S324 - the list's own checksum
                md5_failed += 1
                continue
            _write_bytes_atomically(self._raw_dir / name, body)
            state["done"][name] = {
                "md5": entry.md5,
                "size_bytes": len(body),
                "verified_utc": self._clock.now_utc().isoformat(),
            }
            self._write_state(state)
            downloaded += 1
        return FetchReport(
            listed=len(entries),
            in_window=len(window),
            downloaded=downloaded,
            already_done=already_done,
            md5_failed=md5_failed,
            remaining=len(window) - already_done - downloaded,
        )

    def done_paths(self, since: date, until: date) -> list[Path]:
        """Verified files on disk stamped within [since, until], oldest first. No network."""
        paths = []
        for name in self._read_state()["done"]:
            match = _EXPORT_NAME.match(name)
            if match is None:
                raise GdeltFormatError(f"state file {self._state_path.name} lists {name!r}")
            stamp = datetime.strptime(match.group(1), _STAMP_FORMAT)
            if since <= stamp.date() <= until and (self._raw_dir / name).exists():
                paths.append((stamp, self._raw_dir / name))
        return [path for _, path in sorted(paths)]

    def _read_state(self) -> dict[str, Any]:
        if not self._state_path.exists():
            return {"done": {}}
        try:
            state = json.loads(self._state_path.read_text(encoding="utf-8"))
            if not isinstance(state["done"], dict):
                raise TypeError("done must be an object")
        except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
            raise GdeltFormatError(
                f"unreadable state file {self._state_path}: {error!r}"
            ) from error
        return state

    def _write_state(self, state: dict[str, Any]) -> None:
        atomic_write_lines(self._state_path, [json.dumps(state, sort_keys=True, indent=2)])
