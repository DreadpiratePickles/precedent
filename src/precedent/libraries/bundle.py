"""The library bundle on disk: parquet tables, a build report and a manifest of their hashes.

Layout of `<data dir>/library/`: `events.parquet`, `sessions.parquet`, one `rows_<type>.parquet`
per outcome type, `placebo_asset.parquet`, `markets.parquet`, `build_report.json`, and
`MANIFEST.json` listing every other file with its SHA-256. A bundle is written to a sibling
temporary directory and renamed into place; it is read only after every hash has been checked.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError

from precedent.libraries.common import OUTCOME_TYPES, validate_library_rows
from precedent.libraries.events_table import EVENTS_TABLE_COLUMNS
from precedent.libraries.manifest import (
    MANIFEST_NAME,
    LibraryIntegrityError,
    Manifest,
    ManifestFile,
    ManifestSource,
    sha256_file,
    verify_files,
)
from precedent.libraries.market_resolution import MARKETS_FRAME_COLUMNS

__all__ = [
    "PLACEBO_COLUMNS",
    "LibraryBundle",
    "LibraryMissingError",
    "load_bundle",
    "write_bundle",
]

PLACEBO_COLUMNS = ("outcome_key", "horizon", "value", "response_complete_session")

_DISTRIBUTION = "precedent-engine"
_EVENTS_FILE = "events.parquet"
_SESSIONS_FILE = "sessions.parquet"
_PLACEBO_FILE = "placebo_asset.parquet"
_MARKETS_FILE = "markets.parquet"
_REPORT_FILE = "build_report.json"
_SESSION_INDEX_NAME = "session_date"
_EVENTS_SCHEMA = pa.schema(
    [
        ("event_id", pa.string()),
        ("key", pa.string()),
        ("event_type", pa.string()),
        ("country_group", pa.string()),
        ("available_utc", pa.timestamp("ns", "UTC")),
        ("effective_session", pa.date32()),
        ("effective_position", pa.int64()),
        ("primary_actor", pa.string()),
        ("linked_countries", pa.list_(pa.string())),
        ("story_cluster_id", pa.string()),
        ("is_first_report", pa.bool_()),
        ("one_line", pa.string()),
        ("source", pa.string()),
        ("source_ref", pa.string()),
    ]
)


class LibraryMissingError(FileNotFoundError):
    """No library bundle at the given directory (its `MANIFEST.json` is absent)."""


@dataclass(frozen=True)
class LibraryBundle:
    events: pd.DataFrame
    rows: pd.DataFrame
    placebo_asset: pd.DataFrame
    markets: pd.DataFrame
    sessions: pd.DatetimeIndex
    session_cutoffs_utc: pd.Series
    manifest: Manifest


def _rows_file(outcome_type: str) -> str:
    return f"rows_{outcome_type}.parquet"


def _check_columns(name: str, frame: pd.DataFrame, expected: tuple[str, ...]) -> None:
    if list(frame.columns) != list(expected):
        raise ValueError(
            f"{name} columns must be exactly {list(expected)}, got {list(frame.columns)}"
        )


def _checked_events(events: pd.DataFrame) -> pa.Table:
    _check_columns("events", events, EVENTS_TABLE_COLUMNS)
    return pa.Table.from_pandas(events, schema=_EVENTS_SCHEMA, preserve_index=False)


def _checked_placebo(placebo: pd.DataFrame) -> pd.DataFrame:
    _check_columns("placebo_asset", placebo, PLACEBO_COLUMNS)
    out = pd.DataFrame(
        {
            "outcome_key": placebo["outcome_key"].astype("str"),
            "horizon": placebo["horizon"].astype("int64"),
            "value": placebo["value"].astype("float64"),
            "response_complete_session": placebo["response_complete_session"].astype(
                "datetime64[ns]"
            ),
        }
    )
    if not np.isfinite(out["value"]).all():
        raise ValueError("placebo_asset values must be finite")
    return out.reset_index(drop=True)


def _checked_markets(markets: pd.DataFrame) -> pd.DataFrame:
    _check_columns("markets", markets, MARKETS_FRAME_COLUMNS)
    for column in ("created_utc", "resolved_utc"):
        if markets[column].dt.tz is None:
            raise ValueError(f"markets.{column} must be timezone-aware")
    return markets.reset_index(drop=True)


def _checked_sessions(sessions: pd.DatetimeIndex, cutoffs: pd.Series) -> pd.DataFrame:
    if sessions.tz is not None or (sessions != sessions.normalize()).any():
        raise ValueError("sessions must be timezone-naive midnight dates")
    if not sessions.is_monotonic_increasing or not sessions.is_unique:
        raise ValueError("sessions must be strictly increasing")
    if len(cutoffs) != len(sessions) or not pd.DatetimeIndex(cutoffs.index).equals(sessions):
        raise ValueError("session_cutoffs_utc must be indexed by exactly the sessions")
    if getattr(cutoffs.dtype, "tz", None) is None:
        raise ValueError("session_cutoffs_utc must hold timezone-aware instants")
    return pd.DataFrame(
        {
            "session": pd.Series(sessions.as_unit("ns")),
            "cutoff_utc": cutoffs.dt.tz_convert("UTC").dt.as_unit("ns").reset_index(drop=True),
        }
    )


def _write_parquet(path: Path, table: pa.Table | pd.DataFrame) -> None:
    if isinstance(table, pd.DataFrame):
        table = pa.Table.from_pandas(table, preserve_index=False)
    pq.write_table(table, path)


def write_bundle(
    dest: Path,
    events: pd.DataFrame,
    rows: pd.DataFrame,
    placebo_asset: pd.DataFrame,
    markets: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    session_cutoffs_utc: pd.Series,
    build_report: dict[str, Any],
    sources: list[ManifestSource],
    commit: str,
) -> Manifest:
    """Write the bundle to `dest` through a temporary sibling directory; return its manifest.

    Every input is checked before anything is written; a failure leaves an existing `dest`
    untouched and no temporary directory behind. An existing `dest` is replaced by renaming:
    the old directory is set aside as `<dest>.old`, the new one renamed into place, the old one
    removed. `rows` is split by outcome type into one file per type (an empty type still gets a
    zero-row file) and each part goes through `validate_library_rows`.
    """
    events_table = _checked_events(events)
    checked_rows = validate_library_rows(rows)
    placebo = _checked_placebo(placebo_asset)
    checked_markets = _checked_markets(markets)
    sessions_frame = _checked_sessions(sessions, session_cutoffs_utc)
    created_utc = datetime.now(UTC)
    precedent_version = version(_DISTRIBUTION)

    dest.parent.mkdir(parents=True, exist_ok=True)
    temp = dest.parent / f".{dest.name}.tmp-{os.getpid()}"
    shutil.rmtree(temp, ignore_errors=True)
    temp.mkdir()
    try:
        counts: dict[str, int | None] = {}
        _write_parquet(temp / _EVENTS_FILE, events_table)
        counts[_EVENTS_FILE] = len(events)
        _write_parquet(temp / _SESSIONS_FILE, sessions_frame)
        counts[_SESSIONS_FILE] = len(sessions_frame)
        for outcome_type in OUTCOME_TYPES:
            part = checked_rows[checked_rows["outcome_type"] == outcome_type]
            name = _rows_file(outcome_type)
            _write_parquet(temp / name, part.reset_index(drop=True))
            counts[name] = len(part)
        _write_parquet(temp / _PLACEBO_FILE, placebo)
        counts[_PLACEBO_FILE] = len(placebo)
        _write_parquet(temp / _MARKETS_FILE, checked_markets)
        counts[_MARKETS_FILE] = len(checked_markets)
        report_text = json.dumps(build_report, indent=2, sort_keys=True, default=str) + "\n"
        (temp / _REPORT_FILE).write_text(report_text, encoding="utf-8")
        counts[_REPORT_FILE] = None
        manifest = Manifest(
            files=[
                ManifestFile(path=name, sha256=sha256_file(temp / name), rows=rows_in_file)
                for name, rows_in_file in sorted(counts.items())
            ],
            sources=sources,
            built_from_commit=commit,
            precedent_version=precedent_version,
            created_utc=created_utc,
        )
        (temp / MANIFEST_NAME).write_text(
            json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _swap_into_place(temp, dest)
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
    return manifest


def _swap_into_place(temp: Path, dest: Path) -> None:
    old = dest.with_name(dest.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    replaced = dest.exists()
    if replaced:
        dest.rename(old)
    try:
        temp.rename(dest)
    except BaseException:
        if replaced:
            old.rename(dest)
        raise
    if replaced:
        shutil.rmtree(old)


def _read_manifest(library_dir: Path) -> Manifest:
    path = library_dir / MANIFEST_NAME
    try:
        return Manifest.model_validate_json(path.read_text(encoding="utf-8"))
    except (ValidationError, UnicodeDecodeError) as error:
        raise LibraryIntegrityError(MANIFEST_NAME, f"is not a valid manifest: {error}") from error


def _lists(values: pd.Series) -> pd.Series:
    return pd.Series([list(item) for item in values], dtype=object, index=values.index)


def load_bundle(library_dir: Path) -> LibraryBundle:
    """Read a bundle after verifying every file against its manifest.

    Raises `LibraryMissingError` when `MANIFEST.json` is absent and `LibraryIntegrityError`
    when a file is missing, changed or unlisted; no parquet file is opened before that check
    passes. `linked_countries` come back as Python lists.
    """
    if not (library_dir / MANIFEST_NAME).is_file():
        raise LibraryMissingError(
            f"no library at {library_dir}: {MANIFEST_NAME} is absent "
            "(run `precedent fetch` or `precedent build`)"
        )
    manifest = _read_manifest(library_dir)
    verify_files(library_dir, manifest)
    events = pd.read_parquet(library_dir / _EVENTS_FILE)
    events["linked_countries"] = _lists(events["linked_countries"])
    parts = [
        validate_library_rows(pd.read_parquet(library_dir / _rows_file(outcome_type)))
        for outcome_type in OUTCOME_TYPES
    ]
    sessions_frame = pd.read_parquet(library_dir / _SESSIONS_FILE)
    sessions = pd.DatetimeIndex(sessions_frame["session"], name=_SESSION_INDEX_NAME)
    cutoffs = sessions_frame["cutoff_utc"].set_axis(sessions).rename("cutoff_utc")
    return LibraryBundle(
        events=events,
        rows=pd.concat(parts, ignore_index=True),
        placebo_asset=pd.read_parquet(library_dir / _PLACEBO_FILE),
        markets=pd.read_parquet(library_dir / _MARKETS_FILE),
        sessions=sessions,
        session_cutoffs_utc=cutoffs,
        manifest=manifest,
    )
