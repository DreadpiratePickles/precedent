"""A small committed sample of a built library: a date window, no prices, no question text.

`build_sample` reads a verified library bundle, keeps the events whose effective session lies in
a window with their market-resolution and event-sequence rows, drops every asset row and the
placebo table (no vendor price ships), keeps only the six typed columns of each market, and
writes the result as a bundle plus a `README.md` that states what the files are and cites the
event source. It refuses, before leaving anything behind, a sample over the byte limit.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Literal

import pandas as pd

from precedent.libraries.bundle import PLACEBO_COLUMNS, LibraryBundle, load_bundle, write_bundle
from precedent.libraries.common import empty_library_rows
from precedent.libraries.manifest import Manifest, ManifestSource
from precedent.libraries.market_resolution import MARKETS_FRAME_COLUMNS
from precedent.libraries.sources import CITATIONS, EVENT_SOURCES
from precedent.spine.market_store import VENUES

__all__ = ["GDELT_CITATION", "MAX_SAMPLE_BYTES", "SampleTooLargeError", "build_sample"]

GDELT_CITATION = CITATIONS["gdelt2"]
MAX_SAMPLE_BYTES = 2_000_000
_EVIDENCE_LABELS = ("LIVE", "SYNTHETIC")
_LIBRARY_DIR = "library"
_README = "README.md"
_EVENT_SOURCE_OF = dict(EVENT_SOURCES)


class SampleTooLargeError(ValueError):
    """The sample's files add up to more than the allowed number of bytes."""


def _in_window(sessions: pd.Series, since: date, until: date) -> pd.Series:
    return pd.Series([since <= day <= until for day in sessions], index=sessions.index, dtype=bool)


def _resolved_days(markets: pd.DataFrame) -> pd.Series:
    return markets["resolved_utc"].dt.tz_convert("UTC").dt.date


def _empty_placebo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome_key": pd.Series(dtype="str"),
            "horizon": pd.Series(dtype="int64"),
            "value": pd.Series(dtype="float64"),
            "response_complete_session": pd.Series(dtype="datetime64[ns]"),
        }
    )[list(PLACEBO_COLUMNS)]


def _kept_markets(
    markets: pd.DataFrame, market_rows: pd.DataFrame, since: date, until: date
) -> pd.DataFrame:
    referenced = markets["market_id"].isin(set(market_rows["source_ref"]))
    keep = referenced | _in_window(_resolved_days(markets), since, until)
    return markets.loc[keep, list(MARKETS_FRAME_COLUMNS)].reset_index(drop=True)


def _trimmed_sessions(
    bundle: LibraryBundle,
    events: pd.DataFrame,
    rows: pd.DataFrame,
    markets: pd.DataFrame,
    since: date,
    until: date,
) -> tuple[pd.DatetimeIndex, pd.Series, int]:
    """Sessions from the first on or after `since` to the last any kept record points at.

    Returns the sessions, their cutoffs and the offset of the first kept session in the
    source sessions (`effective_position` is an index into `sessions`, so it moves by it).
    """
    sessions = bundle.sessions
    start = int(sessions.searchsorted(pd.Timestamp(since), side="left"))
    ends = [
        rows["response_complete_session"].max(),
        markets["response_complete_session"].max(),
    ]
    if len(events):
        ends.append(pd.Timestamp(max(events["effective_session"])))
    known = [end for end in ends if pd.notna(end)]
    last = max(known) if len(rows) and known else pd.Timestamp(until)
    stop = int(sessions.searchsorted(last, side="right"))
    stop = max(stop, start)
    return sessions[start:stop], bundle.session_cutoffs_utc.iloc[start:stop], start


def _source_rows(source: ManifestSource, events: pd.DataFrame, markets: pd.DataFrame) -> list[date]:
    """The dates of the kept records that belong to `source`."""
    if source.name in VENUES:
        prefix = markets["market_id"].str.split(":", n=1).str[0]
        return list(_resolved_days(markets)[prefix == source.name])
    event_source = _EVENT_SOURCE_OF.get(source.name, source.name)
    return list(events.loc[events["source"] == event_source, "effective_session"])


def _clipped_sources(
    sources: Sequence[ManifestSource],
    events: pd.DataFrame,
    markets: pd.DataFrame,
    since: date,
    until: date,
) -> list[ManifestSource]:
    kept: list[ManifestSource] = []
    for source in sources:
        days = _source_rows(source, events, markets)
        if not days:
            continue
        first, last = max(source.first_date, since), min(source.last_date, until)
        if first > last:  # a market kept only because a kept row cites it
            first, last = min(days), max(days)
        update = {"first_date": first, "last_date": last, "rows": len(days)}
        kept.append(source.model_copy(update=update))
    return kept


def _readme_lines(
    evidence_label: str, citation: str, since: date, until: date, manifest: Manifest
) -> list[str]:
    banner = (
        f"{evidence_label} — GDELT 2.0 events and resolved Polymarket/Kalshi market outcomes "
        f"(ids, typed keys and dates; no question text), {since} to {until}"
    )
    lines = [banner, "", citation]
    lines += [
        f"- {source.name}: {source.first_date} to {source.last_date}, {source.rows} rows"
        for source in manifest.sources
    ]
    return lines


def build_sample(
    library_dir: Path,
    out_dir: Path,
    since: date,
    until: date,
    evidence_label: Literal["LIVE", "SYNTHETIC"],
    citation: str,
    max_bytes: int = MAX_SAMPLE_BYTES,
) -> Manifest:
    """Write the sample of `library_dir` for `[since, until]` to `out_dir`; return its manifest.

    `out_dir / "library"` is a bundle of the events in the window with their market-resolution
    and event-sequence rows, the markets those rows cite or that resolved in the window (six
    columns, no text), the sessions the records need, an empty `placebo_asset` and no asset row.
    `out_dir / "README.md"` opens with the evidence label, then the citation, then one line per
    source. A sample over `max_bytes` raises `SampleTooLargeError` before `out_dir` exists; an
    earlier `out_dir / "library"` is replaced only once the new one has passed that check.
    """
    if evidence_label not in _EVIDENCE_LABELS:
        raise ValueError(
            f"evidence_label must be one of {_EVIDENCE_LABELS}, got {evidence_label!r}"
        )
    if since > until:
        raise ValueError(f"since {since} is after until {until}")
    bundle = load_bundle(library_dir)
    keep = _in_window(bundle.events["effective_session"], since, until)
    events = bundle.events.loc[keep].reset_index(drop=True)
    kept_rows = bundle.rows[
        bundle.rows["event_id"].isin(set(events["event_id"]))
        & (bundle.rows["outcome_type"] != "asset")
    ].reset_index(drop=True)
    market_rows = kept_rows[kept_rows["outcome_type"] == "market_resolution"]
    markets = _kept_markets(bundle.markets, market_rows, since, until)
    sessions, cutoffs, offset = _trimmed_sessions(bundle, events, kept_rows, markets, since, until)
    events = events.assign(effective_position=events["effective_position"].astype("int64") - offset)
    sources = _clipped_sources(bundle.manifest.sources, events, markets, since, until)
    commit = bundle.manifest.built_from_commit
    report = {
        "sample_of": commit,
        "since": since.isoformat(),
        "until": until.isoformat(),
        "events": len(events),
        "rows": len(kept_rows),
    }

    temp = out_dir.parent / f".{out_dir.name}.tmp-{os.getpid()}"
    shutil.rmtree(temp, ignore_errors=True)
    try:
        manifest = write_bundle(
            temp / _LIBRARY_DIR,
            events,
            kept_rows if len(kept_rows) else empty_library_rows(),
            _empty_placebo(),
            markets,
            sessions,
            cutoffs,
            report,
            sources,
            commit,
        )
        size = sum((temp / _LIBRARY_DIR / entry.path).stat().st_size for entry in manifest.files)
        if size > max_bytes:
            raise SampleTooLargeError(f"sample is {size} bytes, limit {max_bytes}")
        out_dir.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(out_dir / _LIBRARY_DIR, ignore_errors=True)
        (temp / _LIBRARY_DIR).rename(out_dir / _LIBRARY_DIR)
    finally:
        shutil.rmtree(temp, ignore_errors=True)
    readme = _readme_lines(evidence_label, citation, since, until, manifest)
    (out_dir / _README).write_text("\n".join(readme) + "\n", encoding="utf-8")
    return manifest
