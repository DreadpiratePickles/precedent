"""`precedent build`: read the spine's stores and write the verified library bundle.

Reading order: event records per parser version (GDELT records deduplicated across runs), typed
markets per venue; sessions from the NYSE calendar spanning every record and market; then the
events table, the three libraries, the asset placebo and the bundle. Nothing is fetched and no
model is called.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, date, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, ValidationError

from precedent.config import ConfigError, data_dir
from precedent.engine.params import EngineParams, load_engine_params
from precedent.libraries.asset import (
    AssetLibraryReport,
    build_asset_library,
    sample_asset_placebo,
)
from precedent.libraries.bundle import PLACEBO_COLUMNS, write_bundle
from precedent.libraries.common import empty_library_rows
from precedent.libraries.event_sequence import SequenceLibraryReport, build_event_sequence_library
from precedent.libraries.events_table import EventsTableReport, build_events_table
from precedent.libraries.manifest import LibraryIntegrityError, ManifestSource
from precedent.libraries.market_resolution import (
    MarketLibraryReport,
    build_market_resolution_library,
)
from precedent.libraries.sources import CITATIONS, EVENT_SOURCES
from precedent.spine.edgar_8k_events import edgar_parser_version
from precedent.spine.event_record import EventRecord
from precedent.spine.event_store import EventStore
from precedent.spine.gdelt_events import dedupe_gdelt_records, gdelt_parser_version
from precedent.spine.market_record import MarketRecord
from precedent.spine.market_store import VENUES, MarketStore
from precedent.spine.price_provider import ParquetPriceProvider, load_returns
from precedent.spine.session_calendar import default_cutoffs_for, load_nyse_calendar
from precedent.spine.typers.rules import RulesTyper
from precedent.spine.typers.select import typer_from_env

__all__ = ["BuildReport", "NoEventsError", "build_libraries", "current_commit", "run"]

USAGE_ERROR = 2
LIBRARY_DIR_NAME = "library"
LOOKAHEAD_DAYS = 120
PRICE_LOOKBACK_DAYS = 400
_GDELT_PARSER_PREFIX = "rules-gdelt"
_NO_MARKET_EVENT = "no_market_event"
_NO_PRICES = "no prices"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
_COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}")


class NoEventsError(RuntimeError):
    """There is nothing to build from: no event record, or none placed on a session."""


class BuildReport(BaseModel):
    """What one build read, kept and skipped; written to `build_report.json`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    events: EventsTableReport
    asset: AssetLibraryReport | None
    asset_not_built_reason: str | None
    market_resolution: MarketLibraryReport
    event_sequence: SequenceLibraryReport


def current_commit() -> str:
    """`git rev-parse HEAD` of this checkout, or `unknown` when git or the checkout is absent."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_REPOSITORY_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return "unknown"
    commit = result.stdout.strip()
    if result.returncode == 0 and _COMMIT_PATTERN.fullmatch(commit):
        return commit
    return "unknown"


def _read_records(root: Path, parser_versions: tuple[str, ...]) -> list[EventRecord]:
    store = EventStore(root / "events")
    records: list[EventRecord] = []
    for parser_version in parser_versions:
        stored = store.read(parser_version)
        if parser_version.startswith(_GDELT_PARSER_PREFIX):
            stored = dedupe_gdelt_records(stored)
        records.extend(stored)
    return records


def _read_markets(root: Path, typer_version: str) -> list[MarketRecord]:
    store = MarketStore(root / "markets")
    return [market for venue in VENUES for market in store.read(venue, typer_version)]


def _session_cutoffs(
    records: list[EventRecord], markets: list[MarketRecord], lookback_days: int
) -> pd.Series:
    """Sessions from the earliest date, `lookback_days` earlier, to 120 days past the latest.

    The lookback gives the market model its estimation history; the calendar's first year
    floors the start.
    """
    dates: list[date] = [r.effective_session for r in records if r.effective_session is not None]
    dates += [market.resolved_utc.astimezone(UTC).date() for market in markets]
    if lookback_days:
        dates.append(min(dates) - timedelta(days=lookback_days))
    return default_cutoffs_for(dates, load_nyse_calendar(), LOOKAHEAD_DAYS)


def _price_symbols(prices_dir: Path) -> set[str]:
    return {path.stem for path in prices_dir.glob("*.parquet")} if prices_dir.is_dir() else set()


def _impacts_by_event(records: list[EventRecord]) -> dict[str, list[str]]:
    impacts: dict[str, list[str]] = {}
    for record in records:
        found = impacts.setdefault(record.article_id, [])
        found += [impact.instrument for impact in record.resolved_impacts]
    return impacts


def _empty_placebo() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "outcome_key": pd.Series(dtype="str"),
            "horizon": pd.Series(dtype="int64"),
            "value": pd.Series(dtype="float64"),
            "response_complete_session": pd.Series(dtype="datetime64[ns]"),
        }
    )


def _tradable_symbols(root: Path, params: EngineParams) -> set[str]:
    """Symbols that can be assets: none unless prices exist for the market and one other."""
    available = _price_symbols(root / "prices")
    tradable = available - {params.market}
    return tradable if params.market in available else set()


def _asset_part(
    root: Path,
    params: EngineParams,
    events: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    impacts: dict[str, list[str]],
    tradable: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, AssetLibraryReport | None]:
    """Asset rows, placebo rows and the report; no report when there are no usable prices."""
    prices_dir = root / "prices"
    market = params.market
    if not tradable:
        return empty_library_rows(), _empty_placebo(), None
    panel = [symbol for symbol in dict.fromkeys(params.asset_panel) if symbol in tradable]
    from_impacts = {s for symbols in impacts.values() for s in symbols} & tradable
    returns = load_returns(
        [market, *dict.fromkeys([*panel, *sorted(from_impacts)])],
        sessions[0].strftime("%Y-%m-%d"),
        sessions[-1].strftime("%Y-%m-%d"),
        "open_to_open",
        ParquetPriceProvider(prices_dir),
    ).reindex(sessions)

    def assets_for_event(row: pd.Series) -> list[str]:
        wanted = [*params.asset_panel, *impacts.get(str(row["event_id"]), [])]
        return [symbol for symbol in dict.fromkeys(wanted) if symbol in tradable]

    rows, report = build_asset_library(
        events, returns, market, sessions, assets_for_event, params.asset_horizons
    )
    placebo = _placebo(params, events, returns, sessions, panel)
    return rows, placebo, report


def _placebo(
    params: EngineParams,
    events: pd.DataFrame,
    returns: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    panel: list[str],
) -> pd.DataFrame:
    rng = np.random.default_rng(params.placebo_seed)
    exclude = {int(p) for p in events.loc[events["is_first_report"], "effective_position"]}
    samples = [
        sample_asset_placebo(
            returns[symbol],
            returns[params.market],
            sessions,
            horizon,
            params.placebo_samples,
            rng,
            exclude,
        ).assign(outcome_key=symbol, horizon=horizon)
        for symbol in panel
        for horizon in params.asset_horizons
    ]
    samples = [sample for sample in samples if len(sample)]
    if not samples:
        return _empty_placebo()
    return pd.concat(samples, ignore_index=True)[list(PLACEBO_COLUMNS)]


def _sources(events: pd.DataFrame, markets: pd.DataFrame) -> list[ManifestSource]:
    sources: list[ManifestSource] = []
    for name, event_source in EVENT_SOURCES:
        sessions = events.loc[events["source"] == event_source, "effective_session"]
        if len(sessions):
            sources.append(_source(name, min(sessions), max(sessions), len(sessions)))
    resolved_days = markets["resolved_utc"].dt.tz_convert("UTC").dt.date
    for venue in VENUES:
        days = resolved_days[markets["market_id"].str.startswith(f"{venue}:")]
        if len(days):
            sources.append(_source(venue, min(days), max(days), len(days)))
    return sources


def _source(name: str, first: date, last: date, rows: int) -> ManifestSource:
    return ManifestSource(
        name=name, first_date=first, last_date=last, rows=rows, citation=CITATIONS[name]
    )


def build_libraries(
    data_dir: Path,
    params: EngineParams,
    commit: str,
    parser_versions: tuple[str, ...] | None = None,
    typer_version: str | None = None,
) -> BuildReport:
    """Build the libraries from the stores under `data_dir` and write `data_dir / "library"`.

    `parser_versions` default to the live GDELT and EDGAR versions; `typer_version` to the rules
    typer's. Raises `NoEventsError` (naming `precedent ingest`) when no record is stored or none
    can be placed on a session. The asset library needs `<SYMBOL>.parquet` for `params.market`
    and one other symbol under `data_dir / "prices"`; without them `asset` is None and
    `asset_not_built_reason` is `no prices`. The market symbol is never an asset row and never
    a placebo symbol (its abnormal return against itself is zero). When prices exist the session
    axis starts 400 calendar days before the earliest event or market date, so the market model
    has its estimation window for the first events; without prices it starts at that date.
    """
    if parser_versions is None:
        parser_versions = (gdelt_parser_version(), edgar_parser_version())
    if typer_version is None:
        typer_version = RulesTyper().version
    records = _read_records(data_dir, parser_versions)
    if not records:
        raise NoEventsError(
            f"no event records under {data_dir / 'events'} for parser versions "
            f"{list(parser_versions)}; run `precedent ingest` first"
        )
    if all(record.effective_session is None for record in records):
        raise NoEventsError(
            f"none of the {len(records)} event records is aligned to a session; "
            "run `precedent ingest` again"
        )
    markets = _read_markets(data_dir, typer_version)
    tradable = _tradable_symbols(data_dir, params)
    cutoffs = _session_cutoffs(records, markets, PRICE_LOOKBACK_DAYS if tradable else 0)
    sessions = pd.DatetimeIndex(cutoffs.index)
    events, events_report = build_events_table(records, sessions, params.story_window_sessions)
    if events_report.kept == 0:
        raise NoEventsError(
            f"none of the {len(records)} event records falls on a session of the calendar"
        )
    first_reports = events[events["is_first_report"]]
    event_types = tuple(sorted(set(first_reports["event_type"]) - {_NO_MARKET_EVENT}))
    sequence_rows, sequence_report = build_event_sequence_library(
        events,
        sessions,
        event_types,
        params.sequence_horizons,
        feed_end_position=int(events["effective_position"].max()),
    )
    market_rows, markets_frame, market_report = build_market_resolution_library(
        events, markets, sessions, cutoffs
    )
    asset_rows, placebo, asset_report = _asset_part(
        data_dir, params, events, sessions, _impacts_by_event(records), tradable
    )
    report = BuildReport(
        events=events_report,
        asset=asset_report,
        asset_not_built_reason=None if asset_report is not None else _NO_PRICES,
        market_resolution=market_report,
        event_sequence=sequence_report,
    )
    parts = [part for part in (asset_rows, market_rows, sequence_rows) if len(part)]
    write_bundle(
        data_dir / LIBRARY_DIR_NAME,
        events=events,
        rows=pd.concat(parts, ignore_index=True) if parts else empty_library_rows(),
        placebo_asset=placebo,
        markets=markets_frame,
        sessions=sessions,
        session_cutoffs_utc=cutoffs,
        build_report=report.model_dump(),
        sources=_sources(events, markets_frame),
        commit=commit,
    )
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precedent build")
    parser.add_argument("--data-dir", type=Path, default=None, metavar="PATH")
    return parser


def run(argv: list[str]) -> int:
    """Build the libraries; print the report as one JSON line and return 0, or 2 on any refusal."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_request:
        return exit_request.code if isinstance(exit_request.code, int) else USAGE_ERROR
    root: Path = args.data_dir if args.data_dir is not None else data_dir()
    try:
        params = load_engine_params()
    except (ValidationError, ValueError, OSError) as error:
        print(
            f"error: engine parameters are invalid: {' '.join(str(error).split())}", file=sys.stderr
        )
        return USAGE_ERROR
    try:
        typer_version = typer_from_env().version
        report = build_libraries(root, params, current_commit(), typer_version=typer_version)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return USAGE_ERROR
    except (NoEventsError, LibraryIntegrityError) as error:
        print(f"error: {error}", file=sys.stderr)
        return USAGE_ERROR
    counts: dict[str, Any] = report.model_dump()
    print(json.dumps(counts, sort_keys=True))
    return 0
