"""The nightly dream: one deterministic command that builds claims, finds tensions and shifts,
and writes what changed since the previous session as a note and a structured twin.

For session `t` the steps are: (1) build rule claims from the records and links visible at `t`'s
cutoff and append the new ones; (2) expire the claims that have run past the dated rule;
(3) run the tension and shift detectors as of the cutoff; (4) close the tensions whose claims no
longer all hold; (5) write `dreams/dream_<t>.md` and `.jsonl`. The note is read back from the
stores, never from what this run appended, so a re-run, or a backfill of an old session, gives
the same bytes. No wall-clock time reaches any output: stored times are session cutoffs.
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from precedent import config
from precedent.dream.claim_schema import Shift
from precedent.dream.claim_store import (
    ClaimStore,
    ConflictError,
    ResolutionStore,
    ShiftStore,
    TensionStore,
    claims_asof,
)
from precedent.dream.detectors import abnormal_returns_panel, run_detectors
from precedent.dream.inputs import DreamStores, load_detector_params
from precedent.dream.report_render import (
    SECTIONS,
    build_sections,
    jsonl_lines,
    markdown_lines,
)
from precedent.dream.rule_claims import claims_from_events, claims_from_links
from precedent.dream.shifts import run_shift_detectors
from precedent.ripple.link_store import Link, LinkStore
from precedent.spine.event_record import EventRecord
from precedent.spine.event_store import EventStore
from precedent.spine.gdelt_events import SOURCE_NAME as GDELT_SOURCE
from precedent.spine.gdelt_events import dedupe_gdelt_records
from precedent.spine.jsonl_io import atomic_write_lines, read_models
from precedent.spine.price_provider import ParquetPriceProvider, load_returns
from precedent.spine.session_calendar import NyseCalendar, default_cutoffs_for, load_nyse_calendar

OK = 0
STORE_FAILURE = 1
USAGE_ERROR = 2
DEFAULT_OUT = Path("stages") / "52_dream_report" / "output"
EXPIRY_REASON = "rule:annual_15_months"
RETURN_KIND = "close_to_close"  # a return is indexed by the session it became known on


class UsageError(Exception):
    """A bad flag, a date that is not a session, or an events directory with nothing in it."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precedent dream",
        description="Write the nightly dream for one session. Observations, not forecasts.",
    )
    parser.add_argument("--date", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="stores and dreams/")
    parser.add_argument("--events", type=Path, help="event store root (default: <data>/events)")
    parser.add_argument(
        "--parser-version", help="read events_<version>.jsonl (needed when several exist)"
    )
    parser.add_argument("--links", type=Path, help="Ripple link store root (optional)")
    parser.add_argument("--prices", type=Path, help="directory of <SYMBOL>.parquet (optional)")
    parser.add_argument("--market", help="the market symbol in --prices")
    parser.add_argument("--json", action="store_true", help="print the summary as one JSON line")
    return parser


def run(argv: list[str]) -> int:
    """Write the dream: 0 written, 2 usage, 1 a store conflict or an invalid input."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as request:
        return request.code if isinstance(request.code, int) else USAGE_ERROR
    try:
        summary = _write_dream(args)
    except UsageError as error:
        print(f"error: {error}", file=sys.stderr)
        return USAGE_ERROR
    except (ConflictError, ValueError, OSError) as error:
        print(f"error: {error} (stores under {args.out})", file=sys.stderr)
        return STORE_FAILURE
    if args.json:
        print(json.dumps(summary, sort_keys=True))
    else:
        counts = ", ".join(f"{n} {key.replace('_', ' ')}" for key, n in summary["counts"].items())
        print(f"dream {summary['session']}: {counts}\nwrote {summary['markdown']}")
    return OK


def diff_between(dream_a: Path, dream_b: Path) -> str:
    """A unified diff of two dream notes, for the morning view across several sessions."""
    lines_a = dream_a.read_text(encoding="utf-8").splitlines(keepends=True)
    lines_b = dream_b.read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(lines_a, lines_b, fromfile=dream_a.name, tofile=dream_b.name)
    )


# --- reading the inputs --------------------------------------------------------------------------


def _session_cutoff(session: date, calendar: NyseCalendar) -> datetime:
    try:
        cutoffs = default_cutoffs_for([session], calendar, 0)
        return cutoffs.loc[pd.Timestamp(session)].to_pydatetime()
    except (KeyError, ValueError) as error:
        raise UsageError(f"{session} is not a session on the NYSE calendar") from error


def _load_events(root: Path, parser_version: str | None) -> list[EventRecord]:
    """The records of one parser version, with GDELT repeats collapsed (spine rule 9).

    Two parser versions of one article are never merged: with several `events_*.jsonl` files
    the caller must name one.
    """
    if not root.is_dir():
        raise UsageError(f"events directory {root} does not exist")
    files = sorted(root.glob("events_*.jsonl"))
    found = [path.stem.removeprefix("events_") for path in files]
    if parser_version is not None:
        chosen = EventStore(root).path_for(parser_version)
        if chosen not in files:
            raise UsageError(f"no events for parser version {parser_version!r}; found: {found}")
        files = [chosen]
    elif len(files) > 1:
        raise UsageError(
            f"several parser versions under {root}: {found}; pick one with --parser-version"
        )
    records = [r for path in files for r in read_models(path, EventRecord)]
    if not records:
        raise UsageError(f"no event records under {root}")
    kept = {id(r) for r in dedupe_gdelt_records([r for r in records if r.source == GDELT_SOURCE])}
    return [r for r in records if r.source != GDELT_SOURCE or id(r) in kept]


def _visible_at(records: list[EventRecord], cutoff: datetime) -> list[EventRecord]:
    """The records a reader at `cutoff` had: available and ingested by then."""
    return [
        record
        for record in records
        if record.available_utc is not None
        and record.available_utc <= cutoff
        and record.ingested_utc <= cutoff
    ]


def _load_links(root: Path | None) -> tuple[LinkStore | None, list[Link]]:
    if root is None:
        root = config.data_dir() / "ripple" / "links"
        if not (root / "links.jsonl").exists():
            return None, []
    elif not root.is_dir():
        raise UsageError(f"links directory {root} does not exist")
    store = LinkStore(root)
    return store, store.read_all()


def _price_inputs(
    args: argparse.Namespace, axis: pd.Series, params: dict[str, Any]
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The returns and the abnormal-returns panel, built once for this invocation."""
    if args.prices is None:
        return None, None
    symbols = sorted(path.stem for path in args.prices.glob("*.parquet"))
    if args.market not in symbols:
        raise UsageError(f"market {args.market!r} has no parquet file in {args.prices}")
    returns = load_returns(
        symbols, "1900-01-01", args.date.isoformat(), RETURN_KIND, ParquetPriceProvider(args.prices)
    )
    window = params["abnormal_returns"]
    panel = abnormal_returns_panel(
        returns, args.market, axis.index, tuple(window["est_window"]), window["min_obs"]
    )
    # Prices that stop before the session (or leave every subject NaN on it) are skipped and
    # counted by the detector; only returns with no session in common at all raise (D43).
    on_session = panel.xs(pd.Timestamp(args.date), level=0)["abnormal_return"]
    if on_session.isna().all():
        return None, None
    return returns, panel


# --- the night ---


def _write_dream(args: argparse.Namespace) -> dict[str, Any]:
    if (args.prices is None) != (args.market is None):
        raise UsageError("--prices and --market go together")
    params = load_detector_params()
    calendar = load_nyse_calendar()
    session: date = args.date
    cutoff = _session_cutoff(session, calendar)
    events_root = args.events if args.events is not None else config.data_dir() / "events"
    records = _load_events(events_root, args.parser_version)
    link_store, all_links = _load_links(args.links)

    visible = _visible_at(records, cutoff)
    links = [link for link in all_links if link.valid_from_utc <= cutoff]
    dates = {r.available_utc.date() for r in visible if r.available_utc is not None} | {session}
    dates |= {link.valid_from_utc.date() for link in links}
    story_window = params["story_window_sessions"]
    full = default_cutoffs_for(sorted(dates), calendar, story_window * 3 + 14)
    axis = full.loc[: pd.Timestamp(session)]

    out = args.out
    claim_store, tension_store = ClaimStore(out / "claims"), TensionStore(out / "tensions")
    shift_store, resolution_store = ShiftStore(out / "shifts"), ResolutionStore(out / "resolutions")
    claim_store.append(
        [*claims_from_events(visible, story_window, full), *claims_from_links(links)]
    )
    _expire_annual_claims(claim_store, cutoff, params["annual_claim_expiry_months"])

    returns, panel = _price_inputs(args, axis, params)
    prices_missing = args.prices is not None and panel is None
    stores = DreamStores(
        claim_store=claim_store,
        tension_store=tension_store,
        shift_store=shift_store,
        resolution_store=resolution_store,
        event_records=visible,
        link_store=link_store,
        returns=returns,
        market=args.market,
        sessions=axis.index,
        session_cutoffs_utc=axis,
    )
    tensions, detected = run_detectors(stores, session, axis, params, abnormal_returns=panel)
    tension_store.append(tensions)
    shifts, shift_report = run_shift_detectors(stores, session, axis, params)
    _record_shifts(shift_store, shifts, session, cutoff)
    _close_lapsed_tensions(claim_store, tension_store, session, cutoff)

    previous = axis.iloc[-2].to_pydatetime() if len(axis) > 1 else None
    sections = build_sections(
        claim_store.read_all(),
        tension_store.latest(),
        shift_store.latest(),
        resolution_store.latest(),
        links,
        session,
        cutoff,
        previous,
    )
    counts = {key: len(sections[kind]) for kind, key, _ in SECTIONS}
    skipped = {
        "no_prices": detected.skipped_no_prices,
        "link_without_claim": detected.skipped_without_claim,
        **{f"shift_{reason}": n for reason, n in shift_report.skipped.items()},
    }
    header = {
        "type": "header",
        "session": session.isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "documents_ingested": detected.documents_in_session,
        "feed_gap": detected.feed_gap,
        "counts": counts,
        "skipped": skipped,
    }
    if prices_missing:
        header["prices_none_for"] = session.isoformat()
    dreams = out / "dreams"
    jsonl_path, md_path = dreams / f"dream_{session}.jsonl", dreams / f"dream_{session}.md"
    atomic_write_lines(jsonl_path, jsonl_lines(header, sections))
    atomic_write_lines(md_path, markdown_lines(header, sections))
    return {
        "session": header["session"],
        "markdown": str(md_path),
        "jsonl": str(jsonl_path),
        "counts": counts,
        "skipped": skipped,
    }


def _expire_annual_claims(store: ClaimStore, cutoff: datetime, months: int) -> None:
    """Supersede, at the cutoff, every open-ended claim visible now that began `months` ago."""
    visible = claims_asof(store, cutoff)
    oldest_kept = pd.Timestamp(cutoff) - pd.DateOffset(months=months)
    due = visible[visible["valid_to_utc"].isna() & (visible["valid_from_utc"] <= oldest_kept)]
    for claim_id in sorted(due["claim_id"]):
        store.supersede(claim_id, None, cutoff, EXPIRY_REASON)


def _record_shifts(store: ShiftStore, shifts: list[Shift], session: date, cutoff: datetime) -> None:
    """Append unknown shifts; re-fire a known one; leave one a later session already recorded."""
    last_recorded = {s.shift_id: s.recorded_utc for s in store.read_all()}
    store.append([s for s in shifts if s.shift_id not in last_recorded])
    for shift in sorted(
        (s for s in shifts if s.shift_id in last_recorded), key=lambda s: s.shift_id
    ):
        if last_recorded[shift.shift_id] <= cutoff:
            store.refire(shift.shift_id, shift.statistic, session, cutoff)


def _close_lapsed_tensions(
    claims: ClaimStore, tensions: TensionStore, session: date, cutoff: datetime
) -> None:
    """Close each open tension, opened by this session or an earlier one, that has lapsed.

    A tension lapses when any of its claims is no longer visible at the cutoff.
    """
    visible = set(claims_asof(claims, cutoff)["claim_id"])
    for tension in sorted(tensions.latest(), key=lambda t: t.tension_id):
        if (
            tension.status != "closed"
            and tension.opened_session <= session
            and not set(tension.claim_ids) <= visible
        ):
            tensions.close(tension.tension_id, session, cutoff)
