"""The nightly dream: one deterministic command that builds claims, finds tensions and shifts,
and writes what changed since the previous session as a note and a structured twin.

For session `t` the steps are: (1) build rule claims from the records and links visible at `t`'s
cutoff and append the new ones; (2) expire the claims that have run past the dated rule;
(3) run the tension and shift detectors as of the cutoff and store what they found (a tension is
its group: new, restated members, or reopened; D72); (4) close the tensions that no longer hold,
each by its kind's own test; (5) write `dreams/dream_<t>.md` and `.jsonl`. The note is read back
from the stores, never from what this run appended, so a re-run, or a backfill of an old
session, gives the same bytes. No wall-clock time reaches any output: stored times are session
cutoffs.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from precedent import config
from precedent.dream.claim_schema import Shift, Tension, TensionKind
from precedent.dream.claim_store import (
    ClaimStore,
    ConflictError,
    ResolutionStore,
    ShiftStore,
    TensionStore,
    claims_asof,
    tensions_asof,
)
from precedent.dream.detectors import abnormal_returns_panel, run_detectors
from precedent.dream.inputs import DETECTOR_PARAMS_FILE, DreamStores, load_detector_params
from precedent.dream.locking import DreamLockedError, exclusive_lock
from precedent.dream.report_render import (
    SECTIONS,
    build_sections,
    jsonl_lines,
    markdown_lines,
    not_run_items,
)
from precedent.dream.rule_claims import claims_from_events, claims_from_links
from precedent.dream.shifts import run_shift_detectors
from precedent.ripple.link_store import GRAPH_COLUMNS, Link, LinkStore, graph_asof
from precedent.spine.event_record import EventRecord
from precedent.spine.event_store import EventStore
from precedent.spine.gdelt_events import SOURCE_NAME as GDELT_SOURCE
from precedent.spine.gdelt_events import dedupe_gdelt_records
from precedent.spine.jsonl_io import atomic_write_lines, read_models
from precedent.spine.price_provider import ParquetPriceProvider, load_returns
from precedent.spine.session_calendar import (
    NyseCalendar,
    default_cutoffs_for,
    load_nyse_calendar,
    weekday_sessions,
)

OK = 0
STORE_FAILURE = 1
USAGE_ERROR = 2
DEFAULT_OUT = Path("stages") / "52_dream_report" / "output"
EXPIRY_REASON = "rule:annual_15_months"
RETURN_KIND = "close_to_close"  # a return is indexed by the session it became known on


MANIFEST_FILE = "manifest.json"
STORE_DIRS = ("claims", "tensions", "shifts", "resolutions")
_VOCABULARY_FILE = Path("dream") / "claim_vocabulary.json"


class UsageError(Exception):
    """A bad flag, a date that is not a session, or an events directory with nothing in it.

    Also a session out of order or after a gap (D70) and a store root built by other versions.
    """


class ReplayWouldChangeError(Exception):
    """Rebuilding an existing dream would append to a store: the vintage is protected (D70)."""


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
    parser.add_argument(
        "--allow-gap",
        action="store_true",
        help="dream a session after skipped ones, on purpose; the dream says how many were skipped",
    )
    parser.add_argument("--json", action="store_true", help="print the summary as one JSON line")
    return parser


def run(argv: list[str]) -> int:
    """Write the dream: 0 written, 2 usage, 1 a store conflict or an invalid input."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as request:
        return request.code if isinstance(request.code, int) else USAGE_ERROR
    try:
        with exclusive_lock(args.out):
            summary = _write_dream(args)
    except UsageError as error:
        print(f"error: {error}", file=sys.stderr)
        return USAGE_ERROR
    except (ReplayWouldChangeError, DreamLockedError) as error:
        print(f"error: {error}", file=sys.stderr)
        return STORE_FAILURE
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


def _load_events(root: Path, parser_version: str | None) -> tuple[list[EventRecord], str]:
    """The records of one parser version and its name, with GDELT repeats collapsed (spine rule 9).

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
    version = files[0].stem.removeprefix("events_")
    return [r for r in records if r.source != GDELT_SOURCE or id(r) in kept], version


def _visible_at(records: list[EventRecord], cutoff: datetime) -> list[EventRecord]:
    """The records a reader at `cutoff` had: available by then (D69).

    Knowledge time is the spine's `available_utc` (the later of published and first seen, plus
    latency). `ingested_utc` is audit-only there, so it is no gate: a nightly that ingests after
    the cutoff, or a bulk backfill, must still see the session's own documents.
    """
    return [
        record
        for record in records
        if record.available_utc is not None and record.available_utc <= cutoff
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


def _load_price_returns(args: argparse.Namespace) -> pd.DataFrame | None:
    """The returns of every symbol in `--prices` up to the session; None without `--prices`."""
    if args.prices is None:
        return None
    symbols = sorted(path.stem for path in args.prices.glob("*.parquet"))
    if args.market not in symbols:
        raise UsageError(f"market {args.market!r} has no parquet file in {args.prices}")
    return load_returns(
        symbols, "1900-01-01", args.date.isoformat(), RETURN_KIND, ParquetPriceProvider(args.prices)
    )


def _price_axis_start(
    returns: pd.DataFrame, first: date, session: date, window_start: int, calendar: NyseCalendar
) -> date | None:
    """The first price date the session axis must reach back to (D68, M13), or None.

    The market model needs `-window_start` sessions of returns before a session, so the axis
    reaches back to the earliest price date, but no further than that many sessions before the
    first event, link or session: decades of prices would otherwise stretch the axis every night.
    """
    floor = max(
        first - timedelta(days=int(abs(window_start) * 1.6) + 30), date(calendar.first_year, 1, 1)
    )
    sessions = weekday_sessions(floor, first, calendar)
    earliest = (
        sessions[-(abs(window_start) + 1)] if len(sessions) > abs(window_start) else sessions[0]
    )
    stamps = returns.index[(returns.index >= earliest) & (returns.index <= pd.Timestamp(session))]
    return stamps.min().date() if len(stamps) else None


def _price_panel(
    args: argparse.Namespace,
    returns: pd.DataFrame | None,
    axis: pd.Series,
    params: dict[str, Any],
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """The returns and the abnormal-returns panel, built once for this invocation."""
    if returns is None:
        return None, None
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


# --- the output root: manifest, order, replay (D70) ----------------------------------------------


@dataclass(frozen=True)
class _Night:
    """What `_compute` produced for one session, before any dream file is written."""

    summary: dict[str, Any]
    jsonl: list[str]
    markdown: list[str]


def _sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _current_manifest(parser_version: str, params: dict[str, Any]) -> dict[str, Any]:
    """What a store root is built with: the parser version, the vocabulary and the parameters."""
    references = config.references_dir()
    return {
        "parser_version": parser_version,
        "claim_vocabulary_sha256": _sha256_of(references / _VOCABULARY_FILE),
        "detector_params_sha256": _sha256_of(references / DETECTOR_PARAMS_FILE),
        "story_window_sessions": params["story_window_sessions"],
    }


def _read_manifest(out: Path) -> dict[str, Any] | None:
    path = out / MANIFEST_FILE
    if not path.exists():
        return None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        date.fromisoformat(manifest["last_session"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cannot read {path}: {error!r}") from error
    return manifest


def _check_manifest(stored: dict[str, Any] | None, current: dict[str, Any], out: Path) -> None:
    """Refuse a store root built by another parser version, vocabulary or parameter file."""
    if stored is None:
        return
    changed = [
        f"{key} was {stored.get(key)!r}, now {value!r}"
        for key, value in current.items()
        if stored.get(key) != value
    ]
    if changed:
        raise UsageError(
            f"{out} was built with other versions ({'; '.join(changed)}); "
            "use a fresh --out, or a declared migration"
        )


def _sessions_between(first: date, last: date, calendar: NyseCalendar) -> int:
    """NYSE sessions strictly between two dates."""
    start, end = first + timedelta(days=1), last - timedelta(days=1)
    return len(weekday_sessions(start, end, calendar)) if start <= end else 0


def _dream_dates(out: Path) -> list[date]:
    return sorted(
        date.fromisoformat(path.stem.removeprefix("dream_"))
        for path in (out / "dreams").glob("dream_*.md")
    )


def _plan(
    stored: dict[str, Any] | None, session: date, allow_gap: bool, out: Path, calendar: NyseCalendar
) -> bool:
    """Whether this run is a replay of an existing dream; raise UsageError when it is out of order.

    A session is dreamed once, in order. `t` is a replay when it is not after the last session
    dreamed and its dream exists. After a gap the run needs `--allow-gap`.
    """
    if stored is None:
        return False
    last = date.fromisoformat(stored["last_session"])
    if session <= last:
        if (out / "dreams" / f"dream_{session}.md").exists():
            return True
        raise UsageError(
            f"{session} is before the last session dreamed here ({last}) and has no dream; "
            "a session is dreamed in order, so use a fresh --out for it"
        )
    skipped = _sessions_between(last, session, calendar)
    if skipped and not allow_gap:
        raise UsageError(
            f"the last session dreamed here is {last}; {session} would skip {skipped} session(s). "
            "Dream the missed sessions in order, or pass --allow-gap to skip them on purpose"
        )
    return False


# --- the night ---


def _write_dream(args: argparse.Namespace) -> dict[str, Any]:
    if (args.prices is None) != (args.market is None):
        raise UsageError("--prices and --market go together")
    params = load_detector_params()
    calendar = load_nyse_calendar()
    session: date = args.date
    _session_cutoff(session, calendar)  # UsageError for a date that is not a session
    events_root = args.events if args.events is not None else config.data_dir() / "events"
    records, parser_version = _load_events(events_root, args.parser_version)
    out = args.out
    manifest = _current_manifest(parser_version, params)
    stored = _read_manifest(out)
    _check_manifest(stored, manifest, out)
    replay = _plan(stored, session, args.allow_gap, out, calendar)
    previous = [day for day in _dream_dates(out) if day < session]
    skipped_sessions = _sessions_between(previous[-1], session, calendar) if previous else 0
    link_store, all_links = _load_links(args.links)
    if replay:
        night = _replay(args, params, calendar, records, link_store, all_links, skipped_sessions)
    else:
        night = _compute(
            args, out, params, calendar, records, link_store, all_links, skipped_sessions
        )
        dreams = out / "dreams"
        atomic_write_lines(dreams / f"dream_{session}.jsonl", night.jsonl)
        atomic_write_lines(dreams / f"dream_{session}.md", night.markdown)
        atomic_write_lines(
            out / MANIFEST_FILE,
            [json.dumps({"last_session": session.isoformat(), **manifest}, sort_keys=True)],
        )
    return night.summary


def _replay(
    args: argparse.Namespace,
    params: dict[str, Any],
    calendar: NyseCalendar,
    records: list[EventRecord],
    link_store: LinkStore | None,
    all_links: list[Link],
    skipped_sessions: int,
) -> _Night:
    """Rebuild an existing dream on a scratch copy of the stores; it must append nothing.

    A past dream is a vintage: if the inputs have changed so that rebuilding it would add a line
    to a store, nothing is written and the command exits 1 naming the stores.
    """
    out: Path = args.out
    with tempfile.TemporaryDirectory(prefix="dream_replay_") as scratch_name:
        scratch = Path(scratch_name)
        for name in STORE_DIRS:
            if (out / name).is_dir():
                shutil.copytree(out / name, scratch / name)
        night = _compute(
            args, scratch, params, calendar, records, link_store, all_links, skipped_sessions
        )
        changed = sorted(
            f"{name}/{path.name}"
            for name in STORE_DIRS
            for path in {*(scratch / name).glob("*.jsonl"), *(out / name).glob("*.jsonl")}
            if _bytes_of(scratch / name / path.name) != _bytes_of(out / name / path.name)
        )
    if changed:
        raise ReplayWouldChangeError(
            f"the dream for {args.date} already exists and rebuilding it would change "
            f"{', '.join(changed)} under {out}; nothing was written. A past dream is not "
            "rewritten: use a fresh --out for a new vintage"
        )
    dreams = out / "dreams"
    atomic_write_lines(dreams / f"dream_{args.date}.jsonl", night.jsonl)
    atomic_write_lines(dreams / f"dream_{args.date}.md", night.markdown)
    return night


def _bytes_of(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def _compute(
    args: argparse.Namespace,
    out: Path,
    params: dict[str, Any],
    calendar: NyseCalendar,
    records: list[EventRecord],
    link_store: LinkStore | None,
    all_links: list[Link],
    skipped_sessions: int,
) -> _Night:
    """Run the night against the stores under `out` and return the note; writes only stores."""
    session: date = args.date
    cutoff = _session_cutoff(session, calendar)
    visible = _visible_at(records, cutoff)
    links = [link for link in all_links if link.valid_from_utc <= cutoff]
    dates = {r.available_utc.date() for r in visible if r.available_utc is not None} | {session}
    dates |= {link.valid_from_utc.date() for link in links}
    all_returns = _load_price_returns(args)
    if all_returns is not None:
        start = _price_axis_start(
            all_returns, min(dates), session, params["abnormal_returns"]["est_window"][0], calendar
        )
        if start is not None:
            dates.add(start)
    story_window = params["story_window_sessions"]
    full = default_cutoffs_for(sorted(dates), calendar, story_window * 3 + 14)
    axis = full.loc[: pd.Timestamp(session)]

    claim_store, tension_store = ClaimStore(out / "claims"), TensionStore(out / "tensions")
    shift_store, resolution_store = ShiftStore(out / "shifts"), ResolutionStore(out / "resolutions")
    claim_store.append(
        [*claims_from_events(visible, story_window, full), *claims_from_links(links)]
    )
    _expire_annual_claims(claim_store, cutoff, params["annual_claim_expiry_months"])

    returns, panel = _price_panel(args, all_returns, axis, params)
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
    _record_tensions(tension_store, tensions, session, cutoff)
    shifts, shift_report = run_shift_detectors(stores, session, axis, params)
    _record_shifts(shift_store, shifts, session, cutoff)
    graph = (
        graph_asof(link_store, pd.Timestamp(cutoff))
        if link_store is not None
        else pd.DataFrame(columns=GRAPH_COLUMNS)
    )
    _close_lapsed_tensions(claim_store, tension_store, tensions, graph, session, cutoff)

    previous = axis.iloc[-2].to_pydatetime() if len(axis) > 1 else None
    sections = build_sections(
        claim_store.read_all(),
        tension_store.read_all(),
        shift_store.read_all(),
        resolution_store.read_all(),
        links,
        session,
        cutoff,
        previous,
    )
    skipped = {
        "no_prices": detected.skipped_no_prices,
        "link_without_claim": detected.skipped_without_claim,
        **{f"shift_{reason}": n for reason, n in shift_report.skipped.items()},
    }
    sections["not_run"] = not_run_items(
        session=session,
        prices_given=args.prices is not None,
        prices_found=not prices_missing,
        feed_gap=detected.feed_gap,
        has_link_store=link_store is not None,
        skipped=skipped,
    )
    counts = {key: len(sections[kind]) for kind, key, _ in SECTIONS}
    header = {
        "type": "header",
        "session": session.isoformat(),
        "cutoff_utc": cutoff.isoformat(),
        "documents_in_session": detected.documents_in_session,
        "feed_gap": detected.feed_gap,
        "counts": counts,
        "skipped": skipped,
    }
    if prices_missing:
        header["prices_none_for"] = session.isoformat()
    if skipped_sessions:
        header["gap_sessions_skipped"] = skipped_sessions
    dreams = args.out / "dreams"
    summary = {
        "session": header["session"],
        "markdown": str(dreams / f"dream_{session}.md"),
        "jsonl": str(dreams / f"dream_{session}.jsonl"),
        "counts": counts,
        "skipped": skipped,
    }
    return _Night(summary, jsonl_lines(header, sections), markdown_lines(header, sections))


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


def _record_tensions(
    store: TensionStore, found: list[Tension], session: date, cutoff: datetime
) -> None:
    """Store what the detectors found: a new tension, a changed membership, or a recurrence.

    A tension is its group (D72). A group already open as of the cutoff keeps its record and, when
    its claims changed, gets a `restate_members` line; one closed as of the cutoff is reopened.
    The comparison is with the store as of this cutoff, so a replay adds nothing.
    """
    stored = {t.tension_id: t for t in tensions_asof(store, cutoff)}
    unseen: list[Tension] = []
    for tension in sorted(found, key=lambda t: t.tension_id):
        known = stored.get(tension.tension_id)
        if known is None:
            unseen.append(tension)
        elif known.status == "closed":
            store.reopen(tension.tension_id, session, tension.claim_ids, cutoff)
        elif sorted(known.claim_ids) != sorted(tension.claim_ids):
            store.restate_members(tension.tension_id, tension.claim_ids, cutoff)
    store.append(unseen)


def _close_lapsed_tensions(
    claims: ClaimStore,
    tensions: TensionStore,
    found: list[Tension],
    graph: pd.DataFrame,
    session: date,
    cutoff: datetime,
) -> None:
    """Close each open tension, opened by this session or an earlier one, that no longer holds.

    The test is the kind's own (D72). A value_conflict holds while the visible claims of its
    group still conflict across two documents (it is among `found`); an asymmetric_link holds
    while a customer claim of its pair is visible and no supplier_of link back exists; a
    text_vs_price or stale_reliance tension holds while its claims are all visible.
    """
    visible = set(claims_asof(claims, cutoff)["claim_id"])
    conflicting = {t.tension_id for t in found if t.kind is TensionKind.value_conflict}
    supplied_back = {
        (row.source_node, row.receiver_node)
        for row in graph[graph["relation"] == "supplier_of"].itertuples()
    }
    for tension in sorted(tensions_asof(tensions, cutoff), key=lambda t: t.tension_id):
        if tension.status == "closed" or tension.opened_session > session:
            continue
        if tension.kind is TensionKind.value_conflict:
            holds = tension.tension_id in conflicting
        elif tension.kind is TensionKind.asymmetric_link:
            source, receiver = tension.group_key.split("|customer_of|")
            holds = set(tension.claim_ids) <= visible and (receiver, source) not in supplied_back
        else:
            holds = set(tension.claim_ids) <= visible
        if not holds:
            tensions.close(tension.tension_id, session, cutoff)
