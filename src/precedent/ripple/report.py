"""Research: the daily Ripple map, `precedent ripple --date YYYY-MM-DD [--top N] [--out PATH]`.

For one NYSE session it lists the largest price shocks and, for each, the companies the dated link
graph says it reaches, by channel, with the evidence for every link. It is history, not a forecast:
only derived statistics (abnormal returns and propagated sizes) are shown, never a price.

Reads `<data_dir>/ripple/`: `links/links.jsonl` (only through `graph_asof`; the one graph the
run published, named in the manifest and in the map's header), `shocks.parquet`,
`abnormal_returns.parquet` and `MANIFEST.json`, whose SHA-256 entries are checked before use.
Exit codes: 0 rendered, 1 data error (including a date outside the ledger's start to end),
2 usage error (including a date that is not a session).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from precedent.config import data_dir
from precedent.ripple.graph_builders import LEDGER_MANIFEST_NAME, write_text_atomically
from precedent.ripple.link_store import RULES_EXTRACTOR_PREFIX, LinkStore, Relation, graph_asof
from precedent.ripple.params import load_ripple_params
from precedent.ripple.propagate import build_adjacency
from precedent.ripple.ripple_factors import CHANNEL_OF_RELATION
from precedent.spine.session_calendar import calendar_cutoffs, load_nyse_calendar

OK = 0
DATA_ERROR = 1
USAGE_ERROR = 2
NO_SHOCKS_LINE = "No shocks on this session."
NOT_APPLICABLE = "n/a"
NO_FACTOR_SIZE = "n/a (no factor channel)"
COUNTRY_CHANNEL = Relation.COUNTRY_EXPOSURE.value
_CHANNEL_ORDER = ["ripple_customer", "ripple_supplier", "ripple_peer", "ripple_commodity"]
_CHANNEL_TITLES = {
    "ripple_customer": "customer channel (ripple_customer)",
    "ripple_supplier": "supplier channel (ripple_supplier)",
    "ripple_peer": "peer channel (ripple_peer)",
    "ripple_commodity": "commodity channel (ripple_commodity)",
    COUNTRY_CHANNEL: "country_exposure (no factor channel)",
}
_BANNERS = {
    "SYNTHETIC": "SYNTHETIC — built from synthetic data; it shows the code runs, "
    "not anything about markets.",
    "LIVE-DATA": "LIVE — built from live data; derived statistics only, "
    "history and not a forecast.",
    "LIVE-MODEL": "LIVE — built from live data and model-extracted links; "
    "derived statistics only, history and not a forecast.",
    "NO LIVE RUN": "NO LIVE RUN — the stores are marked as not built from a live run.",
}
_SHOCK_FILE = "shocks.parquet"
_ABNORMAL_FILE = "abnormal_returns.parquet"
_LINKS_DIR = "links"
_LINKS_FILE = "links/links.jsonl"
_DATE_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}")


class DataError(Exception):
    """A store is unreadable, inconsistent or marked with an unknown label (exit 1)."""


@dataclass(frozen=True)
class LinkRow:
    receiver: str
    relation: str
    size: float | None  # None: the relation has no factor channel
    move: float | None  # None: the receiver has no abnormal return on the session
    doc_id: str
    evidence: str


def _parse_date(text: str) -> date:
    if not _DATE_PATTERN.fullmatch(text):
        raise argparse.ArgumentTypeError(f"{text!r} is not a date in the form YYYY-MM-DD")
    try:
        return date.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not a valid date: {error}") from error


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from error
    if value < 1:
        raise argparse.ArgumentTypeError(f"--top must be at least 1, got {value}")
    return value


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precedent ripple",
        description="The daily Ripple map: the largest shocks of a session and what they reach.",
    )
    parser.add_argument("--date", required=True, type=_parse_date, help="session, YYYY-MM-DD")
    parser.add_argument("--top", type=_positive_int, default=10, help="shocks shown (default 10)")
    parser.add_argument("--out", type=Path, default=None, help="write the map here, else stdout")
    return parser


def _read_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise DataError(f"cannot read {path}: {error}") from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise DataError(f"{path} must be a JSON object with a `files` entry")
    return manifest


def _banner_for(manifest: dict[str, Any]) -> str:
    label = manifest.get("evidence_label")
    if not isinstance(label, str) or label not in _BANNERS:
        raise DataError(f"unknown evidence_label {label!r} in {LEDGER_MANIFEST_NAME}")
    return _BANNERS[label]


def _checked_bytes(ripple_dir: Path, name: str, manifest: dict[str, Any]) -> tuple[bytes, Any]:
    """A listed file's bytes, after its presence and SHA-256 are checked; and its manifest entry."""
    entry = manifest["files"].get(name)
    if not isinstance(entry, dict) or "sha256" not in entry:
        raise DataError(f"{LEDGER_MANIFEST_NAME} does not list {name} with a sha256")
    path = ripple_dir / name
    try:
        content = path.read_bytes()
    except OSError as error:
        raise DataError(f"{name} is listed in the manifest but cannot be read: {error}") from error
    if hashlib.sha256(content).hexdigest() != entry["sha256"]:
        raise DataError(f"{name} does not match the SHA-256 in {LEDGER_MANIFEST_NAME}")
    return content, entry


def _check_links_file(ripple_dir: Path, manifest: dict[str, Any]) -> str:
    """Check the published link store against the manifest; return the published graph's name."""
    content, entry = _checked_bytes(ripple_dir, _LINKS_FILE, manifest)
    rows = sum(1 for line in content.decode("utf-8", errors="replace").splitlines() if line.strip())
    if "rows" in entry and entry["rows"] != rows:
        raise DataError(f"{_LINKS_FILE} has {rows} rows, the manifest says {entry['rows']}")
    graph = manifest.get("published_graph")
    if not isinstance(graph, str) or not graph.strip():
        raise DataError(f"{LEDGER_MANIFEST_NAME} does not name the published graph")
    return graph


def _check_date_in_range(session: date, manifest: dict[str, Any]) -> None:
    """A date outside the ledger's start to end would read as "no shocks": stop instead."""
    start, end = manifest.get("start"), manifest.get("end")
    if start is None and end is None:
        return
    try:
        first, last = date.fromisoformat(str(start)), date.fromisoformat(str(end))
    except ValueError as error:
        raise DataError(
            f"{LEDGER_MANIFEST_NAME} start and end must be YYYY-MM-DD: {error}"
        ) from error
    if not first <= session <= last:
        raise DataError(
            f"{session.isoformat()} is outside the Ripple ledger's range "
            f"{first.isoformat()} to {last.isoformat()}"
        )


def _read_checked_parquet(ripple_dir: Path, name: str, manifest: dict[str, Any]) -> pd.DataFrame:
    content, entry = _checked_bytes(ripple_dir, name, manifest)
    try:
        frame = pd.read_parquet(io.BytesIO(content))
    except (OSError, ValueError) as error:
        raise DataError(f"{name} is not readable parquet: {error}") from error
    if "rows" in entry and entry["rows"] != len(frame):
        raise DataError(f"{name} has {len(frame)} rows, the manifest says {entry['rows']}")
    if "datetime" not in frame.columns:
        raise DataError(f"{name} has no `datetime` column")
    if getattr(frame["datetime"].dt, "tz", None) is not None:
        raise DataError(f"{name} `datetime` must hold naive session labels")
    return frame


def _session_shocks(frame: pd.DataFrame, session: pd.Timestamp) -> pd.Series:
    """The session's shocks as a float Series indexed by instrument; duplicates and NaN stop."""
    missing = {"instrument", "shock"} - set(frame.columns)
    if missing:
        raise DataError(f"{_SHOCK_FILE} has no column {sorted(missing)[0]!r}")
    today = frame[frame["datetime"] == session]
    if today["instrument"].duplicated().any():
        name = today["instrument"][today["instrument"].duplicated()].iloc[0]
        raise DataError(f"{_SHOCK_FILE} has two shocks for {name!r} on {session.date()}")
    values = pd.Series(
        today["shock"].to_numpy(dtype="float64"), index=today["instrument"].to_numpy()
    )
    if not values.map(math.isfinite).all():
        raise DataError(f"{_SHOCK_FILE} holds a non-finite shock on {session.date()}")
    return values


def _session_moves(frame: pd.DataFrame, session: pd.Timestamp) -> pd.Series:
    """Each instrument's abnormal return on the session; empty when the session has no row."""
    today = frame[frame["datetime"] == session].drop(columns="datetime")
    if len(today) > 1:
        raise DataError(f"{_ABNORMAL_FILE} has {len(today)} rows for {session.date()}")
    if today.empty:
        return pd.Series(dtype="float64")
    return today.iloc[0].astype("float64")


def _propagated_sizes(graph: pd.DataFrame, shocks: pd.Series, alpha: float) -> dict[int, float]:
    """Hop-1 size alpha * A_c[receiver, source] * shock for each channel link, keyed by graph row.

    `A_c` is the row-normalised adjacency of the channel's links. When one source reaches one
    receiver through two relations of the same channel, the pair's entry is split between them by
    weight, so the rows add up to the pair's term and nothing is counted twice.
    """
    sizes: dict[int, float] = {}
    channels = graph["relation"].map(CHANNEL_OF_RELATION)
    for channel in _CHANNEL_ORDER:
        part = graph[channels == channel]
        if part.empty:
            continue
        nodes = sorted(set(part["source_node"]) | set(part["receiver_node"]))
        position = {name: index for index, name in enumerate(nodes)}
        adjacency, _ = build_adjacency(part, nodes)
        pair_weight = part.groupby(["source_node", "receiver_node"])["weight"].transform("sum")
        for row_id, row in part.iterrows():
            source = row["source_node"]
            if source not in shocks.index:
                continue
            entry = float(adjacency[position[row["receiver_node"]], position[source]])
            share = row["weight"] / pair_weight[row_id] if pair_weight[row_id] > 0 else 0.0
            sizes[int(row_id)] = alpha * entry * share * float(shocks[source])
    return sizes


def _escape(text: str) -> str:
    return " ".join(text.replace("|", "\\|").split())


def _evidence(row: pd.Series) -> str:
    if str(row["extractor_version"]).startswith(RULES_EXTRACTOR_PREFIX):
        return f"rule: {_escape(str(row['extractor_version']))}"
    return f'"{_escape(str(row["evidence_span"]))}"'


def _signed(value: float | None) -> str:
    return NOT_APPLICABLE if value is None else f"{value + 0.0:+.4f}"


def _link_rows(
    graph: pd.DataFrame, instrument: str, sizes: dict[int, float], moves: pd.Series
) -> dict[str, list[LinkRow]]:
    """The links whose source is `instrument`, grouped by channel and sorted for display."""
    grouped: dict[str, list[LinkRow]] = {}
    for row_id, row in graph[graph["source_node"] == instrument].iterrows():
        if row["relation"] == COUNTRY_CHANNEL:
            channel = COUNTRY_CHANNEL
        elif row["relation"] in CHANNEL_OF_RELATION:
            channel = CHANNEL_OF_RELATION[row["relation"]]
        else:
            raise DataError(f"unknown link relation {row['relation']!r}")
        move = moves.get(row["receiver_node"])
        grouped.setdefault(channel, []).append(
            LinkRow(
                receiver=row["receiver_node"],
                relation=row["relation"],
                size=sizes.get(int(row_id)) if channel != COUNTRY_CHANNEL else None,
                move=None if move is None or pd.isna(move) else float(move),
                doc_id=_escape(str(row["source_doc_id"])),
                evidence=_evidence(row),
            )
        )
    for rows in grouped.values():
        rows.sort(key=lambda r: (-abs(r.size or 0.0), r.receiver, r.relation, r.doc_id))
    return grouped


def _render_shock(instrument: str, shock: float, grouped: dict[str, list[LinkRow]]) -> list[str]:
    lines = [f"## {instrument} shock {_signed(shock)}", ""]
    if not grouped:
        return [*lines, "No links reach this name at the cutoff.", ""]
    for channel in [*_CHANNEL_ORDER, COUNTRY_CHANNEL]:
        rows = grouped.get(channel)
        if not rows:
            continue
        lines += [
            f"### {_CHANNEL_TITLES[channel]}",
            "",
            "| Reaches | Relation | Propagated size | Move already made | Source | Evidence |",
            "|---|---|---|---|---|---|",
        ]
        for row in rows:
            size = NO_FACTOR_SIZE if channel == COUNTRY_CHANNEL else _signed(row.size)
            lines.append(
                f"| -> {row.receiver} | {row.relation} | {size} | {_signed(row.move)} "
                f"| doc: {row.doc_id} | {row.evidence} |"
            )
        lines.append("")
    return lines


def render_map(
    banner: str,
    session: date,
    cutoff_utc: pd.Timestamp,
    shocks: pd.Series,
    top: int,
    graph: pd.DataFrame,
    moves: pd.Series,
    alpha: float,
    published_graph: str | None = None,
) -> str:
    """The Markdown map; deterministic for the same inputs. The header names `published_graph`."""
    header = [
        banner,
        "",
        f"# Ripple map for {session.isoformat()}",
        "",
        "History, not a forecast. Sizes are derived statistics: a shock is a market-model "
        "abnormal return, a propagated size is alpha times the receiver's normalised link "
        "weight times the shock, and the move already made is the receiver's own abnormal "
        "return on this session. No price is shown. The factor also carries a pooled hop-2 "
        "term that is not split per link here.",
        "",
        f"Links known at the cutoff {cutoff_utc.isoformat()}; alpha {alpha}."
        + (f" Link graph: {published_graph}." if published_graph else ""),
        "",
    ]
    if shocks.empty:
        return "\n".join([*header, NO_SHOCKS_LINE, ""])
    ranked = sorted(shocks.items(), key=lambda item: (-abs(item[1]), item[0]))
    shown = ranked[:top]
    body = [f"Showing {len(shown)} of {len(ranked)} shocks, largest absolute size first.", ""]
    sizes = _propagated_sizes(graph, shocks, alpha) if not graph.empty else {}
    for instrument, shock in shown:
        body += _render_shock(instrument, shock, _link_rows(graph, instrument, sizes, moves))
    return "\n".join([*header, *body])


def _build_map(session: date, top: int, cutoff_utc: pd.Timestamp) -> str:
    ripple_dir = data_dir() / "ripple"
    manifest_path = ripple_dir / LEDGER_MANIFEST_NAME
    if not manifest_path.exists():
        return "\n".join(
            [
                "NO LIVE RUN — no Ripple data has been built under the configured data directory.",
                "",
                f"# Ripple map for {session.isoformat()}",
                "",
                f"No Ripple data has been built under {ripple_dir}, so there is nothing to map.",
                "",
            ]
        )
    manifest = _read_manifest(manifest_path)
    banner = _banner_for(manifest)
    _check_date_in_range(session, manifest)
    links_path = LinkStore(ripple_dir / _LINKS_DIR).path
    if not links_path.is_file():  # an absent store would read as "no links"
        raise DataError(f"the link store {links_path} is missing")
    published_graph = _check_links_file(ripple_dir, manifest)
    stamp = pd.Timestamp(session)
    shocks = _session_shocks(_read_checked_parquet(ripple_dir, _SHOCK_FILE, manifest), stamp)
    moves = _session_moves(_read_checked_parquet(ripple_dir, _ABNORMAL_FILE, manifest), stamp)
    try:
        graph = graph_asof(LinkStore(ripple_dir / _LINKS_DIR), cutoff_utc)
    except (OSError, ValueError) as error:
        raise DataError(f"cannot read the link store {links_path}: {error}") from error
    try:
        alpha = float(load_ripple_params()["alpha"])
    except (OSError, ValueError, KeyError) as error:
        raise DataError(f"cannot read the Ripple parameters: {error}") from error
    return render_map(
        banner, session, cutoff_utc, shocks, top, graph, moves, alpha, published_graph
    )


def run(argv: list[str]) -> int:
    """Entry point for `precedent ripple`; returns the process exit code."""
    try:
        args = _build_parser().parse_args(argv)
    except SystemExit as stop:
        return stop.code if isinstance(stop.code, int) else OK
    try:
        calendar = load_nyse_calendar()
    except ValueError as error:
        print(f"data error: {error}", file=sys.stderr)
        return DATA_ERROR
    try:
        cutoff = calendar_cutoffs(pd.DatetimeIndex([pd.Timestamp(args.date)]), calendar).iloc[0]
    except ValueError as error:
        print(f"{args.date.isoformat()} is not an NYSE session: {error}", file=sys.stderr)
        return USAGE_ERROR
    try:
        text = _build_map(args.date, args.top, cutoff.tz_convert("UTC"))
    except DataError as error:
        print(f"data error: {error}", file=sys.stderr)
        return DATA_ERROR
    if args.out is None:
        sys.stdout.write(text)
    else:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            write_text_atomically(args.out, text)
        except OSError as error:
            print(f"data error: cannot write {args.out}: {error}", file=sys.stderr)
            return DATA_ERROR
    return OK
