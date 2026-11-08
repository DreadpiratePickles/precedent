"""Milestone R-M1: the graph ladder with its placebos and diffusion curve (an owner-gated run).

Reads prices (Parquet provider) and `rules-8k-v1` events, builds price shocks, the sector graph
(G1) and the rolling correlation graph (G2), shuffles each into its degree-preserving placebo,
and scores every graph at 1, 5, 20 and 60 sessions beside a price-only baseline. This file only
parses arguments and wires library calls; the computation lives in `precedent.ripple`.

The real run on live prices is an owner gate. `--dry-run` checks that every input is present,
prints the plan and writes nothing. Qlib starts worker processes, so under the macOS `spawn`
start method this module must only run under `if __name__ == "__main__"`.

Exit codes: 0 success, 1 data error (message on stderr), 2 usage error.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from precedent import config
from precedent.config import ConfigError
from precedent.ripple.control_graphs import sector_links
from precedent.ripple.graph_builders import (
    EVIDENCE_LABELS,
    abnormal_return_frame,
    placebo_graph_for_cutoff,
    rolling_correlation_links,
    snapshot_graph_for_cutoff,
    write_ripple_ledger,
)
from precedent.ripple.link_store import LinkStore
from precedent.ripple.params import load_ripple_params
from precedent.ripple.shocks import price_shocks
from precedent.spine.edgar_8k_events import edgar_parser_version
from precedent.spine.event_store import EventStore
from precedent.spine.price_provider import ParquetPriceProvider, load_returns
from precedent.spine.session_calendar import calendar_cutoffs, load_nyse_calendar

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_CHOICES = ("g1", "g2")
DEFAULT_HORIZONS = "1,5,20,60"
# Proposals, like every Ripple parameter: nothing here is tuned on a result.
DEFAULT_LGB_PARAMS: dict[str, Any] = {
    "n_estimators": 200,
    "num_leaves": 31,
    "learning_rate": 0.05,
    "min_child_samples": 50,
    "n_jobs": 1,
    "random_state": 7,
    "verbose": -1,
}
DEFAULT_FOLD_SESSIONS = {"train_len": 500, "valid_len": 125, "test_len": 125, "step": 125}


class InputError(RuntimeError):
    """An input is missing or unusable; the message says which."""


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not an integer") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"{text!r} must be at least 1")
    return value


def _horizons(text: str) -> tuple[int, ...]:
    return tuple(_positive_int(part) for part in text.split(",") if part.strip())


def _graphs(text: str) -> tuple[str, ...]:
    chosen = tuple(part.strip() for part in text.split(",") if part.strip())
    unknown = [name for name in chosen if name not in GRAPH_CHOICES]
    if not chosen or unknown:
        raise argparse.ArgumentTypeError(f"graphs must be a comma list from {GRAPH_CHOICES}")
    return chosen


def _iso_date(text: str) -> str:
    try:
        return dt.date.fromisoformat(text).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"{text!r} is not a YYYY-MM-DD date") from None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run_ripple_m1.py", description=__doc__.split("\n")[0])
    parser.add_argument("--provider-dir", type=Path, help="Parquet prices (default <data>/prices)")
    parser.add_argument("--qlib-dir", type=Path, help="Qlib provider (default <data>/qlib_us)")
    parser.add_argument("--events-dir", type=Path, help="event store (default <data>/events)")
    parser.add_argument("--parser-version", help="event parser version (default: the 8-K rules)")
    parser.add_argument("--sector-map", type=Path, help='JSON {"INSTRUMENT": "sector"} for G1')
    parser.add_argument("--start", type=_iso_date, required=True)
    parser.add_argument("--end", type=_iso_date, required=True)
    parser.add_argument("--market", default="SPY", help="market instrument (default SPY)")
    parser.add_argument("--out-dir", type=Path, help="default stages/32_ripple_evaluate/output/ID")
    parser.add_argument("--rebuild-every", type=_positive_int, default=20)
    parser.add_argument("--corr-window", type=_positive_int, help="G2 window (default: params)")
    parser.add_argument("--corr-top-k", type=_positive_int, help="G2 peers (default: params)")
    parser.add_argument("--seed", type=int, default=7, help="placebo shuffle seed")
    parser.add_argument("--horizons", type=_horizons, default=_horizons(DEFAULT_HORIZONS))
    parser.add_argument("--graphs", type=_graphs, default=GRAPH_CHOICES, help="default g1,g2")
    for name, default in DEFAULT_FOLD_SESSIONS.items():
        parser.add_argument(f"--{name.replace('_', '-')}", type=_positive_int, default=default)
    parser.add_argument("--evidence-label", choices=EVIDENCE_LABELS, default="LIVE-DATA")
    parser.add_argument("--publish", action="store_true", help="write <data>/ripple/ ledger files")
    parser.add_argument("--dry-run", action="store_true", help="check inputs, write nothing")
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _resolve_defaults(args: argparse.Namespace) -> None:
    data = config.data_dir()
    args.provider_dir = args.provider_dir or data / "prices"
    args.qlib_dir = args.qlib_dir or data / "qlib_us"
    args.events_dir = args.events_dir or data / "events"
    args.parser_version = args.parser_version or edgar_parser_version()
    args.run_id = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    default_out = REPO_ROOT / "stages" / "32_ripple_evaluate" / "output" / args.run_id
    args.out_dir = args.out_dir or default_out
    params = load_ripple_params()
    args.corr_window = args.corr_window or params["correlation_window_sessions"]
    args.corr_top_k = args.corr_top_k or params["correlation_top_k"]
    args.ripple_params = params


def _read_sector_map(path: Path) -> dict[str, str]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise InputError(f"cannot read the sector map {path}: {error}") from error
    if not isinstance(raw, dict) or not raw:
        raise InputError(f"the sector map {path} must be a non-empty JSON object")
    if not all(isinstance(k, str) and isinstance(v, str) for k, v in raw.items()):
        raise InputError(f"the sector map {path} must map instrument names to sector names")
    return raw


def check_inputs(args: argparse.Namespace) -> list[str]:
    """Raise InputError for the first unusable input; otherwise return the plan, one line each."""
    if args.start > args.end:
        raise InputError(f"--start {args.start} is after --end {args.end}")
    symbols = sorted(path.stem for path in args.provider_dir.glob("*.parquet"))
    if not symbols:
        raise InputError(f"no price files (*.parquet) in {args.provider_dir}")
    if args.market not in symbols:
        raise InputError(f"market {args.market!r} has no price file in {args.provider_dir}")
    if not (args.qlib_dir / "calendars" / "day.txt").is_file():
        raise InputError(f"{args.qlib_dir} is not a Qlib provider (no calendars/day.txt)")
    events_file = EventStore(args.events_dir).path_for(args.parser_version)
    if not events_file.is_file():
        raise InputError(
            f"no events for parser version {args.parser_version!r} in {args.events_dir} "
            f"(expected {events_file.name})"
        )
    if "g1" in args.graphs:
        if args.sector_map is None:
            raise InputError("graph g1 needs --sector-map")
        _read_sector_map(args.sector_map)
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise InputError(f"{args.out_dir} already exists and is not empty; runs never overwrite")
    return [
        f"prices: {len(symbols)} instruments, market {args.market}, {args.start} to {args.end}",
        f"events: parser version {args.parser_version}",
        f"graphs: {', '.join(f'{name} and {name}_placebo' for name in args.graphs)}"
        f" (placebo seed {args.seed}; G2 window {args.corr_window}, top {args.corr_top_k},"
        f" rebuilt every {args.rebuild_every} sessions)",
        f"horizons (sessions): {', '.join(str(h) for h in args.horizons)}; baseline: M0 price only",
        f"label: {args.evidence_label}; ledger files written: {'yes' if args.publish else 'no'}",
        f"output: {args.out_dir.name}/ (report.md, ladder.csv, links/, per-run manifests)",
    ]


def _build_graphs(
    args: argparse.Namespace, returns: Any, cutoffs: Any, symbols: Sequence[str]
) -> tuple[dict[str, Callable], dict[str, int], dict[str, str]]:
    """The named graph functions (each graph, then its placebo), link counts and store hashes."""
    graphs: dict[str, Callable] = {}
    counts: dict[str, int] = {}
    store_hashes: dict[str, str] = {}
    after_last = cutoffs.iloc[-1].to_pydatetime() + dt.timedelta(days=1)
    for name in args.graphs:
        store = LinkStore(args.out_dir / "links" / name)
        if name == "g1":
            sector_of = {
                instrument: sector
                for instrument, sector in _read_sector_map(args.sector_map).items()
                if instrument in symbols
            }
            links = sector_links(sector_of, cutoffs.iloc[0].to_pydatetime(), after_last)
            changes = [cutoffs.iloc[0]]
            counts["instruments with a sector"] = len(sector_of)
        else:
            links, changes = rolling_correlation_links(
                returns,
                args.market,
                cutoffs,
                args.corr_window,
                args.corr_top_k,
                args.rebuild_every,
            )
            counts["g2 snapshots"] = len(changes)
        store.append(links)
        counts[f"{name} links"] = len(links)
        store_hashes[name] = _sha256(store.path)
        base = snapshot_graph_for_cutoff(store, changes)
        graphs[name] = base
        graphs[f"{name}_placebo"] = placebo_graph_for_cutoff(base, args.seed)
    return graphs, counts, store_hashes


def run(
    args: argparse.Namespace,
    price_loader_factory: Callable[[int], Any] | None,
    lgb_params: Mapping[str, Any] | None,
) -> int:
    plan = check_inputs(args)
    if args.dry_run:
        print("dry run: nothing is read beyond file names and nothing is written")
        print("\n".join(plan))
        return 0
    # Research imports are deferred so --dry-run works on a core install.
    from precedent.ripple.evaluate import (
        evaluate_graph_ladder,
        label_expr_for,
        render_report,
        write_text_atomically,
    )
    from precedent.spine.ablation import code_commit
    from precedent.spine.purged_splits import purged_walk_forward_segments
    from precedent.spine.qlib_dataset import make_alpha158_price_loader
    from precedent.spine.qlib_provider import init_qlib

    provider = ParquetPriceProvider(args.provider_dir)
    symbols = sorted(path.stem for path in args.provider_dir.glob("*.parquet"))
    sessions = provider.sessions(args.start, args.end)
    if len(sessions) == 0:
        raise InputError(f"no sessions between {args.start} and {args.end}")
    returns = load_returns(symbols, args.start, args.end, "close_to_close", provider)
    if not returns.index.equals(sessions):
        raise InputError("returns and sessions disagree; the price files have uneven calendars")
    cutoffs = calendar_cutoffs(sessions, load_nyse_calendar())
    frame, report = EventStore(args.events_dir).to_frame(args.parser_version)
    in_window = frame["effective_session"].isin(sessions)
    shocks, dropped = price_shocks(frame[in_window], returns, args.market, sessions)
    if shocks.empty:
        raise InputError("no price shock could be built from the events; nothing to evaluate")
    abnormal = abnormal_return_frame(returns, args.market, sessions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    graphs, link_counts, store_hashes = _build_graphs(args, returns, cutoffs, symbols)
    init_qlib(args.qlib_dir, args.out_dir / "qlib_work")
    loader_factory = price_loader_factory or (
        lambda horizon: make_alpha158_price_loader(label_expr_for(horizon))
    )
    params = args.ripple_params
    fold_sessions = {name: getattr(args, name) for name in DEFAULT_FOLD_SESSIONS}
    ladder = evaluate_graph_ladder(
        graphs,
        shocks,
        loader_factory,
        lambda label_horizon: purged_walk_forward_segments(
            sessions, label_horizon=label_horizon, **fold_sessions
        ),
        args.out_dir,
        args.horizons,
        session_cutoffs_utc=cutoffs,
        sessions=sessions,
        abnormal_returns=abnormal,
        lgb_params=lgb_params or DEFAULT_LGB_PARAMS,
        rungs_dir=config.references_dir() / "rungs",
        parser_version=args.parser_version,
        alpha=params["alpha"],
        hops=params["hops"],
        half_life_sessions=params["half_life_sessions"],
        max_age_sessions=params["max_age_sessions"],
    )
    counts = {
        "sessions": len(sessions),
        "instruments": len(symbols),
        "event rows": report.impact_rows,
        "event rows outside the session window": int((~in_window).sum()),
        "shocks": len(shocks),
        "shocks dropped (no column or no fit)": dropped,
        **link_counts,
    }
    notes = [
        f"parser version: {args.parser_version}",
        f"placebo seed: {args.seed}; G2 window {args.corr_window}, top {args.corr_top_k}, "
        f"rebuilt every {args.rebuild_every} sessions",
        f"fold sessions: {fold_sessions}; baseline price only (M0)",
        f"ripple parameters (proposed): {json.dumps(params, sort_keys=True)}",
    ]
    write_text_atomically(
        args.out_dir / "report.md",
        render_report(ladder, evidence_label=args.evidence_label, counts=counts, notes=notes),
    )
    manifest = {
        "run_id": args.run_id,
        "evidence_label": args.evidence_label,
        "parser_version": args.parser_version,
        "start": args.start,
        "end": args.end,
        "market": args.market,
        "seed": args.seed,
        "horizons": list(args.horizons),
        "graphs": list(args.graphs),
        "counts": counts,
        "links_sha256": store_hashes,
        "events_file_sha256": _sha256(EventStore(args.events_dir).path_for(args.parser_version)),
        "code_commit": code_commit(REPO_ROOT),
    }
    write_text_atomically(
        args.out_dir / "run_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    if args.publish:
        ledger = write_ripple_ledger(config.data_dir() / "ripple", shocks, abnormal, manifest)
        print(f"ledger written: {ledger.parent.name}/{ledger.name} and its parquet files")
    print(f"report: {args.out_dir.name}/report.md")
    return 0


def main(
    argv: Sequence[str] | None = None,
    *,
    price_loader_factory: Callable[[int], Any] | None = None,
    lgb_params: Mapping[str, Any] | None = None,
) -> int:
    """Run the milestone. The two keyword seams let tests use a small loader and few trees."""
    args = build_parser().parse_args(argv)  # exits 2 on a usage error
    try:
        _resolve_defaults(args)
        return run(args, price_loader_factory, lgb_params)
    except (InputError, ConfigError, ValueError, OSError) as error:
        print(f"run_ripple_m1: {type(error).__name__}: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
