"""`precedent ask`: the product command, and `run_card`, the function other tracks call.

`run_card` loads the library bundle, types the text (if it is text), selects the past events that
match the key, and builds a `CardModel` from their outcomes. Nothing here forecasts: every number
is a summary of what happened after similar events, behind the time guard of `find_precedents`.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import ValidationError

from precedent import config
from precedent.config import ConfigError
from precedent.engine.aggregate import PrecedentSummary, effective_sample_size, summarize_responses
from precedent.engine.card import (
    KEY_FLAGS_HINT,
    CardModel,
    CardRow,
    NearestPrecedent,
    SourceCoverage,
    render_json,
    render_png,
    render_text,
)
from precedent.engine.keys import EventKey, key_from_flags, typed_as_line
from precedent.engine.params import EngineParams, load_engine_params
from precedent.engine.retrieval import PrecedentQuery, find_precedents, select_precedent_events
from precedent.libraries.build import LIBRARY_DIR_NAME
from precedent.libraries.bundle import LibraryBundle, LibraryMissingError, load_bundle
from precedent.libraries.common import OUTCOME_TYPES
from precedent.libraries.event_sequence import unconditional_sequence_rate
from precedent.libraries.manifest import LibraryIntegrityError
from precedent.libraries.market_resolution import unconditional_market_rate
from precedent.spine.event_key import load_country_groups
from precedent.spine.typers.base import Typer
from precedent.spine.typers.rules import RulesTyper
from precedent.spine.typers.select import typer_from_env

__all__ = ["refused_card_without_library", "run", "run_card"]

USAGE_ERROR = 2
UNTYPED = "UNTYPED"
ALL_OUTCOME_TYPES = "all"
DEFAULT_HORIZON = 5
_END_OF_DAY = dt.time(23, 59, 59)
_EXAMPLES_PARTS = ("docs", "examples")
_NO_PRICES = "no prices"
_NO_LIBRARY_ROWS = "no library rows"


# --- the card -----------------------------------------------------------------------------------


def _check_request(outcome_type: str, horizon: int, params: EngineParams) -> None:
    if outcome_type != ALL_OUTCOME_TYPES and outcome_type not in OUTCOME_TYPES:
        raise ValueError(
            f"unknown outcome_type {outcome_type!r}; expected one of "
            f"{(ALL_OUTCOME_TYPES, *OUTCOME_TYPES)}"
        )
    allowed = sorted(set(params.asset_horizons) | set(params.sequence_horizons))
    if horizon not in allowed:
        raise ValueError(f"unsupported horizon {horizon!r}; expected one of {allowed}")


def _as_date(value: object) -> dt.date:
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def _coverage(bundle: LibraryBundle) -> list[SourceCoverage]:
    return [
        SourceCoverage(name=s.name, first_date=s.first_date, last_date=s.last_date, rows=s.rows)
        for s in bundle.manifest.sources
    ]


def _card(
    *,
    params: EngineParams,
    headline: str,
    typed_as: str,
    typed_reason: str | None,
    key: str | None,
    asof_utc: pd.Timestamp,
    horizon: int,
    outcome_type: str,
    n: int = 0,
    n_eff: float = 0.0,
    earliest: dt.date | None = None,
    latest: dt.date | None = None,
    rows: list[CardRow] | None = None,
    not_built: dict[str, str] | None = None,
    nearest: list[NearestPrecedent] | None = None,
    covers: list[SourceCoverage] | None = None,
    hint: str | None = None,
) -> CardModel:
    return CardModel(
        headline=headline,
        typed_as=typed_as,
        typed_reason=typed_reason,
        key=key,
        asof_utc=asof_utc.to_pydatetime(),
        horizon=horizon,
        outcome_type=outcome_type,
        n=n,
        n_eff=n_eff,
        earliest=earliest,
        latest=latest,
        refused=not rows,
        min_neff=params.min_neff,
        rows=rows or [],
        not_built=not_built or {},
        nearest=nearest or [],
        covers=covers or [],
        hint=hint,
    )


def _untyped_card(
    text: str, reason: str, asof: pd.Timestamp, horizon: int, outcome_type: str, covers: list
) -> CardModel:
    return _card(
        params=load_engine_params(),
        headline=text,
        typed_as=UNTYPED,
        typed_reason=reason,
        key=None,
        asof_utc=asof,
        horizon=horizon,
        outcome_type=outcome_type,
        covers=covers,
        hint=KEY_FLAGS_HINT,
    )


def _resolve_query(
    query: PrecedentQuery | str, typer: Typer, asof_utc: pd.Timestamp | None
) -> tuple[PrecedentQuery | None, str, str | None, str | None, pd.Timestamp]:
    """(query or None when untyped, headline, typed_reason, untyped reason, asof)."""
    if isinstance(query, PrecedentQuery):
        # a caller may hand over a plain `datetime` (the card's own `asof_utc`); use a Timestamp
        available = pd.Timestamp(query.available_utc)
        query = dataclasses.replace(query, available_utc=available)
        return query, typed_as_line(query.key), None, None, available
    asof = asof_utc if asof_utc is not None else pd.Timestamp.now(tz="UTC")
    typed = typer.type_text(query)
    if typed.status != "typed" or typed.key is None:
        return None, query, typed.reason, typed.reason, asof
    return PrecedentQuery(typed.key, asof), query, typed.reason, None, asof


def _check_asof(asof_utc: pd.Timestamp | None) -> None:
    if asof_utc is not None and (asof_utc is pd.NaT or asof_utc.tzinfo is None):
        raise ValueError(f"asof_utc must be a timezone-aware timestamp, got {asof_utc!r}")


def run_card(
    query: PrecedentQuery | str,
    horizon: int,
    outcome_type: str,
    data_dir: Path,
    typer: Typer,
    *,
    asof_utc: pd.Timestamp | None = None,
) -> CardModel:
    """The precedent card for `query` at `horizon` from the library under `data_dir / library`.

    A `str` is typed with `typer`; text the typer cannot type gives the untyped card (refused,
    with the key-flags hint). `asof_utc` is the moment a text query became available (default
    now); a `PrecedentQuery` carries its own. Raises `ValueError` for an unknown `outcome_type`
    or an unsupported `horizon`, `LibraryMissingError` or `LibraryIntegrityError` for a library
    that is absent or altered (the library is read before any text is typed).
    """
    params = load_engine_params()
    _check_request(outcome_type, horizon, params)
    _check_asof(asof_utc)
    bundle = load_bundle(data_dir / LIBRARY_DIR_NAME)
    precedent_query, headline, typed_reason, untyped_reason, asof = _resolve_query(
        query, typer, asof_utc
    )
    if precedent_query is None:
        assert untyped_reason is not None
        return _untyped_card(
            headline, untyped_reason, asof, horizon, outcome_type, _coverage(bundle)
        )
    return _precedent_card(
        precedent_query, headline, typed_reason, horizon, outcome_type, bundle, params
    )


def refused_card_without_library(
    query: PrecedentQuery | str,
    horizon: int,
    outcome_type: str,
    typer: Typer,
    *,
    asof_utc: pd.Timestamp | None = None,
) -> CardModel:
    """The refused card for a query when there is no library to search (`covers` is empty)."""
    params = load_engine_params()
    _check_request(outcome_type, horizon, params)
    _check_asof(asof_utc)
    precedent_query, headline, typed_reason, untyped_reason, asof = _resolve_query(
        query, typer, asof_utc
    )
    if precedent_query is None:
        assert untyped_reason is not None
        return _untyped_card(headline, untyped_reason, asof, horizon, outcome_type, [])
    return _card(
        params=params,
        headline=headline,
        typed_as=typed_as_line(precedent_query.key),
        typed_reason=typed_reason,
        key=precedent_query.key.as_string(),
        asof_utc=asof,
        horizon=horizon,
        outcome_type=outcome_type,
    )


def _precedent_card(
    query: PrecedentQuery,
    headline: str,
    typed_reason: str | None,
    horizon: int,
    outcome_type: str,
    bundle: LibraryBundle,
    params: EngineParams,
) -> CardModel:
    selected = select_precedent_events(query, bundle.events, top_k=params.top_k)
    n = len(selected)
    positions = selected["effective_position"].to_numpy()
    n_eff = effective_sample_size(np.ones(n), positions, horizon)
    refused = n_eff < params.min_neff
    sessions = [_as_date(value) for value in selected["effective_session"]]
    nearest = [
        NearestPrecedent(
            date=_as_date(event.effective_session),
            one_line=event.one_line,
            similarity=float(event.similarity),
            source_ref=event.source_ref,
        )
        for event in selected.head(params.nearest_shown).itertuples(index=False)
    ]
    rows: list[CardRow] = []
    not_built: dict[str, str] = {}
    if not refused:
        rows, not_built = _outcome_rows(query, selected, horizon, outcome_type, bundle, params)
    card = _card(
        params=params,
        headline=headline,
        typed_as=typed_as_line(query.key),
        typed_reason=typed_reason,
        key=query.key.as_string(),
        asof_utc=query.available_utc,
        horizon=horizon,
        outcome_type=outcome_type,
        n=n,
        n_eff=n_eff,
        earliest=min(sessions) if sessions else None,
        latest=max(sessions) if sessions else None,
        rows=rows,
        not_built=not_built,
        nearest=nearest,
        covers=_coverage(bundle),
    )
    # `_card` marks a card refused when it has no rows; an unrefused card with no row to show
    # (every outcome of the type still pending) stays unrefused.
    return card.model_copy(update={"refused": bool(refused)})


# --- outcome rows -------------------------------------------------------------------------------


def _outcome_rows(
    query: PrecedentQuery,
    selected: pd.DataFrame,
    horizon: int,
    outcome_type: str,
    bundle: LibraryBundle,
    params: EngineParams,
) -> tuple[list[CardRow], dict[str, str]]:
    asked = OUTCOME_TYPES if outcome_type == ALL_OUTCOME_TYPES else (outcome_type,)
    position_of = selected.set_index("event_id")["effective_position"]
    rows: list[CardRow] = []
    not_built: dict[str, str] = {}
    for kind in asked:
        if not (bundle.rows["outcome_type"] == kind).any():
            not_built[kind] = _NO_PRICES if kind == "asset" else _NO_LIBRARY_ROWS
            continue
        found = find_precedents(
            query, bundle.events, bundle.rows, bundle.session_cutoffs_utc, horizon, kind,
            top_k=params.top_k,
        )
        found = found.assign(window_start=found["event_id"].map(position_of))
        of_type = [
            _card_row(kind, key, group, horizon, query, bundle, params)
            for key, group in found.groupby("outcome_key", sort=True)
        ]
        of_type.sort(key=lambda row: (-row.n, row.outcome_key))
        rows.extend(of_type[: params.card_rows_per_type])
    return rows, not_built


def _card_row(
    kind: str,
    outcome_key: str,
    group: pd.DataFrame,
    horizon: int,
    query: PrecedentQuery,
    bundle: LibraryBundle,
    params: EngineParams,
) -> CardRow:
    is_binary = kind != "asset"
    summary = summarize_responses(
        group["value"].to_numpy(),
        np.ones(len(group)),
        group["window_start"].to_numpy(),
        horizon,
        _placebo(kind, outcome_key, horizon, query, bundle),
        is_binary=is_binary,
        shrink_k=params.shrink_k,
        min_neff=params.min_neff,
        z=params.wilson_z,
    )
    row_horizon = None if kind == "market_resolution" else horizon
    common = {
        "outcome_type": kind,
        "outcome_key": outcome_key,
        "horizon": row_horizon,
        "is_binary": is_binary,
        "n": summary.n,
        "n_eff": summary.n_eff,
        "refused": summary.refused,
    }
    if summary.refused:
        return CardRow(
            **common, k=summary.k, rate=None, interval=None, median=None, centre_shrunk=None,
            band=None, vs_ordinary=None,
        )
    return _summary_row(common, summary)


def _summary_row(common: dict, summary: PrecedentSummary) -> CardRow:
    vs_ordinary = None
    if summary.inside_ordinary_range is not None:
        vs_ordinary = "inside" if summary.inside_ordinary_range else "outside"
    if summary.mode == "binary":
        return CardRow(
            **common, k=summary.k, rate=summary.rate, interval=summary.interval, median=None,
            centre_shrunk=None, band=None, vs_ordinary=vs_ordinary,
        )
    assert summary.quantiles is not None
    return CardRow(
        **common, k=None, rate=None, interval=None, median=summary.median,
        centre_shrunk=summary.centre_shrunk, band=(summary.quantiles[0.1], summary.quantiles[0.9]),
        vs_ordinary=vs_ordinary,
    )


def _placebo(
    kind: str, outcome_key: str, horizon: int, query: PrecedentQuery, bundle: LibraryBundle
) -> np.ndarray | float | None:
    """The ordinary range for a row: what happens regardless of the event, known by `asof`."""
    if kind == "asset":
        return _asset_placebo(outcome_key, horizon, query.available_utc, bundle)
    if kind == "market_resolution":
        return unconditional_market_rate(
            bundle.markets, outcome_key, query.available_utc, bundle.session_cutoffs_utc
        )
    closed = int((bundle.session_cutoffs_utc <= query.available_utc).sum())
    if closed == 0:
        return None
    return unconditional_sequence_rate(
        bundle.events,
        outcome_key.split("/")[0],
        query.key.country_group,
        horizon,
        closed - 1,
        load_country_groups(),
    )


def _asset_placebo(
    outcome_key: str, horizon: int, available_utc: pd.Timestamp, bundle: LibraryBundle
) -> np.ndarray | None:
    placebo = bundle.placebo_asset
    chosen = placebo.loc[(placebo["outcome_key"] == outcome_key) & (placebo["horizon"] == horizon)]
    if chosen.empty:
        return None
    sessions = pd.DatetimeIndex(chosen["response_complete_session"])
    closed = (bundle.session_cutoffs_utc.loc[sessions] <= available_utc).to_numpy()
    values = chosen["value"].to_numpy()[closed]
    return values if values.size else None


# --- the command --------------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precedent ask",
        description="Past outcomes of events like this one. History, not a forecast.",
    )
    parser.add_argument("text", nargs="?", help="the event, in words")
    key = parser.add_argument_group("or the event as a key")
    key.add_argument("--event-type")
    key.add_argument("--actor-kind")
    key.add_argument("--severity-band")
    schedule = key.add_mutually_exclusive_group()
    schedule.add_argument("--scheduled", dest="is_scheduled", action="store_const", const=True)
    schedule.add_argument("--unscheduled", dest="is_scheduled", action="store_const", const=False)
    key.add_argument("--country-group")
    key.add_argument("--direction-sign", type=int, choices=(-1, 0, 1))
    parser.add_argument("--asof", help="ISO date (end of that day, UTC) or datetime; default now")
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--outcome-type", default=ALL_OUTCOME_TYPES, choices=(ALL_OUTCOME_TYPES, *OUTCOME_TYPES)
    )
    parser.add_argument("--json", action="store_true", help="print the card as JSON")
    parser.add_argument("--card", type=Path, metavar="PATH.png", help="also draw the card as PNG")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true", help="rules typer, bundled sample")
    return parser


def _key_from_args(args: argparse.Namespace) -> EventKey | None:
    """The key the flags describe; None when no key flag was given. Raises `ValueError`."""
    given = {
        "--event-type": args.event_type,
        "--actor-kind": args.actor_kind,
        "--severity-band": args.severity_band,
        "--scheduled|--unscheduled": args.is_scheduled,
    }
    optional_given = args.country_group is not None or args.direction_sign is not None
    if not optional_given and all(value is None for value in given.values()):
        return None
    missing = [flag for flag, value in given.items() if value is None]
    if missing:
        raise ValueError(f"a key query also needs {', '.join(missing)}")
    return key_from_flags(
        args.event_type,
        args.actor_kind,
        args.severity_band,
        args.is_scheduled,
        country_group=args.country_group or "none",
        direction_sign=args.direction_sign or 0,
    )


def _parse_asof(text: str) -> pd.Timestamp:
    try:
        day = dt.date.fromisoformat(text)
    except ValueError:
        day = None
    if day is not None:
        return pd.Timestamp(dt.datetime.combine(day, _END_OF_DAY, tzinfo=dt.UTC))
    try:
        stamp = pd.Timestamp(text)
    except ValueError as error:
        raise ValueError(f"--asof is not an ISO date or datetime: {text!r}") from error
    if stamp is pd.NaT:
        raise ValueError(f"--asof is not an ISO date or datetime: {text!r}")
    return stamp.tz_localize("UTC") if stamp.tzinfo is None else stamp.tz_convert("UTC")


def _usage_error(message: str) -> int:
    print(f"usage error: {message}", file=sys.stderr)
    return USAGE_ERROR


def _is_under_docs_examples(path: Path) -> bool:
    parts = path.resolve().parts
    return any(parts[i : i + 2] == _EXAMPLES_PARTS for i in range(len(parts) - 1))


def _is_synthetic_library(data_dir: Path, dry_run: bool) -> bool:
    """A sample or fixture library: the dry run, the sample dir, or a build report that says so."""
    if dry_run or data_dir.resolve() == config.sample_dir().resolve():
        return True
    report = json.loads((data_dir / LIBRARY_DIR_NAME / "build_report.json").read_text("utf-8"))
    return report.get("synthetic") is True


def _display_path(path: Path) -> str:
    """The path with the home directory written as `~`, so no personal path is committed."""
    resolved = path.expanduser().resolve()
    try:
        return "~/" + resolved.relative_to(Path.home().resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_sidecar(png: Path, card: CardModel, data_dir: Path, dry_run: bool) -> None:
    banner = "SYNTHETIC" if _is_synthetic_library(data_dir, dry_run) else "LIVE"
    text = (
        f"{banner} — precedent ask card\n"
        f"query: {card.headline}; data dir: {_display_path(data_dir)}\n"
    )
    png.with_suffix(".md").write_text(text, encoding="utf-8")


def _output(card: CardModel, args: argparse.Namespace, data_dir: Path) -> int:
    print(render_json(card) if args.json else render_text(card))
    if card.typed_as == UNTYPED:
        print(KEY_FLAGS_HINT, file=sys.stderr)
    if args.card is None:
        return 0
    try:
        render_png(card, args.card)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return USAGE_ERROR
    if _is_under_docs_examples(args.card):
        _write_sidecar(args.card, card, data_dir, args.dry_run)
    return 0


def _typer_for(args: argparse.Namespace) -> Typer:
    if args.dry_run:
        return RulesTyper()
    return typer_from_env()


def run(argv: list[str]) -> int:
    """Print the card for TEXT or a key; 0 for any card, 2 when no card could be made."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_request:
        return exit_request.code if isinstance(exit_request.code, int) else USAGE_ERROR
    try:
        key = _key_from_args(args)
        asof = _parse_asof(args.asof) if args.asof is not None else None
    except ValueError as error:
        return _usage_error(str(error))
    if (key is None) == (args.text is None):
        return _usage_error("give either TEXT or a key (--event-type, --actor-kind, "
                            "--severity-band, --scheduled|--unscheduled), not both and not neither")
    try:
        params = load_engine_params()
        _check_request(args.outcome_type, args.horizon, params)
    except (ValueError, ValidationError, OSError) as error:
        return _usage_error(" ".join(str(error).split()))
    if key is not None:
        query: PrecedentQuery | str = PrecedentQuery(
            key, asof if asof is not None else pd.Timestamp.now(tz="UTC")
        )
        typer: Typer = RulesTyper()  # a key needs no typing; this one is never called
    else:
        query = args.text
        try:
            typer = _typer_for(args)
        except ConfigError as error:
            print(f"config error: {error}\n{KEY_FLAGS_HINT}", file=sys.stderr)
            return USAGE_ERROR
    if args.data_dir is not None:
        data_dir = args.data_dir
    else:
        data_dir = config.sample_dir() if args.dry_run else config.data_dir()
    try:
        card = run_card(query, args.horizon, args.outcome_type, data_dir, typer, asof_utc=asof)
    except LibraryMissingError as error:
        if not args.dry_run:
            print(
                f"error: {error}\nget a library with `precedent fetch` (the published snapshot) "
                "or `precedent build` (from your own stores), or try the bundled sample with "
                "`precedent ask --dry-run`",
                file=sys.stderr,
            )
            return USAGE_ERROR
        card = refused_card_without_library(
            query, args.horizon, args.outcome_type, typer, asof_utc=asof
        )
    except LibraryIntegrityError as error:
        print(f"error: the library is not intact: {error}", file=sys.stderr)
        return USAGE_ERROR
    except ConfigError as error:
        print(f"config error: {error}\n{KEY_FLAGS_HINT}", file=sys.stderr)
        return USAGE_ERROR
    return _output(card, args, data_dir)
