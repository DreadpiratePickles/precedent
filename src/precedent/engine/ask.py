"""`precedent ask`: the product command line.

Card assembly is in `engine/card_build.py`; `run_card`, `refused_card_without_library` and
`UNTYPED` are re-exported here because the MCP server, the Dream brief and the tests import them
from this module.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from precedent import config
from precedent.config import ConfigError
from precedent.engine.card import (
    KEY_FLAGS_HINT,
    CardModel,
    render_json,
    render_png,
    render_text,
)
from precedent.engine.card_build import (
    ALL_OUTCOME_TYPES,
    UNTYPED,
    _check_request,
    _placebo,  # re-exported: tests reach the placebo through this module
    refused_card_without_library,
    run_card,
)
from precedent.engine.keys import EventKey, key_from_flags
from precedent.engine.params import load_engine_params
from precedent.engine.retrieval import PrecedentQuery
from precedent.libraries.build import LIBRARY_DIR_NAME
from precedent.libraries.bundle import LibraryMissingError
from precedent.libraries.common import OUTCOME_TYPES
from precedent.libraries.manifest import LibraryIntegrityError
from precedent.spine.typers.base import Typer
from precedent.spine.typers.rules import RulesTyper
from precedent.spine.typers.select import typer_from_env

__all__ = ["UNTYPED", "refused_card_without_library", "run", "run_card"]

_RE_EXPORTED = (_placebo,)
USAGE_ERROR = 2
DEFAULT_HORIZON = 5
_END_OF_DAY = dt.time(23, 59, 59)
_EXAMPLES_PARTS = ("docs", "examples")


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


def _evidence_banner(data_dir: Path, library_loaded: bool) -> str:
    """`NO LIVE RUN` with no library loaded; else `SYNTHETIC` exactly when the loaded bundle's
    build report says `"synthetic": true`, and `LIVE` otherwise (Ruling P-36)."""
    if not library_loaded:
        return "NO LIVE RUN"
    try:
        report = json.loads((data_dir / LIBRARY_DIR_NAME / "build_report.json").read_text("utf-8"))
    except ValueError:  # not JSON: not a synthetic marker
        return "LIVE"
    synthetic = isinstance(report, dict) and report.get("synthetic") is True
    return "SYNTHETIC" if synthetic else "LIVE"


def _display_path(path: Path) -> str:
    """The path with the home directory written as `~`, so no personal path is committed."""
    resolved = path.expanduser().resolve()
    try:
        return "~/" + resolved.relative_to(Path.home().resolve()).as_posix()
    except ValueError:
        return str(path)


def _write_sidecar(png: Path, card: CardModel, data_dir: Path, library_loaded: bool) -> None:
    banner = _evidence_banner(data_dir, library_loaded)
    text = (
        f"{banner} — precedent ask card\n"
        f"query: {card.headline}; data dir: {_display_path(data_dir)}\n"
    )
    png.with_suffix(".md").write_text(text, encoding="utf-8")


def _output(
    card: CardModel, args: argparse.Namespace, data_dir: Path, library_loaded: bool
) -> int:
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
        _write_sidecar(args.card, card, data_dir, library_loaded)
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
    library_loaded = True
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
        library_loaded = False
        card = refused_card_without_library(
            query, args.horizon, args.outcome_type, typer, asof_utc=asof
        )
    except LibraryIntegrityError as error:
        print(f"error: the library is not intact: {error}", file=sys.stderr)
        return USAGE_ERROR
    except ConfigError as error:
        print(f"config error: {error}\n{KEY_FLAGS_HINT}", file=sys.stderr)
        return USAGE_ERROR
    return _output(card, args, data_dir, library_loaded)
