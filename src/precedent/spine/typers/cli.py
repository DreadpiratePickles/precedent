"""`precedent typers score`: score a typer on the gold set and write a bannered report."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from precedent.config import ConfigError
from precedent.spine.typers.score import (
    Banner,
    GoldCase,
    TypingScore,
    load_gold_cases,
    render_score_markdown,
    score_typer,
)
from precedent.spine.typers.select import TYPER_NAMES, typer_by_name

USAGE_ERROR = 2
_STATUS_PATH = Path("docs/STATUS.md")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="precedent typers")
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("score", help="score a typer on the gold set")
    score.add_argument("--typer", choices=TYPER_NAMES, default="rules")
    score.add_argument("--cases", type=Path, default=None, help="gold case file")
    score.add_argument("--split", choices=("dev", "heldout", "all"), default="all")
    score.add_argument("--out", type=Path, default=None)
    score.add_argument("--max-calls", type=int, default=None, help="required for model typers")
    score.add_argument("--dry-run", action="store_true", help="model typers: open no socket")
    return parser


def _banner(typer_name: str, dry_run: bool) -> Banner:
    if typer_name == "rules":
        return "SYNTHETIC"
    return "NO LIVE RUN" if dry_run else "LIVE"


def _split_scores(typer, cases: list[GoldCase], split: str) -> dict[str, TypingScore]:
    if split != "all":
        return {split: score_typer(typer, [c for c in cases if c.split == split])}
    return {
        "dev": score_typer(typer, [c for c in cases if c.split == "dev"]),
        "heldout": score_typer(typer, [c for c in cases if c.split == "heldout"]),
        "all": score_typer(typer, cases),
    }


def _print_scores(scores: dict[str, TypingScore]) -> None:
    for name, score in scores.items():
        print(f"{name}: {score.n_cases} cases, typer {score.typer_version}")
        print(f"  {'field':<16}{'precision':>10}{'recall':>8}")
        for field, value in [("status", score.status), *score.fields.items()]:
            print(f"  {field:<16}{value.precision:>10.3f}{value.recall:>8.3f}")


def _record_live_run(typer_name: str, version: str, out: Path) -> None:
    if not _STATUS_PATH.exists():
        return
    today = datetime.now(UTC).date().isoformat()
    row = (
        f"| {today} | typing score, {typer_name} typer {version} | LIVE-MODEL | `{out}` "
        "| synthetic gold set, live model calls |\n"
    )
    with _STATUS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(row)


def run(argv: list[str]) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    if args.typer != "rules" and args.max_calls is None:
        print("a model typer requires an explicit --max-calls N", file=sys.stderr)
        return USAGE_ERROR
    try:
        typer = typer_by_name(args.typer, args.max_calls, args.dry_run)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return USAGE_ERROR
    cases = load_gold_cases(args.cases)
    scores = _split_scores(typer, cases, args.split)
    banner = _banner(args.typer, args.dry_run)
    overall = scores.get("all") or next(iter(scores.values()))
    out = args.out or Path("docs/examples") / f"typing_score_{args.typer}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_score_markdown(overall, banner, scores), encoding="utf-8")
    _print_scores(scores)
    print(f"wrote {out}")
    if banner == "LIVE":
        _record_live_run(args.typer, typer.version, out)
    return 0
