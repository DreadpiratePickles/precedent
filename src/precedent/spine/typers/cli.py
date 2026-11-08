"""`precedent typers score`: score a typer on the gold set and write a bannered report."""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from precedent.config import ConfigError, references_dir
from precedent.spine.typers.base import TypedText, Typer
from precedent.spine.typers.score import (
    Banner,
    GoldCase,
    TypingScore,
    load_gold_cases,
    render_score_markdown,
    score_results,
)
from precedent.spine.typers.select import TYPER_NAMES, typer_by_name

USAGE_ERROR = 2
_EXIT_FAILURE = 1


def repo_root() -> Path:
    """The checkout root: the parent of the references directory."""
    return references_dir().parent


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


def _banner(typer_name: str, dry_run: bool, typer: Typer) -> Banner:
    """LIVE only when the typer reports calls made; typers without `calls_made` use the name."""
    if typer_name == "rules":
        return "SYNTHETIC"
    calls_made = getattr(typer, "calls_made", None)
    if calls_made is not None:
        return "LIVE" if calls_made > 0 else "NO LIVE RUN"
    return "NO LIVE RUN" if dry_run else "LIVE"


def _split_scores(
    typer: Typer, cases: list[GoldCase], split: str
) -> dict[str, TypingScore]:
    """Type each case exactly once; every split's score comes from those stored results."""
    results: list[TypedText] = [typer.type_text(c.text) for c in cases]
    pairs = list(zip(cases, results, strict=True))

    def score(name: str) -> TypingScore:
        chosen = [(c, r) for c, r in pairs if name == "all" or c.split == name]
        return score_results([c for c, _ in chosen], [r for _, r in chosen], typer.version)

    names = ("dev", "heldout", "all") if split == "all" else (split,)
    return {name: score(name) for name in names}


def _print_scores(scores: dict[str, TypingScore]) -> None:
    for name, score in scores.items():
        print(f"{name}: {score.n_cases} cases, typer {score.typer_version}")
        print(f"  {'field':<16}{'precision':>10}{'recall':>8}")
        for field, value in [("status", score.status), *score.fields.items()]:
            print(f"  {field:<16}{value.precision:>10.3f}{value.recall:>8.3f}")


def _status_path() -> Path:
    return repo_root() / "docs" / "STATUS.md"


def _record_live_run(typer_name: str, version: str, out: Path) -> None:
    """Append a LIVE-MODEL row to docs/STATUS.md; raise OSError when it cannot be written."""
    path = _status_path()
    today = datetime.now(UTC).date().isoformat()
    row = (
        f"| {today} | typing score, {typer_name} typer {version} | LIVE-MODEL | `{out}` "
        "| synthetic gold set, live model calls |\n"
    )
    existing = path.read_text(encoding="utf-8")
    separator = "" if existing.endswith("\n") or not existing else "\n"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(separator + row)


def run(argv: list[str]) -> int:
    try:
        args = _parser().parse_args(argv)
    except SystemExit as exit_request:
        return int(exit_request.code or 0)
    if args.typer != "rules" and args.max_calls is None:
        print("a model typer requires an explicit --max-calls N", file=sys.stderr)
        return USAGE_ERROR
    if args.typer != "rules" and not args.dry_run and not _status_path().is_file():
        print(
            f"cannot record a LIVE-MODEL run: {_status_path()} not found; no model was called",
            file=sys.stderr,
        )
        return _EXIT_FAILURE
    try:
        typer = typer_by_name(args.typer, args.max_calls, args.dry_run)
    except ConfigError as error:
        print(f"config error: {error}", file=sys.stderr)
        return USAGE_ERROR
    cases = load_gold_cases(args.cases)
    scores = _split_scores(typer, cases, args.split)
    banner = _banner(args.typer, args.dry_run, typer)
    overall = scores.get("all") or next(iter(scores.values()))
    out = args.out or repo_root() / "docs" / "examples" / f"typing_score_{args.typer}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_score_markdown(overall, banner, scores), encoding="utf-8")
    _print_scores(scores)
    print(f"wrote {out}")
    if banner == "LIVE":
        try:
            _record_live_run(args.typer, typer.version, out)
        except OSError as error:
            print(f"could not record the LIVE-MODEL run: {error}", file=sys.stderr)
            return _EXIT_FAILURE
    return 0
