"""`precedent dream brief`: the morning page, three precedent cards plus the overnight diff.

The three headlines are the editor's choice (a text file, one per line). Each is handed to
`run_card` as of the session's cutoff, so a re-run gives the same bytes; the card owns refusal,
the coverage line and the disclaimer. The Dream note is embedded with its headings demoted. This
module does no card logic and calls no model beyond the typer inside `run_card`.
"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import date
from pathlib import Path

import pandas as pd

from precedent import config
from precedent.config import ConfigError
from precedent.dream.report_render import DISCLAIMER as DREAM_DISCLAIMER
from precedent.engine.ask import run_card
from precedent.engine.card import CardModel, render_text
from precedent.libraries.bundle import LibraryMissingError
from precedent.libraries.manifest import LibraryIntegrityError
from precedent.spine.jsonl_io import atomic_write_lines
from precedent.spine.session_calendar import default_cutoffs_for, load_nyse_calendar
from precedent.spine.typers.select import typer_from_env

OK = 0
DATA_FAILURE = 1
USAGE_ERROR = 2
HEADLINE_COUNT = 3
DEFAULT_DREAM_DIR = Path("stages") / "52_dream_report" / "output" / "dreams"
DEFAULT_OUT = Path("stages") / "52_dream_report" / "output" / "briefs"
DEFAULT_HORIZON = 5
DEFAULT_OUTCOME_TYPE = "event_sequence"
OUTCOME_TYPES = ("asset", "event_sequence", "market_resolution")
TEMPLATE_NAME = "brief_template.md"
README_NAME = "README.md"
BANNER_TOKENS = ("SYNTHETIC —", "LIVE —", "NO LIVE RUN —")
SYNTHETIC_BANNER = "SYNTHETIC — built from a synthetic sample library; not market data."
_README_LABEL = re.compile(r"^(SYNTHETIC|NO LIVE RUN|LIVE)\b")
EVIDENCE_CHOICES = ("LIVE", "SYNTHETIC")
DEMOTE_LEVELS = 2
MAX_HEADING_LEVEL = 6
_HEADING = re.compile(r"^(#{1,6})(\s)")
_PLACEHOLDER = re.compile(r"\{\{(\w*)\}\}")


class BriefError(Exception):
    """An input the brief cannot be built from; the command exits 2 and writes nothing."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precedent dream brief",
        description="Write the morning brief: three base-rate cards and the overnight diff.",
    )
    parser.add_argument("--date", type=date.fromisoformat, required=True, metavar="YYYY-MM-DD")
    parser.add_argument(
        "--headlines", type=Path, required=True, help="three non-empty lines, chosen by the editor"
    )
    parser.add_argument("--dream-dir", type=Path, default=DEFAULT_DREAM_DIR)
    parser.add_argument("--data-dir", type=Path, help="default: PRECEDENT_DATA_DIR")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument(
        "--outcome-type", choices=OUTCOME_TYPES, default=DEFAULT_OUTCOME_TYPE, dest="outcome_type"
    )
    parser.add_argument("--mechanism", type=Path, help="a short note; an empty file adds nothing")
    parser.add_argument(
        "--evidence",
        choices=EVIDENCE_CHOICES,
        help="the evidence label, for a library whose README.md has none: a human statement of "
        "where it came from, never inferred; it may not contradict a README label",
    )
    return parser


def run(argv: list[str]) -> int:
    """Write `<out>/brief_<date>.md`: 0 written; 2 bad usage — a missing library, bad flags
    (nothing written); 1 a data or store fault — a corrupt library, a failed write (D76/M4,
    narrowed by D78)."""
    try:
        args = _parser().parse_args(argv)
    except SystemExit as request:
        return request.code if isinstance(request.code, int) else USAGE_ERROR
    try:
        text = _build(args)
    except (BriefError, ConfigError, ValueError, LibraryMissingError) as e:
        print(f"error: {e}", file=sys.stderr)
        return USAGE_ERROR
    except LibraryIntegrityError as e:
        print(f"error: {e}", file=sys.stderr)
        return DATA_FAILURE
    target = args.out / f"brief_{args.date.isoformat()}.md"
    try:
        atomic_write_lines(target, text.splitlines())
    except OSError as error:
        print(f"error: cannot write {target}: {error}", file=sys.stderr)
        return DATA_FAILURE
    print(f"wrote {target}")
    return OK


def _build(args: argparse.Namespace) -> str:
    data_dir = args.data_dir if args.data_dir is not None else config.data_dir()
    headlines = _read_headlines(args.headlines)
    dream = _read_dream(args.dream_dir / f"dream_{args.date.isoformat()}.md")
    mechanism = _read_mechanism(args.mechanism)
    asof_utc = _session_cutoff(args.date)
    template = (config.references_dir() / TEMPLATE_NAME).read_text(encoding="utf-8")
    typer = typer_from_env()
    cards = [
        run_card(headline, args.horizon, args.outcome_type, data_dir, typer, asof_utc=asof_utc)
        for headline in headlines
    ]
    return render_brief(
        args.date, _banner(data_dir, args.evidence), cards, dream, mechanism, template
    )


def _read_headlines(path: Path) -> list[str]:
    try:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    except OSError as error:
        raise BriefError(f"cannot read headlines {path}: {error}") from error
    headlines = [line for line in lines if line]
    if len(headlines) != HEADLINE_COUNT:
        raise BriefError(
            f"{path} must hold exactly {HEADLINE_COUNT} non-empty lines, found {len(headlines)}"
        )
    return headlines


def _read_dream(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise BriefError(f"no Dream file for the date: cannot read {path}: {error}") from error


def _read_mechanism(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
    except OSError as error:
        raise BriefError(f"cannot read mechanism note {path}: {error}") from error
    return text or None


def _session_cutoff(session: date) -> pd.Timestamp:
    """The committed calendar's cutoff for the brief's date, as a tz-aware UTC timestamp."""
    try:
        cutoffs = default_cutoffs_for([session], load_nyse_calendar(), 0)
        return pd.Timestamp(cutoffs.loc[pd.Timestamp(session)]).tz_convert("UTC")
    except (KeyError, ValueError) as error:
        raise BriefError(f"{session} is not a session on the NYSE calendar") from error


def _readme_label(data_dir: Path) -> str | None:
    """The evidence label on the first line of the data dir's README: SYNTHETIC, LIVE or
    NO LIVE RUN; None when there is no README or its first line carries none."""
    try:
        first = (data_dir / README_NAME).read_text(encoding="utf-8").lstrip().splitlines()[0]
    except (OSError, IndexError):
        return None
    found = _README_LABEL.match(first)
    return found.group(1) if found else None


def _banner(data_dir: Path, evidence: str | None = None) -> str:
    """The banner passes through the data dir's README label, never inferred from its path (D57,
    D64). A library with no label needs `--evidence` (a human statement of provenance); a flag
    that contradicts the README is refused. Only the data dir's final component is printed."""
    label = _readme_label(data_dir)
    if label is None and evidence is None:
        raise BriefError(
            f"{data_dir / README_NAME} carries no evidence label (SYNTHETIC, LIVE or NO LIVE "
            f"RUN on its first line); label it, or pass --evidence {'|'.join(EVIDENCE_CHOICES)}"
        )
    if label is not None and evidence is not None and evidence != label:
        raise BriefError(f"--evidence {evidence} contradicts the README label {label}")
    label = label or evidence
    name = data_dir.resolve().name
    if label == "SYNTHETIC":
        return SYNTHETIC_BANNER
    if label == "NO LIVE RUN":
        return f"NO LIVE RUN — cards built from the library under {name}; no live run made it."
    return f"LIVE — cards built from the library under {name}."


def _embed_dream(markdown: str) -> str:
    """Drop the note's banner and closing disclaimer lines, demote its headings two levels."""
    lines = markdown.splitlines()
    if lines and lines[0].startswith(BANNER_TOKENS):
        lines = lines[1:]
    while lines and not lines[-1].strip():
        lines.pop()
    if lines and lines[-1].strip() == DREAM_DISCLAIMER:
        lines.pop()
    demoted = [_demote(line) for line in lines]
    return "\n".join(demoted).strip("\n")


def _demote(line: str) -> str:
    found = _HEADING.match(line)
    if found is None:
        return line
    level = min(len(found.group(1)) + DEMOTE_LEVELS, MAX_HEADING_LEVEL)
    return "#" * level + line[len(found.group(1)) :]


def _fenced(card: CardModel) -> str:
    return f"```text\n{render_text(card)}\n```"


def render_brief(
    session: date,
    banner: str,
    cards: list[CardModel],
    dream_markdown: str,
    mechanism: str | None,
    template: str,
) -> str:
    """Fill the template; a placeholder with no value raises `ValueError` (nothing is left bare).

    Substitution is one pass over the template, so inserted text is never read for placeholders.
    """
    if len(cards) != HEADLINE_COUNT:
        raise ValueError(f"the brief takes {HEADLINE_COUNT} cards, got {len(cards)}")
    values = {
        "banner": banner,
        "date": session.isoformat(),
        "card_1": _fenced(cards[0]),
        "card_2": _fenced(cards[1]),
        "card_3": _fenced(cards[2]),
        "dream_diff": _embed_dream(dream_markdown),
        "mechanism_section": f"## One mechanism\n\n{mechanism.strip()}\n\n" if mechanism else "",
    }
    unfilled = sorted({n for n in _PLACEHOLDER.findall(template) if n not in values})
    if unfilled:
        raise ValueError(f"brief template has unfilled placeholders: {unfilled}")
    text = _PLACEHOLDER.sub(lambda m: values[m.group(1)], template)
    return text if text.endswith("\n") else text + "\n"
