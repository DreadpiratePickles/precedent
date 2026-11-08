"""Score a typer against the hand-labelled gold set: precision and recall per field."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from precedent.config import references_dir
from precedent.spine.event_key import EventKey
from precedent.spine.typers.base import TYPING_FIELDS, TypedText, Typer

Banner = Literal["SYNTHETIC", "LIVE", "NO LIVE RUN"]

_BANNER_SENTENCES: dict[str, str] = {
    "SYNTHETIC": "typer {version} scored on the committed synthetic gold set, no external call",
    "LIVE": "typer {version} scored with live model calls on the committed synthetic gold set",
    "NO LIVE RUN": "typer {version} dry run on the committed synthetic gold set, no model called",
}


class GoldCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    kind: Literal["market_question", "headline"]
    split: Literal["dev", "heldout"]
    expected_status: Literal["typed", "untyped"]
    expected_key: str | None
    source: str


class FieldScore(BaseModel):
    """Precision is correct over predicted `typed`; recall is correct over expected `typed`.
    A ratio with a zero denominator is reported as 0.0."""

    precision: float
    recall: float
    n_predicted: int
    n_expected: int


class TypingScore(BaseModel):
    typer_version: str
    n_cases: int
    status: FieldScore
    fields: dict[str, FieldScore]


def load_gold_cases(path: Path | None = None) -> list[GoldCase]:
    source = path or references_dir() / "gold" / "typing_cases.json"
    return [GoldCase.model_validate(item) for item in json.loads(source.read_text("utf-8"))]


def _ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def _field_score(correct: int, n_predicted: int, n_expected: int) -> FieldScore:
    return FieldScore(
        precision=_ratio(correct, n_predicted),
        recall=_ratio(correct, n_expected),
        n_predicted=n_predicted,
        n_expected=n_expected,
    )


def score_results(
    cases: list[GoldCase], results: list[TypedText], typer_version: str
) -> TypingScore:
    """Score already-typed results; `results[i]` is the typer's answer for `cases[i]`."""
    if len(cases) != len(results):
        raise ValueError(f"{len(cases)} cases but {len(results)} results")
    n_expected = sum(c.expected_status == "typed" for c in cases)
    n_predicted = 0
    status_correct = 0
    field_correct = dict.fromkeys(TYPING_FIELDS, 0)
    for case, result in zip(cases, results, strict=True):
        if result.status != "typed" or result.key is None:
            continue
        n_predicted += 1
        if case.expected_status != "typed" or case.expected_key is None:
            continue
        status_correct += 1
        expected = EventKey.from_string(case.expected_key)
        for name in TYPING_FIELDS:
            if getattr(result.key, name) == getattr(expected, name):
                field_correct[name] += 1
    return TypingScore(
        typer_version=typer_version,
        n_cases=len(cases),
        status=_field_score(status_correct, n_predicted, n_expected),
        fields={
            name: _field_score(correct, n_predicted, n_expected)
            for name, correct in field_correct.items()
        },
    )


def score_typer(typer: Typer, cases: list[GoldCase]) -> TypingScore:
    """Type every case exactly once, then score."""
    return score_results(cases, [typer.type_text(c.text) for c in cases], typer.version)


def _table(name: str, score: TypingScore) -> list[str]:
    lines = [
        f"### {name} ({score.n_cases} cases)",
        "",
        "| field | precision | recall | predicted typed | expected typed |",
        "|---|---:|---:|---:|---:|",
    ]
    rows = [("status", score.status), *score.fields.items()]
    for field, value in rows:
        lines.append(
            f"| {field} | {value.precision:.3f} | {value.recall:.3f} "
            f"| {value.n_predicted} | {value.n_expected} |"
        )
    return [*lines, ""]


def render_score_markdown(
    score: TypingScore,
    banner: Banner,
    by_split: Mapping[str, TypingScore] | None = None,
) -> str:
    """Line 1 is the banner and a short sentence; then one table per split (default: `score`)."""
    sentence = _BANNER_SENTENCES[banner].format(version=score.typer_version)
    lines = [f"{banner} — {sentence}", "", "## Typing score", ""]
    for name, split_score in (by_split or {"all": score}).items():
        lines.extend(_table(name, split_score))
    return "\n".join(lines)
