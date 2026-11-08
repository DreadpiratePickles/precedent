"""Tests for gold-set scoring, the gold set's shape, and the score command."""
import json

import pytest

from precedent.spine.event_key import EventKey
from precedent.spine.typers import cli as typers_cli
from precedent.spine.typers.base import TypedText
from precedent.spine.typers.cli import run
from precedent.spine.typers.fake import FakeTyper
from precedent.spine.typers.rules import RulesTyper
from precedent.spine.typers.score import (
    GoldCase,
    load_gold_cases,
    render_score_markdown,
    score_typer,
)

K1 = EventKey("central_bank", "central_bank", ("country",), "US", "mid", True, 1)
K1_IN_DM = EventKey("central_bank", "central_bank", ("country",), "DM", "mid", True, 1)
K3 = EventKey("earnings", "company", ("company",), "none", "mid", True, 1)
UNTYPED = TypedText(status="untyped", key=None, reason="r", evidence_span=None, typer_version="v")


def typed(key):
    return TypedText(status="typed", key=key, reason="r", evidence_span="x", typer_version="v")


def case(case_id, key):
    return GoldCase(
        id=case_id, text=case_id, kind="headline", split="dev",
        expected_status="untyped" if key is None else "typed",
        expected_key=None if key is None else key.as_string(),
        source="SYNTHETIC: written for this repository",
    )


FOUR_CASES = [case("a", K1), case("b", K1), case("c", None), case("d", None)]


def test_scores_match_hand_counts():
    fake = FakeTyper({"a": typed(K1), "b": typed(K1_IN_DM), "c": typed(K3), "d": UNTYPED})
    score = score_typer(fake, FOUR_CASES)
    assert score.n_cases == 4 and score.typer_version == "fake-v1"
    assert (score.status.precision, score.status.recall) == pytest.approx((2 / 3, 1.0))
    assert (score.fields["event_type"].precision, score.fields["event_type"].recall) == pytest.approx((2 / 3, 1.0))
    assert (score.fields["country_group"].precision, score.fields["country_group"].recall) == pytest.approx((1 / 3, 0.5))
    assert score.fields["country_group"].n_predicted == 3 and score.fields["country_group"].n_expected == 2


def test_a_typer_that_predicts_nothing_scores_zero_not_a_division_error():
    score = score_typer(FakeTyper({}), FOUR_CASES)
    assert (score.status.precision, score.status.recall) == (0.0, 0.0)


def test_gold_set_has_the_promised_shape():
    cases = load_gold_cases()
    assert len(cases) == 60 and sum(c.split == "heldout" for c in cases) == 20
    assert sum(c.kind == "market_question" for c in cases) >= 30 and sum(c.kind == "headline" for c in cases) >= 20
    assert sum(c.expected_status == "untyped" for c in cases) >= 10
    assert all(EventKey.from_string(c.expected_key) for c in cases if c.expected_key is not None)
    assert all(c.source == "SYNTHETIC: written for this repository" for c in cases)


def test_render_puts_the_banner_on_line_one_and_a_table_per_split():
    score = score_typer(FakeTyper({}), FOUR_CASES)
    text = render_score_markdown(score, "SYNTHETIC", {"dev": score, "heldout": score})
    assert text.splitlines()[0].startswith("SYNTHETIC — ")
    assert "dev" in text and "heldout" in text and "event_type" in text
    assert render_score_markdown(score, "NO LIVE RUN").startswith("NO LIVE RUN — ")


def test_score_command_writes_a_bannered_report(tmp_path):
    out = tmp_path / "typing_score_rules.md"
    assert run(["score", "--typer", "rules", "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert text.startswith("SYNTHETIC —")
    assert all(name in text for name in ("dev", "heldout"))


def test_score_command_on_one_split_prints_precision_and_recall(tmp_path, capsys):
    out = tmp_path / "s.md"
    assert run(["score", "--typer", "rules", "--split", "dev", "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "event_type" in printed and "precision" in printed and "recall" in printed
    assert "heldout" not in out.read_text(encoding="utf-8")


def test_score_command_reads_a_custom_case_file(tmp_path):
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps([c.model_dump() for c in FOUR_CASES]), encoding="utf-8")
    out = tmp_path / "s.md"
    assert run(["score", "--cases", str(cases), "--split", "all", "--out", str(out)]) == 0


def test_a_model_typer_requires_an_explicit_max_calls(tmp_path, capsys):
    assert run(["score", "--typer", "anthropic", "--out", str(tmp_path / "s.md")]) == 2
    assert "--max-calls" in capsys.readouterr().err


def test_a_live_run_gets_a_live_banner_and_a_status_row(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "docs").mkdir()
    status = tmp_path / "docs" / "STATUS.md"
    status.write_text("| Date | Result | Label | Evidence | Notes |\n|---|---|---|---|---|\n", encoding="utf-8")
    monkeypatch.setattr(typers_cli, "typer_by_name", lambda name, max_calls, dry_run: FakeTyper({}))
    out = tmp_path / "docs" / "examples" / "typing_score_anthropic.md"
    assert run(["score", "--typer", "anthropic", "--max-calls", "5", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("LIVE —")
    assert "LIVE-MODEL" in status.read_text(encoding="utf-8")


def test_rules_typer_scores_in_range_on_the_gold_set():
    score = score_typer(RulesTyper(), load_gold_cases())
    assert 0.0 <= score.status.precision <= 1.0 and score.n_cases == 60
