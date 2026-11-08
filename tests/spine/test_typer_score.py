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


class CountingTyper(FakeTyper):
    """Counts every call; `calls_made` is the attribute the CLI reads to choose the banner."""

    def __init__(self, calls_made_override=None):
        super().__init__({})
        self.calls = 0
        self._override = calls_made_override

    @property
    def calls_made(self):
        return self.calls if self._override is None else self._override

    def type_text(self, text):
        self.calls += 1
        return super().type_text(text)


@pytest.fixture
def fake_repo(tmp_path, monkeypatch):
    (tmp_path / "docs").mkdir()
    status = tmp_path / "docs" / "STATUS.md"
    status.write_text("| Date | Result | Label | Evidence | Notes |\n|---|---|---|---|---|", "utf-8")
    monkeypatch.setattr(typers_cli, "repo_root", lambda: tmp_path)
    return tmp_path


def _use_typer(monkeypatch, typer):
    monkeypatch.setattr(typers_cli, "typer_by_name", lambda name, max_calls, dry_run: typer)


def test_split_all_types_each_case_exactly_once(fake_repo, monkeypatch):
    counting = CountingTyper()
    _use_typer(monkeypatch, counting)
    out = fake_repo / "s.md"
    assert run(["score", "--typer", "anthropic", "--max-calls", "60", "--out", str(out)]) == 0
    assert counting.calls == 60


def test_a_live_run_gets_a_live_banner_and_a_status_row_even_without_a_trailing_newline(
    fake_repo, monkeypatch
):
    _use_typer(monkeypatch, CountingTyper())
    out = fake_repo / "docs" / "examples" / "typing_score_anthropic.md"
    assert run(["score", "--typer", "anthropic", "--max-calls", "70", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("LIVE —")
    rows = (fake_repo / "docs" / "STATUS.md").read_text(encoding="utf-8").splitlines()
    assert rows[-1].startswith("|") and "LIVE-MODEL" in rows[-1] and rows[-2].startswith("|---")


def test_a_model_typer_that_made_no_calls_is_not_banner_live(fake_repo, monkeypatch):
    class NoCalls(CountingTyper):
        calls_made = 0

    _use_typer(monkeypatch, NoCalls())
    out = fake_repo / "s.md"
    assert run(["score", "--typer", "ollama", "--max-calls", "5", "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("NO LIVE RUN —")
    assert "LIVE-MODEL" not in (fake_repo / "docs" / "STATUS.md").read_text(encoding="utf-8")


def test_a_live_run_without_a_status_file_fails_loudly_before_any_call(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(typers_cli, "repo_root", lambda: tmp_path)
    counting = CountingTyper()
    _use_typer(monkeypatch, counting)
    code = run(["score", "--typer", "anthropic", "--max-calls", "5", "--out", str(tmp_path / "s.md")])
    assert code == 1 and counting.calls == 0
    assert "STATUS.md" in capsys.readouterr().err


def test_rules_typer_dev_split_is_pinned():
    dev = [c for c in load_gold_cases() if c.split == "dev"]
    score = score_typer(RulesTyper(), dev)
    assert score.n_cases == 40
    assert (score.status.precision, score.status.recall) == pytest.approx((1.0, 1.0))
    assert (score.fields["event_type"].precision, score.fields["event_type"].recall) == pytest.approx((1.0, 1.0))
    assert score.fields["country_group"].recall == pytest.approx(1.0)
    assert score.fields["direction_sign"].recall == pytest.approx(1.0)
    assert score.fields["actor_kind"].recall == pytest.approx(30 / 33)
    assert score.fields["impacted_kinds"].recall == pytest.approx(31 / 33)
