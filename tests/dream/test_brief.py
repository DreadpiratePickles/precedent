"""The morning brief: three precedent cards, the overnight Dream diff, an optional mechanism."""

import importlib.util
import json
import re
import shutil
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from library_root_synthetic import write_library_root

from precedent.dream import brief
from precedent.dream.report_render import DISCLAIMER as DREAM_DISCLAIMER
from precedent.engine.card import DISCLAIMER as CARD_DISCLAIMER
from precedent.engine.card import CardModel, render_text
from precedent.libraries.bundle import LibraryMissingError
from precedent.libraries.manifest import LibraryIntegrityError

DATE = "2024-06-03"
ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "dream" / "fixtures"
HEADLINES_FILE = FIX / f"headlines_{DATE}.txt"
HEADLINES = HEADLINES_FILE.read_text(encoding="utf-8").splitlines()
CUTOFF = pd.Timestamp("2024-06-03T20:00:00Z")  # 16:00 New York, the committed calendar's cutoff
BRIEF_LAST_LINE = (
    "Tensions and shifts are observations about documents and prices, not forecasts. "
    "Base rates are historical frequencies, not predictions."
)
SYNTHETIC_README = "SYNTHETIC — a synthetic sample library; not market data.\n"


@pytest.fixture(autouse=True)
def rules_typer(monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "rules")  # deterministic, never a model


@pytest.fixture
def data_dir(tmp_path) -> Path:
    root = write_library_root(tmp_path / "data")
    (root / "README.md").write_text(SYNTHETIC_README, encoding="utf-8")
    return root


def run_brief(tmp_path, data_dir, *extra, headlines=HEADLINES_FILE):
    return brief.run(
        [
            "--date", DATE, "--headlines", str(headlines), "--dream-dir", str(FIX),
            "--data-dir", str(data_dir), "--out", str(tmp_path / "out"), *extra,
        ]
    )  # fmt: skip


def brief_text(tmp_path) -> str:
    return (tmp_path / "out" / f"brief_{DATE}.md").read_text(encoding="utf-8")


def card_for(headline: str, *, refused: bool, n: int = 20) -> CardModel:
    return CardModel(
        headline=headline,
        typed_as="central_bank / central_bank / US / severity high / unscheduled",
        typed_reason=None,
        key=None,
        asof_utc=CUTOFF.to_pydatetime(),
        horizon=5,
        outcome_type="event_sequence",
        n=n,
        n_eff=float(n),
        earliest=None,
        latest=None,
        refused=refused,
        min_neff=10.0,
        rows=[],
        not_built={},
        nearest=[],
        covers=[],
        hint=None,
    )


@pytest.fixture
def three_cards() -> list[CardModel]:
    return [card_for(h, refused=False) for h in HEADLINES]


# --- the real pipeline, no patching -------------------------------------------------------------


def test_line_one_is_an_evidence_banner(tmp_path, data_dir):
    assert run_brief(tmp_path, data_dir) == 0
    first = brief_text(tmp_path).splitlines()[0]
    assert first.startswith(("SYNTHETIC —", "LIVE —", "NO LIVE RUN —"))
    assert first == "SYNTHETIC — built from a synthetic sample library; not market data."


def test_real_cards_carry_the_cards_own_refusal_coverage_and_disclaimer(tmp_path, data_dir):
    assert run_brief(tmp_path, data_dir) == 0
    text = brief_text(tmp_path)
    # the card owns these lines; the brief only embeds them
    assert text.count("library covers") == 3 and text.count("REFUSED") >= 2
    assert text.count("PRECEDENT  ·  ") == 3


def test_the_example_headlines_are_typed_one_to_the_library_key(tmp_path, data_dir):
    run_brief(tmp_path, data_dir)
    assert "independent=20" in brief_text(tmp_path)  # the Fed headline found the 20 events


# --- with run_card patched ----------------------------------------------------------------------


def test_three_cards_appear_in_order_with_their_mandatory_lines(
    tmp_path, data_dir, monkeypatch, three_cards
):
    monkeypatch.setattr(
        brief, "run_card",
        lambda q, h, o, d, t, *, asof_utc=None: three_cards[HEADLINES.index(q)],
    )  # fmt: skip
    assert run_brief(tmp_path, data_dir) == 0
    text = brief_text(tmp_path)
    positions = [text.index(render_text(c)) for c in three_cards]
    assert positions == sorted(positions)
    assert text.count("library covers") >= 3 and text.count(CARD_DISCLAIMER) == 3


def test_a_refused_card_stays_in_the_brief(tmp_path, data_dir, monkeypatch):
    cards = [card_for(HEADLINES[0], refused=True, n=3)] + [
        card_for(h, refused=False) for h in HEADLINES[1:]
    ]
    monkeypatch.setattr(
        brief, "run_card", lambda q, h, o, d, t, *, asof_utc=None: cards[HEADLINES.index(q)]
    )
    assert run_brief(tmp_path, data_dir) == 0
    assert "REFUSED" in brief_text(tmp_path)


def test_run_card_gets_the_documented_arguments_and_the_cutoff_as_asof(
    tmp_path, data_dir, monkeypatch, three_cards
):
    seen = []

    def fake(q, h, o, d, t, *, asof_utc=None):
        seen.append((q, h, o, Path(d), t.version, asof_utc))
        return three_cards[0]

    monkeypatch.setattr(brief, "run_card", fake)
    run_brief(tmp_path, data_dir, "--horizon", "10", "--outcome-type", "asset")
    assert [s[:3] for s in seen] == [(h, 10, "asset") for h in HEADLINES]
    assert all(s[3] == data_dir and s[4].startswith("rules-typing") for s in seen)
    assert {s[5] for s in seen} == {CUTOFF}  # point in time, never wall-clock (D56)


def test_defaults_are_horizon_5_and_event_sequence(tmp_path, data_dir, monkeypatch, three_cards):
    seen = []
    monkeypatch.setattr(
        brief, "run_card",
        lambda q, h, o, d, t, *, asof_utc=None: seen.append((h, o)) or three_cards[0],
    )  # fmt: skip
    run_brief(tmp_path, data_dir)
    assert set(seen) == {(5, "event_sequence")}


@pytest.mark.parametrize("error", [ValueError("bad horizon")])
def test_a_run_card_error_is_exit_2_and_writes_nothing(
    tmp_path, data_dir, monkeypatch, capsys, error
):
    def failing(q, h, o, d, t, *, asof_utc=None):
        raise error

    monkeypatch.setattr(brief, "run_card", failing)
    assert run_brief(tmp_path, data_dir) == 2
    assert str(error) in capsys.readouterr().err
    assert (
        list((tmp_path / "out").glob("brief_*.md")) == [] if (tmp_path / "out").exists() else True
    )


@pytest.mark.parametrize(
    "error",
    [LibraryMissingError("no library"), LibraryIntegrityError("MANIFEST.json", "sha256 mismatch")],
)
def test_a_library_fault_is_exit_1_and_writes_nothing(
    tmp_path, data_dir, monkeypatch, capsys, error
):
    """M4 (Opus final review / D76): a library fault is a data fault, so it exits 1 like the
    dream command's store failures; only usage errors exit 2."""

    def failing(q, h, o, d, t, *, asof_utc=None):
        raise error

    monkeypatch.setattr(brief, "run_card", failing)
    assert run_brief(tmp_path, data_dir) == 1
    assert capsys.readouterr().err.startswith("error:")
    assert (
        list((tmp_path / "out").glob("brief_*.md")) == [] if (tmp_path / "out").exists() else True
    )


def test_a_write_fault_is_exit_1_and_leaves_no_partial_brief(tmp_path, data_dir, monkeypatch):
    """M4 (Opus final review / D76): the brief is written atomically, so a write fault exits 1
    and no half-written brief exists."""

    def full_disk(path, lines):
        raise OSError("disk full")

    monkeypatch.setattr(brief, "atomic_write_lines", full_disk)
    assert run_brief(tmp_path, data_dir) == 1
    assert not (tmp_path / "out").exists()


def test_a_missing_library_is_exit_1_through_the_real_run_card(tmp_path, capsys):
    assert run_brief(tmp_path, tmp_path / "empty") == 1
    assert capsys.readouterr().err.startswith("error:")


def test_the_default_out_is_the_stage_briefs_dir(tmp_path, data_dir, monkeypatch):
    """M8 (Opus final review / D76): the default --out is the stage output, never the committed
    docs/examples; the committed example is written only by an explicit --out."""
    monkeypatch.chdir(tmp_path)
    code = brief.run(
        [
            "--date", DATE, "--headlines", str(HEADLINES_FILE), "--dream-dir", str(FIX),
            "--data-dir", str(data_dir),
        ]
    )
    assert code == 0
    assert (tmp_path / "stages" / "52_dream_report" / "output" / "briefs" / f"brief_{DATE}.md").exists()
    assert not (tmp_path / "docs").exists()


# --- banner from the data dir's README (D57) ----------------------------------------------------


def test_a_live_readme_gives_the_live_banner_naming_only_the_final_path_component(
    tmp_path, data_dir
):
    (data_dir / "README.md").write_text("LIVE — the built sample.\n", encoding="utf-8")
    run_brief(tmp_path, data_dir)
    first = brief_text(tmp_path).splitlines()[0]
    assert first == f"LIVE — cards built from the library under {data_dir.name}."
    assert str(data_dir.parent) not in brief_text(tmp_path).split("\n\n")[0]  # D64: no abs path


def test_a_no_live_run_readme_passes_through_to_the_banner(tmp_path, data_dir):
    (data_dir / "README.md").write_text("NO LIVE RUN — a designed shape.\n", encoding="utf-8")
    assert run_brief(tmp_path, data_dir) == 0
    first = brief_text(tmp_path).splitlines()[0]
    assert first.startswith("NO LIVE RUN —") and data_dir.name in first
    assert not first.startswith("LIVE")


def test_a_readme_without_a_label_is_refused_and_writes_nothing(tmp_path, data_dir, capsys):
    (data_dir / "README.md").write_text("A sample library.\n", encoding="utf-8")
    assert run_brief(tmp_path, data_dir) == 2
    error = capsys.readouterr().err
    assert "no evidence label" in error and "--evidence" in error
    assert not (tmp_path / "out" / f"brief_{DATE}.md").exists()


def test_no_readme_is_refused_the_same_way(tmp_path, data_dir, capsys):
    (data_dir / "README.md").unlink()
    assert run_brief(tmp_path, data_dir) == 2
    assert "no evidence label" in capsys.readouterr().err
    assert not (tmp_path / "out" / f"brief_{DATE}.md").exists()


@pytest.mark.parametrize("label", ["LIVE", "SYNTHETIC"])
def test_the_evidence_flag_labels_a_library_whose_readme_has_none(tmp_path, data_dir, label):
    (data_dir / "README.md").unlink()
    assert run_brief(tmp_path, data_dir, "--evidence", label) == 0
    assert brief_text(tmp_path).splitlines()[0].startswith(f"{label} —")


@pytest.mark.parametrize(
    ("readme", "flag"),
    [
        ("SYNTHETIC — x\n", "LIVE"),
        ("LIVE — x\n", "SYNTHETIC"),
        ("NO LIVE RUN — x\n", "LIVE"),
        ("NO LIVE RUN — x\n", "SYNTHETIC"),
    ],
)
def test_a_flag_that_contradicts_the_readme_label_is_refused(
    tmp_path, data_dir, capsys, readme, flag
):
    (data_dir / "README.md").write_text(readme, encoding="utf-8")
    assert run_brief(tmp_path, data_dir, "--evidence", flag) == 2
    assert "contradicts" in capsys.readouterr().err
    assert not (tmp_path / "out" / f"brief_{DATE}.md").exists()


def test_a_flag_equal_to_the_readme_label_is_accepted(tmp_path, data_dir):
    assert run_brief(tmp_path, data_dir, "--evidence", "SYNTHETIC") == 0


def test_the_path_never_decides_the_banner(tmp_path, data_dir):
    sample = tmp_path / "elsewhere" / "data" / "sample"  # named like the sample, labelled LIVE
    shutil.copytree(data_dir, sample)
    (sample / "README.md").write_text("LIVE — a built sample.\n", encoding="utf-8")
    run_brief(tmp_path, sample)
    assert brief_text(tmp_path).splitlines()[0].startswith("LIVE —")


# --- the Dream diff (D59) -----------------------------------------------------------------------


def test_dream_diff_is_embedded_demoted_and_without_its_own_banner_or_disclaimer(
    tmp_path, data_dir
):
    run_brief(tmp_path, data_dir)
    text = brief_text(tmp_path)
    # the fixture is the real output for the generated stores (D67): three opened tensions
    assert "## Overnight" in text and "#### Opened tensions (3)" in text
    assert "### Dream for 2024-06-03" in text
    assert "\n## Opened tensions" not in text and "\n# Dream for" not in text
    assert text.count("SYNTHETIC —") == 1  # the Dream fixture's own banner line is dropped
    assert text.count(DREAM_DISCLAIMER) == 1  # only inside the brief's own closing line
    assert text.rstrip().splitlines()[-1] == BRIEF_LAST_LINE
    assert text.endswith("\n")


def test_demotion_handles_every_dream_heading_level(tmp_path, data_dir):
    dream_dir = tmp_path / "dreams"
    dream_dir.mkdir()
    (dream_dir / f"dream_{DATE}.md").write_text(
        f"# Dream for {DATE}\n\n## Opened tensions (0)\n\n{DREAM_DISCLAIMER}\n", encoding="utf-8"
    )
    code = brief.run(
        ["--date", DATE, "--headlines", str(HEADLINES_FILE), "--dream-dir", str(dream_dir),
         "--data-dir", str(data_dir), "--out", str(tmp_path / "out")]
    )  # fmt: skip
    assert code == 0
    assert "### Dream for 2024-06-03\n\n#### Opened tensions (0)" in brief_text(tmp_path)


# --- the mechanism slot -------------------------------------------------------------------------


def test_empty_mechanism_slot_omits_the_section_and_a_filled_one_adds_it(tmp_path, data_dir):
    run_brief(tmp_path, data_dir)
    assert "## One mechanism" not in brief_text(tmp_path)
    blank = tmp_path / "blank.md"
    blank.write_text("\n  \n", encoding="utf-8")
    run_brief(tmp_path, data_dir, "--mechanism", str(blank))
    assert "## One mechanism" not in brief_text(tmp_path)
    note = tmp_path / "mech.md"
    note.write_text("Figure: oil shock to airline margins.\n", encoding="utf-8")
    run_brief(tmp_path, data_dir, "--mechanism", str(note))
    text = brief_text(tmp_path)
    assert "## One mechanism\n\nFigure: oil shock to airline margins.\n\n" in text
    assert text.index("## Overnight") < text.index("## One mechanism")
    assert text.rstrip().splitlines()[-1] == BRIEF_LAST_LINE


def test_a_missing_mechanism_file_is_exit_2(tmp_path, data_dir):
    assert run_brief(tmp_path, data_dir, "--mechanism", str(tmp_path / "nope.md")) == 2
    assert not (tmp_path / "out" / f"brief_{DATE}.md").exists()


# --- exit 2 writes nothing ----------------------------------------------------------------------


def test_missing_dream_file_is_exit_2_and_writes_nothing(tmp_path, data_dir, capsys):
    code = brief.run(
        ["--date", "2030-01-01", "--headlines", str(HEADLINES_FILE), "--dream-dir", str(FIX),
         "--data-dir", str(data_dir), "--out", str(tmp_path / "out")]
    )  # fmt: skip
    assert code == 2 and "dream_2030-01-01.md" in capsys.readouterr().err
    assert (
        list((tmp_path / "out").glob("brief_*.md")) == [] if (tmp_path / "out").exists() else True
    )


@pytest.mark.parametrize("lines", ["a\nb\n", "a\nb\nc\nd\n", "", "a\n\nb\n"])
def test_headlines_must_be_exactly_three(tmp_path, data_dir, lines):
    bad = tmp_path / "bad.txt"
    bad.write_text(lines, encoding="utf-8")
    assert run_brief(tmp_path, data_dir, headlines=bad) == 2
    assert not (tmp_path / "out" / f"brief_{DATE}.md").exists()


def test_a_date_that_is_not_a_session_is_exit_2(tmp_path, data_dir):
    dream_dir = tmp_path / "dreams"
    dream_dir.mkdir()
    (dream_dir / "dream_2024-06-02.md").write_text("# Dream for 2024-06-02\n", encoding="utf-8")
    code = brief.run(
        ["--date", "2024-06-02", "--headlines", str(HEADLINES_FILE), "--dream-dir", str(dream_dir),
         "--data-dir", str(data_dir), "--out", str(tmp_path / "out")]
    )  # fmt: skip
    assert code == 2 and not (tmp_path / "out").exists()


# --- rendering ----------------------------------------------------------------------------------


def test_no_placeholder_is_left_and_the_run_is_idempotent(tmp_path, data_dir):
    run_brief(tmp_path, data_dir)
    first = (tmp_path / "out" / f"brief_{DATE}.md").read_bytes()
    run_brief(tmp_path, data_dir)
    assert b"{{" not in first and (tmp_path / "out" / f"brief_{DATE}.md").read_bytes() == first


def test_render_brief_refuses_a_template_with_an_unknown_placeholder(three_cards):
    with pytest.raises(ValueError, match="unfilled"):
        brief.render_brief(date(2024, 6, 3), "b", three_cards, "# d", None, "{{banner}} {{nope}}")


def test_render_brief_does_not_rescan_inserted_text(three_cards):
    out = brief.render_brief(
        date(2024, 6, 3), "{{date}}", three_cards, "# {{card_1}}", None,
        "{{banner}}\n{{dream_diff}}\n{{card_1}}{{card_2}}{{card_3}}{{date}}{{mechanism_section}}",
    )  # fmt: skip
    assert out.splitlines()[0] == "{{date}}" and "### {{card_1}}" in out


# --- the committed example (D58) ----------------------------------------------------------------


def _example_script():
    path = ROOT / "scripts" / "build_brief_example.py"
    spec = importlib.util.spec_from_file_location("build_brief_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_dream_fixture_is_the_real_renderer_output_with_a_banner(tmp_path):
    # D67: the example's overnight part is what `precedent dream` writes, not a hand-written shape
    import sys

    sys.path.insert(0, str(ROOT / "tests" / "dream"))
    from example_dream import fixture_text

    fixture = (FIX / f"dream_{DATE}.md").read_text(encoding="utf-8")
    assert fixture == fixture_text(tmp_path)
    assert fixture.startswith("SYNTHETIC —")
    assert "event_direction positive" in fixture and "event_direction 1.0" not in fixture  # F-4


def test_the_committed_example_is_reproduced_byte_for_byte(tmp_path):
    assert _example_script().build(tmp_path) == 0
    committed = ROOT / "docs" / "examples" / f"brief_{DATE}.md"
    assert (tmp_path / f"brief_{DATE}.md").read_bytes() == committed.read_bytes()
    assert committed.read_text(encoding="utf-8").startswith("SYNTHETIC —")
    assert json.dumps(re.findall(r"\{\{", committed.read_text(encoding="utf-8"))) == "[]"
