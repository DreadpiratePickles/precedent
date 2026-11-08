from pathlib import Path

REQUIRED_HEADINGS = [
    "## Objective",
    "## Inputs",
    "## Process",
    "## Outputs",
    "## Verify",
    "## Approval",
    "## Failure Behavior",
]
STAGES = (
    "20_precedent_libraries",
    "21_precedent_evaluate",
    "22_precedent_factors",
    "23_precedent_card_mcp",
    "24_library_snapshot",
)
ROOT = Path(__file__).resolve().parents[2]


def test_precedent_stage_contracts_are_complete_and_single_pass():
    for stage in STAGES:
        text = (ROOT / "stages" / stage / "CONTEXT.md").read_text()
        assert [h for h in REQUIRED_HEADINGS if h not in text] == [], stage
        assert "single pass" in text.lower(), stage
        assert len(text.splitlines()) <= 60, stage


def test_card_contract_names_the_mandatory_lines_and_owner_gates():
    card = (ROOT / "stages" / "23_precedent_card_mcp" / "CONTEXT.md").read_text()
    assert "library covers" in card
    assert "These are past outcomes of similar events, not a forecast." in card
    assert "owner gate" in card.lower()
    snapshot = (ROOT / "stages" / "24_library_snapshot" / "CONTEXT.md").read_text()
    assert "owner gate" in snapshot.lower() and "SHA-256" in snapshot


def test_evaluate_contract_names_the_rungs_and_a_strict_loop_oracle():
    text = (ROOT / "stages" / "21_precedent_evaluate" / "CONTEXT.md").read_text()
    objective = text.split("## Objective")[1].split("## Inputs")[0]
    assert all(rung in objective for rung in ("B0", "B1", "B2", "B3"))
    assert "canonical_text" in text and "embedder" in text
    loop = text.split("## Loop")[1]
    oracle = next(line for line in loop.splitlines() if line.startswith("Oracle:"))
    assert "lower than" in oracle and "0.5%" in oracle and "not worse" not in oracle
    assert "uv run ruff check . && uv run pytest -q" in oracle
    assert "B2" in loop and "contamination" in loop.lower()
    assert "embedding_weight" in loop
