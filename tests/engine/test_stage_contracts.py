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
