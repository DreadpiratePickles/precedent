"""Tests for the template graphs: vocabulary, reachability, lookup and loader refusals."""

import json
from pathlib import Path

import pytest

from precedent.config import references_dir
from precedent.mechanism.graph_schema import descendants, load_params, node_vocabulary
from precedent.mechanism.templates import TemplateError, load_templates, template_for
from precedent.spine.event_key import EventKey
from precedent.spine.event_schema import EventType

ASSET_PANEL = load_params().asset_panel
VOCAB = node_vocabulary(
    references_dir() / "mechanism" / "intermediates.json",
    ASSET_PANEL,
    [e.value for e in EventType],
)
TEMPLATES_DIR = references_dir() / "mechanism" / "templates"
EXPECTED_TYPES = {
    "central_bank",
    "macro_data_release",
    "trade_or_sanctions",
    "commodity_supply",
    "earnings",
    "guidance",
    "armed_conflict",
    "natural_disaster",
}


def key_with(event_type: str = "other", direction_sign: int = 1) -> EventKey:
    return EventKey(
        event_type=event_type,
        actor_kind="central_bank",
        impacted_kinds=("market",),
        country_group="US",
        severity_band="high",
        is_scheduled=True,
        direction_sign=direction_sign,
    )


def test_every_template_validates_against_the_vocabulary():
    templates = load_templates(TEMPLATES_DIR, VOCAB)
    assert set(templates) == EXPECTED_TYPES
    assert all(g.graph_version == "template-v1" for g in templates.values())
    assert all(g.graph_id == f"template:{t}" for t, g in templates.items())
    assert all(g.event_node_id == f"event:{t}:direction_sign" for t, g in templates.items())


def test_template_reaches_at_least_one_asset_from_the_event_node():
    for graph in load_templates(TEMPLATES_DIR, VOCAB).values():
        assert descendants(graph, graph.event_node_id) & set(ASSET_PANEL)


def test_unknown_event_type_has_no_template():
    assert template_for(key_with(event_type="other"), load_templates(TEMPLATES_DIR, VOCAB)) is None


def test_template_for_carries_the_query_key_through():
    key = key_with(event_type="central_bank", direction_sign=-1)
    result = template_for(key, load_templates(TEMPLATES_DIR, VOCAB))
    assert result is not None
    assert result.event_key == key
    assert result.graph_version == "template-v1"


def test_central_bank_template_has_the_plan_edges():
    graph = load_templates(TEMPLATES_DIR, VOCAB)["central_bank"]
    got = {(e.source, e.target, e.sign) for e in graph.edges}
    event = "event:central_bank:direction_sign"
    assert got == {
        (event, "rates_expectation", -1),
        ("rates_expectation", "TLT", -1),
        ("rates_expectation", "risk_appetite", -1),
        ("risk_appetite", "SPY", 1),
        ("risk_appetite", "HYG", 1),
        ("rates_expectation", "UUP", 1),
    }


def test_template_with_an_id_outside_the_vocabulary_is_refused_with_the_file_name(
    tmp_path: Path,
):
    bad = tmp_path / "central_bank.json"
    bad.write_text(
        json.dumps(
            {
                "event_type": "central_bank",
                "graph_version": "template-v1",
                "rationale": "test",
                "edges": [
                    {
                        "source": "event:central_bank:direction_sign",
                        "target": "made_up_node",
                        "sign": 1,
                        "lag_band": "same_session",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(TemplateError) as excinfo:
        load_templates(tmp_path, VOCAB)
    assert "central_bank.json" in str(excinfo.value)
    assert "made_up_node" in str(excinfo.value)


def test_every_rationale_states_the_spines_direction_sign_convention():
    files = sorted(TEMPLATES_DIR.glob("*.json"))
    assert {f.stem for f in files} == EXPECTED_TYPES
    for file in files:
        rationale = json.loads(file.read_text(encoding="utf-8"))["rationale"]
        assert "1 favourable, -1 unfavourable" in rationale, file.name
        assert "references/typing_prompt_v1.txt rule 8" in rationale, file.name
