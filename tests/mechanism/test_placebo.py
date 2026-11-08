"""Tests for the degree-preserving placebo graph and the M4 rung files."""

from pathlib import Path

import pytest

from precedent.config import references_dir
from precedent.mechanism.evaluate import PlaceboError, shuffle_edges_preserving_degree
from precedent.mechanism.graph_schema import (
    CausalGraph,
    Edge,
    Node,
    NodeKind,
    derive_kind,
    load_params,
    node_vocabulary,
)
from precedent.mechanism.templates import load_templates
from precedent.spine.event_schema import EventType

ASSET_PANEL = load_params().asset_panel
VOCAB = node_vocabulary(
    references_dir() / "mechanism" / "intermediates.json",
    ASSET_PANEL,
    [e.value for e in EventType],
)
TEMPLATES = load_templates(references_dir() / "mechanism" / "templates", VOCAB)
TEMPLATE = TEMPLATES["central_bank"]
RUNGS_DIR = references_dir() / "rungs"


def degrees(graph: CausalGraph) -> dict[str, tuple[int, int]]:
    """Per node: (in-degree, out-degree)."""
    return {
        node.id: (
            sum(edge.target == node.id for edge in graph.edges),
            sum(edge.source == node.id for edge in graph.edges),
        )
        for node in graph.nodes
    }


def edge_set(graph: CausalGraph) -> set[tuple[str, str]]:
    return {(edge.source, edge.target) for edge in graph.edges}


def test_placebo_keeps_degrees_and_stays_a_valid_dag():
    placebo = shuffle_edges_preserving_degree(TEMPLATE, seed=5)
    assert degrees(placebo) == degrees(TEMPLATE) and placebo.graph_version == "placebo-5"
    assert edge_set(placebo) != edge_set(TEMPLATE)
    assert placebo.graph_id == f"{TEMPLATE.graph_id}:placebo-5"
    assert placebo.event_key == TEMPLATE.event_key
    assert [n.id for n in placebo.nodes] == [n.id for n in TEMPLATE.nodes]
    # Re-validation checks the DAG, vocabulary and kind rules independently of the shuffler.
    CausalGraph.model_validate(placebo.model_dump())
    assert len(placebo.topological_order()) == len(placebo.nodes)


def test_placebo_edges_carry_no_evidence():
    placebo = shuffle_edges_preserving_degree(TEMPLATE, seed=2)
    assert all(e.source_doc_id is None and e.evidence_span is None for e in placebo.edges)
    assert all(e.support == 1 for e in placebo.edges)
    assert all(e.sign in (-1, 1) for e in placebo.edges)


def test_same_seed_is_identical_and_seeds_differ():
    assert shuffle_edges_preserving_degree(TEMPLATE, 7) == shuffle_edges_preserving_degree(
        TEMPLATE, 7
    )
    distinct = {frozenset(edge_set(shuffle_edges_preserving_degree(TEMPLATE, s))) for s in range(5)}
    assert len(distinct) >= 2


# Star-shaped templates admit no degree-preserving swap: every pair of edges shares an
# endpoint or would make a self-loop, so there is no placebo (Task 5b must treat M-P as absent).
UNREWIRABLE = {"earnings", "macro_data_release"}


@pytest.mark.parametrize("event_type", sorted(TEMPLATES))
def test_placebo_respects_event_and_asset_kind_rules(event_type):
    template = TEMPLATES[event_type]
    if event_type in UNREWIRABLE:
        for seed in range(10):
            with pytest.raises(PlaceboError, match=template.graph_id):
                shuffle_edges_preserving_degree(template, seed)
        return
    for seed in range(10):
        placebo = shuffle_edges_preserving_degree(template, seed)
        kinds = {n.id: n.kind for n in placebo.nodes}
        assert degrees(placebo) == degrees(template)
        assert edge_set(placebo) != edge_set(template)
        for edge in placebo.edges:
            assert kinds[edge.target] is not NodeKind.EVENT
            assert not (
                kinds[edge.source] is NodeKind.ASSET and kinds[edge.target] is not NodeKind.ASSET
            )


def test_a_graph_with_one_edge_cannot_be_rewired():
    node_ids = ["event:central_bank:direction_sign", "rates_expectation"]
    single = CausalGraph(
        graph_id="template:one_edge",
        graph_version="template-v1",
        event_key=TEMPLATE.event_key,
        nodes=[Node(id=i, kind=derive_kind(i)) for i in node_ids],
        edges=[
            Edge(
                source=node_ids[0],
                target=node_ids[1],
                sign=1,
                lag_band="days",
                source_doc_id=None,
                evidence_span=None,
            )
        ],
    )
    with pytest.raises(PlaceboError, match="template:one_edge"):
        shuffle_edges_preserving_degree(single, seed=0)


def test_rung_files_exist_and_name_themselves():
    for name in ("m4_mechanism_template", "m4_mechanism_placebo"):
        text = Path(RUNGS_DIR / f"{name}.yaml").read_text(encoding="utf-8")
        assert text.startswith(f"name: {name}\n")
        assert "panel_prefixes: [mech_]" in text
