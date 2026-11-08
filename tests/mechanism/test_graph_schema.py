"""Tests for the causal graph schema: vocabulary, validation rules and the reference schema."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from precedent.config import references_dir
from precedent.mechanism.graph_schema import (
    CausalGraph,
    Edge,
    Node,
    NodeKind,
    derive_kind,
    graph_json_schema,
    is_acyclic,
    load_params,
    node_vocabulary,
)
from precedent.spine.event_key import EventKey
from precedent.spine.event_schema import EventType

ROOT = Path(__file__).resolve().parents[2]
KEY = EventKey(
    event_type="central_bank",
    actor_kind="central_bank",
    impacted_kinds=("market",),
    country_group="US",
    severity_band="mid",
    is_scheduled=True,
    direction_sign=1,
)


def graph(edges, version: str = "template-v1", key: EventKey = KEY) -> CausalGraph:
    """Build a graph from (source, target[, evidence span]) tuples; kinds come from the ids."""
    rows = [(*edge, None) if len(edge) == 2 else edge for edge in edges]
    node_ids = list(dict.fromkeys(i for source, target, _ in rows for i in (source, target)))
    return CausalGraph(
        graph_id="g1",
        graph_version=version,
        event_key=key,
        nodes=[Node(id=i, kind=derive_kind(i)) for i in node_ids],
        edges=[
            Edge(
                source=source,
                target=target,
                sign=1,
                lag_band="days",
                source_doc_id=None if span is None else "doc-1",
                evidence_span=span,
            )
            for source, target, span in rows
        ],
    )


G = graph([("event:central_bank", "rates_expectation"), ("rates_expectation", "SPY")])


def test_cycle_is_rejected():
    with pytest.raises(ValidationError, match="cycle"):
        graph(
            edges=[
                ("event:central_bank", "rates_expectation"),
                ("rates_expectation", "risk_appetite"),
                ("risk_appetite", "rates_expectation"),
            ]
        )


def test_self_loop_is_a_cycle():
    with pytest.raises(ValidationError, match="cycle"):
        graph(
            edges=[("event:central_bank", "rates_expectation"), ("risk_appetite", "risk_appetite")]
        )


def test_asset_cannot_cause_an_event_variable():
    with pytest.raises(ValidationError):
        graph(edges=[("SPY", "event:central_bank")])


def test_asset_cannot_cause_an_intermediate():
    with pytest.raises(ValidationError, match="asset"):
        graph(edges=[("event:central_bank", "SPY"), ("SPY", "risk_appetite")])


def test_asset_may_cause_another_asset():
    graph(edges=[("event:central_bank", "SPY"), ("SPY", "QQQ")])


def test_unknown_node_is_rejected_not_added():
    with pytest.raises(ValidationError, match="my_new_variable"):
        graph(edges=[("event:central_bank", "my_new_variable"), ("my_new_variable", "SPY")])


def test_edge_endpoint_missing_from_nodes_is_rejected():
    dumped = G.model_dump(mode="json")
    dumped["edges"][1]["target"] = "QQQ"
    with pytest.raises(ValidationError, match="not in nodes"):
        CausalGraph.model_validate(dumped)


def test_node_kind_must_match_its_id():
    with pytest.raises(ValidationError, match="kind"):
        CausalGraph(
            graph_id="g1",
            graph_version="template-v1",
            event_key=KEY,
            nodes=[Node(id="event:central_bank", kind=NodeKind.ASSET)],
            edges=[],
        )


def test_graph_needs_an_event_node():
    with pytest.raises(ValidationError, match="event node"):
        graph(edges=[("rates_expectation", "SPY")])


def test_event_node_must_name_the_key_event_type():
    with pytest.raises(ValidationError, match="event_type"):
        graph(edges=[("event:earnings", "rates_expectation")])


def test_event_node_helpers():
    g = graph(
        edges=[
            ("event:central_bank", "rates_expectation"),
            ("event:central_bank:direction_sign", "rates_expectation"),
        ]
    )
    assert g.event_node_ids == ["event:central_bank", "event:central_bank:direction_sign"]
    assert g.event_node_id == "event:central_bank"


def test_topological_order_and_parents():
    order = G.topological_order()
    assert order.index("event:central_bank") < order.index("rates_expectation") < order.index("SPY")
    assert G.parents("SPY") == ["rates_expectation"]
    assert G.parents("event:central_bank") == []


def test_duplicate_edge_is_rejected():
    with pytest.raises(ValidationError, match="duplicate"):
        graph(edges=[("event:central_bank", "SPY"), ("event:central_bank", "SPY")])


def test_model_graph_needs_evidence_on_every_edge():
    with pytest.raises(ValidationError, match="evidence"):
        graph(edges=[("event:central_bank", "rates_expectation", None)], version="llm-graph-v1")
    with pytest.raises(ValidationError, match="evidence"):
        graph(edges=[("event:central_bank", "rates_expectation", "  ")], version="llm-graph-v1")
    graph(edges=[("event:central_bank", "rates_expectation", "rates fell")], version="llm-graph-v1")
    graph(edges=[("event:central_bank", "rates_expectation", None)], version="template-v1")


def test_placebo_graph_is_exempt_from_the_evidence_rule():
    graph(edges=[("event:central_bank", "rates_expectation", None)], version="placebo-3")
    with pytest.raises(ValidationError, match="evidence"):
        graph(edges=[("event:central_bank", "rates_expectation", None)], version="llm-graph-v1")


def test_is_acyclic_reports_cycles():
    ids = ["rates_expectation", "risk_appetite"]

    def edge(source: str, target: str) -> Edge:
        return Edge(
            source=source,
            target=target,
            sign=1,
            lag_band="days",
            source_doc_id=None,
            evidence_span=None,
        )

    assert is_acyclic(ids, [edge(ids[0], ids[1])])
    assert not is_acyclic(ids, [edge(ids[0], ids[1]), edge(ids[1], ids[0])])


def test_edge_rejects_unknown_fields_and_enum_values():
    base = {"source": "a", "target": "b", "sign": 1, "lag_band": "days"}
    for extra in ({"weight": 0.5}, {"probability": 0.5}, {"confidence": 0.5}):
        with pytest.raises(ValidationError):
            Edge.model_validate({**base, **extra, "source_doc_id": None, "evidence_span": None})
    for bad in ({"sign": 0}, {"sign": 2}, {"lag_band": "hours"}):
        with pytest.raises(ValidationError):
            Edge.model_validate({**base, **bad, "source_doc_id": None, "evidence_span": None})


def test_no_weight_probability_or_confidence_field_exists():
    banned = ("weight", "prob", "confidence")
    for model in (Edge, Node, CausalGraph):
        assert [f for f in model.model_fields if any(b in f for b in banned)] == []


def test_vocabulary_contents():
    params = load_params()
    vocab = node_vocabulary(
        references_dir() / "mechanism" / "intermediates.json",
        params.asset_panel,
        [member.value for member in EventType],
    )
    assert "event:central_bank" in vocab
    assert "event:earnings:severity_band" in vocab
    assert "event:mna:direction_sign" in vocab
    assert "event:armed_conflict:is_scheduled" in vocab
    assert "event:central_bank:horizon" not in vocab
    for name in ("rates_expectation", "risk_appetite", "oil_supply", "usd_demand", "credit_spread"):
        assert name in vocab
    for sector in ["XLB", "XLC", "XLE", "XLF", "XLI", "XLK", "XLP", "XLRE", "XLU", "XLV", "XLY"]:
        assert f"sector_demand:{sector}" in vocab
    assert "sector_demand:<sector>" not in vocab
    assert "sector_demand:XLZ" not in vocab
    assert {"SPY", "TLT", "UUP", "EFA"} <= vocab


def test_derive_kind():
    assert derive_kind("event:mna") is NodeKind.EVENT
    assert derive_kind("SPY") is NodeKind.ASSET
    assert derive_kind("credit_spread") is NodeKind.INTERMEDIATE
    assert derive_kind("sector_demand:XLE") is NodeKind.INTERMEDIATE


def test_params_file():
    params = load_params()
    assert params.n_paths == 10000
    assert len(params.asset_panel) == 22 and len(set(params.asset_panel)) == 22
    assert "engine.asset_panel" in json.dumps(params.notes)
    with pytest.raises(ValidationError):
        type(params).model_validate({**params.model_dump(), "extra_field": 1})


def test_params_loader_reads_a_given_path(tmp_path):
    path = tmp_path / "params.json"
    path.write_text(json.dumps({"notes": {}, "n_paths": 5, "asset_panel": ["SPY"]}))
    assert load_params(path).n_paths == 5


def test_graph_round_trips_through_json():
    assert CausalGraph.model_validate_json(G.model_dump_json()) == G


def test_reference_schema_file_matches_the_code():
    stored = json.loads((ROOT / "references/mechanism/graph_schema_v1.json").read_text())
    assert stored == graph_json_schema()
