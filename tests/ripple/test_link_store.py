import pandas as pd
import pytest
from pydantic import ValidationError

from precedent.ripple.link_store import Link, LinkStore, Relation, graph_asof
from precedent.ripple.params import load_ripple_params

GRAPH_COLUMNS = [
    "source_node",
    "receiver_node",
    "relation",
    "weight",
    "valid_from_utc",
    "valid_to_utc",
    "source_doc_id",
    "evidence_span",
    "extractor_version",
]


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def link(
    source,
    receiver,
    relation,
    valid_from="2020-01-01T00:00:00Z",
    valid_to="2030-01-01T00:00:00Z",
    weight=1.0,
    doc="DOC",
    extractor="rules-test-v1",
    evidence=None,
) -> Link:
    return Link(
        source_node=source,
        receiver_node=receiver,
        relation=relation,
        weight=weight,
        valid_from_utc=valid_from,
        valid_to_utc=valid_to,
        source_doc_id=doc,
        evidence_span=evidence,
        extractor_version=extractor,
    )


def test_link_is_invisible_before_its_document_and_after_expiry(tmp_path):
    store = LinkStore(tmp_path)
    store.append(
        [
            link(
                "SUP",
                "CUS",
                "supplier_of",
                valid_from="2020-03-02T21:30:00Z",
                valid_to="2021-06-02T21:30:00Z",
            )
        ]
    )
    assert graph_asof(store, utc("2020-03-02 21:00:00")).empty  # accepted after that day's cutoff
    assert len(graph_asof(store, utc("2020-03-03 21:00:00"))) == 1
    assert graph_asof(store, utc("2021-06-03 21:00:00")).empty


def test_a_restated_link_extends_validity_without_duplicating(tmp_path):
    store = LinkStore(tmp_path)
    store.append(
        [
            link(
                "SUP",
                "CUS",
                "supplier_of",
                "2020-03-02T21:30:00Z",
                "2021-06-02T21:30:00Z",
                doc="10K-2019",
            ),
            link(
                "SUP",
                "CUS",
                "supplier_of",
                "2021-03-01T21:30:00Z",
                "2022-06-01T21:30:00Z",
                doc="10K-2020",
            ),
        ]
    )
    graph = graph_asof(store, utc("2021-04-01 21:00:00"))
    assert len(graph) == 1 and graph.iloc[0].source_doc_id == "10K-2020"


def test_model_extracted_link_needs_evidence():
    with pytest.raises(ValidationError):
        link(
            "SUP",
            "CUS",
            "supplier_of",
            "2020-03-02T21:30:00Z",
            "2021-06-02T21:30:00Z",
            extractor="llm-links-v1",
            evidence=None,
        )


def test_model_extracted_link_with_evidence_is_accepted():
    made = link("SUP", "CUS", "supplier_of", extractor="llm-links-v1", evidence="sells chips to CUS")
    assert made.evidence_span == "sells chips to CUS"


def test_model_extracted_link_with_blank_evidence_is_rejected():
    with pytest.raises(ValidationError):
        link("SUP", "CUS", "supplier_of", extractor="llm-links-v1", evidence="   ")


def test_naive_timestamp_is_rejected():
    with pytest.raises(ValidationError):
        link("SUP", "CUS", "supplier_of", valid_from="2020-03-02T21:30:00")


@pytest.mark.parametrize("weight", [-0.01, 1.01])
def test_weight_outside_zero_one_is_rejected(weight):
    with pytest.raises(ValidationError):
        link("SUP", "CUS", "supplier_of", weight=weight)


@pytest.mark.parametrize("valid_to", ["2020-03-02T21:30:00Z", "2020-03-01T00:00:00Z"])
def test_valid_to_not_after_valid_from_is_rejected(valid_to):
    with pytest.raises(ValidationError):
        link("SUP", "CUS", "supplier_of", valid_from="2020-03-02T21:30:00Z", valid_to=valid_to)


def test_unknown_relation_and_extra_field_are_rejected():
    with pytest.raises(ValidationError):
        link("SUP", "CUS", "frenemy_of")
    with pytest.raises(ValidationError):
        Link.model_validate({**link("A", "B", "peer_of").model_dump(), "surprise": 1})


def test_exact_duplicate_append_is_skipped_and_not_stored_twice(tmp_path):
    store = LinkStore(tmp_path)
    first = link("SUP", "CUS", "supplier_of")
    report = store.append([first, link("SUP", "CUS", "supplier_of")])  # twin inside one batch
    assert (report.appended, report.duplicates_skipped) == (1, 1)
    again = store.append([first, link("A", "B", "peer_of")])
    assert (again.appended, again.duplicates_skipped) == (1, 1)
    assert len(store.read_all()) == 2
    assert len((tmp_path / "links.jsonl").read_text().splitlines()) == 2


def test_a_link_differing_only_in_document_is_not_a_duplicate(tmp_path):
    store = LinkStore(tmp_path)
    report = store.append([link("A", "B", "peer_of", doc="D1"), link("A", "B", "peer_of", doc="D2")])
    assert (report.appended, report.duplicates_skipped) == (2, 0)


def test_store_creates_parents_and_reads_empty_when_absent(tmp_path):
    store = LinkStore(tmp_path / "deep" / "links")
    assert store.read_all() == []
    store.append([link("A", "B", "peer_of")])
    assert (tmp_path / "deep" / "links" / "links.jsonl").is_file()
    assert [made.relation for made in store.read_all()] == [Relation.PEER_OF]


def test_graph_asof_with_naive_cutoff_raises(tmp_path):
    store = LinkStore(tmp_path)
    with pytest.raises(ValueError, match="tz-aware"):
        graph_asof(store, pd.Timestamp("2021-01-01"))


def test_graph_is_half_open_at_both_ends(tmp_path):
    store = LinkStore(tmp_path)
    store.append([link("A", "B", "peer_of", "2020-01-01T00:00:00Z", "2020-02-01T00:00:00Z")])
    assert len(graph_asof(store, utc("2020-01-01 00:00:00"))) == 1
    assert graph_asof(store, utc("2020-02-01 00:00:00")).empty


def test_relation_filter(tmp_path):
    store = LinkStore(tmp_path)
    store.append([link("A", "B", "peer_of"), link("A", "B", "supplier_of")])
    only_peers = graph_asof(store, utc("2021-01-01"), Relation.PEER_OF)
    assert only_peers["relation"].tolist() == ["peer_of"]
    assert len(graph_asof(store, utc("2021-01-01"))) == 2


def test_different_relations_between_the_same_nodes_are_kept_apart(tmp_path):
    store = LinkStore(tmp_path)
    store.append([link("A", "B", "peer_of", doc="D1"), link("A", "B", "supplier_of", doc="D2")])
    assert len(graph_asof(store, utc("2021-01-01"))) == 2


def test_empty_frame_has_exact_columns(tmp_path):
    store = LinkStore(tmp_path)
    assert list(graph_asof(store, utc("2021-01-01")).columns) == GRAPH_COLUMNS
    store.append([link("A", "B", "peer_of", "2020-01-01T00:00:00Z", "2020-02-01T00:00:00Z")])
    late = graph_asof(store, utc("2025-01-01"))
    assert late.empty and list(late.columns) == GRAPH_COLUMNS


def test_visible_frame_columns_values_and_order(tmp_path):
    store = LinkStore(tmp_path)
    store.append([link("A", "B", "customer_of", weight=0.25, doc="D1", extractor="llm-v1", evidence="x")])
    graph = graph_asof(store, utc("2021-01-01"))
    assert list(graph.columns) == GRAPH_COLUMNS
    row = graph.iloc[0]
    assert row["relation"] == "customer_of" and row["weight"] == 0.25
    assert row["evidence_span"] == "x" and row["extractor_version"] == "llm-v1"
    assert row["valid_from_utc"] == utc("2020-01-01")


def test_load_ripple_params_returns_the_proposed_values():
    params = load_ripple_params()
    assert params["alpha"] == 0.5 and params["hops"] == 2
    assert params["half_life_sessions"] == 3.0 and params["max_age_sessions"] == 20
    assert params["annual_link_expiry_months"] == 15
    assert params["correlation_window_sessions"] == 250 and params["correlation_top_k"] == 10
    assert params["proposed"] is True
