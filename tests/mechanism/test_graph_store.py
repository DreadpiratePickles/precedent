"""Tests for the append-only graph store: idempotent, atomic, conflict-refusing."""

import pytest

from precedent.mechanism.graph_schema import CausalGraph, Edge, Node, derive_kind
from precedent.mechanism.graph_store import GraphConflictError, GraphStore
from precedent.spine.event_key import EventKey

KEY = EventKey("central_bank", "central_bank", ("market",), "US", "mid", True, 1)
OTHER_KEY = EventKey("central_bank", "central_bank", ("market",), "US", "high", True, -1)


def make_graph(key: EventKey = KEY, version: str = "template-v1", tail: str = "SPY") -> CausalGraph:
    pairs = [("event:central_bank", "rates_expectation"), ("rates_expectation", tail)]
    ids = list(dict.fromkeys(i for pair in pairs for i in pair))
    return CausalGraph(
        graph_id=f"{key.as_string()}@{version}",
        graph_version=version,
        event_key=key,
        nodes=[Node(id=i, kind=derive_kind(i)) for i in ids],
        edges=[
            Edge(
                source=s, target=t, sign=1, lag_band="days", source_doc_id=None, evidence_span=None
            )
            for s, t in pairs
        ],
    )


G = make_graph()


def test_store_is_idempotent_and_refuses_conflicts(tmp_path):
    store = GraphStore(tmp_path)
    assert store.append([G]).added == 1
    assert store.append([G]).identical_skipped == 1
    with pytest.raises(GraphConflictError):
        store.append([G.model_copy(update={"edges": G.edges[:-1]})])
    assert store.read("template-v1") == [G]


def test_conflict_is_all_or_nothing(tmp_path):
    store = GraphStore(tmp_path)
    store.append([G])
    fresh = make_graph(OTHER_KEY)
    changed = G.model_copy(update={"edges": G.edges[:-1]})
    with pytest.raises(GraphConflictError):
        store.append([fresh, changed])
    assert store.read("template-v1") == [G]


def test_for_key_finds_by_key_and_version(tmp_path):
    store = GraphStore(tmp_path)
    store.append([G, make_graph(OTHER_KEY)])
    assert store.for_key(KEY, "template-v1") == G
    assert store.for_key(OTHER_KEY, "template-v1") == make_graph(OTHER_KEY)
    assert store.for_key(KEY, "no-such-version") is None
    unknown = EventKey("central_bank", "central_bank", ("market",), "EM", "low", False, 0)
    assert store.for_key(unknown, "template-v1") is None


def test_one_file_per_version_and_missing_reads_empty(tmp_path):
    store = GraphStore(tmp_path)
    assert store.read("template-v1") == []
    store.append([G])
    assert store.path_for("template-v1").exists()
    assert store.path_for("template-v1") != store.path_for("template-v2")
    assert [p.name for p in tmp_path.iterdir()] == [store.path_for("template-v1").name]


def test_slug_collision_is_refused(tmp_path):
    store = GraphStore(tmp_path)
    with pytest.raises(ValueError, match="slug"):
        store.append([make_graph(version="template-a/b"), make_graph(version="template-a_b")])


def test_malformed_line_names_file_and_line(tmp_path):
    store = GraphStore(tmp_path)
    store.append([G])
    with store.path_for("template-v1").open("a", encoding="utf-8") as handle:
        handle.write("{not json}\n")
    with pytest.raises(ValueError, match="line 2"):
        store.read("template-v1")
