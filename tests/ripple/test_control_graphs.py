import numpy as np
import pandas as pd
import pytest

from precedent.ripple.control_graphs import (
    correlation_links,
    correlation_links_as_links,
    sector_links,
    shuffle_by_relation,
    shuffle_preserving_degree,
)
from precedent.ripple.link_store import Link, Relation


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


@pytest.fixture
def returns_frame() -> pd.DataFrame:
    """400 sessions of 12 instruments plus MKT; two latent factors give real residual structure."""
    rng = np.random.default_rng(11)
    sessions = pd.bdate_range("2020-01-01", periods=400)
    market = rng.normal(0.0, 0.01, size=400)
    factor_a = rng.normal(0.0, 0.01, size=400)
    factor_b = rng.normal(0.0, 0.01, size=400)
    columns = {"MKT": market}
    for position in range(12):
        loading = factor_a if position < 6 else factor_b
        columns[f"S{position:02d}"] = (
            0.8 * market + 0.9 * loading + rng.normal(0.0, 0.005, size=400)
        )
    return pd.DataFrame(columns, index=sessions)


@pytest.fixture
def graph_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    nodes = [f"N{i}" for i in range(10)]
    edges: set[tuple[str, str]] = set()
    while len(edges) < 30:
        source, receiver = rng.choice(nodes, size=2, replace=False)
        edges.add((str(source), str(receiver)))
    ordered = sorted(edges)
    return pd.DataFrame(
        {
            "source_node": [s for s, _ in ordered],
            "receiver_node": [r for _, r in ordered],
            "relation": "peer_of",
            "weight": np.linspace(0.1, 1.0, len(ordered)),
        }
    )


# --- sector graph ---------------------------------------------------------------------------


def test_sector_links_are_equal_weight_peer_links_per_ordered_pair():
    sector_of = {"A": "tech", "B": "tech", "C": "tech", "D": "energy", "E": "energy", "F": "solo"}
    links = sector_links(sector_of, utc("2020-01-01"), utc("2030-01-01"))
    assert all(isinstance(item, Link) for item in links)
    assert len(links) == 3 * 2 + 2 * 1
    assert {item.relation for item in links} == {Relation.PEER_OF}
    assert {item.weight for item in links} == {1.0}
    assert {item.extractor_version for item in links} == {"rules-sector-v1"}
    assert {item.evidence_span for item in links} == {None}
    assert all(item.source_node != item.receiver_node for item in links)
    assert not any("F" in (item.source_node, item.receiver_node) for item in links)
    tech = [item for item in links if item.source_doc_id == "sector:tech"]
    assert {(item.source_node, item.receiver_node) for item in tech} == {
        (a, b) for a in "ABC" for b in "ABC" if a != b
    }
    assert links == sector_links(sector_of, utc("2020-01-01"), utc("2030-01-01"))


def test_sector_links_reject_naive_timestamps():
    with pytest.raises(ValueError):
        sector_links({"A": "x", "B": "x"}, pd.Timestamp("2020-01-01").to_pydatetime(), utc("2030-01-01"))


# --- correlation graph ----------------------------------------------------------------------


def test_correlation_graph_uses_only_past_returns(returns_frame):
    before = correlation_links(returns_frame, "MKT", asof_pos=300)
    noisy = returns_frame.copy()
    noisy.iloc[301:] = np.random.default_rng(0).normal(size=noisy.iloc[301:].shape)
    assert correlation_links(noisy, "MKT", asof_pos=300) == before
    assert before


def test_correlation_graph_window_edges(returns_frame):
    before = correlation_links(returns_frame, "MKT", asof_pos=300, window=250)
    outside = returns_frame.copy()
    outside.iloc[300 - 250] = 5.0  # position 50: one before the window start (51)
    assert correlation_links(outside, "MKT", asof_pos=300, window=250) == before
    inside = returns_frame.copy()
    inside.iloc[300 - 250 + 1, 1] = 0.5  # position 51: first row of the window
    assert correlation_links(inside, "MKT", asof_pos=300, window=250) != before


def test_correlations_are_positive_capped_and_exclude_the_market(returns_frame):
    pairs = correlation_links(returns_frame, "MKT", asof_pos=399, top_k=3)
    assert pairs
    assert all(isinstance(value, float) and value > 0 for _, _, value in pairs)
    assert all("MKT" not in (source, receiver) for source, receiver, _ in pairs)
    assert all(source != receiver for source, receiver, _ in pairs)
    per_receiver: dict[str, list[float]] = {}
    for _, receiver, value in pairs:
        per_receiver.setdefault(receiver, []).append(value)
    assert all(len(values) <= 3 for values in per_receiver.values())
    assert all(values == sorted(values, reverse=True) for values in per_receiver.values())
    assert [receiver for _, receiver, _ in pairs] == sorted(receiver for _, receiver, _ in pairs)


def test_residual_correlation_finds_the_latent_factor_groups(returns_frame):
    pairs = correlation_links(returns_frame, "MKT", asof_pos=399, top_k=2)
    group_of = {f"S{i:02d}": 0 if i < 6 else 1 for i in range(12)}
    assert all(group_of[source] == group_of[receiver] for source, receiver, _ in pairs)


def test_correlation_links_too_early_or_past_the_end_raise(returns_frame):
    with pytest.raises(ValueError, match="window"):
        correlation_links(returns_frame, "MKT", asof_pos=100, window=250)
    with pytest.raises(ValueError, match="asof_pos"):
        correlation_links(returns_frame, "MKT", asof_pos=400)


def test_zero_variance_column_gets_no_links(returns_frame):
    flat = returns_frame.copy()
    flat["FLAT"] = 0.0
    pairs = correlation_links(flat, "MKT", asof_pos=399)
    assert all("FLAT" not in (source, receiver) for source, receiver, _ in pairs)


def test_correlation_pairs_wrap_into_valid_links(returns_frame):
    pairs = correlation_links(returns_frame, "MKT", asof_pos=399, top_k=2)
    asof_session = returns_frame.index[399]
    links = correlation_links_as_links(pairs, asof_session, utc("2021-07-01"), utc("2021-07-02"))
    assert len(links) == len(pairs)
    assert {item.relation for item in links} == {Relation.PEER_OF}
    assert {item.extractor_version for item in links} == {"rules-correlation-v1"}
    assert {item.source_doc_id for item in links} == {f"corr:{asof_session.date().isoformat()}"}
    assert all(0.0 < item.weight <= 1.0 for item in links)
    assert [item.weight for item in links] == [value for _, _, value in pairs]


# --- degree-preserving placebo --------------------------------------------------------------


def test_shuffle_keeps_degrees_and_changes_edges(graph_frame):
    shuffled = shuffle_preserving_degree(graph_frame, seed=7)
    assert shuffled.groupby("source_node").size().to_dict() == graph_frame.groupby("source_node").size().to_dict()
    assert shuffled.groupby("receiver_node").size().to_dict() == graph_frame.groupby("receiver_node").size().to_dict()
    assert set(map(tuple, shuffled[["source_node", "receiver_node"]].values)) != set(map(tuple, graph_frame[["source_node", "receiver_node"]].values))
    assert (shuffled.source_node != shuffled.receiver_node).all()
    pd.testing.assert_frame_equal(shuffled, shuffle_preserving_degree(graph_frame, seed=7))


def test_shuffle_has_no_duplicate_pairs_and_keeps_columns_with_the_source_row(graph_frame):
    shuffled = shuffle_preserving_degree(graph_frame, seed=3)
    assert list(shuffled.columns) == list(graph_frame.columns)
    assert len(shuffled) == len(graph_frame)
    assert not shuffled.duplicated(["source_node", "receiver_node"]).any()
    assert shuffled.index.tolist() == list(range(len(graph_frame)))
    assert shuffled.equals(
        shuffled.sort_values(["source_node", "receiver_node"]).reset_index(drop=True)
    )
    # each source keeps the multiset of its non-node columns (weights travel with the source)
    for source, group in graph_frame.groupby("source_node"):
        assert sorted(group.weight) == sorted(shuffled[shuffled.source_node == source].weight)


def test_shuffle_differs_across_seeds_and_leaves_input_untouched(graph_frame):
    original = graph_frame.copy()
    first = shuffle_preserving_degree(graph_frame, seed=1)
    second = shuffle_preserving_degree(graph_frame, seed=2)
    pd.testing.assert_frame_equal(graph_frame, original)
    assert not first[["source_node", "receiver_node"]].equals(second[["source_node", "receiver_node"]])


@pytest.mark.parametrize(
    ("sources", "receivers"),
    [(["A", "B"], ["B", "A"]), (["A", "A"], ["B", "C"])],
)
def test_shuffle_of_a_graph_too_small_to_shuffle_raises(sources, receivers):
    tiny = pd.DataFrame({"source_node": sources, "receiver_node": receivers, "weight": [1.0, 1.0]})
    with pytest.raises(ValueError, match="too small"):
        shuffle_preserving_degree(tiny, seed=0)


def test_shuffle_rejects_self_links_and_duplicate_pairs():
    selfish = pd.DataFrame({"source_node": ["A", "B"], "receiver_node": ["A", "C"]})
    with pytest.raises(ValueError, match="self-links"):
        shuffle_preserving_degree(selfish, seed=0)
    doubled = pd.DataFrame({"source_node": ["A", "A"], "receiver_node": ["B", "B"]})
    with pytest.raises(ValueError, match="duplicate"):
        shuffle_preserving_degree(doubled, seed=0)


# --- per-relation placebo (final review I3) -------------------------------------------------


def two_relation_graph() -> pd.DataFrame:
    """Customer and supplier edges over the same nodes; some pairs carry both relations."""
    rng = np.random.default_rng(9)
    nodes = [f"N{i}" for i in range(9)]
    rows = []
    for relation in ("customer_of", "supplier_of"):
        edges: set[tuple[str, str]] = set()
        while len(edges) < 18:
            source, receiver = rng.choice(nodes, size=2, replace=False)
            edges.add((str(source), str(receiver)))
        rows += [
            {"source_node": s, "receiver_node": r, "relation": relation, "weight": 1.0}
            for s, r in sorted(edges)
        ]
    graph = pd.DataFrame(rows)
    pairs = graph.groupby(["source_node", "receiver_node"]).size()
    assert (pairs > 1).any()  # the fixture does exercise a pair named under two relations
    return graph


def degrees(graph: pd.DataFrame) -> dict[str, dict]:
    return {
        column: graph.groupby(["relation", column]).size().to_dict()
        for column in ("source_node", "receiver_node")
    }


def test_shuffle_keeps_per_relation_in_and_out_degrees():
    graph = two_relation_graph()
    shuffled = shuffle_preserving_degree(graph, seed=4)
    assert degrees(shuffled) == degrees(graph)
    assert len(shuffled) == len(graph)
    assert not shuffled.duplicated(["source_node", "receiver_node", "relation"]).any()
    assert (shuffled.source_node != shuffled.receiver_node).all()
    for relation, group in graph.groupby("relation"):
        before = set(zip(group.source_node, group.receiver_node, strict=True))
        after_group = shuffled[shuffled.relation == relation]
        assert set(zip(after_group.source_node, after_group.receiver_node, strict=True)) != before


def test_a_pair_under_two_relations_is_not_a_duplicate():
    graph = pd.DataFrame(
        {
            "source_node": ["A", "A", "B", "C", "B", "C"],
            "receiver_node": ["B", "B", "C", "A", "A", "B"],
            "relation": ["customer_of", "supplier_of", "customer_of", "customer_of",
                         "supplier_of", "supplier_of"],
        }
    )
    shuffled, unchanged = shuffle_by_relation(graph, seed=1)
    assert len(shuffled) == 6 and unchanged >= 0
    still_duplicated = graph.duplicated(["source_node", "receiver_node", "relation"])
    assert not still_duplicated.any()


def test_the_same_pair_twice_under_one_relation_is_still_a_duplicate():
    graph = pd.DataFrame(
        {
            "source_node": ["A", "A"],
            "receiver_node": ["B", "B"],
            "relation": ["customer_of", "customer_of"],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        shuffle_by_relation(graph, seed=0)


def test_a_relation_no_swap_can_change_is_returned_unchanged_and_counted():
    mutual = pd.DataFrame(
        {
            "source_node": ["A", "B"],
            "receiver_node": ["B", "A"],
            "relation": ["supplier_of", "supplier_of"],
            "weight": [0.5, 0.7],
        }
    )
    graph = pd.concat([two_relation_graph().query("relation == 'customer_of'"), mutual])
    shuffled, unchanged = shuffle_by_relation(graph, seed=2)
    assert unchanged == 1
    kept = shuffled[shuffled.relation == "supplier_of"]
    assert set(zip(kept.source_node, kept.receiver_node, strict=True)) == {("A", "B"), ("B", "A")}
    assert degrees(shuffled) == degrees(graph)


def test_a_two_node_mutual_graph_does_not_crash_the_relation_shuffle():
    mutual = pd.DataFrame(
        {"source_node": ["A", "B"], "receiver_node": ["B", "A"], "relation": ["peer_of"] * 2}
    )
    shuffled, unchanged = shuffle_by_relation(mutual, seed=0)
    assert unchanged == 1
    pd.testing.assert_frame_equal(shuffled, mutual)
    with pytest.raises(ValueError, match="too small"):  # the strict function still refuses
        shuffle_preserving_degree(mutual, seed=0)


def test_a_single_receiver_graph_is_unshufflable_and_counted():
    one_receiver = pd.DataFrame(
        {
            "source_node": ["A", "B", "C"],
            "receiver_node": ["Z", "Z", "Z"],
            "relation": ["customer_of"] * 3,
        }
    )
    _, unchanged = shuffle_by_relation(one_receiver, seed=0)
    assert unchanged == 1


def test_a_shufflable_graph_never_comes_back_unchanged_whatever_the_seed():
    # Three independent edges: random swaps sometimes cancel out; the shuffle must still change.
    graph = pd.DataFrame(
        {"source_node": ["A", "C", "E"], "receiver_node": ["B", "D", "F"], "relation": "peer_of"}
    )
    key = ["source_node", "receiver_node"]
    original = set(map(tuple, graph[key].to_numpy()))
    for seed in range(300):
        shuffled = shuffle_preserving_degree(graph, seed=seed)
        assert set(map(tuple, shuffled[key].to_numpy())) != original, seed


def test_too_small_message_names_the_real_cause():
    tiny = pd.DataFrame({"source_node": ["A"], "receiver_node": ["B"]})
    with pytest.raises(ValueError, match="too small or too constrained"):
        shuffle_preserving_degree(tiny, seed=0)
