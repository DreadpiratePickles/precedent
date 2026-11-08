"""Research: adjacency building and two-hop shock propagation (needs scipy)."""

import pytest

pytest.importorskip("scipy")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from precedent.ripple.propagate import (  # noqa: E402
    AdjacencyReport,
    build_adjacency,
    propagate_shock,
)


def links_frame(pairs, weight=1.0):
    """Links as `graph_asof` frames them: each pair is (source, receiver)."""
    return pd.DataFrame(
        {
            "source_node": [source for source, _ in pairs],
            "receiver_node": [receiver for _, receiver in pairs],
            "relation": "peer_of",
            "weight": weight,
        }
    )


def test_two_hop_chain():
    adjacency, _ = build_adjacency(links_frame([("A", "B"), ("B", "C")]), ["A", "B", "C"])  # (source, receiver)
    hop1, hop2 = propagate_shock(adjacency, np.array([1.0, 0.0, 0.0]))
    assert hop1.tolist() == [0.0, 0.5, 0.0] and hop2.tolist() == [0.0, 0.0, 0.25]


def test_incoming_weights_are_normalized():
    adjacency, _ = build_adjacency(links_frame([("A", "C"), ("B", "C")]), ["A", "B", "C"])
    hop1, _ = propagate_shock(adjacency, np.array([1.0, 0.0, 0.0]))
    assert hop1[2] == pytest.approx(0.25)  # half of C's incoming weight, times alpha


def test_no_shock_returns_to_its_origin():
    adjacency, _ = build_adjacency(links_frame([("A", "B"), ("B", "A")]), ["A", "B"])
    _, hop2 = propagate_shock(adjacency, np.array([1.0, 0.0]))
    assert hop2[0] == 0.0


def test_unknown_nodes_and_isolated_receivers_are_counted():
    _, report = build_adjacency(links_frame([("A", "B"), ("ZZZ", "B")]), ["A", "B", "C"])
    assert (report.links_dropped_unknown_node, report.receivers_without_links) == (1, 2)


def test_report_counts_links_used():
    _, report = build_adjacency(links_frame([("A", "B"), ("ZZZ", "B")]), ["A", "B", "C"])
    assert isinstance(report, AdjacencyReport)
    assert report.links_used == 1


def test_duplicate_links_sum_before_normalisation():
    graph = pd.concat(
        [
            links_frame([("A", "C")], weight=1.0),
            links_frame([("A", "C")], weight=1.0),
            links_frame([("B", "C")], weight=2.0),
        ],
        ignore_index=True,
    )
    adjacency, report = build_adjacency(graph, ["A", "B", "C"])
    assert adjacency[2, 0] == pytest.approx(0.5) and adjacency[2, 1] == pytest.approx(0.5)
    assert report.links_used == 3


def test_negative_weight_raises():
    with pytest.raises(ValueError, match="weight"):
        build_adjacency(links_frame([("A", "B")], weight=-1.0), ["A", "B"])


def test_non_finite_weight_raises():
    with pytest.raises(ValueError, match="weight"):
        build_adjacency(links_frame([("A", "B")], weight=float("nan")), ["A", "B"])


def test_duplicate_node_names_raise():
    with pytest.raises(ValueError, match="duplicate"):
        build_adjacency(links_frame([("A", "B")]), ["A", "B", "A"])


def test_shock_of_wrong_length_raises():
    adjacency, _ = build_adjacency(links_frame([("A", "B")]), ["A", "B"])
    with pytest.raises(ValueError, match="length"):
        propagate_shock(adjacency, np.array([1.0, 0.0, 0.0]))


def test_non_finite_shock_raises():
    adjacency, _ = build_adjacency(links_frame([("A", "B")]), ["A", "B"])
    with pytest.raises(ValueError, match="finite"):
        propagate_shock(adjacency, np.array([np.nan, 0.0]))


def test_three_hops_raises():
    adjacency, _ = build_adjacency(links_frame([("A", "B")]), ["A", "B"])
    with pytest.raises(ValueError, match="hops"):
        propagate_shock(adjacency, np.array([1.0, 0.0]), hops=3)


def test_one_hop_gives_a_zero_second_term():
    adjacency, _ = build_adjacency(links_frame([("A", "B"), ("B", "C")]), ["A", "B", "C"])
    hop1, hop2 = propagate_shock(adjacency, np.array([1.0, 0.0, 0.0]), hops=1)
    assert hop1.tolist() == [0.0, 0.5, 0.0]
    assert hop2.tolist() == [0.0, 0.0, 0.0]


def test_empty_graph_gives_an_all_zero_matrix():
    graph = links_frame([])
    adjacency, report = build_adjacency(graph, ["A", "B"])
    assert adjacency.shape == (2, 2) and adjacency.nnz == 0
    assert (report.links_used, report.receivers_without_links) == (0, 2)
    hop1, hop2 = propagate_shock(adjacency, np.array([1.0, 1.0]))
    assert hop1.tolist() == [0.0, 0.0] and hop2.tolist() == [0.0, 0.0]


def test_row_sums_are_one_or_zero_for_a_random_graph():
    rng = np.random.default_rng(7)
    nodes = [f"N{i}" for i in range(30)]
    pairs = [(nodes[i], nodes[j]) for i, j in rng.integers(0, 30, size=(80, 2))]
    graph = links_frame(pairs)
    graph["weight"] = rng.uniform(0.1, 3.0, size=len(graph))
    adjacency, _ = build_adjacency(graph, nodes)
    sums = np.asarray(adjacency.sum(axis=1)).ravel()
    assert np.all(np.isclose(sums, 1.0) | (sums == 0.0))


def test_triangle_keeps_paths_to_other_nodes_and_drops_none_to_origin():
    # A -> B -> C -> A with the shock on A: the only two-hop path from A ends at C.
    adjacency, _ = build_adjacency(
        links_frame([("A", "B"), ("B", "C"), ("C", "A")]), ["A", "B", "C"]
    )
    hop1, hop2 = propagate_shock(adjacency, np.array([1.0, 0.0, 0.0]))
    assert hop1.tolist() == [0.0, 0.5, 0.0]
    assert hop2[0] == 0.0 and hop2[2] == pytest.approx(0.25)


def test_a_shocked_third_node_still_receives_the_two_hop_path():
    # A -> B -> C with C also shocked: the path A -> B -> C is kept; only i -> j -> i is removed.
    adjacency, _ = build_adjacency(links_frame([("A", "B"), ("B", "C")]), ["A", "B", "C"])
    _, hop2 = propagate_shock(adjacency, np.array([1.0, 0.0, 1.0]))
    assert hop2[2] == pytest.approx(0.25)


def test_returned_arrays_are_one_dimensional_float64():
    adjacency, _ = build_adjacency(links_frame([("A", "B"), ("B", "A")]), ["A", "B"])
    hop1, hop2 = propagate_shock(adjacency, np.array([1.0, 0.0]))
    assert hop1.shape == (2,) and hop2.shape == (2,)
    assert hop1.dtype == np.float64 and hop2.dtype == np.float64
