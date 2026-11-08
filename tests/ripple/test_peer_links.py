import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from precedent.ripple.link_store import LinkStore, Relation, graph_asof
from precedent.ripple.peer_links import (
    embed_business_sections,
    peer_links_from_embeddings,
)

V1 = np.array([1.0, 0.0, 0.0])
V1_CLOSE = np.array([0.95, 0.1, 0.0])
V_FAR = np.array([0.0, 0.0, 1.0])
V_OPPOSITE = np.array([-1.0, 0.0, 0.0])

ACCEPTED = {
    "A": "2020-02-01 21:30",
    "B": "2020-03-15 21:30",
    "C": "2020-01-10 21:30",
}


def utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="UTC")


def accepted(names=("A", "B", "C")) -> dict[str, pd.Timestamp]:
    return {name: utc(ACCEPTED[name]) for name in names}


def test_peer_link_starts_at_the_later_filing():
    links = peer_links_from_embeddings(
        {"A": V1, "B": V1_CLOSE, "C": V_FAR},
        {
            "A": utc("2020-02-01 21:30"),
            "B": utc("2020-03-15 21:30"),
            "C": utc("2020-01-10 21:30"),
        },
        top_k=1,
    )
    a_to_b = next(link for link in links if {link.source_node, link.receiver_node} == {"A", "B"})
    assert a_to_b.valid_from_utc == utc("2020-03-15 21:30")


def test_link_fields_follow_the_decisions():
    links = peer_links_from_embeddings(
        {"A": V1, "B": V1_CLOSE, "C": V_FAR}, accepted(), top_k=1, expiry_months=12
    )
    a_to_b = next(
        link for link in links if link.source_node == "A" and link.receiver_node == "B"
    )
    assert a_to_b.relation is Relation.PEER_OF
    assert a_to_b.extractor_version == "rules-10k-similarity-v1"
    assert a_to_b.evidence_span is None
    assert a_to_b.source_doc_id == "10k-similarity:A|B"
    assert a_to_b.valid_to_utc == utc("2021-03-15 21:30")
    expected = float(V1 @ V1_CLOSE / (np.linalg.norm(V1) * np.linalg.norm(V1_CLOSE)))
    assert a_to_b.weight == pytest.approx(expected)


def test_self_links_are_never_emitted_and_weights_stay_in_unit_interval():
    embeddings = {"A": V1, "B": V1_CLOSE, "C": V1 * 2.0, "D": V_FAR}
    links = peer_links_from_embeddings(embeddings, accepted(("A", "B", "C")) | {
        "D": utc("2020-04-01 21:30")
    })
    assert links
    assert all(link.source_node != link.receiver_node for link in links)
    assert all(0.0 < link.weight <= 1.0 for link in links)


def test_top_k_is_respected_per_receiver():
    names = ["A", "B", "C", "D", "E"]
    rng = np.random.default_rng(3)
    embeddings = {name: np.abs(rng.normal(size=6)) + 0.1 for name in names}
    stamps = {name: utc("2020-01-01 21:30") for name in names}
    links = peer_links_from_embeddings(embeddings, stamps, top_k=2)
    for receiver in names:
        assert sum(1 for link in links if link.receiver_node == receiver) == 2


def test_non_positive_similarity_is_dropped():
    links = peer_links_from_embeddings(
        {"A": V1, "B": V_OPPOSITE, "C": V_FAR}, accepted(), top_k=5
    )
    assert links == []


def test_a_pair_can_link_in_both_directions():
    links = peer_links_from_embeddings({"A": V1, "B": V1_CLOSE}, accepted(("A", "B")), top_k=1)
    assert {(link.source_node, link.receiver_node) for link in links} == {("A", "B"), ("B", "A")}


def test_order_is_by_receiver_then_descending_weight_then_source_and_ties_break_by_name():
    embeddings = {"A": V1, "B": V1_CLOSE, "C": V1_CLOSE.copy(), "D": V1}
    stamps = {name: utc("2020-01-01 21:30") for name in embeddings}
    links = peer_links_from_embeddings(embeddings, stamps, top_k=2)
    keys = [(link.receiver_node, -link.weight, link.source_node) for link in links]
    assert keys == sorted(keys)
    again = peer_links_from_embeddings(dict(reversed(list(embeddings.items()))), stamps, top_k=2)
    assert [link.model_dump() for link in again] == [link.model_dump() for link in links]
    # A's two best sources are D (identical) then B/C tied; the tie goes to the smaller name.
    assert [link.source_node for link in links if link.receiver_node == "A"] == ["D", "B"]


def test_links_round_trip_through_the_store_and_are_invisible_before_the_later_filing(tmp_path):
    links = peer_links_from_embeddings(
        {"A": V1, "B": V1_CLOSE, "C": V_FAR}, accepted(), top_k=1
    )
    store = LinkStore(tmp_path)
    assert store.append(links).appended == len(links)
    assert store.read_all() == links
    before = graph_asof(store, utc("2020-03-15 21:29"), Relation.PEER_OF)
    assert before.empty
    after = graph_asof(store, utc("2020-03-15 21:30"), Relation.PEER_OF)
    assert not after.empty
    assert set(zip(after["source_node"], after["receiver_node"], strict=True)) == {
        ("A", "B"),
        ("B", "A"),
    }
    expired = graph_asof(store, utc("2021-06-15 21:30"), Relation.PEER_OF)
    assert expired.empty


def test_existing_links_are_unchanged_when_an_unrelated_company_is_added():
    base = peer_links_from_embeddings({"A": V1, "B": V1_CLOSE}, accepted(("A", "B")), top_k=1)
    wider = peer_links_from_embeddings(
        {"A": V1, "B": V1_CLOSE, "D": V_FAR},
        accepted(("A", "B")) | {"D": utc("2020-06-01 21:30")},
        top_k=1,
    )
    kept = [link for link in wider if {link.source_node, link.receiver_node} == {"A", "B"}]
    assert [link.model_dump() for link in kept] == [link.model_dump() for link in base]
    assert len(wider) == len(base)


def test_defaults_agree_with_the_params_file():
    params_path = Path(__file__).resolve().parents[2] / "references" / "ripple_params.json"
    params = json.loads(params_path.read_text(encoding="utf-8"))["ripple"]
    links = peer_links_from_embeddings(
        {f"N{i}": V1 + 0.01 * i * V_FAR for i in range(14)},
        {f"N{i}": utc("2020-01-01 00:00") for i in range(14)},
    )
    assert max(
        sum(1 for link in links if link.receiver_node == f"N{i}") for i in range(14)
    ) == params["correlation_top_k"]
    one = links[0]
    expected_to = one.valid_from_utc + pd.DateOffset(months=params["annual_link_expiry_months"])
    assert one.valid_to_utc == expected_to


@pytest.mark.parametrize(
    ("embeddings", "stamps", "kwargs", "match"),
    [
        ({"A": V1, "B": V1_CLOSE}, {"A": utc("2020-01-01")}, {}, "B"),
        ({"A": V1}, {"A": utc("2020-01-01")}, {}, "two"),
        ({"A": V1, "B": np.zeros(3)}, {"A": utc("2020-01-01"), "B": utc("2020-01-01")}, {}, "B"),
        ({"A": V1, "B": np.array([1.0, np.nan, 0.0])},
         {"A": utc("2020-01-01"), "B": utc("2020-01-01")}, {}, "B"),
        ({"A": V1, "B": np.array([1.0, 0.0])},
         {"A": utc("2020-01-01"), "B": utc("2020-01-01")}, {}, "B"),
        ({"A": V1, "B": np.ones((2, 3))},
         {"A": utc("2020-01-01"), "B": utc("2020-01-01")}, {}, "B"),
        ({"A": V1, "B": V1_CLOSE},
         {"A": utc("2020-01-01"), "B": pd.Timestamp("2020-01-01")}, {}, "B"),
        ({"A": V1, "B": V1_CLOSE},
         {"A": utc("2020-01-01"), "B": utc("2020-01-01")}, {"top_k": 0}, "top_k"),
        ({"A": V1, "B": V1_CLOSE},
         {"A": utc("2020-01-01"), "B": utc("2020-01-01")}, {"expiry_months": 0}, "expiry_months"),
    ],
)
def test_invalid_inputs_are_rejected_with_the_offending_name(embeddings, stamps, kwargs, match):
    with pytest.raises(ValueError, match=match):
        peer_links_from_embeddings(embeddings, stamps, **kwargs)


def test_non_utc_timestamps_are_normalised_to_utc():
    links = peer_links_from_embeddings(
        {"A": V1, "B": V1_CLOSE},
        {
            "A": pd.Timestamp("2020-02-01 16:30", tz="America/New_York"),
            "B": pd.Timestamp("2020-02-01 21:00", tz="UTC"),
        },
        top_k=1,
    )
    assert {link.valid_from_utc for link in links} == {utc("2020-02-01 21:30")}


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):
        self.calls.append(list(texts))
        return np.array([[float(len(text)), 1.0] for text in texts])


def test_embed_business_sections_makes_one_batched_call_in_sorted_name_order():
    embedder = FakeEmbedder()
    result = embed_business_sections({"B": "bb", "A": "a", "C": "cccc"}, embedder)
    assert embedder.calls == [["a", "bb", "cccc"]]
    assert list(result) == ["A", "B", "C"]
    assert result["C"].tolist() == [4.0, 1.0]


def test_embed_business_sections_rejects_a_wrong_shaped_result():
    class BadEmbedder:
        def embed(self, texts):
            return np.ones((len(texts) - 1, 2))

    with pytest.raises(ValueError, match="rows"):
        embed_business_sections({"A": "a", "B": "b"}, BadEmbedder())


def test_embed_business_sections_rejects_empty_input_and_blank_text():
    with pytest.raises(ValueError, match="no sections"):
        embed_business_sections({}, FakeEmbedder())
    with pytest.raises(ValueError, match="B"):
        embed_business_sections({"A": "a", "B": "  "}, FakeEmbedder())


# Point-in-time top-k (final review I1): the graph at t may only use filings accepted by t.
FILINGS = {  # name: (vector, acceptance instant), top_k=1 and top_k=2 both used below
    "R": (V1, utc("2020-01-10 21:30")),
    "B": (np.array([0.8, 0.6, 0.0]), utc("2020-02-10 21:30")),
    "C": (np.array([0.99, 0.1, 0.0]), utc("2020-03-10 21:30")),
    "D": (np.array([0.5, 0.5, 0.7]), utc("2020-04-10 21:30")),
    "E": (np.array([0.3, 0.1, 0.9]), utc("2020-05-10 21:30")),
}


def build(filings, top_k, **kwargs):
    return peer_links_from_embeddings(
        {name: vector for name, (vector, _) in filings.items()},
        {name: stamp for name, (_, stamp) in filings.items()},
        top_k=top_k,
        **kwargs,
    )


def edges_at(links, instant, tmp_path):
    store = LinkStore(tmp_path)
    store.append(links)
    frame = graph_asof(store, instant, Relation.PEER_OF)
    return {
        (row.source_node, row.receiver_node): row.weight for row in frame.itertuples(index=False)
    }


@pytest.mark.parametrize("top_k", [1, 2])
def test_truncated_build_read_at_t_equals_the_full_build_read_at_t(tmp_path, top_k):
    full = build(FILINGS, top_k)
    cutoffs = [
        utc("2020-02-10 21:30"),  # exactly at B's acceptance
        utc("2020-03-01 00:00"),
        utc("2020-03-10 21:30"),
        utc("2020-04-20 00:00"),
        utc("2020-06-01 00:00"),
    ]
    for position, instant in enumerate(cutoffs):
        known = {n: f for n, f in FILINGS.items() if f[1] <= instant}
        if len(known) < 2:
            continue
        truncated = build(known, top_k)
        full_edges = edges_at(full, instant, tmp_path / f"full{top_k}-{position}")
        truncated_edges = edges_at(truncated, instant, tmp_path / f"cut{top_k}-{position}")
        assert truncated_edges == full_edges, (top_k, instant)


def test_a_later_filer_displaces_a_link_and_ends_it_at_the_later_acceptance(tmp_path):
    # R and B file first; B is R's peer. C files later and is more similar to R: with top_k=1
    # the B->R link ends when C is accepted, and C->R starts then.
    filings = {name: FILINGS[name] for name in ("R", "B", "C")}
    links = build(filings, 1)
    b_to_r = next(link for link in links if (link.source_node, link.receiver_node) == ("B", "R"))
    c_to_r = next(link for link in links if (link.source_node, link.receiver_node) == ("C", "R"))
    assert b_to_r.valid_from_utc == utc("2020-02-10 21:30")
    assert b_to_r.valid_to_utc == utc("2020-03-10 21:30")  # displaced, before the 15-month expiry
    assert c_to_r.valid_from_utc == utc("2020-03-10 21:30")
    assert edges_at(links, utc("2020-03-01 00:00"), tmp_path / "before")[("B", "R")] > 0.0
    after = edges_at(links, utc("2020-03-10 21:30"), tmp_path / "after")
    assert ("B", "R") not in after and ("C", "R") in after
    # R is the only receiver B->R touched: B's own top-1 is not R forever either.
    assert all(link.valid_from_utc < link.valid_to_utc for link in links)


def test_a_link_that_would_only_exist_in_the_full_universe_is_not_visible_earlier(tmp_path):
    # The final review's reproduction: B filed before t, C (more similar to R) after t.
    # The old static top-k gave R no link from B at t at all.
    links = build({name: FILINGS[name] for name in ("R", "B", "C")}, 1)
    at_t = edges_at(links, utc("2020-02-20 00:00"), tmp_path)
    assert set(at_t) == {("B", "R"), ("R", "B")}


def test_link_expires_after_the_expiry_months_when_never_displaced():
    links = build({name: FILINGS[name] for name in ("R", "B")}, 1, expiry_months=6)
    assert {link.valid_to_utc for link in links} == {utc("2020-08-10 21:30")}


def test_a_displacement_after_expiry_does_not_extend_the_link():
    late_c = (FILINGS["C"][0], utc("2020-05-10 21:30"))
    links = build({"R": FILINGS["R"], "B": FILINGS["B"], "C": late_c}, 1, expiry_months=1)
    b_to_r = next(link for link in links if (link.source_node, link.receiver_node) == ("B", "R"))
    assert b_to_r.valid_to_utc == utc("2020-03-10 21:30")


def test_filings_accepted_at_the_same_instant_do_not_displace_each_other():
    stamps = {name: utc("2020-01-01 21:30") for name in ("A", "B", "C")}
    links = peer_links_from_embeddings(
        {"A": V1, "B": V1_CLOSE, "C": V1_CLOSE.copy()}, stamps, top_k=2
    )
    assert len(links) == 6
    assert {link.valid_from_utc for link in links} == {utc("2020-01-01 21:30")}
    assert all(link.valid_to_utc == utc("2021-04-01 21:30") for link in links)
