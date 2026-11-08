"""The ladder's wiring pieces: abnormal-return frame, rebuilt correlation graph, cached readers."""

import datetime as dt
import json

import numpy as np
import pandas as pd
import pytest

from precedent.ripple.graph_builders import (
    abnormal_return_frame,
    placebo_graph_for_cutoff,
    rolling_correlation_links,
    snapshot_graph_for_cutoff,
    write_ripple_ledger,
)
from precedent.ripple.link_store import Link, LinkStore, Relation, graph_asof
from precedent.spine.abnormal_returns import abnormal_return_at
from precedent.spine.pit_alignment import build_session_cutoffs

SESSIONS = pd.bdate_range("2021-01-04", periods=320)
CUTOFFS = build_session_cutoffs(SESSIONS, dt.time(16, 0))
FAR_FUTURE = pd.Timestamp("2035-01-01", tz="UTC")


def market_returns(seed=3):
    rng = np.random.default_rng(seed)
    names = ["MKT", "A", "B", "C", "D", "E"]
    market = rng.normal(0, 0.01, len(SESSIONS))
    common = rng.normal(0, 0.01, len(SESSIONS))
    data = {"MKT": market}
    for name in names[1:]:
        data[name] = 0.8 * market + rng.normal(0, 0.01, len(SESSIONS))
    data["A"] = data["A"] + common  # A and B share a residual factor the market does not explain
    data["B"] = data["B"] + common
    return pd.DataFrame(data, index=SESSIONS)


def test_abnormal_return_frame_matches_abnormal_return_at_and_leaves_out_the_market():
    returns = market_returns()
    frame = abnormal_return_frame(returns, "MKT", SESSIONS)
    assert list(frame.columns) == ["A", "B", "C", "D", "E"]
    assert frame.index.equals(SESSIONS)
    for position in (0, 100, 200, 319):
        expected = abnormal_return_at(returns["C"], returns["MKT"], position)
        assert frame["C"].iloc[position] == pytest.approx(expected, nan_ok=True)
    assert frame["C"].iloc[:120].isna().all()  # too little history for the market model
    assert frame["C"].iloc[200:].notna().all()


def test_abnormal_return_frame_recovers_a_planted_shock():
    returns = market_returns()
    returns.iloc[250, returns.columns.get_loc("D")] += 0.05
    frame = abnormal_return_frame(returns, "MKT", SESSIONS)
    assert frame["D"].iloc[250] > 0.04
    assert abs(frame["E"].iloc[250]) < 0.04


def test_abnormal_return_frame_rejects_misaligned_inputs():
    returns = market_returns()
    with pytest.raises(ValueError, match="sessions"):
        abnormal_return_frame(returns.iloc[1:], "MKT", SESSIONS)
    with pytest.raises(ValueError, match="market"):
        abnormal_return_frame(returns, "SPY", SESSIONS)


def test_rolling_correlation_links_snapshots_and_validity_windows(tmp_path):
    returns = market_returns()
    links, snapshot_cutoffs = rolling_correlation_links(
        returns, "MKT", CUTOFFS, window=100, top_k=2, rebuild_every=50
    )
    positions = list(range(99, 320, 50))
    assert snapshot_cutoffs == [CUTOFFS.iloc[p] for p in positions]
    store = LinkStore(tmp_path)
    store.append(links)
    assert graph_asof(store, CUTOFFS.iloc[98]).empty  # nothing before the first window closes
    first = graph_asof(store, CUTOFFS.iloc[99])
    assert set(first["source_doc_id"]) == {f"corr:{SESSIONS[99].date().isoformat()}"}
    second = graph_asof(store, CUTOFFS.iloc[149])
    assert set(second["source_doc_id"]) == {f"corr:{SESSIONS[149].date().isoformat()}"}
    still_second = graph_asof(store, CUTOFFS.iloc[198])
    assert set(still_second["source_doc_id"]) == set(second["source_doc_id"])
    last = graph_asof(store, CUTOFFS.iloc[-1])
    assert set(last["source_doc_id"]) == {f"corr:{SESSIONS[299].date().isoformat()}"}
    assert (first["relation"] == "peer_of").all()
    assert (first["extractor_version"] == "rules-correlation-v1").all()
    assert first.groupby("receiver_node").size().max() <= 2


def test_rolling_correlation_links_finds_the_planted_peer():
    returns = market_returns()
    links, _ = rolling_correlation_links(
        returns, "MKT", CUTOFFS, window=250, top_k=1, rebuild_every=100
    )
    pairs = {(link.source_node, link.receiver_node) for link in links}
    assert ("B", "A") in pairs and ("A", "B") in pairs


def test_rolling_correlation_snapshot_does_not_read_later_returns():
    returns = market_returns()
    full, _ = rolling_correlation_links(
        returns, "MKT", CUTOFFS, window=100, top_k=3, rebuild_every=50
    )
    cut = 149
    truncated, _ = rolling_correlation_links(
        returns.iloc[: cut + 1], "MKT", CUTOFFS.iloc[: cut + 1], window=100, top_k=3,
        rebuild_every=50,
    )
    doc = f"corr:{SESSIONS[cut].date().isoformat()}"

    def keyed(links):
        return sorted(
            (link.source_node, link.receiver_node, link.weight)
            for link in links
            if link.source_doc_id == doc
        )

    assert keyed(full) == keyed(truncated) and keyed(full)


def test_rolling_correlation_links_rejects_bad_arguments():
    returns = market_returns()
    with pytest.raises(ValueError, match="does not fit"):
        rolling_correlation_links(returns, "MKT", CUTOFFS, window=400, top_k=2, rebuild_every=20)
    with pytest.raises(ValueError, match="rebuild_every"):
        rolling_correlation_links(returns, "MKT", CUTOFFS, window=100, top_k=2, rebuild_every=0)
    with pytest.raises(ValueError, match="session_cutoffs_utc"):
        rolling_correlation_links(
            returns, "MKT", CUTOFFS.iloc[1:], window=100, top_k=2, rebuild_every=20
        )


class CountingStore(LinkStore):
    reads = 0

    def read_all(self):
        type(self).reads += 1
        return super().read_all()


def make_link(source, receiver, valid_from, valid_to=FAR_FUTURE):
    return Link(
        source_node=source,
        receiver_node=receiver,
        relation=Relation.PEER_OF,
        weight=1.0,
        valid_from_utc=valid_from,
        valid_to_utc=valid_to,
        source_doc_id="doc",
        evidence_span=None,
        extractor_version="rules-1",
    )


def test_snapshot_reader_equals_graph_asof_and_reads_the_store_once_per_stretch(tmp_path):
    CountingStore.reads = 0
    store = CountingStore(tmp_path)
    first_change, second_change = CUTOFFS.iloc[100], CUTOFFS.iloc[200]
    store.append(
        [
            make_link("A", "B", first_change, second_change),
            make_link("C", "D", second_change),
        ]
    )
    cached = snapshot_graph_for_cutoff(store, [first_change, second_change])
    CountingStore.reads = 0
    for cutoff in CUTOFFS.iloc[90:300]:
        pd.testing.assert_frame_equal(cached(cutoff), graph_asof(store, cutoff))
    # 210 cutoffs over three stretches: one read per stretch by the cache, plus one per direct call
    assert CountingStore.reads == 3 + 210


def test_snapshot_reader_only_reads_the_store_when_a_stretch_is_new(tmp_path):
    CountingStore.reads = 0
    store = CountingStore(tmp_path)
    store.append([make_link("A", "B", CUTOFFS.iloc[10])])
    cached = snapshot_graph_for_cutoff(store, [CUTOFFS.iloc[10]])
    for cutoff in CUTOFFS.iloc[20:60]:
        cached(cutoff)
    assert CountingStore.reads == 1


def test_snapshot_reader_requires_increasing_change_cutoffs(tmp_path):
    with pytest.raises(ValueError, match="increasing"):
        snapshot_graph_for_cutoff(LinkStore(tmp_path), [CUTOFFS.iloc[5], CUTOFFS.iloc[1]])


def ring_graph(n=20):
    rows = [
        {
            "source_node": f"N{i:02d}",
            "receiver_node": f"N{(i + 1) % n:02d}",
            "relation": "peer_of",
            "weight": 1.0,
        }
        for i in range(n)
    ] + [
        {
            "source_node": f"N{i:02d}",
            "receiver_node": f"N{(i + 3) % n:02d}",
            "relation": "peer_of",
            "weight": 0.5,
        }
        for i in range(n)
    ]
    return pd.DataFrame(rows)


def test_placebo_keeps_degrees_is_deterministic_and_is_shuffled_once_per_graph():
    graph = ring_graph()
    calls = []

    def graph_for_cutoff(cutoff):
        calls.append(cutoff)
        return graph

    placebo = placebo_graph_for_cutoff(graph_for_cutoff, seed=7)
    first = placebo(CUTOFFS.iloc[0])
    again = placebo(CUTOFFS.iloc[5])
    assert first is again  # the same graph gets the same placebo object
    other = placebo_graph_for_cutoff(lambda cutoff: graph, seed=7)(CUTOFFS.iloc[0])
    pd.testing.assert_frame_equal(first, other)
    for column in ("source_node", "receiver_node"):
        assert (
            first[column].value_counts().sort_index()
            .equals(graph[column].value_counts().sort_index())
        )
    key = ["source_node", "receiver_node"]
    assert set(map(tuple, first[key].to_numpy())) != set(map(tuple, graph[key].to_numpy()))
    assert len(calls) == 2


def test_placebo_of_a_graph_too_small_to_shuffle_is_the_graph():
    empty = pd.DataFrame(columns=["source_node", "receiver_node", "relation", "weight"])
    one = ring_graph().iloc[:1]
    assert placebo_graph_for_cutoff(lambda cutoff: empty, seed=7)(CUTOFFS.iloc[0]).empty
    pd.testing.assert_frame_equal(
        placebo_graph_for_cutoff(lambda cutoff: one, seed=7)(CUTOFFS.iloc[0]), one
    )


def test_placebo_of_an_unshufflable_snapshot_never_raises_and_is_counted():
    mutual = pd.DataFrame(
        {
            "source_node": ["A", "B"],
            "receiver_node": ["B", "A"],
            "relation": ["peer_of", "peer_of"],
            "weight": [1.0, 1.0],
        }
    )
    ring = ring_graph()
    graphs = {0: mutual, 1: ring}

    def graph_for_cutoff(cutoff):
        return graphs[0 if cutoff == CUTOFFS.iloc[0] else 1]

    placebo = placebo_graph_for_cutoff(graph_for_cutoff, seed=7)
    assert placebo.placebo_unshuffled == 0
    pd.testing.assert_frame_equal(placebo(CUTOFFS.iloc[0]), mutual)
    placebo(CUTOFFS.iloc[0])  # the same snapshot read again is counted once
    assert placebo.placebo_unshuffled == 1
    shuffled = placebo(CUTOFFS.iloc[3])
    assert not shuffled[["source_node", "receiver_node"]].equals(
        ring[["source_node", "receiver_node"]]
    )
    assert placebo.placebo_unshuffled == 1


def test_placebo_shuffles_each_relation_separately():
    rows = [
        {"source_node": f"N{i}", "receiver_node": f"N{(i + k) % 8}", "relation": relation,
         "weight": 1.0}
        for relation, k in (("customer_of", 1), ("supplier_of", 3))
        for i in range(8)
    ]
    graph = pd.DataFrame(rows)
    placebo = placebo_graph_for_cutoff(lambda cutoff: graph, seed=5)(CUTOFFS.iloc[0])
    for column in ("source_node", "receiver_node"):
        before = graph.groupby(["relation", column]).size().to_dict()
        assert placebo.groupby(["relation", column]).size().to_dict() == before


def test_ledger_writer_round_trips_and_records_hashes(tmp_path):
    shocks = pd.Series(
        [0.04, -0.03],
        index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(["2021-06-01", "2021-06-02"]).as_unit("ns"), ["A", "B"]],
            names=["datetime", "instrument"],
        ),
        name="shock",
    )
    abnormal = pd.DataFrame({"A": [0.01, np.nan], "B": [0.0, 0.02]}, SESSIONS[:2])
    manifest_path = write_ripple_ledger(
        tmp_path / "ripple", shocks, abnormal, {"evidence_label": "SYNTHETIC", "run_id": "r1"}
    )
    ripple_dir = tmp_path / "ripple"
    assert manifest_path == ripple_dir / "MANIFEST.json"
    back = pd.read_parquet(ripple_dir / "shocks.parquet")
    assert list(back.columns) == ["datetime", "instrument", "shock"]
    assert back["shock"].tolist() == [0.04, -0.03]
    abnormal_back = pd.read_parquet(ripple_dir / "abnormal_returns.parquet")
    assert abnormal_back["A"].isna().tolist() == [False, True]
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["evidence_label"] == "SYNTHETIC" and manifest["run_id"] == "r1"
    assert set(manifest["files"]) == {"shocks.parquet", "abnormal_returns.parquet"}
    assert all(len(entry["sha256"]) == 64 for entry in manifest["files"].values())
    assert not list(ripple_dir.glob("*.tmp")) and not list(ripple_dir.glob(".*"))


def test_ledger_writer_requires_an_evidence_label(tmp_path):
    shocks = pd.Series(
        [0.04],
        index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(["2021-06-01"]).as_unit("ns"), ["A"]],
            names=["datetime", "instrument"],
        ),
    )
    with pytest.raises(ValueError, match="evidence_label"):
        write_ripple_ledger(tmp_path / "ripple", shocks, pd.DataFrame({"A": [0.0]}), {})
    assert not (tmp_path / "ripple").exists()


def _ledger_inputs():
    shocks = pd.Series(
        [0.04],
        index=pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(["2021-06-01"]).as_unit("ns"), ["A"]],
            names=["datetime", "instrument"],
        ),
        name="shock",
    )
    return shocks, pd.DataFrame({"A": [0.01]}, SESSIONS[:1])


def _link(source, receiver):
    return Link(
        source_node=source,
        receiver_node=receiver,
        relation=Relation.PEER_OF,
        weight=0.5,
        valid_from_utc=dt.datetime(2021, 1, 1, tzinfo=dt.UTC),
        valid_to_utc=dt.datetime(2022, 1, 1, tzinfo=dt.UTC),
        source_doc_id="d",
        evidence_span=None,
        extractor_version="rules-test",
    )


def test_ledger_writer_publishes_one_graphs_links_with_a_hash_and_replaces_them(tmp_path):
    import hashlib

    shocks, abnormal = _ledger_inputs()
    ripple_dir = tmp_path / "ripple"
    write_ripple_ledger(
        ripple_dir, shocks, abnormal, {"evidence_label": "SYNTHETIC"},
        links=[_link("A", "B"), _link("B", "A")], published_graph="g2",
    )
    manifest = json.loads((ripple_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    path = ripple_dir / "links" / "links.jsonl"
    assert manifest["published_graph"] == "g2"
    assert manifest["files"]["links/links.jsonl"] == {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": 2,
    }
    assert len(LinkStore(ripple_dir / "links").read_all()) == 2
    # a later publish replaces the store instead of appending to it
    write_ripple_ledger(
        ripple_dir, shocks, abnormal, {"evidence_label": "SYNTHETIC"},
        links=[_link("A", "C")], published_graph="g1",
    )
    assert [(x.source_node, x.receiver_node) for x in LinkStore(ripple_dir / "links").read_all()] == [
        ("A", "C")
    ]
    manifest = json.loads((ripple_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["published_graph"] == "g1" and manifest["files"]["links/links.jsonl"]["rows"] == 1


def test_ledger_writer_makes_the_link_file_even_when_the_graph_is_empty(tmp_path):
    shocks, abnormal = _ledger_inputs()
    write_ripple_ledger(
        tmp_path / "ripple", shocks, abnormal, {"evidence_label": "SYNTHETIC"},
        links=[], published_graph="g1",
    )
    assert (tmp_path / "ripple" / "links" / "links.jsonl").read_bytes() == b""
    manifest = json.loads((tmp_path / "ripple" / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["files"]["links/links.jsonl"]["rows"] == 0


@pytest.mark.parametrize(
    "kwargs", [{"links": []}, {"published_graph": "g1"}, {"links": [], "published_graph": " "}]
)
def test_ledger_writer_needs_links_and_a_graph_name_together(tmp_path, kwargs):
    shocks, abnormal = _ledger_inputs()
    with pytest.raises(ValueError):
        write_ripple_ledger(
            tmp_path / "ripple", shocks, abnormal, {"evidence_label": "SYNTHETIC"}, **kwargs
        )
    assert not (tmp_path / "ripple").exists()
