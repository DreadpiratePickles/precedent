"""Research: the Ripple factor panel and the control rung files (needs scipy)."""

import pytest

pytest.importorskip("scipy")

import datetime as dt  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from precedent.ripple.link_store import Link, LinkStore, Relation, graph_asof  # noqa: E402
from precedent.ripple.ripple_factors import (  # noqa: E402
    CHANNEL_OF_RELATION,
    PANEL_COLUMNS,
    build_ripple_factor_panel,
)
from precedent.spine.pit_alignment import build_session_cutoffs  # noqa: E402

SESSIONS = pd.bdate_range("2021-01-04", periods=320)
CUTOFFS = build_session_cutoffs(SESSIONS, dt.time(16, 0))
CUT_SESSIONS = [SESSIONS[302], SESSIONS[310], SESSIONS[315]]
END_OF_TIME = pd.Timestamp("2030-01-01", tz="UTC")


def make_link(source, receiver, relation, valid_from, weight=1.0):
    return Link(
        source_node=source,
        receiver_node=receiver,
        relation=relation,
        weight=weight,
        valid_from_utc=valid_from,
        valid_to_utc=END_OF_TIME,
        source_doc_id="doc-1",
        evidence_span=None,
        extractor_version="rules-1",
    )


def graph_function(links, root: Path):
    """A `graph_for_cutoff` that reads the links through the real `graph_asof` visibility rule."""
    store = LinkStore(root)
    store.append(links)
    return lambda cutoff: graph_asof(store, cutoff)


def shock_series(entries):
    """Shocks from (session position, instrument, value) triples."""
    index = pd.MultiIndex.from_arrays(
        [
            pd.DatetimeIndex([SESSIONS[position] for position, _, _ in entries]).as_unit("ns"),
            pd.Index([instrument for _, instrument, _ in entries], dtype=str),
        ],
        names=["datetime", "instrument"],
    )
    return pd.Series([value for _, _, value in entries], index=index, name="shock").sort_index()


def abnormal_frame(columns, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.0, 0.01, (len(SESSIONS), len(columns))), SESSIONS, columns)


def inputs_for(links, shocks, tmp_path, abnormal=None, **options):
    return {
        "shocks": shocks,
        "graph_for_cutoff": graph_function(links, tmp_path),
        "session_cutoffs_utc": CUTOFFS,
        "sessions": SESSIONS,
        "abnormal_returns": abnormal if abnormal is not None else abnormal_frame(list("ABCD")),
        **options,
    }


@pytest.fixture
def panel_inputs(tmp_path):
    valid_from = CUTOFFS.iloc[289] - pd.Timedelta(days=1)
    links = [
        make_link("A", "B", Relation.CUSTOMER_OF, valid_from),
        make_link("B", "C", Relation.SUPPLIER_OF, valid_from),
        make_link("A", "C", Relation.PEER_OF, valid_from),
        # visible only after the first two shock sessions, inside the cut range
        make_link("A", "D", Relation.CUSTOMER_OF, CUTOFFS.iloc[306]),
    ]
    shocks = shock_series(
        [(300, "A", 0.04), (300, "B", -0.02), (305, "A", 0.03), (305, "B", 0.01), (308, "A", 0.02)]
    )
    return inputs_for(links, shocks, tmp_path)


@pytest.fixture
def panel_inputs_late_link(tmp_path):
    early = CUTOFFS.iloc[289] - pd.Timedelta(days=1)
    links = [
        make_link("A", "B", Relation.CUSTOMER_OF, early),
        make_link("A", "RECEIVER", Relation.CUSTOMER_OF, CUTOFFS.iloc[310]),
    ]
    abnormal = abnormal_frame(["A", "B", "RECEIVER"])
    return inputs_for(links, shock_series([(300, "A", 0.04)]), tmp_path, abnormal)


@pytest.fixture
def panel_inputs_no_shocks(panel_inputs):
    return {**panel_inputs, "shocks": panel_inputs["shocks"].iloc[0:0]}


def truncate_inputs(inputs, cut):
    """The same inputs as they were known on session `cut`: nothing later is visible."""
    cut_cutoff = CUTOFFS.loc[cut]
    shocks = inputs["shocks"]
    abnormal = inputs["abnormal_returns"].copy()
    abnormal.loc[abnormal.index > cut] = np.nan
    base = inputs["graph_for_cutoff"]

    def graph_known_at_cut(cutoff):
        graph = base(cutoff)
        return graph[graph["valid_from_utc"] <= cut_cutoff].reset_index(drop=True)

    return {
        **inputs,
        "shocks": shocks[shocks.index.get_level_values("datetime") <= cut],
        "abnormal_returns": abnormal,
        "graph_for_cutoff": graph_known_at_cut,
    }


def test_ripple_panel_passes_truncation_invariance(panel_inputs):
    for cut in CUT_SESSIONS:
        full = build_ripple_factor_panel(**panel_inputs).loc[:cut]
        truncated = build_ripple_factor_panel(**truncate_inputs(panel_inputs, cut)).loc[:cut]
        assert not full.empty
        pd.testing.assert_frame_equal(full, truncated)


def test_truncation_test_has_teeth_against_a_later_graph_lookup(panel_inputs):
    """Negative control: a builder reading the graph at the last cutoff must fail the comparison."""
    cut = CUT_SESSIONS[0]

    def wrong_panel(inputs):
        graph_of = inputs["graph_for_cutoff"]
        last_cutoff = CUTOFFS.iloc[-1]
        return build_ripple_factor_panel(
            **{**inputs, "graph_for_cutoff": lambda _: graph_of(last_cutoff)}
        )

    full = wrong_panel(panel_inputs).loc[:cut]
    truncated = wrong_panel(truncate_inputs(panel_inputs, cut)).loc[:cut]
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(full, truncated)


def test_a_link_filed_later_does_not_carry_an_earlier_shock(panel_inputs_late_link):
    panel = build_ripple_factor_panel(**panel_inputs_late_link)
    assert "RECEIVER" not in panel.index.get_level_values("instrument")
    assert "B" in panel.index.get_level_values("instrument")  # the earlier link still carries it


def test_no_shocks_gives_an_empty_panel_with_the_right_columns(panel_inputs_no_shocks):
    panel = build_ripple_factor_panel(**panel_inputs_no_shocks)
    assert panel.empty and list(panel.columns)[0] == "ripple_customer"
    assert list(panel.columns) == PANEL_COLUMNS
    assert list(panel.index.names) == ["datetime", "instrument"]
    assert (panel.dtypes == "float64").all()


def test_chain_values_by_hand(tmp_path):
    valid_from = CUTOFFS.iloc[100] - pd.Timedelta(days=1)
    links = [
        make_link("A", "B", Relation.CUSTOMER_OF, valid_from),
        make_link("B", "C", Relation.CUSTOMER_OF, valid_from),
    ]
    abnormal = abnormal_frame(list("ABC"))
    inputs = inputs_for(links, shock_series([(200, "A", 0.04)]), tmp_path, abnormal)
    panel = build_ripple_factor_panel(**inputs)
    t, t3 = SESSIONS[200], SESSIONS[203]
    assert panel.loc[(t, "B"), "ripple_customer"] == pytest.approx(0.02)  # alpha * 0.04
    assert panel.loc[(t, "C"), "ripple_hop2"] == pytest.approx(0.01)  # alpha^2 * 0.04
    assert panel.loc[(t, "C"), "ripple_supplier"] == 0.0
    assert panel.loc[(t, "B"), "ripple_unabsorbed"] == pytest.approx(0.02)
    assert panel.loc[(t3, "B"), "ripple_customer"] == pytest.approx(0.01)  # one half-life
    car = abnormal["B"].iloc[201:204].sum()
    assert panel.loc[(t3, "B"), "ripple_unabsorbed"] == pytest.approx(0.02 - car)
    assert (t, "A") not in panel.index  # the shocked source is no one's receiver


def test_two_shocks_reaching_one_receiver_sum(tmp_path):
    valid_from = CUTOFFS.iloc[100] - pd.Timedelta(days=1)
    links = [
        make_link("A", "C", Relation.PEER_OF, valid_from),
        make_link("B", "C", Relation.PEER_OF, valid_from),
    ]
    shocks = shock_series([(200, "A", 0.04), (200, "B", 0.02)])
    panel = build_ripple_factor_panel(**inputs_for(links, shocks, tmp_path))
    # C's incoming weights are 1/2 each: alpha * (0.5 * 0.04 + 0.5 * 0.02)
    assert panel.loc[(SESSIONS[200], "C"), "ripple_peer"] == pytest.approx(0.015)
    assert panel.index.is_unique


def test_shocks_on_different_sessions_overlap_by_summing(tmp_path):
    valid_from = CUTOFFS.iloc[100] - pd.Timedelta(days=1)
    links = [make_link("A", "B", Relation.PEER_OF, valid_from)]
    shocks = shock_series([(200, "A", 0.04), (203, "A", 0.04)])
    abnormal = abnormal_frame(list("AB"))
    panel = build_ripple_factor_panel(**inputs_for(links, shocks, tmp_path, abnormal))
    assert panel.loc[(SESSIONS[203], "B"), "ripple_peer"] == pytest.approx(0.02 + 0.01)
    # the receiver's own CAR is subtracted once per shock, each from its own t + 1:
    # (0.02 - CAR(201..203)) from the first shock, (0.02 - 0) from the second
    expected = 0.04 - abnormal["B"].iloc[201:204].sum()
    assert panel.loc[(SESSIONS[203], "B"), "ripple_unabsorbed"] == pytest.approx(expected)


def test_commodity_relations_share_one_channel(tmp_path):
    valid_from = CUTOFFS.iloc[100] - pd.Timedelta(days=1)
    links = [
        make_link("OIL", "A", Relation.INPUT_COMMODITY, valid_from),
        make_link("OIL", "B", Relation.OUTPUT_COMMODITY, valid_from),
    ]
    abnormal = abnormal_frame(["A", "B", "OIL"])
    panel = build_ripple_factor_panel(
        **inputs_for(links, shock_series([(200, "OIL", 0.1)]), tmp_path, abnormal)
    )
    at_t = panel.loc[SESSIONS[200]]
    assert at_t["ripple_commodity"].tolist() == [pytest.approx(0.05)] * 2
    assert (at_t.drop(columns=["ripple_commodity", "ripple_unabsorbed"]) == 0.0).all().all()


def test_country_exposure_links_give_no_rows(tmp_path):
    valid_from = CUTOFFS.iloc[100] - pd.Timedelta(days=1)
    links = [make_link("TW", "A", Relation.COUNTRY_EXPOSURE, valid_from)]
    panel = build_ripple_factor_panel(
        **inputs_for(links, shock_series([(200, "TW", 0.1)]), tmp_path)
    )
    assert panel.empty and list(panel.columns) == PANEL_COLUMNS
    assert Relation.COUNTRY_EXPOSURE.value not in CHANNEL_OF_RELATION


def test_panel_never_reaches_past_the_last_session(tmp_path):
    valid_from = CUTOFFS.iloc[100] - pd.Timedelta(days=1)
    links = [make_link("A", "B", Relation.PEER_OF, valid_from)]
    panel = build_ripple_factor_panel(
        **inputs_for(links, shock_series([(318, "A", 0.04)]), tmp_path)
    )
    dates = panel.index.get_level_values("datetime")
    assert dates.max() == SESSIONS[-1] and len(panel) == 2


def test_age_limit_and_nan_abnormal_returns(tmp_path):
    valid_from = CUTOFFS.iloc[100] - pd.Timedelta(days=1)
    links = [make_link("A", "B", Relation.PEER_OF, valid_from)]
    abnormal = abnormal_frame(["A", "B"])
    abnormal.iloc[201, 1] = np.nan  # an unknown abnormal return counts as zero
    options = {"max_age_sessions": 2}
    panel = build_ripple_factor_panel(
        **inputs_for(links, shock_series([(200, "A", 0.04)]), tmp_path, abnormal, **options)
    )
    assert panel.index.get_level_values("datetime").max() == SESSIONS[202]
    expected = 0.02 - abnormal.iloc[202, 1]
    assert panel.loc[(SESSIONS[202], "B"), "ripple_unabsorbed"] == pytest.approx(expected)
    assert not panel.isna().any().any()


def test_panel_passes_the_qlib_factor_panel_check(panel_inputs):
    pytest.importorskip("qlib")
    from precedent.spine.qlib_dataset import _check_factor_panel

    _check_factor_panel("ripple_", build_ripple_factor_panel(**panel_inputs))


@pytest.mark.parametrize(
    "change",
    [
        {"half_life_sessions": 0.0},
        {"max_age_sessions": -1},
        {"sessions": SESSIONS.tz_localize("UTC")},
        {"sessions": SESSIONS[::-1]},
        {"abnormal_returns": abnormal_frame(["A"]).iloc[:-1]},
        {"session_cutoffs_utc": CUTOFFS.dt.tz_localize(None)},
        {"session_cutoffs_utc": CUTOFFS.iloc[:-1]},
    ],
    ids=[
        "half_life",
        "max_age",
        "tz_sessions",
        "unsorted",
        "short_returns",
        "naive_cutoffs",
        "short_cutoffs",
    ],
)
def test_bad_inputs_are_rejected(panel_inputs, change):
    with pytest.raises(ValueError):
        build_ripple_factor_panel(**{**panel_inputs, **change})


def test_infinite_abnormal_returns_are_rejected(panel_inputs):
    abnormal = panel_inputs["abnormal_returns"].copy()
    abnormal.iloc[5, 0] = np.inf
    with pytest.raises(ValueError, match="finite"):
        build_ripple_factor_panel(**{**panel_inputs, "abnormal_returns": abnormal})


def test_an_unknown_relation_is_rejected(panel_inputs):
    base = panel_inputs["graph_for_cutoff"]

    def graph_with_bogus_relation(cutoff):
        graph = base(cutoff).copy()
        graph.loc[0, "relation"] = "bogus_relation"
        return graph

    with pytest.raises(ValueError, match="bogus_relation"):
        build_ripple_factor_panel(**{**panel_inputs, "graph_for_cutoff": graph_with_bogus_relation})


def test_a_tz_aware_shock_datetime_is_rejected(panel_inputs):
    shocks = panel_inputs["shocks"].copy()
    shocks.index = shocks.index.set_levels(shocks.index.levels[0].tz_localize("UTC"), level=0)
    with pytest.raises(ValueError, match="naive"):
        build_ripple_factor_panel(**{**panel_inputs, "shocks": shocks})


def test_a_shock_outside_the_calendar_is_rejected(panel_inputs):
    shocks = panel_inputs["shocks"].copy()
    shocks.index = shocks.index.set_levels(shocks.index.levels[0] + pd.Timedelta(hours=5), level=0)
    with pytest.raises(ValueError, match="sessions"):
        build_ripple_factor_panel(**{**panel_inputs, "shocks": shocks})


def test_the_shipped_ripple_rungs_load():
    pytest.importorskip("yaml")
    pytest.importorskip("lightgbm")
    pytest.importorskip("qlib")
    from precedent.spine.ablation import load_rungs

    rungs = {rung.name: rung for rung in load_rungs(Path("references/rungs"))}
    for name in ("m3_ripple_g1", "m3_ripple_g2", "m3_ripple_placebo"):
        assert rungs[name].panel_prefixes == ["news_", "ripple_"]
