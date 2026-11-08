"""Tests for `score_mechanism_ladder`: M-T and M-P against Precedent's P-B1 on identical units."""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from mechanism_world import World, make_world

import precedent.mechanism.evaluate as evaluate_module
from precedent.config import references_dir
from precedent.engine.evaluate import score_precedent_ladder
from precedent.mechanism.evaluate import (
    precedent_scored_units,
    score_mechanism_ladder,
    shuffle_edges_preserving_degree,
)
from precedent.mechanism.graph_schema import (
    CausalGraph,
    Edge,
    Node,
    NodeKind,
    derive_kind,
    load_params,
    node_vocabulary,
)
from precedent.mechanism.panel import (
    NoPrecedentError,
    PanelError,
    build_guarded_panel,
    event_values_of,
    mechanism_query,
)
from precedent.mechanism.templates import load_templates, template_for
from precedent.spine.event_key import EventKey
from precedent.spine.event_schema import EventType

COLUMNS = [
    "rung",
    "event_type",
    "quantile",
    "pinball_loss",
    "coverage_80",
    "n_scored",
    "n_refused",
    "n_underdetermined",
]
HORIZON = 5
N_PATHS = 500
PLACEBO_SEED = 1
EVENT_NODE = "event:central_bank:direction_sign"

VOCABULARY = node_vocabulary(
    references_dir() / "mechanism" / "intermediates.json",
    load_params().asset_panel,
    [e.value for e in EventType],
)
TEMPLATES = load_templates(references_dir() / "mechanism" / "templates", VOCABULARY)


def template_of(key: EventKey) -> CausalGraph | None:
    return template_for(key, TEMPLATES)


def placebo_of(key: EventKey) -> CausalGraph | None:
    graph = template_of(key)
    return None if graph is None else shuffle_edges_preserving_degree(graph, PLACEBO_SEED)


def ladder(world: World, graphs_by_rung, **kwargs) -> pd.DataFrame:
    kwargs.setdefault("horizon", HORIZON)
    kwargs.setdefault("n_paths", N_PATHS)
    return score_mechanism_ladder(
        world.events,
        world.rows,
        world.session_cutoffs_utc,
        world.returns,
        world.market,
        graphs_by_rung,
        **kwargs,
    )


def median_loss(table: pd.DataFrame) -> pd.Series:
    return table[table["quantile"] == 0.5].groupby("rung")["pinball_loss"].mean()


def small_graph(*assets: str, isolated: tuple[str, ...] = ()) -> CausalGraph:
    """event -> rates_expectation -> each of `assets`; the `isolated` assets hang off nothing."""
    ids = [EVENT_NODE, "rates_expectation", *assets, *isolated]
    pairs = [(EVENT_NODE, "rates_expectation"), *(("rates_expectation", a) for a in assets)]
    edges = [
        Edge(
            source=source,
            target=target,
            sign=-1,
            lag_band="same_session",
            source_doc_id=None,
            evidence_span=None,
        )
        for source, target in pairs
    ]
    return CausalGraph(
        graph_id="test:small",
        graph_version="template-v1",
        event_key=_KEY_FOR_GRAPHS,
        nodes=[Node(id=i, kind=derive_kind(i)) for i in ids],
        edges=edges,
    )


_KEY_FOR_GRAPHS = make_world().key(1)


@pytest.fixture(scope="module")
def world() -> World:
    return make_world()


@pytest.fixture(scope="module")
def table(world: World) -> pd.DataFrame:
    return ladder(world, {"M-T": template_of, "M-P": placebo_of})


# -------------------------------------------------------------- the planted mechanism


def test_the_true_structure_beats_a_graph_that_leaves_the_event_out(world, table):
    """The world's response runs only through the event: a graph with no event path loses.

    `M-0` keeps the event and one latent but connects no asset to them, so every asset is
    forecast by its intercept alone.
    """
    unconnected = ladder(world, {"M-0": lambda key: small_graph(isolated=("TLT", "HYG", "UUP"))})
    both = pd.concat([table, unconnected])
    median = median_loss(both)
    assert median["M-T"] < 0.6 * median["M-0"]


def test_on_the_central_bank_template_the_placebo_forecasts_like_the_template(world, table):
    """Characterisation, not a goal: M-P cannot lose to M-T here, so the ladder cannot rank them.

    The panel's only exogenous column is `direction_sign` (severity and scheduling are matched
    by the key, hence constant). Every connected node of a degree-preserving rewiring is still a
    descendant of the event through single-parent latents, and each asset has its own free slope
    on its latent, so the placebo's forecasts equal the template's (differences of order 1e-17).
    If this test fails, the fit or the placebo changed; re-read the 5b report before relying on
    an M-T versus M-P comparison.
    """
    key = world.key(1)
    template, placebo = template_of(key), placebo_of(key)
    pairs = {g.graph_id: {(e.source, e.target) for e in g.edges} for g in (template, placebo)}
    assert pairs[template.graph_id] != pairs[placebo.graph_id]  # the rewiring is real
    both = table[table["quantile"] == 0.5].set_index("rung")
    assert both.loc["M-P", "pinball_loss"] == pytest.approx(
        both.loc["M-T", "pinball_loss"], rel=1e-9
    )
    assert both.loc["M-P", "coverage_80"] == both.loc["M-T", "coverage_80"]


# ----------------------------------------------------------------- identical units


def test_mechanism_and_precedent_are_scored_on_identical_events(table):
    pb1 = table[(table["rung"] == "P-B1") & (table["quantile"] == 0.5)].set_index("event_type")
    for rung in ("M-T", "M-P"):
        m = table[(table["rung"] == rung) & (table["quantile"] == 0.5)].set_index("event_type")
        assert list(m.index) == list(pb1.index)
        assert ((m["n_scored"] + m["n_refused"]) == pb1["n_scored"]).all()
    m_t = table[(table["rung"] == "M-T") & (table["quantile"] == 0.5)].set_index("event_type")
    assert pb1["n_scored"].sum() > 0
    assert (m_t["n_refused"] == 0).all()  # past the warm-up the mechanism refuses nothing
    assert (m_t["n_scored"] == pb1["n_scored"]).all()


def test_precedent_unit_helper_agrees_with_the_engine_ladder(world):
    units = precedent_scored_units(
        world.events, world.rows, world.session_cutoffs_utc, horizon=HORIZON
    )
    engine = score_precedent_ladder(
        world.events,
        world.rows,
        world.session_cutoffs_utc,
        "asset",
        rungs=("B1",),
        horizon=HORIZON,
    )
    engine_scored = engine[engine["quantile"] == 0.5].set_index("event_type")["n_scored"]
    type_of = world.events.set_index("event_id")["event_type"]
    helper_scored = pd.Series(
        {event_id: len(assets) for event_id, assets in units.items()}, dtype="int64"
    )
    by_type = helper_scored.groupby(type_of.reindex(helper_scored.index)).sum()
    assert engine_scored.sum() > 0
    assert by_type.to_dict() == engine_scored.to_dict()
    assert all("SPY" not in assets for assets in units.values())  # the market is not a unit


def test_events_are_visited_in_time_order(world, monkeypatch):
    seen: list[pd.Timestamp] = []
    real = evaluate_module.build_guarded_panel

    def recording(query, *args, **kwargs):
        seen.append(query.available_utc)
        return real(query, *args, **kwargs)

    monkeypatch.setattr(evaluate_module, "build_guarded_panel", recording)
    ladder(world, {"M-T": template_of})
    assert len(seen) > 10 and seen == sorted(seen)


# --------------------------------------------------------------------- refusals


def test_a_rung_with_no_graph_refuses_every_unit(world, table):
    absent = ladder(world, {"M-N": lambda key: None})
    pb1 = table[(table["rung"] == "P-B1") & (table["quantile"] == 0.5)].set_index("event_type")
    m = absent[(absent["rung"] == "M-N") & (absent["quantile"] == 0.5)].set_index("event_type")
    assert (m["n_scored"] == 0).all()
    assert (m["n_refused"] == pb1["n_scored"]).all()
    assert m["pinball_loss"].isna().all() and m["coverage_80"].isna().all()


def test_a_unit_whose_asset_is_not_in_the_graph_is_refused(world):
    result = ladder(world, {"M-S": lambda key: small_graph("TLT")})
    units = precedent_scored_units(
        world.events, world.rows, world.session_cutoffs_utc, horizon=HORIZON
    )
    tlt_units = sum("TLT" in assets for assets in units.values())
    all_units = sum(len(assets) for assets in units.values())
    m = result[(result["rung"] == "M-S") & (result["quantile"] == 0.5)].iloc[0]
    assert tlt_units > 0
    assert (m["n_scored"], m["n_refused"]) == (tlt_units, all_units - tlt_units)


def test_no_precedent_or_a_thin_panel_refuses_the_event(world, monkeypatch):
    real = evaluate_module.build_guarded_panel
    calls = {"n": 0}

    def flaky(query, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            raise NoPrecedentError("no precedent passed the time guard")
        panel = real(query, *args, **kwargs)
        if calls["n"] % 3 == 1:
            return panel
        return dataclasses.replace(
            panel,
            event_values=panel.event_values.iloc[:3],
            asset_returns=panel.asset_returns.iloc[:3],
        )

    monkeypatch.setattr(evaluate_module, "build_guarded_panel", flaky)
    result = ladder(world, {"M-T": template_of})
    m = result[(result["rung"] == "M-T") & (result["quantile"] == 0.5)].iloc[0]
    pb1 = result[(result["rung"] == "P-B1") & (result["quantile"] == 0.5)].iloc[0]
    assert m["n_refused"] > 0 and m["n_scored"] > 0
    assert m["n_scored"] + m["n_refused"] == pb1["n_scored"]


def test_another_panel_error_is_not_swallowed(world):
    wrong_type = TEMPLATES["armed_conflict"]
    with pytest.raises(PanelError, match="not for the query's event type"):
        ladder(world, {"M-X": lambda key: wrong_type})


def test_underdetermined_scored_units_are_counted(world, monkeypatch):
    real = evaluate_module.fit_sem

    def flagged(graph, panel, *args, **kwargs):
        fitted = real(graph, panel, *args, **kwargs)
        return dataclasses.replace(fitted, underdetermined=frozenset({"TLT"}))

    monkeypatch.setattr(evaluate_module, "fit_sem", flagged)
    result = ladder(world, {"M-T": template_of})
    m = result[(result["rung"] == "M-T") & (result["quantile"] == 0.5)].iloc[0]
    units = precedent_scored_units(
        world.events, world.rows, world.session_cutoffs_utc, horizon=HORIZON
    )
    assert m["n_underdetermined"] == sum("TLT" in assets for assets in units.values()) > 0
    pb1 = result[(result["rung"] == "P-B1") & (result["quantile"] == 0.5)].iloc[0]
    assert pb1["n_underdetermined"] == 0


# ------------------------------------------------------------- shape and contract


def test_columns_dtypes_order_and_determinism(world, table):
    assert list(table.columns) == COLUMNS
    for column in ("n_scored", "n_refused", "n_underdetermined"):
        assert table[column].dtype == np.dtype("int64")
    for column in ("quantile", "pinball_loss", "coverage_80"):
        assert table[column].dtype == np.dtype("float64")
    assert set(table["rung"]) == {"M-T", "M-P", "P-B1"}
    assert set(table["quantile"]) == {0.1, 0.5, 0.9}
    keys = list(zip(table["rung"], table["event_type"], table["quantile"], strict=True))
    assert keys == sorted(keys)
    assert not table.duplicated(["rung", "event_type", "quantile"]).any()
    assert table["coverage_80"].between(0.0, 1.0).all()
    again = ladder(world, {"M-T": template_of, "M-P": placebo_of})
    pd.testing.assert_frame_equal(table, again)


def test_coverage_is_computed_even_when_the_band_quantiles_are_not_requested(world):
    result = ladder(world, {"M-T": template_of}, quantiles=(0.5,))
    m = result[result["rung"] == "M-T"]
    assert list(m["quantile"]) == [0.5] and m["coverage_80"].notna().all()


def test_the_precedent_rung_name_is_reserved(world):
    with pytest.raises(ValueError, match="P-B1"):
        ladder(world, {"P-B1": template_of})


def test_an_empty_event_table_gives_an_empty_typed_table(world):
    empty = score_mechanism_ladder(
        world.events.iloc[0:0],
        world.rows,
        world.session_cutoffs_utc,
        world.returns,
        world.market,
        {"M-T": template_of},
        horizon=HORIZON,
        n_paths=N_PATHS,
    )
    assert list(empty.columns) == COLUMNS and len(empty) == 0
    assert empty["n_scored"].dtype == np.dtype("int64")


def test_the_module_says_where_mech_rungs_are_scored():
    doc = evaluate_module.__doc__
    assert "run_ablation" in doc and "mech_" in doc and "load_rungs" in doc


# ----------------------------------------------------------- shared event mapping


def test_event_values_of_is_the_mapping_the_panel_uses(world):
    key = EventKey(
        event_type="central_bank",
        actor_kind="central_bank",
        impacted_kinds=("index",),
        country_group="US",
        severity_band="high",
        is_scheduled=True,
        direction_sign=-1,
    )
    nodes = [
        "event:central_bank:direction_sign",
        "event:central_bank:severity_band",
        "event:central_bank:is_scheduled",
        "event:central_bank",
    ]
    assert event_values_of(key, nodes) == {nodes[0]: -1, nodes[1]: 2, nodes[2]: 1, nodes[3]: -1}
    with pytest.raises(PanelError, match="not for the query's event type"):
        event_values_of(key, ["event:armed_conflict:direction_sign"])
    # the panel's own event columns are exactly this mapping, evaluated on each precedent's key
    graph = TEMPLATES["central_bank"]
    query_key = world.key(0)
    query = mechanism_query(query_key, world.available_utc_at(600))
    panel = build_guarded_panel(
        query,
        world.events,
        world.rows,
        world.session_cutoffs_utc,
        world.returns,
        world.market,
        [n.id for n in graph.nodes if n.kind is NodeKind.ASSET],
        graph.event_node_ids,
        HORIZON,
        30,
    )
    keys = world.events.set_index("event_id")["key"].loc[panel.event_values.index]
    expected = [
        event_values_of(EventKey.from_string(text), [EVENT_NODE])[EVENT_NODE] for text in keys
    ]
    assert panel.event_values[EVENT_NODE].tolist() == expected
