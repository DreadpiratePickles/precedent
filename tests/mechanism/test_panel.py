"""Tests for the time-guarded panel built from the Precedent engine on the synthetic world."""

import numpy as np
import pandas as pd
import pytest
from mechanism_world import World, make_world

from precedent.config import references_dir
from precedent.engine.retrieval import PrecedentQuery, find_precedents
from precedent.mechanism.fit_sem import GuardedPanel, fit_sem
from precedent.mechanism.graph_schema import (
    CausalGraph,
    Edge,
    Node,
    NodeKind,
    derive_kind,
    load_params,
    node_vocabulary,
)
from precedent.mechanism.panel import PanelError, build_guarded_panel, mechanism_query
from precedent.mechanism.templates import load_templates
from precedent.spine.event_key import EventKey
from precedent.spine.event_schema import EventType

SIGN = "event:central_bank:direction_sign"
SEVERITY = "event:central_bank:severity_band"
SCHEDULED = "event:central_bank:is_scheduled"
BARE = "event:central_bank"
TOP_K = 30
ALL = 1000  # a top_k that selects every event


@pytest.fixture(scope="module")
def graph() -> CausalGraph:
    vocabulary = node_vocabulary(
        references_dir() / "mechanism" / "intermediates.json",
        load_params().asset_panel,
        [e.value for e in EventType],
    )
    return load_templates(references_dir() / "mechanism" / "templates", vocabulary)["central_bank"]


@pytest.fixture(scope="module")
def world() -> World:
    return make_world()


def asset_ids(graph: CausalGraph) -> list[str]:
    return [n.id for n in graph.nodes if n.kind is NodeKind.ASSET]


def build(
    world: World,
    graph: CausalGraph,
    position: int,
    horizon: int,
    *,
    events: pd.DataFrame | None = None,
    rows: pd.DataFrame | None = None,
    returns: pd.DataFrame | None = None,
    direction_sign: int = 1,
    top_k: int = TOP_K,
) -> GuardedPanel:
    """The panel for a query effective at `position`, using the mechanism (direction-free) query."""
    query = mechanism_query(world.key(direction_sign), world.available_utc_at(position))
    return build_guarded_panel(
        query,
        world.events if events is None else events,
        world.rows if rows is None else rows,
        world.session_cutoffs_utc,
        world.returns if returns is None else returns,
        world.market,
        asset_ids(graph),
        graph.event_node_ids,
        horizon,
        top_k,
    )


def drop_event(world: World, event_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        world.events[world.events["event_id"] != event_id].reset_index(drop=True),
        world.rows[world.rows["event_id"] != event_id].reset_index(drop=True),
    )


# ---------------------------------------------------------------- the time guard


def test_fit_uses_only_time_guarded_rows(graph):
    """An analog whose response window finishes after the query cannot move the fit.

    top_k is large so that dropping the analog does not change which other events are selected.
    """
    spacing, horizon = 12, 20
    query_position = make_world().event_positions[30]
    analog_position = query_position - spacing  # window completes at q + 9 > q at h = 20
    shocked = make_world(shocks={analog_position: 0.5})
    analog_id = shocked.event_id_at(analog_position)
    assert (shocked.rows["event_id"] == analog_id).any()  # the row exists: only the guard hides it
    with_analog = build(shocked, graph, query_position, horizon, top_k=ALL)
    events, rows = drop_event(shocked, analog_id)
    without_analog = build(
        shocked, graph, query_position, horizon, events=events, rows=rows, top_k=ALL
    )
    assert analog_id not in with_analog.event_values.index
    fit_with, fit_without = fit_sem(graph, with_analog), fit_sem(graph, without_analog)
    assert fit_with.coefficients == fit_without.coefficients
    assert fit_with.intercepts == fit_without.intercepts
    assert np.array_equal(fit_with.residual_cov, fit_without.residual_cov)
    # Control: once the window has finished (a later query) the same analog does move the fit.
    later = build(shocked, graph, query_position + spacing, horizon, top_k=ALL)
    later_without = build(
        shocked, graph, query_position + spacing, horizon, events=events, rows=rows, top_k=ALL
    )
    assert analog_id in later.event_values.index
    assert fit_sem(graph, later).coefficients != fit_sem(graph, later_without).coefficients


def test_panel_index_is_exactly_what_find_precedents_returned(world, graph):
    horizon = 20
    position = world.event_positions[30]
    query = mechanism_query(world.key(1), world.available_utc_at(position))
    found = find_precedents(
        query, world.events, world.rows, world.session_cutoffs_utc, horizon, "asset", TOP_K
    )
    panel = build(world, graph, position, horizon)
    assert panel.event_values.index.tolist() == found["event_id"].drop_duplicates().tolist()
    assert panel.asset_returns.index.equals(panel.event_values.index)
    unfinished = world.event_id_at(position - 12)
    assert unfinished not in panel.event_values.index  # selected by top_k, dropped by the guard
    assert panel.n_obs == TOP_K - 2  # the query's own event and the unfinished analog
    assert panel.horizon == horizon
    assert panel.asof_utc == world.available_utc_at(position)
    assert list(panel.asset_returns.columns) == asset_ids(graph)
    assert list(panel.event_values.columns) == graph.event_node_ids


def test_unobserved_asset_is_an_all_nan_column(world, graph):
    panel = build(world, graph, world.event_positions[30], 5)
    unobserved = [a for a in asset_ids(graph) if a not in world.assets]
    assert unobserved
    assert panel.asset_returns[unobserved].isna().all().all()
    assert panel.asset_returns[["SPY", "TLT", "HYG", "UUP"]].notna().all().all()


def test_asset_values_are_the_library_values(world, graph):
    position = world.event_positions[30]
    panel = build(world, graph, position, 5)
    rows = world.rows[(world.rows["horizon"] == 5) & (world.rows["outcome_key"] == "TLT")]
    expected = rows.set_index("event_id")["value"].reindex(panel.asset_returns.index)
    assert np.allclose(panel.asset_returns["TLT"].to_numpy(), expected.to_numpy())


# ---------------------------------------------------------------- the market column


def test_market_column_is_the_raw_cumulative_return_over_the_window(world, graph):
    horizon = 5
    panel = build(world, graph, world.event_positions[30], horizon)
    positions = world.events.set_index("event_id")["effective_position"]
    for event_id in panel.asset_returns.index[:5]:
        p = int(positions[event_id])
        by_hand = world.returns["SPY"].iloc[p + 2 : p + 2 + horizon].sum()
        assert panel.asset_returns.loc[event_id, "SPY"] == pytest.approx(by_hand, abs=1e-15)
    # A different horizon uses a different window.
    long_panel = build(world, graph, world.event_positions[30], 20)
    event_id = long_panel.asset_returns.index[0]
    p = int(positions[event_id])
    assert long_panel.asset_returns.loc[event_id, "SPY"] == pytest.approx(
        world.returns["SPY"].iloc[p + 2 : p + 22].sum(), abs=1e-15
    )


def test_market_column_is_nan_when_a_window_return_is_missing(world, graph):
    position = world.event_positions[30]
    clean = build(world, graph, position, 5)
    victim = clean.asset_returns.index[0]
    p = int(world.events.set_index("event_id").loc[victim, "effective_position"])
    holey = world.returns.copy()
    holey.iloc[p + 3, holey.columns.get_loc("SPY")] = np.nan
    panel = build(world, graph, position, 5, returns=holey)
    assert np.isnan(panel.asset_returns.loc[victim, "SPY"])
    others = panel.asset_returns.drop(index=victim)
    assert others["SPY"].notna().all()
    assert panel.asset_returns["TLT"].equals(clean.asset_returns["TLT"])  # not bridged, not zeroed


def test_market_not_among_assets_adds_no_market_column(world):
    position = world.event_positions[30]
    query = mechanism_query(world.key(1), world.available_utc_at(position))
    panel = build_guarded_panel(
        query,
        world.events,
        world.rows,
        world.session_cutoffs_utc,
        world.returns,
        "SPY",
        ["TLT", "HYG"],
        [SIGN],
        5,
        TOP_K,
    )
    assert list(panel.asset_returns.columns) == ["TLT", "HYG"]


def test_market_missing_from_returns_is_a_panel_error(world, graph):
    with pytest.raises(PanelError, match="SPY"):
        build(world, graph, world.event_positions[30], 5, returns=world.returns.drop(columns="SPY"))


def test_returns_on_another_axis_is_a_panel_error(world, graph):
    with pytest.raises(PanelError, match="session"):
        build(world, graph, world.event_positions[30], 5, returns=world.returns.iloc[:-1])


# ---------------------------------------------------------------- the event columns


@pytest.mark.parametrize("band, ordinal", [("low", 0), ("mid", 1), ("high", 2)])
@pytest.mark.parametrize("scheduled", [True, False])
def test_event_values_are_direction_severity_ordinal_and_schedule_flag(
    graph, band, ordinal, scheduled
):
    small = make_world(severity_band=band, is_scheduled=scheduled)
    query = mechanism_query(small.key(1), small.available_utc_at(small.event_positions[30]))
    panel = build_guarded_panel(
        query,
        small.events,
        small.rows,
        small.session_cutoffs_utc,
        small.returns,
        small.market,
        asset_ids(graph),
        [SIGN, SEVERITY, SCHEDULED, BARE],
        5,
        TOP_K,
    )
    signs = small.events.set_index("event_id")["key"].map(
        lambda text: EventKey.from_string(text).direction_sign
    )
    expected = signs.reindex(panel.event_values.index)
    assert panel.event_values[SIGN].tolist() == expected.tolist()
    assert panel.event_values[BARE].tolist() == expected.tolist()  # a bare id means direction_sign
    assert set(panel.event_values[SIGN]) == {-1, 1}  # the fit sees both signs
    assert (panel.event_values[SEVERITY] == ordinal).all()
    assert (panel.event_values[SCHEDULED] == int(scheduled)).all()


def test_unknown_event_field_is_a_panel_error_naming_it(world, graph):
    query = mechanism_query(world.key(1), world.available_utc_at(world.event_positions[30]))
    with pytest.raises(PanelError, match="mood"):
        build_guarded_panel(
            query,
            world.events,
            world.rows,
            world.session_cutoffs_utc,
            world.returns,
            world.market,
            asset_ids(graph),
            ["event:central_bank:mood"],
            5,
            TOP_K,
        )


@pytest.mark.parametrize(
    "bad", ["event:armed_conflict:direction_sign", "rates_expectation", "event:a:b:c"]
)
def test_event_column_that_is_not_this_events_is_a_panel_error(world, graph, bad):
    query = mechanism_query(world.key(1), world.available_utc_at(world.event_positions[30]))
    with pytest.raises(PanelError, match="event"):
        build_guarded_panel(
            query,
            world.events,
            world.rows,
            world.session_cutoffs_utc,
            world.returns,
            world.market,
            asset_ids(graph),
            [bad],
            5,
            TOP_K,
        )


def test_duplicate_assets_are_a_panel_error(world):
    query = mechanism_query(world.key(1), world.available_utc_at(world.event_positions[30]))
    with pytest.raises(PanelError, match="duplicate"):
        build_guarded_panel(
            query,
            world.events,
            world.rows,
            world.session_cutoffs_utc,
            world.returns,
            world.market,
            ["TLT", "TLT"],
            [SIGN],
            5,
            TOP_K,
        )


# ---------------------------------------------------------------- the query and the fit


def test_mechanism_query_keeps_every_field_but_the_direction():
    key = EventKey("central_bank", "central_bank", ("index", "company"), "US", "high", False, -1)
    asof = pd.Timestamp("2022-03-01 12:00", tz="UTC")
    query = mechanism_query(key, asof, story_cluster_id="story-7")
    assert query.key == EventKey(
        "central_bank", "central_bank", ("index", "company"), "US", "high", False, 0
    )
    assert query.available_utc == asof
    assert query.story_cluster_id == "story-7"
    assert mechanism_query(key, asof).story_cluster_id is None


def test_mechanism_query_converts_to_utc_and_rejects_naive_time():
    key = EventKey("central_bank", "central_bank", (), "US", "mid", True, 1)
    eastern = pd.Timestamp("2022-03-01 07:00", tz="America/New_York")
    assert mechanism_query(key, eastern).available_utc == pd.Timestamp("2022-03-01 12:00", tz="UTC")
    assert mechanism_query(key, eastern).available_utc.utcoffset() == pd.Timedelta(0)
    with pytest.raises(ValueError, match="timezone"):
        mechanism_query(key, pd.Timestamp("2022-03-01 12:00"))


def test_exact_key_query_would_freeze_the_direction_column(world, graph):
    """Why mechanism_query exists: a signed query selects one sign, so the slope is unidentifiable."""
    position = world.event_positions[30]
    exact = PrecedentQuery(world.key(1), world.available_utc_at(position))
    panel = build_guarded_panel(
        exact,
        world.events,
        world.rows,
        world.session_cutoffs_utc,
        world.returns,
        world.market,
        asset_ids(graph),
        [SIGN],
        5,
        TOP_K,
    )
    assert set(panel.event_values[SIGN]) == {1}
    assert set(build(world, graph, position, 5).event_values[SIGN]) == {-1, 1}


def test_fitted_slopes_recover_the_planted_response(world, graph):
    """End to end: world -> engine -> panel -> SEM agrees with what was planted (loose tolerance)."""
    panel = build(world, graph, world.event_positions[-1], 5)
    fit = fit_sem(graph, panel)
    assert fit.sign_disagreements == ()
    assert fit.coefficients["rates_expectation"][SIGN] < 0  # the template edge is negative
    # rates_expectation reads as -direction, so TLT's slope on it is minus its planted CAR.
    assert fit.coefficients["TLT"]["rates_expectation"] == pytest.approx(
        -world.planted_car("TLT"), abs=0.006
    )
    # The market node is the market's own return: its slope on risk_appetite is the raw effect.
    assert fit.coefficients["SPY"]["risk_appetite"] == pytest.approx(
        world.planted_car("SPY"), abs=0.006
    )


def test_constant_event_column_gets_a_zero_slope_and_changes_nothing_else(world):
    def graph_of(sources: list[str]) -> CausalGraph:
        node_ids = [*sources, "TLT", "HYG"]
        return CausalGraph(
            graph_id="g-const",
            graph_version="template-v1",
            event_key=world.key(1),
            nodes=[Node(id=i, kind=derive_kind(i)) for i in node_ids],
            edges=[
                Edge(
                    source=s,
                    target=t,
                    sign=1,
                    lag_band="same_session",
                    source_doc_id=None,
                    evidence_span=None,
                )
                for s in sources
                for t in ("TLT", "HYG")
            ],
        )

    position = world.event_positions[30]
    query = mechanism_query(world.key(1), world.available_utc_at(position))

    def panel_for(columns: list[str]) -> GuardedPanel:
        return build_guarded_panel(
            query,
            world.events,
            world.rows,
            world.session_cutoffs_utc,
            world.returns,
            world.market,
            ["TLT", "HYG"],
            columns,
            5,
            TOP_K,
        )

    with_const = fit_sem(graph_of([SIGN, SEVERITY]), panel_for([SIGN, SEVERITY]))
    alone = fit_sem(graph_of([SIGN]), panel_for([SIGN]))
    for asset in ("TLT", "HYG"):
        assert with_const.coefficients[asset][SEVERITY] == pytest.approx(0.0, abs=1e-12)
        assert with_const.coefficients[asset][SIGN] == pytest.approx(
            alone.coefficients[asset][SIGN], abs=1e-12
        )
        assert with_const.intercepts[asset] == pytest.approx(alone.intercepts[asset], abs=1e-12)


# ---------------------------------------------------------------- nothing passes


def test_no_matching_precedent_is_a_panel_error(world, graph):
    other = EventKey("armed_conflict", "government", (), "US", "mid", True, 0)
    query = mechanism_query(other, world.available_utc_at(world.event_positions[30]))
    with pytest.raises(PanelError, match="no precedent passed the time guard for armed_conflict"):
        build_guarded_panel(
            query,
            world.events,
            world.rows,
            world.session_cutoffs_utc,
            world.returns,
            world.market,
            asset_ids(graph),
            ["event:armed_conflict:direction_sign"],
            5,
            TOP_K,
        )


def test_every_window_unfinished_is_a_panel_error(world, graph):
    """The first event is selected, but its horizon-20 window ends after the query."""
    with pytest.raises(PanelError, match="no precedent passed the time guard"):
        build(world, graph, world.event_positions[1], 20)


def test_query_story_excludes_that_story_from_the_fit_panel(world, graph):
    """V-5: with story_cluster_id set, a finished earlier event of that story leaves the panel."""
    horizon = 20
    position = world.event_positions[30]
    earlier_id = world.event_id_at(world.event_positions[10])
    story = world.events.set_index("event_id").loc[earlier_id, "story_cluster_id"]
    assert (
        story not in world.events[world.events["event_id"] != earlier_id]["story_cluster_id"].values
    )

    def panel_for(story_cluster_id: str | None) -> GuardedPanel:
        query = mechanism_query(
            world.key(1), world.available_utc_at(position), story_cluster_id=story_cluster_id
        )
        return build_guarded_panel(
            query,
            world.events,
            world.rows,
            world.session_cutoffs_utc,
            world.returns,
            world.market,
            asset_ids(graph),
            graph.event_node_ids,
            horizon,
            ALL,
        )

    assert earlier_id in panel_for(None).event_values.index  # present without the story id
    assert earlier_id not in panel_for(story).event_values.index
