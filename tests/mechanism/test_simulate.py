"""Tests for Monte Carlo rollouts, do-interventions and the shared counter-based PRNG."""

import numpy as np
import pandas as pd
import pytest

from precedent.mechanism.fit_sem import FittedSEM
from precedent.mechanism.graph_schema import CausalGraph, Edge, Node, derive_kind
from precedent.mechanism.simulate import (
    Intervention,
    counterfactual_pair,
    fan,
    hash_normals,
    hash_uniform,
    simulate,
)
from precedent.spine.event_key import EventKey

SIGN = "event:central_bank:direction_sign"
RATES = "rates_expectation"
RISK = "risk_appetite"
DEAD_END = "credit_spread"
KEY = EventKey(
    event_type="central_bank",
    actor_kind="central_bank",
    impacted_kinds=("market",),
    country_group="US",
    severity_band="mid",
    is_scheduled=True,
    direction_sign=1,
)
EVENT = {SIGN: 1.0}
EDGES = [(SIGN, RATES, 1), (RATES, "TLT", -1), (RATES, RISK, -1), (RISK, "SPY", 1)]


def make_graph(edges: list[tuple[str, str, int]]) -> CausalGraph:
    node_ids = list(dict.fromkeys(i for source, target, _ in edges for i in (source, target)))
    return CausalGraph(
        graph_id="g-test",
        graph_version="template-v1",
        event_key=KEY,
        nodes=[Node(id=i, kind=derive_kind(i)) for i in node_ids],
        edges=[
            Edge(
                source=s,
                target=t,
                sign=sign,
                lag_band="same_session",
                source_doc_id=None,
                evidence_span=None,
            )
            for s, t, sign in edges
        ],
    )


def make_fitted(
    graph: CausalGraph, coefficients: dict[str, dict[str, float]], cov: np.ndarray
) -> FittedSEM:
    assets = [n.id for n in graph.nodes if n.kind.value == "asset"]
    return FittedSEM(
        coefficients=coefficients,
        intercepts={node: 0.0 for node in coefficients},
        residual_cov=cov,
        assets=assets,
        n_obs=50,
        underdetermined=frozenset(),
        sign_disagreements=(),
        graph_id=graph.graph_id,
        graph_version=graph.graph_version,
        fitted_asof_utc=pd.Timestamp("2026-11-07", tz="UTC"),
        horizon=1,
    )


COEFFICIENTS = {
    RATES: {SIGN: 1.0},
    "TLT": {RATES: -0.5},
    RISK: {RATES: -1.0},
    "SPY": {RISK: 0.4},
}


@pytest.fixture
def fitted_graph() -> tuple[FittedSEM, CausalGraph]:
    graph = make_graph(EDGES)
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])  # assets in node order: TLT, SPY
    return make_fitted(graph, COEFFICIENTS, cov), graph


@pytest.fixture
def fitted_graph_with_leaf() -> tuple[FittedSEM, CausalGraph]:
    graph = make_graph([*EDGES, (SIGN, DEAD_END, 1)])
    coefficients = {**COEFFICIENTS, DEAD_END: {SIGN: 1.0}}
    return make_fitted(graph, coefficients, np.diag([0.04, 0.09])), graph


# --- PRNG ---------------------------------------------------------------------------------


def test_hash_uniform_pinned_values():
    # Computed by this implementation; the JS twin in the cockpit must reproduce them.
    expected = [0.038885081536, 0.026300315862, 0.852083498961, 0.685263112304, 0.491849179263]
    got = hash_uniform(0, np.arange(5, dtype=np.uint32))
    np.testing.assert_allclose(got, expected, rtol=0, atol=5e-13)


def test_hash_uniform_is_open_unit_interval():
    u = hash_uniform(7, np.arange(100_000, dtype=np.uint32))
    assert u.min() > 0.0 and u.max() < 1.0
    assert u.dtype == np.float64


def test_hash_uniform_depends_on_seed():
    index = np.arange(10, dtype=np.uint32)
    assert not np.array_equal(hash_uniform(0, index), hash_uniform(1, index))


def test_hash_normals_are_standard_and_shaped():
    z = hash_normals(0, 20_000, 1)
    assert z.shape == (20_000, 1)
    assert abs(z.mean()) < 0.03
    assert abs(z.std() - 1.0) < 0.03
    assert hash_normals(0, 5, 3).shape == (5, 3)


def test_hash_normals_index_layout_is_path_major():
    z = hash_normals(2, 4, 3)
    k = np.arange(12, dtype=np.uint32)
    u1, u2 = hash_uniform(2, 2 * k), hash_uniform(2, 2 * k + 1)
    np.testing.assert_array_equal(
        z.reshape(-1), np.sqrt(-2.0 * np.log(u1)) * np.cos(2 * np.pi * u2)
    )


# --- simulate -----------------------------------------------------------------------------


def test_same_seed_same_paths(fitted_graph):
    a, b = simulate(*fitted_graph, EVENT, seed=3), simulate(*fitted_graph, EVENT, seed=3)
    np.testing.assert_array_equal(a.paths, b.paths)
    assert a.seed == 3 and a.n_paths == 10_000 and a.assets == ["TLT", "SPY"]
    assert a.paths.shape == (10_000, 2)
    assert not a.diagonal_fallback and a.unreachable_interventions == []


def test_different_seed_different_paths(fitted_graph):
    a, b = simulate(*fitted_graph, EVENT, seed=3), simulate(*fitted_graph, EVENT, seed=4)
    assert not np.array_equal(a.paths, b.paths)


def test_paths_follow_the_equations(fitted_graph):
    # rates = 1, TLT mean = -0.5, risk = -1, SPY mean = -0.4
    rollout = simulate(*fitted_graph, EVENT, n_paths=20_000, seed=0)
    means = rollout.paths.mean(axis=0)
    assert means[0] == pytest.approx(-0.5, abs=0.01)
    assert means[1] == pytest.approx(-0.4, abs=0.01)
    assert np.cov(rollout.paths.T)[0, 1] == pytest.approx(0.01, abs=0.005)


def test_do_cuts_incoming_edges(fitted_graph):
    # with rates_expectation forced to 0, the event's sign must not reach TLT. The noise is
    # tiny so the brief's 1e-3 median bound is meaningful; factual TLT would sit at -0.5.
    fitted, graph = fitted_graph
    quiet = make_fitted(graph, COEFFICIENTS, np.diag([1e-8, 1e-8]))
    factual = simulate(quiet, graph, EVENT, seed=0)
    assert fan(factual).loc["TLT", 0.5] == pytest.approx(-0.5, abs=1e-3)
    rollout = simulate(quiet, graph, EVENT, interventions=(Intervention(RATES, 0.0),), seed=0)
    assert abs(fan(rollout).loc["TLT", 0.5]) < 1e-3
    assert rollout.interventions == (Intervention(RATES, 0.0),)


def test_do_value_is_broadcast_and_downstream_uses_it(fitted_graph):
    rollout = simulate(
        *fitted_graph, EVENT, interventions=(Intervention(RISK, 2.0),), n_paths=20_000, seed=0
    )
    assert rollout.paths[:, 1].mean() == pytest.approx(0.8, abs=0.01)  # SPY = 0.4 * 2
    assert rollout.paths[:, 0].mean() == pytest.approx(-0.5, abs=0.01)  # TLT untouched


def test_intervening_on_the_event_node_overrides_event_values(fitted_graph):
    rollout = simulate(
        *fitted_graph, EVENT, interventions=(Intervention(SIGN, -1.0),), n_paths=20_000, seed=0
    )
    assert rollout.paths[:, 0].mean() == pytest.approx(0.5, abs=0.01)


def test_missing_event_value_names_the_node(fitted_graph):
    with pytest.raises(ValueError, match=SIGN):
        simulate(*fitted_graph, {})


def test_intervention_on_unknown_node_is_an_error(fitted_graph):
    with pytest.raises(ValueError, match="nope"):
        simulate(*fitted_graph, EVENT, interventions=(Intervention("nope", 1.0),))


def test_n_paths_must_be_positive(fitted_graph):
    with pytest.raises(ValueError, match="n_paths"):
        simulate(*fitted_graph, EVENT, n_paths=0)


def test_graph_mismatch_is_an_error(fitted_graph):
    fitted, graph = fitted_graph
    other = graph.model_copy(update={"graph_id": "g-other"})
    with pytest.raises(ValueError, match="graph_id"):
        simulate(fitted, other, EVENT)


def test_non_positive_definite_cov_falls_back_to_diagonal(fitted_graph):
    fitted, graph = fitted_graph
    bad = make_fitted(graph, COEFFICIENTS, np.array([[0.04, 0.5], [0.5, 0.09]]))
    rollout = simulate(bad, graph, EVENT, n_paths=20_000, seed=0)
    assert rollout.diagonal_fallback is True
    assert np.cov(rollout.paths.T)[0, 1] == pytest.approx(0.0, abs=0.005)
    assert rollout.paths[:, 0].std() == pytest.approx(0.2, abs=0.01)


def test_negative_variance_is_clamped_in_the_fallback(fitted_graph):
    _, graph = fitted_graph
    bad = make_fitted(graph, COEFFICIENTS, np.array([[-0.04, 0.0], [0.0, 0.09]]))
    rollout = simulate(bad, graph, EVENT, n_paths=100, seed=0)
    assert rollout.diagonal_fallback is True
    assert np.isfinite(rollout.paths).all()


# --- unreachable interventions ------------------------------------------------------------


def test_intervention_with_no_asset_descendant_is_reported(fitted_graph_with_leaf):
    rollout = simulate(*fitted_graph_with_leaf, EVENT, interventions=(Intervention(DEAD_END, 1.0),))
    assert rollout.unreachable_interventions == [DEAD_END]


def test_unreachable_intervention_leaves_the_fan_unchanged(fitted_graph_with_leaf):
    plain = simulate(*fitted_graph_with_leaf, EVENT, seed=5)
    forced = simulate(
        *fitted_graph_with_leaf, EVENT, interventions=(Intervention(DEAD_END, 9.0),), seed=5
    )
    pd.testing.assert_frame_equal(fan(plain), fan(forced))
    assert forced.unreachable_interventions == [DEAD_END]


def test_reachable_intervention_is_not_reported(fitted_graph_with_leaf):
    rollout = simulate(*fitted_graph_with_leaf, EVENT, interventions=(Intervention(RATES, 1.0),))
    assert rollout.unreachable_interventions == []


# --- fan ----------------------------------------------------------------------------------


def test_fan_shape_labels_and_order(fitted_graph):
    table = fan(simulate(*fitted_graph, EVENT, seed=0))
    assert list(table.index) == ["TLT", "SPY"]
    assert list(table.columns) == [0.1, 0.25, 0.5, 0.75, 0.9]
    assert (table.diff(axis=1).dropna(axis=1) > 0).all().all()
    custom = fan(simulate(*fitted_graph, EVENT, seed=0), quantiles=(0.05, 0.95))
    assert list(custom.columns) == [0.05, 0.95]


# --- counterfactual pair ------------------------------------------------------------------


def test_counterfactual_pair_shares_noise(fitted_graph):
    factual, cf = counterfactual_pair(
        *fitted_graph, EVENT, Intervention(RATES, 1.0), n_paths=1000, seed=1
    )
    diff = cf.paths - factual.paths
    assert np.allclose(diff.std(axis=0), 0.0, atol=1e-9)  # the difference is deterministic
    assert factual.interventions == () and cf.interventions == (Intervention(RATES, 1.0),)
    assert factual.seed == cf.seed == 1


def test_counterfactual_pair_matches_independent_simulations(fitted_graph):
    factual, cf = counterfactual_pair(
        *fitted_graph, EVENT, Intervention(RISK, 3.0), n_paths=500, seed=2
    )
    np.testing.assert_array_equal(
        factual.paths, simulate(*fitted_graph, EVENT, n_paths=500, seed=2).paths
    )
    np.testing.assert_array_equal(
        cf.paths,
        simulate(
            *fitted_graph, EVENT, interventions=(Intervention(RISK, 3.0),), n_paths=500, seed=2
        ).paths,
    )
