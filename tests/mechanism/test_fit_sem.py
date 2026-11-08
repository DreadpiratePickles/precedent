"""Tests for structural equation fitting over a hand-built guarded panel."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

from precedent.config import references_dir
from precedent.mechanism.fit_sem import FittedSEM, GuardedPanel, fit_sem
from precedent.mechanism.graph_schema import (
    CausalGraph,
    Edge,
    Node,
    derive_kind,
    load_params,
    node_vocabulary,
)
from precedent.mechanism.templates import load_templates
from precedent.spine.event_key import EventKey
from precedent.spine.event_schema import EventType

SIGN = "event:central_bank:direction_sign"
SEVERITY = "event:central_bank:severity_band"
SCHEDULED = "event:central_bank:is_scheduled"
EVENT = "event:central_bank"
ASOF = pd.Timestamp("2026-11-07", tz="UTC")
KEY = EventKey(
    event_type="central_bank",
    actor_kind="central_bank",
    impacted_kinds=("market",),
    country_group="US",
    severity_band="mid",
    is_scheduled=True,
    direction_sign=1,
)


def make_graph(edges: list[tuple[str, str, int]]) -> CausalGraph:
    """Build a template-version graph from (source, target, sign) tuples."""
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


def make_panel(events: dict[str, list[float]], returns: dict[str, list[float]]) -> GuardedPanel:
    n = len(next(iter(events.values())))
    index = pd.Index([f"e{i}" for i in range(n)], dtype=object)
    return GuardedPanel(
        event_values=pd.DataFrame(events, index=index),
        asset_returns=pd.DataFrame(returns, index=index),
        horizon=1,
        asof_utc=ASOF,
    )


def signs(n: int, seed: int = 1) -> np.ndarray:
    return np.random.default_rng(seed).choice([-1.0, 1.0], size=n)


@pytest.fixture
def planted() -> tuple[CausalGraph, GuardedPanel]:
    """event sign -> latent X (+0.02 per sign unit) -> SPY (coefficient 1.0), noise sd 0.001."""
    rng = np.random.default_rng(7)
    direction = signs(200)
    latent = 0.02 * direction
    spy = 1.0 * latent + rng.normal(0.0, 0.001, size=200)
    graph = make_graph([(SIGN, "rates_expectation", 1), ("rates_expectation", "SPY", 1)])
    return graph, make_panel({SIGN: list(direction)}, {"SPY": list(spy)})


def test_recovers_planted_linear_structure(planted):
    graph, panel = planted
    fitted = fit_sem(graph, panel, ridge=1e-6)
    product = (
        fitted.coefficients["SPY"]["rates_expectation"]
        * (fitted.coefficients["rates_expectation"][SIGN])
    )
    assert product == pytest.approx(0.02, abs=2e-3)
    assert fitted.n_obs == 200
    assert fitted.underdetermined == frozenset()
    assert fitted.sign_disagreements == ()
    assert fitted.intercepts["rates_expectation"] == 0.0


def test_too_few_rows_is_intercept_only_and_flagged():
    graph = make_graph([(SIGN, "HYG", 1), (SEVERITY, "HYG", 1), (SCHEDULED, "HYG", 1)])
    panel = make_panel(
        {SIGN: [1.0, -1.0, 1.0], SEVERITY: [0.0, 1.0, 2.0], SCHEDULED: [1.0, 0.0, 1.0]},
        {"HYG": [0.01, -0.02, 0.03]},
    )
    fitted = fit_sem(graph, panel)
    assert "HYG" in fitted.underdetermined and fitted.coefficients["HYG"] == {}
    assert fitted.intercepts["HYG"] == pytest.approx(0.02 / 3)
    assert fitted.sign_disagreements == ()


def test_latent_with_too_few_rows_is_underdetermined_with_zero_score():
    graph = make_graph([(SIGN, "rates_expectation", 1), ("rates_expectation", "TLT", 1)])
    panel = make_panel({SIGN: [1.0, -1.0]}, {"TLT": [0.01, -0.01]})
    fitted = fit_sem(graph, panel)
    assert fitted.coefficients["rates_expectation"] == {}
    assert fitted.intercepts["rates_expectation"] == 0.0
    assert {"rates_expectation", "TLT"} <= fitted.underdetermined


def test_sign_disagreement_is_recorded_not_corrected():
    rng = np.random.default_rng(3)
    direction = signs(120, seed=3)
    rates_up = -direction  # hawkish (-1) raises rates, matching the edge sign of -1
    tlt = 0.01 * rates_up + rng.normal(0.0, 0.0005, size=120)  # data: rates up -> TLT up
    graph = make_graph([(SIGN, "rates_expectation", -1), ("rates_expectation", "TLT", -1)])
    fitted = fit_sem(graph, make_panel({SIGN: list(direction)}, {"TLT": list(tlt)}), ridge=1e-6)
    assert ("rates_expectation", "TLT") in fitted.sign_disagreements
    assert fitted.coefficients["TLT"]["rates_expectation"] > 0
    assert (SIGN, "rates_expectation") not in fitted.sign_disagreements
    assert fitted.coefficients["rates_expectation"][SIGN] < 0  # anchored to the edge sign


@pytest.mark.parametrize("child_sign", [1, -1])
def test_anchor_falls_back_to_the_child_edge_when_the_first_parent_is_constant(child_sign):
    rng = np.random.default_rng(17)
    direction = signs(100, seed=17)
    spy = 0.02 * direction + rng.normal(0.0, 0.001, size=100)
    graph = make_graph(
        [
            (SCHEDULED, "rates_expectation", 1),
            (SIGN, "rates_expectation", 1),
            ("rates_expectation", "SPY", child_sign),
        ]
    )
    panel = make_panel({SCHEDULED: [1.0] * 100, SIGN: list(direction)}, {"SPY": list(spy)})
    fitted = fit_sem(graph, panel, ridge=1e-6)
    assert fitted.coefficients["rates_expectation"][SCHEDULED] == pytest.approx(0.0, abs=1e-9)
    assert np.sign(fitted.coefficients["SPY"]["rates_expectation"]) == child_sign


def test_nan_asset_rows_are_dropped_for_that_asset_only():
    rng = np.random.default_rng(5)
    direction = signs(60, seed=5)
    spy = 0.02 * direction + rng.normal(0.0, 0.002, size=60)
    tlt = -0.01 * direction + rng.normal(0.0, 0.002, size=60)
    tlt_with_gaps = tlt.copy()
    tlt_with_gaps[::3] = np.nan
    graph = make_graph([(SIGN, "SPY", 1), (SIGN, "TLT", -1)])
    padded = fit_sem(
        graph, make_panel({SIGN: list(direction)}, {"SPY": list(spy), "TLT": list(tlt_with_gaps)})
    )
    keep = ~np.isnan(tlt_with_gaps)
    observed_only = fit_sem(
        graph,
        make_panel({SIGN: list(direction[keep])}, {"SPY": list(spy[keep]), "TLT": list(tlt[keep])}),
    )
    assert padded.coefficients["TLT"][SIGN] == pytest.approx(
        observed_only.coefficients["TLT"][SIGN]
    )
    assert padded.intercepts["TLT"] == pytest.approx(observed_only.intercepts["TLT"])
    full = fit_sem(graph, make_panel({SIGN: list(direction)}, {"SPY": list(spy), "TLT": list(tlt)}))
    assert padded.coefficients["SPY"] == full.coefficients["SPY"]
    assert padded.n_obs == 60


def test_all_nan_asset_is_intercept_only_without_warnings():
    direction = signs(30, seed=2)
    graph = make_graph([(SIGN, "SPY", 1), (SIGN, "GLD", 1)])
    panel = make_panel(
        {SIGN: list(direction)}, {"SPY": list(0.01 * direction), "GLD": [np.nan] * 30}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted = fit_sem(graph, panel)
    assert "GLD" in fitted.underdetermined and fitted.coefficients["GLD"] == {}
    assert fitted.intercepts["GLD"] == 0.0
    gld = fitted.assets.index("GLD")
    assert np.all(fitted.residual_cov[gld] == 0.0) and np.all(fitted.residual_cov[:, gld] == 0.0)


def test_latent_ignores_a_descendant_asset_with_no_observations():
    rng = np.random.default_rng(11)
    direction = signs(80, seed=11)
    spy = 0.02 * direction + rng.normal(0.0, 0.001, size=80)
    graph = make_graph(
        [
            (SIGN, "rates_expectation", 1),
            ("rates_expectation", "SPY", 1),
            ("rates_expectation", "GLD", 1),
        ]
    )
    panel = make_panel({SIGN: list(direction)}, {"SPY": list(spy), "GLD": [np.nan] * 80})
    fitted = fit_sem(graph, panel, ridge=1e-6)
    assert "rates_expectation" not in fitted.underdetermined
    assert fitted.coefficients["rates_expectation"][SIGN] > 0


def test_constant_parent_column_gets_zero_coefficient_without_warnings():
    direction = signs(40, seed=4)
    graph = make_graph([(SIGN, "SPY", 1), (SCHEDULED, "SPY", 1)])
    panel = make_panel(
        {SIGN: list(direction), SCHEDULED: [1.0] * 40}, {"SPY": list(0.01 * direction)}
    )
    for ridge in (1.0, 0.0):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            fitted = fit_sem(graph, panel, ridge=ridge)
        assert fitted.coefficients["SPY"][SCHEDULED] == pytest.approx(0.0, abs=1e-12)
        assert np.isfinite(fitted.residual_cov).all()


def test_exactly_parents_plus_two_rows_is_fitted_not_flagged():
    graph = make_graph([(SIGN, "SPY", 1), (SEVERITY, "SPY", 1)])
    panel = make_panel(
        {SIGN: [1.0, -1.0, 1.0, -1.0], SEVERITY: [0.0, 1.0, 2.0, 1.0]},
        {"SPY": [0.02, -0.01, 0.03, -0.02]},
    )
    fitted = fit_sem(graph, panel, ridge=1e-9)
    assert "SPY" not in fitted.underdetermined
    assert set(fitted.coefficients["SPY"]) == {SIGN, SEVERITY}


def test_asset_without_parents_uses_its_mean_and_is_not_flagged():
    graph = make_graph([(SIGN, "XLE", 1), ("SPY", "QQQ", 1)])
    panel = make_panel(
        {SIGN: [1.0, -1.0]}, {"XLE": [0.01, -0.01], "SPY": [0.01, 0.03], "QQQ": [0.02, 0.04]}
    )
    fitted = fit_sem(graph, panel)
    assert fitted.coefficients["SPY"] == {}
    assert fitted.intercepts["SPY"] == pytest.approx(0.02)
    assert "SPY" not in fitted.underdetermined
    assert "XLE" in fitted.underdetermined  # one parent needs three rows, only two exist


def test_residual_cov_is_symmetric_in_asset_order_and_matches_ols_residuals():
    rng = np.random.default_rng(9)
    direction = signs(100, seed=9)
    shared = rng.normal(0.0, 0.002, size=100)
    spy = 0.02 * direction + shared + rng.normal(0.0, 0.001, size=100)
    tlt = -0.01 * direction + shared + rng.normal(0.0, 0.001, size=100)
    graph = make_graph([(SIGN, "SPY", 1), (SIGN, "TLT", -1)])
    fitted = fit_sem(
        graph, make_panel({SIGN: list(direction)}, {"SPY": list(spy), "TLT": list(tlt)}), ridge=0.0
    )
    assert fitted.assets == ["SPY", "TLT"]
    assert fitted.residual_cov.shape == (2, 2)
    assert np.array_equal(fitted.residual_cov, fitted.residual_cov.T)
    design = np.column_stack([np.ones(100), direction])
    residuals = np.vstack(
        [y - design @ np.linalg.lstsq(design, y, rcond=None)[0] for y in (spy, tlt)]
    )
    assert fitted.residual_cov == pytest.approx(np.cov(residuals), rel=1e-6)
    assert fitted.residual_cov[0, 1] > 0  # the planted shared noise shows up


def test_template_graph_fits_every_asset_without_warnings():
    asset_panel = load_params().asset_panel
    vocabulary = node_vocabulary(
        references_dir() / "mechanism" / "intermediates.json",
        asset_panel,
        [e.value for e in EventType],
    )
    graph = load_templates(references_dir() / "mechanism" / "templates", vocabulary)["central_bank"]
    assets = [n.id for n in graph.nodes if n.kind.value == "asset"]
    rng = np.random.default_rng(13)
    direction = signs(150, seed=13)
    returns = {a: list(rng.normal(0.0, 0.01, size=150)) for a in assets}
    returns["SPY"] = list(0.01 * direction + rng.normal(0.0, 0.002, size=150))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fitted = fit_sem(graph, make_panel({SIGN: list(direction)}, returns))
    assert fitted.assets == assets
    assert fitted.graph_id == graph.graph_id and fitted.graph_version == "template-v1"
    assert fitted.underdetermined == frozenset()
    assert fitted.coefficients["SPY"]["risk_appetite"] != 0.0


def test_json_round_trip(planted):
    graph, panel = planted
    fitted = fit_sem(graph, panel)
    restored = FittedSEM.from_json_dict(json.loads(json.dumps(fitted.to_json_dict())))
    assert restored.coefficients == fitted.coefficients
    assert restored.intercepts == fitted.intercepts
    assert np.array_equal(restored.residual_cov, fitted.residual_cov)
    assert restored.assets == fitted.assets
    assert restored.underdetermined == fitted.underdetermined
    assert restored.sign_disagreements == fitted.sign_disagreements
    assert restored.fitted_asof_utc == fitted.fitted_asof_utc
    assert (restored.n_obs, restored.horizon) == (fitted.n_obs, fitted.horizon)
    assert (restored.graph_id, restored.graph_version) == (fitted.graph_id, fitted.graph_version)


def test_panel_rejects_naive_asof():
    frame = pd.DataFrame({SIGN: [1.0]}, index=["e0"])
    with pytest.raises(ValueError, match="tz-aware"):
        GuardedPanel(
            frame, pd.DataFrame({"SPY": [0.1]}, index=["e0"]), 1, pd.Timestamp("2026-11-07")
        )


def test_panel_rejects_non_utc_asof():
    frame = pd.DataFrame({SIGN: [1.0]}, index=["e0"])
    asof = pd.Timestamp("2026-11-07", tz="America/New_York")
    with pytest.raises(ValueError, match="UTC"):
        GuardedPanel(frame, pd.DataFrame({"SPY": [0.1]}, index=["e0"]), 1, asof)


def test_panel_rejects_mismatched_duplicate_or_empty_indexes():
    events = pd.DataFrame({SIGN: [1.0, 1.0]}, index=["a", "b"])
    with pytest.raises(ValueError, match="same index"):
        GuardedPanel(events, pd.DataFrame({"SPY": [0.1, 0.2]}, index=["a", "c"]), 1, ASOF)
    dup = pd.DataFrame({SIGN: [1.0, 1.0]}, index=["a", "a"])
    with pytest.raises(ValueError, match="duplicate"):
        GuardedPanel(dup, pd.DataFrame({"SPY": [0.1, 0.2]}, index=["a", "a"]), 1, ASOF)
    empty = pd.DataFrame({SIGN: []}, index=pd.Index([], dtype=object))
    with pytest.raises(ValueError, match="at least one row"):
        GuardedPanel(empty, pd.DataFrame({"SPY": []}, index=pd.Index([], dtype=object)), 1, ASOF)


def test_missing_event_column_or_asset_column_is_an_error():
    graph = make_graph([(SIGN, "SPY", 1)])
    with pytest.raises(ValueError, match=SIGN):
        fit_sem(graph, make_panel({SEVERITY: [1.0, 2.0, 3.0]}, {"SPY": [0.1, 0.2, 0.3]}))
    with pytest.raises(ValueError, match="SPY"):
        fit_sem(graph, make_panel({SIGN: [1.0, 2.0, 3.0]}, {"TLT": [0.1, 0.2, 0.3]}))


def test_negative_ridge_is_rejected(planted):
    graph, panel = planted
    with pytest.raises(ValueError, match="ridge"):
        fit_sem(graph, panel, ridge=-1.0)
