"""Page and text disclosures of the cockpit: Precedent's numbers and sizes, the underdetermined
equations, the diagonal fallback (verifier findings V-2, V-6), slider ranges (V-7) and the
embedded residual factor (M-1). The page script runs under node against a small DOM stub."""

import dataclasses
import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from test_cockpit import (
    ASOF,
    NODE,
    RATES,
    RISK,
    SIGN,
    BandRow,
    build_result,
    hand_result,
    load_engine_params,
    make_fitted,
    make_graph,
    page_script,
    render_html,
    render_json,
    render_text,
)

from precedent.mechanism.cockpit import _triple
from precedent.mechanism.fit_sem import GuardedPanel, fit_sem
from precedent.mechanism.simulate import Intervention, simulate

DIAGONAL_LINE = (
    "residual covariance is not positive definite: the fan uses its diagonal, so cross-asset "
    "correlation is dropped (each row's own fan is unaffected)"
)
needs_node = pytest.mark.skipif(NODE is None, reason="node not on PATH")

DOM_STUB = r"""
class El {
  constructor(tag) { this.tag = tag; this.children = []; this._text = ""; this.attrs = {};
    this.className = ""; this.classList = { toggle() {} }; this.style = {}; }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(""); }
  append(...nodes) { nodes.forEach((n) => { if (typeof n === "string") { const t = new El("#text"); t._text = n; this.children.push(t); } else this.children.push(n); }); }
  appendChild(n) { this.children.push(n); return n; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  getAttribute(k) { return this.attrs[k]; }
  addEventListener() {}
}
const byId = {};
globalThis.document = {
  getElementById(id) { return byId[id] || (byId[id] = new El(id)); },
  createElement(tag) { return new El(tag); },
  createElementNS(ns, tag) { return new El(tag); },
  querySelectorAll() { return []; },
};
document.getElementById("payload")._text = require("fs").readFileSync(0, "utf8");
"""
DUMP = r"""
const texts = (el) => el.children.map((c) => c.textContent);
console.log(JSON.stringify({
  fans: byId.fans.children.map((row) => texts(row)),
  disclosures: byId.disclosures ? texts(byId.disclosures) : [],
  subtitle: byId.subtitle.textContent,
  controls: byId.controls.children.map((row) => ({
    node: row.children[0].textContent, min: row.children[2].min, max: row.children[2].max,
    step: row.children[2].step, value: row.children[2].value })),
}));
"""


def run_page(html: str, stdin: str) -> dict:
    code = "globalThis.window = globalThis;\n" + DOM_STUB + page_script(html) + DUMP
    done = subprocess.run(
        [NODE, "-e", code], input=stdin, capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def page_controls(result) -> dict:
    """The sliders the page builds, keyed by full node id (the label drops the `event:` prefix)."""
    out = run_page(render_html(result), render_json(result))
    return {
        (SIGN if c["node"] == SIGN.removeprefix("event:") else c["node"]): c
        for c in out["controls"]
    }


def bands_result(**overrides):
    """TLT and SPY survive with bands; HYG's band is refused (a reason row)."""
    graph = make_graph()
    kwargs = {
        "graph": graph,
        "fitted": make_fitted(graph),
        "event_values": {SIGN: -1.0},
        "interventions": (),
        "precedent": [
            BandRow(asset="TLT", horizon=5, p10=-0.009, median=-0.005, p90=0.001,
                    n=17, n_eff=11.5, refused=False),
            BandRow(asset="SPY", horizon=5, p10=-0.020, median=-0.010, p90=0.005,
                    n=30, n_eff=20.0, refused=False),
            BandRow(asset="HYG", horizon=5, p10=None, median=None, p90=None,
                    n=4, n_eff=3.24, refused=True),
        ],
        "refused_assets": ("HYG",),
        "min_neff": load_engine_params().min_neff,
        "seed": 0,
        "n_paths": 2000,
        "asof_utc": ASOF,
        "code_commit": None,
    }  # fmt: skip
    kwargs.update(overrides)
    return build_result(**kwargs)


# --- V-2: the page shows Precedent's numbers and both sample sizes --------------------------


@needs_node
def test_page_rows_show_precedent_p10_median_p90_and_both_sample_sizes():
    result = bands_result()
    out = run_page(render_html(result), render_json(result))
    rows = {row[0]: row for row in out["fans"]}
    tlt, spy = rows["TLT"], rows["SPY"]
    fan = result.rollout_fan
    assert tlt[2] == _triple(fan["TLT"]["0.1"], fan["TLT"]["0.5"], fan["TLT"]["0.9"])
    assert tlt[3] == "-0.9% / -0.5% / +0.1%"  # Precedent's p10 / median / p90
    assert tlt[5] == "n=42/17 n_eff=11.5"  # fit n, band n, n_eff
    assert spy[3] == "-2.0% / -1.0% / +0.5%" and spy[5] == "n=42/30 n_eff=20.0"


@needs_node
def test_page_suppressed_rows_keep_their_reason_and_no_fan():
    result = bands_result()
    out = run_page(render_html(result), render_json(result))
    hyg = next(row for row in out["fans"] if row[0] == "HYG")
    assert hyg == ["HYG", "", "band refused: n_eff=3.2 < min_neff=8.0"]


@needs_node
def test_page_market_row_stays_labelled_raw_beside_its_band():
    graph = make_graph()
    result = bands_result(
        graph=graph,
        precedent=[
            BandRow(asset="SPY", horizon=5, p10=-0.02, median=0.0, p90=0.02,
                    n=30, n_eff=20.0, refused=False, basis="raw_return"),
        ],
        refused_assets=(),
    )  # fmt: skip
    out = run_page(render_html(result), render_json(result))
    spy = next(row for row in out["fans"] if row[0].startswith("SPY"))
    assert spy[0] == "SPY (raw)" and spy[3] == "-2.0% / +0.0% / +2.0% (raw)"


# --- V-6: the text and the page name the underdetermined equations and the fallback ---------


def test_text_names_the_underdetermined_equations():
    lines = render_text(hand_result()).splitlines()
    assert f"underdetermined equations: {RISK}" in lines
    clean = make_fitted(make_graph()).to_json_dict() | {"underdetermined": []}
    assert not any(
        line.startswith("underdetermined equations")
        for line in render_text(hand_result(fitted=clean)).splitlines()
    )


def test_text_prints_one_line_when_the_covariance_fell_back_to_the_diagonal():
    lines = render_text(hand_result(diagonal_fallback=True)).splitlines()
    assert lines.count(DIAGONAL_LINE) == 1
    assert DIAGONAL_LINE not in render_text(hand_result()).splitlines()
    assert lines[-1].startswith("A fitted model")  # the disclaimer stays last


@needs_node
def test_page_mirrors_both_disclosure_lines():
    graph = make_graph()
    not_pd = np.array([[0.04, 0.2, 0.0], [0.2, 0.09, 0.0], [0.0, 0.0, 0.0625]])
    result = bands_result(fitted=make_fitted(graph, cov=not_pd), refused_assets=())
    assert result.diagonal_fallback is True
    out = run_page(render_html(result), render_json(result))
    assert out["disclosures"] == [f"underdetermined equations: {RISK}", DIAGONAL_LINE]
    quiet = bands_result()
    assert run_page(render_html(quiet), render_json(quiet))["disclosures"] == [
        f"underdetermined equations: {RISK}"
    ]


# --- V-7: slider ranges are the fitted panel's 5th to 95th percentile per node ----------------


def planted_panel() -> GuardedPanel:
    rng = np.random.default_rng(3)
    direction = rng.choice([-1.0, 1.0], size=60)
    rates = -1.0 * direction
    returns = {
        "TLT": -0.5 * rates + rng.normal(0, 0.01, 60),
        "SPY": 0.4 * rates + rng.normal(0, 0.01, 60),
        "HYG": 0.2 * rates + rng.normal(0, 0.01, 60),
    }
    index = pd.Index([f"e{i}" for i in range(60)], dtype=object)
    return GuardedPanel(
        event_values=pd.DataFrame({SIGN: direction}, index=index),
        asset_returns=pd.DataFrame(returns, index=index),
        horizon=5,
        asof_utc=pd.Timestamp(ASOF),
    )


def test_fit_records_each_non_asset_nodes_panel_percentiles():
    panel = planted_panel()
    fitted = fit_sem(make_graph(), panel)
    assert set(fitted.node_ranges) == {SIGN, RATES, RISK}
    direction = panel.event_values[SIGN].to_numpy()
    assert fitted.node_ranges[SIGN] == (
        pytest.approx(np.percentile(direction, 5)),
        pytest.approx(np.percentile(direction, 95)),
    )
    # a latent's range is its fitted score over the same rows
    score = fitted.intercepts[RATES] + fitted.coefficients[RATES][SIGN] * direction
    assert fitted.node_ranges[RATES] == (
        pytest.approx(np.percentile(score, 5)),
        pytest.approx(np.percentile(score, 95)),
    )
    restored = type(fitted).from_json_dict(json.loads(json.dumps(fitted.to_json_dict())))
    assert restored.node_ranges == fitted.node_ranges


def ranged_result(**overrides):
    graph = make_graph()
    ranges = {SIGN: (-0.8, 0.9), RATES: (-0.4, 0.6), RISK: (-0.2, 0.3)}
    fitted = dataclasses.replace(make_fitted(graph), node_ranges=ranges)
    kwargs = {
        "graph": graph, "fitted": fitted, "event_values": {SIGN: -1.0},
        "interventions": (), "precedent": [], "min_neff": load_engine_params().min_neff,
        "seed": 0, "n_paths": 500, "asof_utc": ASOF, "code_commit": None,
    }  # fmt: skip
    kwargs.update(overrides)
    return build_result(**kwargs)


def test_payload_embeds_the_ranges_and_the_page_has_no_fixed_range():
    payload = json.loads(render_json(ranged_result()))
    assert payload["slider_ranges"] == {SIGN: [-0.8, 0.9], RATES: [-0.4, 0.6], RISK: [-0.2, 0.3]}
    template = TEMPLATE_TEXT()
    assert "EVENT_RANGES" not in template and "Math.min(-3" not in template


def TEMPLATE_TEXT() -> str:
    from test_cockpit import TEMPLATE_PATH

    return TEMPLATE_PATH.read_text(encoding="utf-8")


@needs_node
def test_slider_range_is_the_panel_percentile_widened_to_include_the_query_value():
    result = ranged_result(interventions=(Intervention(RATES, 1.5),))
    controls = page_controls(result)
    # an event enum moves in whole steps, so its percentile range widens outward to integers
    assert (controls[SIGN]["min"], controls[SIGN]["max"], controls[SIGN]["step"]) == (-1, 1, 1)
    # the do() value 1.5 is outside the percentile range: the range widens to include it
    assert (controls[RATES]["min"], controls[RATES]["max"]) == (-0.4, 1.5)
    assert (controls[RISK]["min"], controls[RISK]["max"]) == (-0.2, 0.3)
    assert controls[RISK]["step"] == 0.008  # a 60th of the span, to one significant figure


@needs_node
def test_a_node_without_a_panel_range_is_pinned_to_its_value_not_a_default_range():
    fitted = dataclasses.replace(make_fitted(make_graph()), node_ranges={})  # none recorded
    result = ranged_result(fitted=fitted)
    controls = page_controls(result)
    assert (controls[RATES]["min"], controls[RATES]["max"]) == (0, 0)
    assert (controls[SIGN]["min"], controls[SIGN]["max"]) == (-1, -1)


# --- M-1: the Cholesky factor is computed once in Python and embedded -------------------------


TICKERS = ["TLT", "SPY", "HYG", "XLE", "GLD", "QQQ", "IWM", "EEM"]


def near_boundary_covariances(count: int, seed: int) -> list[np.ndarray]:
    """Covariances a hair either side of positive definite: rank n - 1 plus a tiny ridge."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(count):
        n = 3 + i % 6
        b = rng.normal(0.0, 0.02, size=(n, n - 1))
        ridge = 10.0 ** rng.uniform(-18, -10) * 4e-4
        out.append(b @ b.T + ridge * np.eye(n))
    return out


def factor_result(cov: np.ndarray, seed: int = 0):
    n = cov.shape[0]
    assets = TICKERS[:n]
    graph = make_graph(edges=[(SIGN, a, 1) for a in assets])
    fitted = make_fitted(graph, coefficients={a: {SIGN: 0.1} for a in assets}, cov=cov)
    return build_result(
        graph=graph, fitted=fitted, event_values={SIGN: -1.0}, interventions=(),
        precedent=[], min_neff=load_engine_params().min_neff,
        seed=seed, n_paths=300, asof_utc=ASOF, code_commit=None,
    )  # fmt: skip


def test_payload_factor_is_the_one_simulate_used():
    pd_cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    not_pd = np.array([[0.04, 0.2], [0.2, 0.09]])
    for cov, expect_fallback in ((pd_cov, False), (not_pd, True)):
        result = factor_result(cov)
        graph, fitted = result.graph, make_fitted(result.graph, cov=cov, coefficients={
            a: {SIGN: 0.1} for a in result.fitted["assets"]})  # fmt: skip
        rollout = simulate(fitted, graph, {SIGN: -1.0}, n_paths=300, seed=0)
        factor = np.array(result.residual_factor)
        assert np.array_equal(factor, rollout.factor)
        assert result.diagonal_fallback is expect_fallback
        expected = np.diag(np.sqrt(np.diag(cov))) if expect_fallback else np.linalg.cholesky(cov)
        assert np.array_equal(factor, expected)
        assert json.loads(render_json(result))["residual_factor"] == expected.tolist()


@needs_node
def test_js_fans_match_python_on_seeded_near_boundary_matrices_and_the_page_never_factorises():
    results = [factor_result(cov, seed=i) for i, cov in enumerate(near_boundary_covariances(120, 11))]
    assert any(r.diagonal_fallback for r in results) and not all(r.diagonal_fallback for r in results)
    html = render_html(results[0])
    assert "cholesky" not in html.lower()  # the twin uses the embedded factor
    program = """
    const payloads = JSON.parse(require("fs").readFileSync(0, "utf8"));
    console.log(JSON.stringify(payloads.map((p) =>
      window.mechanismSimulate(p, p.event_values, [], p.n_paths, p.seed))));
    """
    stdin = json.dumps([json.loads(render_json(r)) for r in results])
    from test_cockpit import run_node

    got = run_node(html, program, stdin)
    for index, (result, fans) in enumerate(zip(results, got, strict=True)):
        for asset, row in result.rollout_fan.items():
            for q, expected in row.items():
                assert fans[asset][q] == pytest.approx(expected, abs=5e-4), (index, asset, q)
