"""Page and text disclosures of the cockpit: Precedent's numbers and sizes, the underdetermined
equations, the diagonal fallback (verifier findings V-2, V-6), slider ranges (V-7) and the
embedded residual factor (M-1). The page script runs under node against a small DOM stub."""

import json
import subprocess

import pytest
from test_cockpit import (
    ASOF,
    NODE,
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
    result = hand_result(diagonal_fallback=True)
    out = run_page(render_html(result), render_json(result))
    assert out["disclosures"] == [f"underdetermined equations: {RISK}", DIAGONAL_LINE]
    quiet = bands_result()
    assert run_page(render_html(quiet), render_json(quiet))["disclosures"] == [
        f"underdetermined equations: {RISK}"
    ]
