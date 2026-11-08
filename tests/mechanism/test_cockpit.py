"""Tests for the cockpit renderers, the JavaScript simulator twin and the mechanism command."""

import json
import re
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from precedent.config import references_dir
from precedent.mechanism import cli
from precedent.mechanism.cli import parse_do_flags, run
from precedent.mechanism.cockpit import (
    DISCLAIMER,
    BandRow,
    InterventionRow,
    MechanismResult,
    build_result,
    render_html,
    render_json,
    render_text,
)
from precedent.mechanism.fit_sem import FittedSEM
from precedent.mechanism.graph_schema import (
    CausalGraph,
    Edge,
    Node,
    derive_kind,
)
from precedent.mechanism.simulate import Intervention, fan, simulate
from precedent.mechanism.templates import load_templates
from precedent.spine.event_key import EventKey
from precedent.spine.typers.base import TypedText

SIGN = "event:central_bank:direction_sign"
RATES = "rates_expectation"
RISK = "risk_appetite"
KEY = EventKey(
    event_type="central_bank",
    actor_kind="central_bank",
    impacted_kinds=("index",),
    country_group="US",
    severity_band="high",
    is_scheduled=False,
    direction_sign=-1,
)
EDGES = [
    (SIGN, RATES, -1),
    (RATES, "TLT", -1),
    (RATES, RISK, -1),
    (RISK, "SPY", 1),
    (RISK, "HYG", 1),
]
COEFFICIENTS = {
    RATES: {SIGN: -1.0},
    "TLT": {RATES: -0.5},
    RISK: {RATES: -1.0},
    "SPY": {RISK: 0.4},
    "HYG": {RISK: 0.2},
}
ASOF = datetime(2026, 11, 7, 12, 0, tzinfo=UTC)
KEY_FLAGS = [
    "--event-type", "central_bank", "--actor-kind", "central_bank",
    "--severity-band", "high", "--unscheduled", "--direction", "-1", "--asof", "2024-06-03",
]  # fmt: skip
NODE = shutil.which("node")


def make_graph(edges=EDGES) -> CausalGraph:
    node_ids = list(dict.fromkeys(i for s, t, _ in edges for i in (s, t)))
    return CausalGraph(
        graph_id="g-cockpit",
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


def make_fitted(graph, coefficients=None, cov=None, disagreements=()) -> FittedSEM:
    coefficients = COEFFICIENTS if coefficients is None else coefficients
    assets = [n.id for n in graph.nodes if n.kind.value == "asset"]
    cov = np.diag([0.04, 0.09, 0.0625][: len(assets)]) if cov is None else cov
    return FittedSEM(
        coefficients=coefficients,
        intercepts={node: 0.0 for node in coefficients},
        residual_cov=cov,
        assets=assets,
        n_obs=42,
        underdetermined=frozenset({RISK}),
        sign_disagreements=tuple(disagreements),
        graph_id=graph.graph_id,
        graph_version=graph.graph_version,
        fitted_asof_utc=pd.Timestamp(ASOF),
        horizon=5,
    )


def quantile_row(p10, median, p90) -> dict[str, float]:
    return {
        "0.1": p10, "0.25": (p10 + median) / 2, "0.5": median,
        "0.75": (median + p90) / 2, "0.9": p90,
    }  # fmt: skip


def hand_result(**overrides) -> MechanismResult:
    graph = make_graph()
    fields = {
        "graph": graph,
        "fitted": make_fitted(graph, disagreements=[(RATES, "TLT")]).to_json_dict(),
        "event_values": {SIGN: -1.0},
        "interventions": [],
        "rollout_fan": {
            "TLT": quantile_row(-0.012, -0.004, 0.003),
            "SPY": quantile_row(0.010, 0.020, 0.030),
            "HYG": quantile_row(-0.010, 0.000, 0.010),
        },
        "precedent": [
            BandRow(asset="TLT", horizon=5, p10=-0.009, median=-0.005, p90=0.001,
                    n=17, n_eff=11.5, refused=False),
            BandRow(asset="SPY", horizon=5, p10=-0.020, median=-0.010, p90=0.005,
                    n=30, n_eff=20.0, refused=False),
            BandRow(asset="HYG", horizon=5, p10=None, median=None, p90=None,
                    n=4, n_eff=3.24, refused=True),
        ],
        "seed": 0,
        "n_paths": 2000,
        "asof_utc": ASOF,
        "unreachable_interventions": [],
        "diagonal_fallback": False,
        "code_commit": "abc1234",
    }  # fmt: skip
    fields.update(overrides)
    return MechanismResult(**fields)


def simulated_result(interventions=(), bands=(), n_paths=4000, seed=0) -> MechanismResult:
    graph = make_graph()
    fitted = make_fitted(graph)
    return build_result(
        graph=graph,
        fitted=fitted,
        event_values={SIGN: -1.0},
        interventions=tuple(interventions),
        precedent=list(bands),
        seed=seed,
        n_paths=n_paths,
        asof_utc=ASOF,
        code_commit=None,
    )


# --- render_text ----------------------------------------------------------------------------


def test_text_header_and_rows_show_both_bands_and_both_sample_sizes():
    lines = render_text(hand_result()).splitlines()
    assert lines[0] == (
        "mechanism      template-v1 for central_bank, fitted on 42 precedents as of 2026-11-07; "
        "1 equations underdetermined"
    )
    assert lines[1].split() == [
        "asset", "window", "mechanism", "p10/median/p90",
        "precedent", "p10/median/p90", "agree?",
    ]  # fmt: skip
    tlt = next(line for line in lines if line.startswith("TLT"))
    assert "-1.2% / -0.4% / +0.3%" in tlt and "-0.9% / -0.5% / +0.1%" in tlt
    assert tlt.split()[1] == "5" and tlt.endswith(" n=42/17")


def test_text_agree_rule_yes_no_and_refused():
    lines = {line.split()[0]: line for line in render_text(hand_result()).splitlines()[2:6]}
    assert "yes" in lines["TLT"]
    assert "no (median outside the other's band)" in lines["SPY"]
    assert "refused: n_eff=3.2" in lines["HYG"] and "n/a" in lines["HYG"]
    assert lines["HYG"].endswith(" n=42/4")


def test_text_agree_is_symmetric_one_median_outside_is_a_no():
    # mechanism median inside the precedent band, precedent median outside the mechanism band
    rows = [
        BandRow(asset="TLT", horizon=5, p10=-0.05, median=0.04, p90=0.06,
                n=9, n_eff=7.0, refused=False),
    ]  # fmt: skip
    base = hand_result()
    result = hand_result(precedent=rows, rollout_fan=base.rollout_fan)
    tlt = next(line for line in render_text(result).splitlines() if line.startswith("TLT"))
    assert "no (median outside the other's band)" in tlt


def test_text_lists_edges_the_data_disagrees_with():
    text = render_text(hand_result())
    assert (
        "edges the data disagrees with: rates_expectation -> TLT "
        "(template -1, fitted -0.500, n=42)" in text
    )
    clean = hand_result(fitted=make_fitted(make_graph()).to_json_dict())
    assert "edges the data disagrees with: none" in render_text(clean)


def test_text_unreachable_intervention_line_and_disclaimer_last():
    plain = render_text(hand_result()).splitlines()
    assert plain[-1] == DISCLAIMER
    assert not any("reaches no asset" in line for line in plain)
    result = hand_result(unreachable_interventions=["credit_spread"])
    lines = render_text(result).splitlines()
    assert (
        "intervention on credit_spread reaches no asset; "
        "the fan equals the no-intervention fan." in lines
    )
    assert lines[-1] == DISCLAIMER


# --- render_json ----------------------------------------------------------------------------


def test_json_round_trips_and_carries_interventions_and_manifest():
    result = simulated_result(interventions=[Intervention(RATES, 0.0)])
    text = render_json(result)
    assert MechanismResult.model_validate_json(text) == result
    payload = json.loads(text)
    assert payload["interventions"] == [{"node": RATES, "value": 0.0}]
    assert payload["fitted"]["graph_version"] == "template-v1"
    assert {"seed", "n_paths", "code_commit", "disclaimer"} <= payload.keys()
    assert payload["fitted"]["fitted_asof_utc"].startswith("2026-11-07")


def test_result_rejects_unknown_fields_and_naive_timestamps():
    with pytest.raises(ValueError):
        hand_result(asof_utc=datetime(2026, 11, 7))
    with pytest.raises(ValueError):
        hand_result(extra_field=1)


def test_build_result_matches_simulate_and_fan():
    result = simulated_result(interventions=[Intervention(RISK, 1.0)])
    graph, fitted = make_graph(), make_fitted(make_graph())
    rollout = simulate(
        fitted, graph, {SIGN: -1.0}, (Intervention(RISK, 1.0),), n_paths=4000, seed=0
    )
    table = fan(rollout)
    for asset in fitted.assets:
        for q in (0.1, 0.5, 0.9):
            assert result.rollout_fan[asset][str(q)] == pytest.approx(table.loc[asset, q])
    assert result.unreachable_interventions == []


# --- render_html ----------------------------------------------------------------------------


def test_html_is_self_contained_and_carries_the_disclaimer():
    html = render_html(simulated_result())
    assert "<script src=" not in html and "fetch(" not in html
    assert "<link" not in html
    assert not re.search(r"""(src|href)\s*=\s*["']?https?:""", html)
    assert "{{" not in html
    assert "not a forecast" in html
    title = re.search(r"<title>(.*?)</title>", html, re.S).group(1)
    assert DISCLAIMER in title and "central_bank" in title
    assert html.count(DISCLAIMER) >= 2  # title and footer


def test_html_embeds_the_payload_as_json_and_survives_a_closing_script_tag():
    result = simulated_result()
    html = render_html(result)
    body = re.search(
        r'<script type="application/json" id="payload">(.*?)</script>', html, re.S
    ).group(1)
    assert MechanismResult.model_validate_json(body) == result
    evil = hand_result(code_commit="</script><b>x")
    assert "</script><b>x" not in render_html(evil)


def test_html_page_builds_one_range_slider_per_event_and_intermediate_node():
    html = render_html(simulated_result())
    assert '"range"' in html and "mechanismControls" in html
    payload = json.loads(
        re.search(r'id="payload">(.*?)</script>', html, re.S).group(1)
    )
    nodes = payload["graph"]["nodes"]
    controlled = [n["id"] for n in nodes if n["kind"] in ("event", "intermediate")]
    assert controlled == [SIGN, RATES, RISK]


# --- JavaScript twin ------------------------------------------------------------------------

needs_node = pytest.mark.skipif(NODE is None, reason="node not on PATH")
PRELUDE = "globalThis.window = globalThis;\n"


def page_script(html: str) -> str:
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    return scripts[0]


def run_node(html: str, program: str, stdin: str) -> dict:
    code = PRELUDE + page_script(html) + "\n" + program
    done = subprocess.run(
        [NODE, "-e", code], input=stdin, capture_output=True, text=True, timeout=120, check=False
    )
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


@needs_node
def test_js_prng_matches_the_pinned_python_values():
    html = render_html(simulated_result())
    program = """
    const P = window.mechanismPrng;
    const out = {
      first: [0, 1, 2, 3, 4].map((i) => P.hashUniform(0, i)),
      wraps: [4294967295, 2147483648, 7].map((i) => P.hashUniform(12345, i)),
      normals: Array.from(P.hashNormals(2, 4, 3)),
    };
    console.log(JSON.stringify(out));
    """
    out = run_node(html, program, "")
    np.testing.assert_allclose(
        out["first"],
        [0.038885081536, 0.026300315862, 0.852083498961, 0.685263112304, 0.491849179263],
        rtol=0, atol=1e-12,
    )  # fmt: skip
    np.testing.assert_allclose(
        out["wraps"], [0.351858842303, 0.458710182807, 0.062277476420], rtol=0, atol=1e-12
    )
    from precedent.mechanism.simulate import hash_normals

    np.testing.assert_allclose(out["normals"], hash_normals(2, 4, 3).reshape(-1), rtol=0, atol=1e-12)


@needs_node
def test_js_simulation_matches_python_on_the_same_seed():
    result = simulated_result(n_paths=4000, seed=0)
    html = render_html(result)
    program = """
    const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
    const out = window.mechanismSimulate(payload, payload.event_values, [], payload.n_paths, payload.seed);
    console.log(JSON.stringify(out));
    """
    out = run_node(html, program, render_json(result))
    graph, fitted = make_graph(), make_fitted(make_graph())
    table = fan(simulate(fitted, graph, {SIGN: -1.0}, n_paths=4000, seed=0))
    for asset in fitted.assets:
        for q in (0.1, 0.25, 0.5, 0.75, 0.9):
            assert out[asset][str(q)] == pytest.approx(table.loc[asset, q], abs=5e-4)
    assert out["SPY"]["0.5"] == pytest.approx(table.loc["SPY", 0.5], abs=5e-4)


@needs_node
def test_js_do_intervention_and_controls_follow_python():
    result = simulated_result(interventions=[Intervention(RATES, 0.0)], n_paths=3000, seed=5)
    html = render_html(result)
    program = """
    const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
    const forced = [{node: "rates_expectation", value: 0.0}];
    const out = {
      sim: window.mechanismSimulate(payload, payload.event_values, forced, 3000, 5),
      controls: window.mechanismControls(payload),
      bad: (() => { try { window.mechanismSimulate(payload, {}, [], 10, 0); return "no error"; }
                    catch (e) { return String(e.message); } })(),
    };
    console.log(JSON.stringify(out));
    """
    out = run_node(html, program, render_json(result))
    table = fan(
        simulate(
            make_fitted(make_graph()), make_graph(), {SIGN: -1.0},
            (Intervention(RATES, 0.0),), n_paths=3000, seed=5,
        )
    )  # fmt: skip
    assert out["sim"]["TLT"]["0.5"] == pytest.approx(table.loc["TLT", 0.5], abs=5e-4)
    assert [c["node"] for c in out["controls"]] == [SIGN, RATES, RISK]
    by_node = {c["node"]: c for c in out["controls"]}
    assert (by_node[SIGN]["min"], by_node[SIGN]["max"], by_node[SIGN]["step"]) == (-1, 1, 1)
    assert (by_node[RATES]["min"], by_node[RATES]["max"], by_node[RATES]["step"]) == (-3, 3, 0.1)
    assert by_node[SIGN]["kind"] == "event" and by_node[RATES]["kind"] == "intermediate"
    assert SIGN in out["bad"]


@needs_node
def test_js_initial_state_keeps_every_intervention_and_matches_the_embedded_fan():
    # an asset and an intermediate intervention, the second beyond the default slider range
    forced = (Intervention("TLT", 0.01), Intervention(RATES, 5.0))
    result = simulated_result(interventions=forced, n_paths=3000, seed=2)
    program = """
    const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
    const state = window.mechanismInitialState(payload);
    const list = Object.keys(state.forced).map((node) => ({node, value: state.forced[node]}));
    console.log(JSON.stringify({
      state,
      controls: window.mechanismControls(payload),
      sim: window.mechanismSimulate(payload, state.events, list, payload.n_paths, payload.seed),
    }));
    """
    out = run_node(render_html(result), program, render_json(result))
    assert out["state"]["forced"] == {"TLT": 0.01, RATES: 5.0}
    assert out["state"]["events"] == {SIGN: -1.0}
    rates = next(c for c in out["controls"] if c["node"] == RATES)
    assert rates["max"] >= 5.0 and rates["value"] == 5.0 and rates["free"] is False
    assert all(c["node"] != "TLT" for c in out["controls"])  # an asset has no slider
    for asset, row in result.rollout_fan.items():
        for q, expected in row.items():
            assert out["sim"][asset][q] == pytest.approx(expected, abs=5e-4)


@needs_node
def test_js_falls_back_to_the_diagonal_like_python():
    graph = make_graph()
    bad_cov = np.array([[0.04, 0.2, 0.0], [0.2, 0.09, 0.0], [0.0, 0.0, 0.0625]])
    fitted = make_fitted(graph, cov=bad_cov)
    result = build_result(
        graph=graph, fitted=fitted, event_values={SIGN: -1.0}, interventions=(),
        precedent=[], seed=1, n_paths=3000, asof_utc=ASOF, code_commit=None,
    )  # fmt: skip
    assert result.diagonal_fallback is True
    program = """
    const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
    console.log(JSON.stringify(window.mechanismSimulate(
      payload, payload.event_values, [], payload.n_paths, payload.seed)));
    """
    out = run_node(render_html(result), program, render_json(result))
    table = fan(simulate(fitted, graph, {SIGN: -1.0}, n_paths=3000, seed=1))
    assert out["SPY"]["0.9"] == pytest.approx(table.loc["SPY", 0.9], abs=5e-4)


# --- the command ----------------------------------------------------------------------------


def template_fitted(graph: CausalGraph) -> FittedSEM:
    assets = [n.id for n in graph.nodes if n.kind.value == "asset"]
    coefficients = {
        node.id: {e.source: 0.4 * e.sign for e in graph.edges if e.target == node.id}
        for node in graph.nodes
        if node.kind.value != "event"
    }
    return FittedSEM(
        coefficients=coefficients,
        intercepts={node: 0.0 for node in coefficients},
        residual_cov=np.eye(len(assets)) * 4e-4,
        assets=assets,
        n_obs=33,
        underdetermined=frozenset(),
        sign_disagreements=(),
        graph_id=graph.graph_id,
        graph_version=graph.graph_version,
        fitted_asof_utc=pd.Timestamp("2024-06-03", tz="UTC"),
        horizon=5,
    )


@pytest.fixture
def stubbed_data_step(monkeypatch):
    """Replace the data step (loader and fit) so the renderers and flag handling run end to end."""
    seen = {}

    def fake_inputs(directory):
        seen["data_dir"] = directory

    def fake(key, graph, asof_utc, horizon, inputs, params):
        seen.update(key=key, asof=asof_utc, horizon=horizon)
        bands = [
            BandRow(asset="TLT", horizon=horizon, p10=-0.01, median=-0.002, p90=0.006,
                    n=12, n_eff=8.0, refused=False),
        ]  # fmt: skip
        return template_fitted(graph), bands

    monkeypatch.setattr(cli, "_load_inputs", fake_inputs)
    monkeypatch.setattr(cli, "_fit_and_band", fake)
    return seen


def test_untyped_text_query_exits_2_and_prints_the_key_flags(capsys, monkeypatch):
    monkeypatch.delenv("PRECEDENT_TYPER", raising=False)
    assert run(["the weather in Lisbon was pleasant"]) == 2
    err = capsys.readouterr().err
    assert "--event-type" in err and "--direction" in err and "untyped" in err


def test_typed_text_query_uses_the_typers_key(capsys, monkeypatch, stubbed_data_step):
    class FakeTyper:
        def type_text(self, text):
            return TypedText(
                status="typed", key=KEY, reason="rules", evidence_span=text, typer_version="t"
            )

    monkeypatch.setattr(cli, "typer_from_env", lambda: FakeTyper())
    assert run(["rate hike"]) == 0
    assert stubbed_data_step["key"] == KEY
    assert "mechanism      template-v1 for central_bank" in capsys.readouterr().out


def test_refused_text_query_exits_2(capsys, monkeypatch):
    class Refuser:
        def type_text(self, text):
            return TypedText(
                status="refused_invalid", key=None, reason="schema: bad", evidence_span=None,
                typer_version="t",
            )  # fmt: skip

    monkeypatch.setattr(cli, "typer_from_env", lambda: Refuser())
    assert run(["x"]) == 2
    assert "schema: bad" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("flags", "bad"),
    [
        (["--event-type", "bogus", "--actor-kind", "central_bank"], "bogus"),
        (["--event-type", "central_bank", "--actor-kind", "wizard"], "wizard"),
        (
            ["--event-type", "central_bank", "--actor-kind", "central_bank",
             "--impacted-kinds", "index,planet"],
            "planet",
        ),
    ],
)  # fmt: skip
def test_bad_key_flags_exit_2_naming_the_value(capsys, flags, bad):
    assert run(flags) == 2
    err = capsys.readouterr().err
    assert bad in err and "Traceback" not in err


def test_do_on_an_event_node_exits_2_and_points_at_the_key_flag(capsys):
    assert run([*KEY_FLAGS, "--do", f"{SIGN}=0"]) == 2
    err = capsys.readouterr().err
    assert SIGN in err and "event variable" in err and "--direction" in err
    with pytest.raises(ValueError, match="event variable"):
        parse_do_flags([f"{SIGN}=1"], make_graph())


def test_result_rejects_an_intervention_on_an_event_node():
    bad = [InterventionRow(node=SIGN, value=0.0)]
    with pytest.raises(ValueError, match="event nodes"):
        hand_result(interventions=bad)
    assert hand_result(interventions=[InterventionRow(node=RATES, value=0.0)])


def test_do_value_must_be_finite(capsys):
    graph = make_graph()
    for raw in ("nan", "inf", "-inf"):
        with pytest.raises(ValueError, match="finite"):
            parse_do_flags([f"{RATES}={raw}"], graph)
    assert run([*KEY_FLAGS, "--do", "rates_expectation=nan"]) == 2
    assert "rates_expectation=nan" in capsys.readouterr().err


def test_zero_n_paths_is_rejected_not_defaulted(capsys, stubbed_data_step):
    assert run([*KEY_FLAGS, "--n-paths", "0"]) == 2
    assert "n_paths" in capsys.readouterr().err


def test_no_arguments_and_text_with_flags_are_usage_errors(capsys):
    assert run([]) == 2
    assert capsys.readouterr().err
    assert run(["some text", "--event-type", "central_bank"]) == 2
    assert capsys.readouterr().err


def test_unknown_do_node_exits_2_and_names_it(capsys):
    assert run([*KEY_FLAGS, "--do", "nonsense_node=1"]) == 2
    assert "nonsense_node" in capsys.readouterr().err


def test_bad_do_value_exits_2(capsys):
    assert run([*KEY_FLAGS, "--do", "rates_expectation=high"]) == 2
    assert "rates_expectation=high" in capsys.readouterr().err
    assert run([*KEY_FLAGS, "--do", "rates_expectation"]) == 2


def test_event_type_without_a_template_exits_2(capsys):
    flags = ["--event-type", "mna", "--actor-kind", "company", "--severity-band", "low",
             "--direction", "1"]  # fmt: skip
    assert run(flags) == 2
    assert "no template graph for event type mna" in capsys.readouterr().err


def test_key_path_without_the_engine_says_not_built_yet(capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "precedent.engine.retrieval", None)
    monkeypatch.setattr(cli, "_load_inputs", lambda directory: None)
    assert run(KEY_FLAGS) == 2
    err = capsys.readouterr().err
    assert err.strip() == (
        "not built yet: mechanism needs precedent.engine.retrieval from the Precedent track"
    )


def test_parse_do_flags_checks_nodes_and_floats():
    graph = make_graph()
    assert parse_do_flags([f"{RATES}=0", f"{RISK}=-1.5"], graph) == (
        Intervention(RATES, 0.0),
        Intervention(RISK, -1.5),
    )
    with pytest.raises(ValueError, match="nope"):
        parse_do_flags(["nope=1"], graph)
    with pytest.raises(ValueError, match="NODE=VALUE"):
        parse_do_flags(["missing_equals"], graph)


def test_run_prints_text_by_default_and_applies_do(capsys, stubbed_data_step):
    assert run([*KEY_FLAGS, "--do", "rates_expectation=0", "--horizon", "5"]) == 0
    out = capsys.readouterr().out
    assert "mechanism p10/median/p90" in out and "precedent p10/median/p90" in out
    assert out.rstrip().splitlines()[-1] == DISCLAIMER
    assert stubbed_data_step["horizon"] == 5
    assert stubbed_data_step["asof"] == datetime(2024, 6, 3, tzinfo=UTC)


def test_run_json_carries_the_intervention(capsys, stubbed_data_step):
    argv = [*KEY_FLAGS, "--do", "rates_expectation=0", "--json", "--seed", "3", "--n-paths", "500"]
    assert run(argv) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["interventions"] == [{"node": "rates_expectation", "value": 0.0}]
    assert payload["seed"] == 3 and payload["n_paths"] == 500
    assert payload["fitted"]["graph_version"] == "template-v1"
    assert payload["event_values"] == {SIGN: -1.0}


def test_run_writes_html_atomically_and_still_prints_text(tmp_path, capsys, stubbed_data_step):
    out = tmp_path / "cockpit.html"
    assert run([*KEY_FLAGS, "--html", str(out), "--n-paths", "200"]) == 0
    html = out.read_text(encoding="utf-8")
    assert "<script src=" not in html and "fetch(" not in html and "not a forecast" in html
    assert "mechanism      template-v1" in capsys.readouterr().out
    assert [p.name for p in tmp_path.iterdir()] == ["cockpit.html"]  # no temp file left behind


def test_data_dir_defaults_to_the_config_default(monkeypatch, tmp_path, stubbed_data_step):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    assert run([*KEY_FLAGS, "--n-paths", "100"]) == 0
    assert stubbed_data_step["data_dir"] == Path(tmp_path)


def test_template_graphs_have_event_nodes_the_cli_can_value():
    templates = load_templates(references_dir() / "mechanism" / "templates", cli._vocabulary())
    assert all(g.event_node_ids for g in templates.values())
