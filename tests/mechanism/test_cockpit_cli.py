"""Tests for the by-key path of `precedent mechanism` on the Precedent engine (Task 7b).

The plan's `populated_stores` fixture does not exist in this tree, and its events would not match
the plan's key (digest section 7). The synthetic world of `mechanism_world.py` replaces it: real
events table, real asset library, real time guard. The one seam is `_load_inputs`, the loader that
reads the library bundle from disk (`precedent.libraries.bundle`); it is monkeypatched to hand back the world's inputs, and everything after it is real. The
loader body is exercised separately against a fake bundle module and real price parquet files.
"""

from __future__ import annotations

import dataclasses
import json
import re
import shutil
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mechanism_world import World, make_world

from precedent.config import references_dir
from precedent.engine.params import load_engine_params
from precedent.mechanism import cli
from precedent.mechanism.cli import MechanismInputs, MechanismRefused, run
from precedent.mechanism.cockpit import DISCLAIMER
from precedent.mechanism.graph_schema import (
    CausalGraph,
    Edge,
    Node,
    NodeKind,
    derive_kind,
    load_params,
)
from precedent.mechanism.templates import load_templates, template_for
from precedent.spine.event_key import EventKey

HORIZON = 5
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not on PATH")
REFUSED_LINE = re.compile(
    r"^refused: the fit has n_eff=(\d+\.\d) below min_neff=8\.0; no mechanism is shown$"
)
BAND_REFUSED_LINE = re.compile(
    r"^refused: Precedent's exact-key band has n_eff=(\d+\.\d) below min_neff=8\.0; "
    r"no mechanism is shown$"
)


@pytest.fixture(scope="module")
def world() -> World:
    return make_world()


def inputs_of(world: World) -> MechanismInputs:
    return MechanismInputs(
        events=world.events,
        rows=world.rows,
        session_cutoffs_utc=world.session_cutoffs_utc,
        returns=world.returns,
    )


@pytest.fixture
def loaded(monkeypatch, world):
    """Replace only the bundle loader; the fit, the band and the renderers are real."""
    seen: dict[str, Path] = {}

    def fake_loader(directory: Path) -> MechanismInputs:
        seen["data_dir"] = directory
        return inputs_of(world)

    monkeypatch.setattr(cli, "_load_inputs", fake_loader)
    return seen


def asof_at(world: World, position: int) -> str:
    return world.sessions[position].date().isoformat()


def key_flags(world: World, position: int = 700, direction: int = 1) -> list[str]:
    """The world's key as flags (impacted kind `index`, US, mid, scheduled) and an --asof."""
    return [
        "--event-type", "central_bank", "--actor-kind", "central_bank",
        "--impacted-kinds", "index", "--country-group", "US", "--severity-band", "mid",
        "--scheduled", "--direction", str(direction), "--asof", asof_at(world, position),
        "--horizon", str(HORIZON),
    ]  # fmt: skip


def graph_of(world: World):
    templates = load_templates(references_dir() / "mechanism" / "templates", cli._vocabulary())
    return template_for(world.key(1), templates)


# --- the by-key path ------------------------------------------------------------------------


def test_key_query_prints_both_bands_and_the_disclaimer(capsys, world, loaded, tmp_path):
    assert run([*key_flags(world), "--data-dir", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "mechanism p10/median/p90" in out and "precedent p10/median/p90" in out
    assert out.rstrip().splitlines()[-1] == DISCLAIMER
    assert re.search(r"mechanism\s+template-v1 for central_bank, fitted on \d+ precedents", out)
    for asset in ("SPY", "TLT", "HYG", "UUP"):
        assert re.search(rf"^{asset}\s+{HORIZON}\s", out, re.M)
    assert loaded["data_dir"] == tmp_path


def test_do_flag_is_applied_and_named(capsys, world, loaded):
    assert run([*key_flags(world), "--do", "rates_expectation=0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["interventions"] == [{"node": "rates_expectation", "value": 0.0}]
    bands = {band["asset"]: band for band in payload["precedent"]}
    assert set(bands) == set(world.assets)  # only assets with data are banded and shown
    assert all(not bands[a]["refused"] for a in world.assets)
    # the market's band is its own cumulative return; every other band is abnormal
    assert bands["SPY"]["basis"] == "raw_return"
    assert all(
        bands[a]["basis"] == "abnormal_return" for a in world.assets if a != "SPY"
    )
    assert payload["refused_assets"] == []
    assert payload["min_neff"] == load_engine_params().min_neff
    # a graph asset the library holds no rows for (the world has four assets) is suppressed
    # and named in no_data_assets, in graph order
    asset_ids = [n.id for n in graph_of(world).nodes if n.kind is NodeKind.ASSET]
    assert payload["no_data_assets"] == [a for a in asset_ids if a not in world.assets]


def test_html_is_self_contained(tmp_path, world, loaded):
    out = tmp_path / "cockpit.html"
    assert run([*key_flags(world), "--html", str(out), "--n-paths", "200"]) == 0
    html = out.read_text(encoding="utf-8")
    assert "<script src=" not in html and "fetch(" not in html and "not a forecast" in html
    assert [p.name for p in tmp_path.iterdir()] == ["cockpit.html"]


@needs_node
def test_js_simulation_matches_python_on_the_same_seed(tmp_path, capsys, world, loaded):
    page = tmp_path / "page.html"
    flags = [*key_flags(world), "--n-paths", "4000", "--seed", "0"]
    assert run([*flags, "--html", str(page), "--json"]) == 0
    python_payload = json.loads(capsys.readouterr().out)
    html = page.read_text(encoding="utf-8")
    scripts = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(scripts) == 1
    program = (
        "globalThis.window = globalThis;\n"
        + scripts[0]
        + """
    const payload = JSON.parse(require("fs").readFileSync(0, "utf8"));
    const out = window.mechanismSimulate(
      payload, payload.event_values, [], payload.n_paths, payload.seed);
    console.log(JSON.stringify(out));
    """
    )
    done = subprocess.run(
        [NODE, "-e", program],
        input=json.dumps(python_payload),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    js_fan = json.loads(done.stdout)
    for asset, quantiles in python_payload["rollout_fan"].items():
        for q, value in quantiles.items():
            assert js_fan[asset][q] == pytest.approx(value, abs=5e-4)
    assert js_fan["SPY"]["0.5"] == pytest.approx(
        python_payload["rollout_fan"]["SPY"]["0.5"], abs=5e-4
    )


def test_band_uses_the_exact_key_and_the_fit_uses_direction_zero(world):
    """The fit sees both signs (its n_obs is larger); the band sees only the queried sign."""
    key, graph, params = world.key(1), graph_of(world), load_engine_params()
    asof = pd.Timestamp(asof_at(world, 700), tz="UTC")
    fitted, bands, no_data, refused = cli._fit_and_band(
        key, graph, asof, HORIZON, inputs_of(world), params
    )
    events = world.events
    cutoffs = world.session_cutoffs_utc
    finished = events["effective_position"].map(lambda p: cutoffs.iloc[p + 1 + HORIZON] <= asof)
    seen = events[events["available_utc"] <= asof]
    top = seen.sort_values("effective_position", ascending=False).head(params.top_k)
    # P-42: the guard comes before the cut, so the fit sees top_k finished events (was 29,
    # the finished ones among the 30 latest unguarded).
    assert int(finished[top.index].sum()) == 29
    assert fitted.n_obs == params.top_k == 30
    signed = events["key"].map(lambda k: EventKey.from_string(k).direction_sign == 1)
    same_sign = events[signed & (events["available_utc"] <= asof)]
    expected = int(finished[same_sign.index].sum())
    assert 8 <= expected < fitted.n_obs
    asset_ids = [n.id for n in graph.nodes if n.kind is NodeKind.ASSET]
    # assets without library rows are suppressed, in graph order; the band covers the rest
    assert no_data == [a for a in asset_ids if a not in world.assets]
    assert [b.asset for b in bands] == [a for a in asset_ids if a in world.assets]
    for band in bands:
        assert band.horizon == HORIZON and not band.refused
        assert band.n == expected and band.n_eff == pytest.approx(float(expected))
        assert band.p10 < band.median < band.p90
        assert band.median == pytest.approx(world.planted_car(band.asset), abs=0.006)
        expected_basis = "raw_return" if band.asset == params.market else "abnormal_return"
        assert band.basis == expected_basis


def test_a_zero_direction_key_gives_the_band_the_fits_precedents(world):
    graph, params = graph_of(world), load_engine_params()
    asof = pd.Timestamp(asof_at(world, 700), tz="UTC")
    fitted, bands, no_data, refused = cli._fit_and_band(
        world.key(0), graph, asof, HORIZON, inputs_of(world), params
    )
    asset_ids = [n.id for n in graph.nodes if n.kind is NodeKind.ASSET]
    assert no_data == [a for a in asset_ids if a not in world.assets]
    assert refused == []
    assert {b.asset for b in bands} == set(world.assets)
    assert all(b.n == fitted.n_obs for b in bands)


def test_asof_is_midnight_utc_of_the_day_so_that_days_close_is_not_seen(world):
    """A response completing at the close of the --asof day is not yet known at its midnight."""
    graph, params = graph_of(world), load_engine_params()
    position = world.event_positions[20]
    completes = position + 1 + HORIZON  # the session whose close finishes this event's window

    def n_obs(session_position: int) -> int:
        asof = pd.Timestamp(asof_at(world, session_position), tz="UTC")
        assert asof.hour == 0  # midnight UTC of the day, as `--asof` is parsed
        return cli._fit_and_band(world.key(1), graph, asof, HORIZON, inputs_of(world), params)[
            0
        ].n_obs

    assert n_obs(completes) == 20  # events 0..19; event 20 completes at this day's close
    assert n_obs(completes + 1) == 21


# --- refusal and usage errors ---------------------------------------------------------------


def test_command_refuses_when_the_fit_has_too_few_effective_precedents(capsys, world, loaded):
    assert run(key_flags(world, position=345)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    line = captured.err.strip()
    match = REFUSED_LINE.match(line)
    assert match, line
    assert float(match.group(1)) < 8.0


def test_command_refuses_with_zero_neff_before_any_precedent_exists(capsys, world, loaded):
    assert run(key_flags(world, position=100)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "n_eff=0.0 below min_neff=8.0" in captured.err


def test_refusal_is_a_typed_error_with_both_numbers():
    error = MechanismRefused(n_eff=4.0, min_neff=8.0)
    assert (error.n_eff, error.min_neff) == (4.0, 8.0)
    assert str(error) == (
        "refused: the fit has n_eff=4.0 below min_neff=8.0; no mechanism is shown"
    )


def test_a_horizon_the_library_does_not_hold_is_a_usage_error(capsys, world, loaded):
    flags = [*key_flags(world), "--horizon", "7"]
    assert run(flags) == 2
    err = capsys.readouterr().err
    assert "horizon 7" in err and "1, 5, 20" in err and "n_eff" not in err


def test_an_exact_key_with_no_precedent_refuses_the_command(capsys, world, monkeypatch):
    """Requirement change, ruling I-2 (recorded in the addendum of
    docs/verification/mechanism-2026-11-08.md and docs/STATUS.md): with no exact-key
    precedent the command refuses — the fan is never
    printed beside a refused band. The fit itself still sees the -1 precedents, which is why
    the refusal names the band (n_eff=0.0), not the fit."""
    # drop every +1 event: the fit (direction 0) keeps the -1 events, the exact +1 key has none
    events = world.events[
        world.events["key"].map(lambda k: EventKey.from_string(k).direction_sign == -1)
    ]
    inputs = MechanismInputs(
        events=events.reset_index(drop=True),
        rows=world.rows,
        session_cutoffs_utc=world.session_cutoffs_utc,
        returns=world.returns,
    )
    params = load_engine_params()
    with pytest.raises(MechanismRefused) as excinfo:
        cli._fit_and_band(
            world.key(1), graph_of(world),
            pd.Timestamp(asof_at(world, 700), tz="UTC"), HORIZON, inputs, params,
        )
    assert excinfo.value.band is True  # the fit passed; the band refused
    assert (excinfo.value.n_eff, excinfo.value.min_neff) == (0.0, params.min_neff)

    monkeypatch.setattr(cli, "_load_inputs", lambda directory: inputs)
    assert run(key_flags(world)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.strip() == str(excinfo.value)


def test_the_command_refuses_when_every_band_is_thin(capsys, world, loaded):
    """At this as-of the fit passes (it is direction-free) but every observed asset's
    exact-key band is below min_neff, so no row survives and the command refuses (ruling
    I-2) — including the assets the library has no rows for, which cannot band either."""
    assert run(key_flags(world, position=430)) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    line = captured.err.strip()
    match = BAND_REFUSED_LINE.match(line)
    assert match, line
    assert 0.0 < float(match.group(1)) < load_engine_params().min_neff


def test_mixed_panel_shows_good_rows_and_the_reason_for_thin_bands(capsys, world, monkeypatch):
    """One thin asset (four rows: observations exist, n_eff below min_neff) among good ones:
    the command succeeds; the good rows show fans, the thin row prints the refusal reason
    with no fan and no fabricated "+0.0%"."""
    # keep HYG rows only for four events that are inside the query's panels at this as-of
    asof = pd.Timestamp(asof_at(world, 700), tz="UTC")
    events = world.events
    cutoffs = world.session_cutoffs_utc
    finished = events["effective_position"].map(lambda p: cutoffs.iloc[p + 1 + HORIZON] <= asof)
    signed = events["key"].map(lambda k: EventKey.from_string(k).direction_sign == 1)
    in_panel = events[finished & signed & (events["available_utc"] <= asof)]
    keep = set(in_panel.sort_values("effective_position")["event_id"].tail(4))
    hyg = (world.rows["outcome_key"] == "HYG") & (world.rows["horizon"] == HORIZON)
    rows = world.rows.loc[~hyg | world.rows["event_id"].isin(keep)]
    inputs = MechanismInputs(
        events=world.events,
        rows=rows.reset_index(drop=True),
        session_cutoffs_utc=world.session_cutoffs_utc,
        returns=world.returns,
    )
    monkeypatch.setattr(cli, "_load_inputs", lambda directory: inputs)

    assert run([*key_flags(world), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["refused_assets"] == ["HYG"]
    assert "HYG" not in payload["rollout_fan"]
    bands = {b["asset"]: b for b in payload["precedent"]}
    assert bands["HYG"]["refused"] is True and bands["HYG"]["n"] == 4
    assert bands["HYG"]["n_eff"] == pytest.approx(4.0)
    assert all(not bands[a]["refused"] for a in ("SPY", "TLT", "UUP"))
    assert "HYG" not in payload["no_data_assets"]  # HYG has observations: it is refused, not no-data

    assert run(key_flags(world)) == 0
    out = capsys.readouterr().out
    lines = {line.split()[0]: line for line in out.splitlines()}
    assert lines["HYG"] == "HYG     5        band refused: n_eff=4.0 < min_neff=8.0"
    assert "%" not in lines["HYG"]  # no fan, no fabricated "+0.0%"
    for asset in ("SPY", "TLT", "UUP"):
        assert "/" in lines[asset]  # a real fan triple
    # the template's other assets have no rows at all: "no data" reason rows
    assert any(line.endswith(" no data") for line in out.splitlines())


def test_every_asset_without_rows_refuses_with_its_own_message(world):
    """NoDataAssets is distinct from MechanismRefused: the library holds events and simply
    no rows for any of the graph's assets."""
    from precedent.mechanism.graph_schema import CausalGraph, Edge, Node, derive_kind

    node_ids = ["event:central_bank:direction_sign", "GLD"]
    graph = CausalGraph(
        graph_id="g-no-data",
        graph_version="template-v1",
        event_key=world.key(1),
        nodes=[Node(id=i, kind=derive_kind(i)) for i in node_ids],
        edges=[Edge(
            source=node_ids[0], target="GLD", sign=-1, lag_band="same_session",
            source_doc_id=None, evidence_span=None,
        )],
    )
    with pytest.raises(cli.NoDataAssets) as excinfo:
        cli._fit_and_band(
            world.key(1), graph, pd.Timestamp(asof_at(world, 700), tz="UTC"),
            HORIZON, inputs_of(world), load_engine_params(),
        )
    assert excinfo.value.assets == ["GLD"]
    assert excinfo.value.horizon == HORIZON
    assert "no data:" in str(excinfo.value) and "n_eff" not in str(excinfo.value)


def test_no_data_assets_exit_2_with_their_own_message(capsys, monkeypatch, tmp_path, world):
    def fake(key, graph, asof_utc, horizon, inputs, params):
        raise cli.NoDataAssets(["XLE", "GLD"], horizon)

    monkeypatch.setattr(cli, "_load_inputs", lambda directory: inputs_of(world))
    monkeypatch.setattr(cli, "_fit_and_band", fake)
    assert run([*key_flags(world), "--data-dir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert err.startswith("no data:") and "XLE, GLD" in err and "n_eff" not in err


# --- the loader -----------------------------------------------------------------------------


def test_loader_without_the_bundle_module_exits_2_naming_it(capsys, monkeypatch, tmp_path, world):
    monkeypatch.setitem(sys.modules, "precedent.libraries.bundle", None)
    assert run([*key_flags(world), "--data-dir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert err.strip() == (
        "not built yet: mechanism needs precedent.libraries.bundle from the Precedent track"
    )


def write_prices(directory: Path, symbol: str, opens: np.ndarray, sessions: pd.DatetimeIndex):
    directory.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({"date": sessions, "open": opens, "close": opens})
    frame.to_parquet(directory / f"{symbol}.parquet")


def fake_bundle_module(monkeypatch, world: World, calls: list[Path]) -> None:
    def load_bundle(library_dir: Path):
        calls.append(library_dir)
        return types.SimpleNamespace(
            events=world.events,
            rows=world.rows,
            sessions=world.sessions,
            session_cutoffs_utc=world.session_cutoffs_utc,
        )

    module = types.ModuleType("precedent.libraries.bundle")
    module.load_bundle = load_bundle
    monkeypatch.setitem(sys.modules, "precedent.libraries.bundle", module)


def test_loader_reads_the_library_and_open_to_open_returns_on_the_session_axis(
    monkeypatch, tmp_path, world
):
    calls: list[Path] = []
    fake_bundle_module(monkeypatch, world, calls)
    spy = world.returns["SPY"].to_numpy()
    opens = 100.0 * np.cumprod(1.0 + np.where(np.arange(len(spy)) == 0, 0.0, spy))
    write_prices(tmp_path / "prices", "SPY", opens, world.sessions)
    inputs = cli._load_inputs(tmp_path)
    assert calls == [tmp_path / "library"]
    assert inputs.events is world.events and inputs.rows is world.rows
    assert inputs.session_cutoffs_utc.index.equals(world.sessions)
    assert inputs.returns.index.equals(world.sessions)
    assert "SPY" in inputs.returns.columns
    assert np.isnan(inputs.returns["SPY"].iloc[0])
    np.testing.assert_allclose(inputs.returns["SPY"].iloc[1:], spy[1:], atol=1e-12)


def test_missing_prices_exit_2_and_a_missing_library_is_reported(
    capsys, monkeypatch, tmp_path, world
):
    fake_bundle_module(monkeypatch, world, [])
    assert run([*key_flags(world), "--data-dir", str(tmp_path)]) == 2
    assert "no prices" in capsys.readouterr().err

    def missing(library_dir: Path):
        raise FileNotFoundError(f"no library at {library_dir}")

    module = types.ModuleType("precedent.libraries.bundle")
    module.load_bundle = missing
    monkeypatch.setitem(sys.modules, "precedent.libraries.bundle", module)
    assert run([*key_flags(world), "--data-dir", str(tmp_path)]) == 2
    assert f"no library at {tmp_path / 'library'}" in capsys.readouterr().err


def test_a_corrupt_library_exits_2_without_a_traceback(capsys, monkeypatch, tmp_path, world):
    """Ruling T7B-8: a bundle whose manifest fails verification is a clean exit 2 (V-10)."""
    from precedent.libraries.manifest import LibraryIntegrityError

    def corrupt(library_dir: Path):
        raise LibraryIntegrityError("MANIFEST.json", "sha256 mismatch")

    module = types.ModuleType("precedent.libraries.bundle")
    module.load_bundle = corrupt
    monkeypatch.setitem(sys.modules, "precedent.libraries.bundle", module)
    assert run([*key_flags(world), "--data-dir", str(tmp_path)]) == 2
    err = capsys.readouterr().err
    assert "MANIFEST.json" in err and "sha256 mismatch" in err and "Traceback" not in err


# --- one asset panel -------------------------------------------------------------------------


def test_mechanism_asset_panel_cannot_drift_from_the_engine_panel():
    assert load_params().asset_panel == list(load_engine_params().asset_panel)


def test_the_template_vocabulary_comes_from_the_engine_asset_panel():
    assets = {n for n in cli._vocabulary() if n in set(load_engine_params().asset_panel)}
    assert assets == set(load_engine_params().asset_panel)


def test_event_values_follow_the_panel_encoding_for_every_key_field():
    """M-5 pin: the CLI's event values are the panel's one encoding — no private severity or
    scheduled map can drift from it — asserted for every severity level and the scheduled
    flag against a graph whose event nodes name those fields (no shipped template names
    them, so an unpaired map would otherwise never be caught)."""
    from precedent.mechanism.panel import event_values_of

    base = make_world(seed=3).key(1)
    severity_node = "event:central_bank:severity_band"
    scheduled_node = "event:central_bank:is_scheduled"
    ids = [severity_node, scheduled_node, "rates_expectation", "SPY"]
    graph = CausalGraph(
        graph_id="test:encoding",
        graph_version="template-v1",
        event_key=base,
        nodes=[Node(id=node_id, kind=derive_kind(node_id)) for node_id in ids],
        edges=[
            Edge(
                source=source,
                target="rates_expectation",
                sign=1,
                lag_band="same_session",
                source_doc_id=None,
                evidence_span=None,
            )
            for source in (severity_node, scheduled_node)
        ]
        + [
            Edge(
                source="rates_expectation",
                target="SPY",
                sign=1,
                lag_band="same_session",
                source_doc_id=None,
                evidence_span=None,
            )
        ],
    )
    for severity in ("low", "mid", "high"):
        for scheduled in (False, True):
            key = dataclasses.replace(base, severity_band=severity, is_scheduled=scheduled)
            expected = {
                node: float(value)
                for node, value in event_values_of(key, graph.event_node_ids).items()
            }
            assert cli._event_values(key, graph) == expected
