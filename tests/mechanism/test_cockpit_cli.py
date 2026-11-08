"""Tests for the by-key path of `precedent mechanism` on the Precedent engine (Task 7b).

The plan's `populated_stores` fixture does not exist in this tree, and its events would not match
the plan's key (digest section 7). The synthetic world of `mechanism_world.py` replaces it: real
events table, real asset library, real time guard. The one seam is `_load_inputs`, the loader that
reads the library bundle from disk (`precedent.libraries.bundle`, Precedent Task 10, not merged
here); it is monkeypatched to hand back the world's inputs, and everything after it is real. The
loader body is exercised separately against a fake bundle module and real price parquet files.
"""

from __future__ import annotations

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
from precedent.mechanism.graph_schema import NodeKind, load_params
from precedent.mechanism.templates import load_templates, template_for
from precedent.spine.event_key import EventKey

HORIZON = 5
NODE = shutil.which("node")
needs_node = pytest.mark.skipif(NODE is None, reason="node not on PATH")
REFUSED_LINE = re.compile(
    r"^refused: the fit has n_eff=(\d+\.\d) below min_neff=8\.0; no mechanism is shown$"
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
    assert set(world.assets) <= set(bands)
    assert all(not bands[a]["refused"] for a in world.assets)
    # a graph asset the library holds no rows for (the world has four assets) is refused, n=0
    absent = [band for asset, band in bands.items() if asset not in world.assets]
    assert absent and all(b["refused"] and b["n"] == 0 for b in absent)


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
    fitted, bands = cli._fit_and_band(key, graph, asof, HORIZON, inputs_of(world), params)
    events = world.events
    cutoffs = world.session_cutoffs_utc
    finished = events["effective_position"].map(lambda p: cutoffs.iloc[p + 1 + HORIZON] <= asof)
    seen = events[events["available_utc"] <= asof]
    top = seen.sort_values("effective_position", ascending=False).head(params.top_k)
    assert fitted.n_obs == int(finished[top.index].sum())
    signed = events["key"].map(lambda k: EventKey.from_string(k).direction_sign == 1)
    same_sign = events[signed & (events["available_utc"] <= asof)]
    expected = int(finished[same_sign.index].sum())
    assert 8 <= expected < fitted.n_obs
    asset_ids = [n.id for n in graph.nodes if n.kind is NodeKind.ASSET]
    assert [b.asset for b in bands] == asset_ids
    assert all(b.refused and b.n == 0 for b in bands if b.asset not in world.assets)
    for band in (b for b in bands if b.asset in world.assets):
        assert band.horizon == HORIZON and not band.refused
        assert band.n == expected and band.n_eff == pytest.approx(float(expected))
        assert band.p10 < band.median < band.p90
        assert band.median == pytest.approx(world.planted_car(band.asset), abs=0.006)


def test_a_zero_direction_key_gives_the_band_the_fits_precedents(world):
    graph, params = graph_of(world), load_engine_params()
    asof = pd.Timestamp(asof_at(world, 700), tz="UTC")
    fitted, bands = cli._fit_and_band(world.key(0), graph, asof, HORIZON, inputs_of(world), params)
    assert all(b.n == fitted.n_obs for b in bands if b.asset in world.assets)


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


def test_an_exact_key_with_no_precedent_refuses_every_band_but_not_the_command(world):
    """No exact-key precedent is a refused band row, not a refusal of the mechanism."""
    graph, params = graph_of(world), load_engine_params()
    asof = pd.Timestamp(asof_at(world, 700), tz="UTC")
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
    fitted, bands = cli._fit_and_band(world.key(1), graph, asof, HORIZON, inputs, params)
    assert fitted.n_obs >= 8
    assert all(b.refused and b.n == 0 and b.n_eff == 0.0 for b in bands)
    assert all(b.p10 is None and b.median is None and b.p90 is None for b in bands)


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


# --- one asset panel -------------------------------------------------------------------------


def test_mechanism_asset_panel_cannot_drift_from_the_engine_panel():
    assert load_params().asset_panel == list(load_engine_params().asset_panel)


def test_the_template_vocabulary_comes_from_the_engine_asset_panel():
    assets = {n for n in cli._vocabulary() if n in set(load_engine_params().asset_panel)}
    assert assets == set(load_engine_params().asset_panel)
