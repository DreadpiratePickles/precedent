"""Renderers for a mechanism result: a text table, JSON, and one self-contained what-if page.

Nothing here fits or samples anything: `build_result` calls the simulator, and the renderers
only lay out what a `MechanismResult` already holds. The page re-runs the simulator in
JavaScript (`cockpit_template.html`), a twin of `simulate.py`.
"""

from __future__ import annotations

import html
import re
from datetime import UTC
from pathlib import Path
from typing import Any, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict

from precedent.mechanism.fit_sem import FittedSEM
from precedent.mechanism.graph_schema import CausalGraph
from precedent.mechanism.simulate import DEFAULT_QUANTILES, Intervention, fan, simulate

DISCLAIMER = "A fitted model of past co-movement under an assumed structure, not a forecast."
TEMPLATE_PATH = Path(__file__).with_name("cockpit_template.html")
_LOW, _MEDIAN, _HIGH = "0.1", "0.5", "0.9"
_NO_AGREEMENT = "n/a"
_DISAGREE = "no (median outside the other's band)"
_ASSET_WIDTH, _WINDOW_WIDTH, _MECHANISM_WIDTH, _PRECEDENT_WIDTH = 8, 9, 30, 27


class BandRow(BaseModel):
    """Precedent's p10/median/p90 for one asset and window, with its sample sizes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: str
    horizon: int
    p10: float | None
    median: float | None
    p90: float | None
    n: int
    n_eff: float
    refused: bool


class InterventionRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node: str
    value: float


class MechanismResult(BaseModel):
    """Everything one mechanism run reports; the three outputs are views of this value.

    `fitted` is `FittedSEM.to_json_dict()` and `rollout_fan` maps asset -> quantile (as the
    string `fan()` columns print, e.g. "0.1") -> simulated value.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    graph: CausalGraph
    fitted: dict[str, Any]
    event_values: dict[str, float]
    interventions: list[InterventionRow]
    rollout_fan: dict[str, dict[str, float]]
    precedent: list[BandRow]
    seed: int
    n_paths: int
    asof_utc: AwareDatetime
    unreachable_interventions: list[str]
    diagonal_fallback: bool
    code_commit: str | None
    disclaimer: Literal[DISCLAIMER] = DISCLAIMER


def build_result(
    *,
    graph: CausalGraph,
    fitted: FittedSEM,
    event_values: dict[str, float],
    interventions: tuple[Intervention, ...],
    precedent: list[BandRow],
    seed: int,
    n_paths: int,
    asof_utc: Any,
    code_commit: str | None,
) -> MechanismResult:
    """Simulate the fitted graph and bundle the fan with its inputs."""
    rollout = simulate(fitted, graph, event_values, interventions, n_paths=n_paths, seed=seed)
    table = fan(rollout, DEFAULT_QUANTILES)
    return MechanismResult(
        graph=graph,
        fitted=fitted.to_json_dict(),
        event_values=dict(event_values),
        interventions=[InterventionRow(node=i.node, value=i.value) for i in interventions],
        rollout_fan={
            asset: {str(q): float(table.loc[asset, q]) for q in DEFAULT_QUANTILES}
            for asset in rollout.assets
        },
        precedent=precedent,
        seed=seed,
        n_paths=n_paths,
        asof_utc=asof_utc,
        unreachable_interventions=list(rollout.unreachable_interventions),
        diagonal_fallback=rollout.diagonal_fallback,
        code_commit=code_commit,
    )


def _pct(value: float) -> str:
    return f"{value:+.1%}"


def _triple(low: float, median: float, high: float) -> str:
    return f"{_pct(low)} / {_pct(median)} / {_pct(high)}"


def _inside(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def _agreement(mechanism: dict[str, float], band: BandRow | None) -> str:
    if band is None or band.refused or None in (band.p10, band.median, band.p90):
        return _NO_AGREEMENT
    assert band.p10 is not None and band.median is not None and band.p90 is not None
    mechanism_median, precedent_median = mechanism[_MEDIAN], band.median
    both_inside = _inside(mechanism_median, band.p10, band.p90) and _inside(
        precedent_median, mechanism[_LOW], mechanism[_HIGH]
    )
    return "yes" if both_inside else _DISAGREE


def _precedent_cell(band: BandRow | None) -> str:
    if band is None:
        return _NO_AGREEMENT
    if band.refused or None in (band.p10, band.median, band.p90):
        return f"refused: n_eff={band.n_eff:.1f}"
    assert band.p10 is not None and band.median is not None and band.p90 is not None
    return _triple(band.p10, band.median, band.p90)


def _row(asset: str, result: MechanismResult, band: BandRow | None) -> str:
    fan_row = result.rollout_fan[asset]
    mechanism = _triple(fan_row[_LOW], fan_row[_MEDIAN], fan_row[_HIGH])
    window = result.fitted["horizon"]
    return (
        f"{asset:<{_ASSET_WIDTH}}{window:<{_WINDOW_WIDTH}}"
        f"{mechanism:<{_MECHANISM_WIDTH}}{_precedent_cell(band):<{_PRECEDENT_WIDTH}}"
        f"{_agreement(fan_row, band)} n={result.fitted['n_obs']}/{band.n if band else 0}"
    )


def _disagreement_line(result: MechanismResult) -> str:
    template_signs = {(e.source, e.target): e.sign for e in result.graph.edges}
    fitted = result.fitted
    parts = [
        f"{source} -> {target} (template {template_signs[(source, target)]:+d}, "
        f"fitted {fitted['coefficients'][target][source]:+.3f}, n={fitted['n_obs']})"
        for source, target in fitted["sign_disagreements"]
    ]
    return "edges the data disagrees with: " + ("; ".join(parts) if parts else "none")


def render_text(result: MechanismResult) -> str:
    """The plain-text block: header, one row per asset, disagreeing edges, disclaimer last."""
    fitted = result.fitted
    bands = {(b.asset, b.horizon): b for b in result.precedent}
    lines = [
        f"mechanism      {fitted['graph_version']} for {result.graph.event_key.event_type}, "
        f"fitted on {fitted['n_obs']} precedents as of "
        f"{result.asof_utc.astimezone(UTC):%Y-%m-%d}; "
        f"{len(fitted['underdetermined'])} equations underdetermined",
        f"{'asset':<{_ASSET_WIDTH}}{'window':<{_WINDOW_WIDTH}}"
        f"{'mechanism p10/median/p90':<{_MECHANISM_WIDTH}}"
        f"{'precedent p10/median/p90':<{_PRECEDENT_WIDTH}}agree?",
    ]
    lines += [
        _row(asset, result, bands.get((asset, fitted["horizon"]))) for asset in fitted["assets"]
    ]
    lines.append(_disagreement_line(result))
    lines += [
        f"intervention on {node} reaches no asset; the fan equals the no-intervention fan."
        for node in result.unreachable_interventions
    ]
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def render_json(result: MechanismResult) -> str:
    """The result as JSON; the manifest fields live in `fitted` and the result itself."""
    return result.model_dump_json(indent=2)


def _json_for_script_tag(payload: str) -> str:
    """JSON that cannot close the surrounding script element or open an HTML comment."""
    return payload.replace("<", "\\u003c")


def render_html(result: MechanismResult) -> str:
    """One self-contained page: the result embedded as JSON, a JavaScript simulator, no server."""
    template = TEMPLATE_PATH.read_text(encoding="utf-8")
    title = f"Mechanism cockpit — {result.graph.event_key.event_type} — {DISCLAIMER}"
    payload = _json_for_script_tag(result.model_dump_json())
    values = {"TITLE": html.escape(title), "PAYLOAD_JSON": payload}
    return re.sub(r"\{\{(TITLE|PAYLOAD_JSON)\}\}", lambda match: values[match[1]], template)
