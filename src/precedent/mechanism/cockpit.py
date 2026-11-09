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

from pydantic import AwareDatetime, BaseModel, ConfigDict, model_validator

from precedent.mechanism.fit_sem import FittedSEM
from precedent.mechanism.graph_schema import CausalGraph
from precedent.mechanism.simulate import DEFAULT_QUANTILES, Intervention, fan, simulate

DISCLAIMER = "A fitted model of past co-movement under an assumed structure, not a forecast."
TEMPLATE_PATH = Path(__file__).with_name("cockpit_template.html")
_LOW, _MEDIAN, _HIGH = "0.1", "0.5", "0.9"
_NO_AGREEMENT = "n/a"
_DISAGREE = "no (median outside the other's band)"
_ASSET_WIDTH, _WINDOW_WIDTH, _MECHANISM_WIDTH, _PRECEDENT_WIDTH = 8, 9, 30, 34


class BandRow(BaseModel):
    """Precedent's p10/median/p90 for one asset and window, with its sample sizes.

    `basis` is `abnormal_return` for every asset but the market: the library has no market
    row, so the market's band is computed from its own cumulative return (raw), while all
    other bands and every fitted equation are abnormal returns against the market model.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: str
    horizon: int
    p10: float | None
    median: float | None
    p90: float | None
    n: int
    n_eff: float
    refused: bool
    basis: Literal["abnormal_return", "raw_return"] = "abnormal_return"


class InterventionRow(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    node: str
    value: float


class MechanismResult(BaseModel):
    """Everything one mechanism run reports; the three outputs are views of this value.

    `fitted` is `FittedSEM.to_json_dict()` and `rollout_fan` maps asset -> quantile (as the
    string `fan()` columns print, e.g. "0.1") -> simulated value, one entry per row that
    survived the band gate. `no_data_assets` and `refused_assets` name the suppressed rows
    (ruling I-2, the addendum of `docs/verification/mechanism-2026-11-08.md`): assets with
    zero observations, and assets whose exact-key band is refused. Suppressed assets are
    simulated like every other node — dropping them when the fan is read out keeps the
    counter-based PRNG draws identical — and the text and the page print the reason ("no
    data", "band refused: n_eff=X < min_neff=Y") instead of a fan. `min_neff` is the
    threshold the rows were gated on (the engine's, from `references/precedent_params.json`).
    `precedent[].basis` marks the market row's band as computed on raw market returns (see
    `BandRow`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    graph: CausalGraph
    fitted: dict[str, Any]
    event_values: dict[str, float]
    interventions: list[InterventionRow]
    rollout_fan: dict[str, dict[str, float]]
    precedent: list[BandRow]
    no_data_assets: list[str] = []
    refused_assets: list[str] = []
    # the engine's value, required: the CLI passes load_engine_params().min_neff and there is
    # no default, so a result can only carry the parameter the engine was configured with
    min_neff: float
    seed: int
    n_paths: int
    asof_utc: AwareDatetime
    unreachable_interventions: list[str]
    diagonal_fallback: bool
    code_commit: str | None
    disclaimer: Literal[DISCLAIMER] = DISCLAIMER

    @model_validator(mode="after")
    def _no_do_on_an_event_variable(self) -> MechanismResult:
        """do() is for intermediate and asset nodes; an event variable is set by its key."""
        events = set(self.graph.event_node_ids)
        bad = [i.node for i in self.interventions if i.node in events]
        if bad:
            raise ValueError(f"interventions on event nodes are not allowed: {bad}")
        return self


def build_result(
    *,
    graph: CausalGraph,
    fitted: FittedSEM,
    event_values: dict[str, float],
    interventions: tuple[Intervention, ...],
    precedent: list[BandRow],
    no_data_assets: tuple[str, ...] = (),
    refused_assets: tuple[str, ...] = (),
    min_neff: float,
    seed: int,
    n_paths: int,
    asof_utc: Any,
    code_commit: str | None,
) -> MechanismResult:
    """Simulate the fitted graph and bundle the fan with its inputs.

    The simulation always covers every graph asset — dropping `no_data_assets` and
    `refused_assets` only when the fan is read out keeps the counter-based PRNG draws
    identical for the rows that survived, in Python and in the page's JavaScript twin.
    """
    rollout = simulate(fitted, graph, event_values, interventions, n_paths=n_paths, seed=seed)
    table = fan(rollout, DEFAULT_QUANTILES)
    suppressed = set(no_data_assets) | set(refused_assets)
    unknown = suppressed - set(rollout.assets)
    if unknown:
        raise ValueError(f"suppressed assets are not asset nodes with a fan: {sorted(unknown)}")
    return MechanismResult(
        graph=graph,
        fitted=fitted.to_json_dict(),
        event_values=dict(event_values),
        interventions=[InterventionRow(node=i.node, value=i.value) for i in interventions],
        rollout_fan={
            asset: {str(q): float(table.loc[asset, q]) for q in DEFAULT_QUANTILES}
            for asset in rollout.assets
            if asset not in suppressed
        },
        precedent=precedent,
        no_data_assets=[asset for asset in rollout.assets if asset in no_data_assets],
        refused_assets=[asset for asset in rollout.assets if asset in refused_assets],
        min_neff=min_neff,
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
    cell = _triple(band.p10, band.median, band.p90)
    return cell + " (raw)" if band.basis == "raw_return" else cell


def _suppressed_reason(
    asset: str, result: MechanismResult, band: BandRow | None
) -> str | None:
    """Why a row shows no fan: "no data" (zero observations) or its refused band's numbers."""
    if asset in result.no_data_assets:
        return "no data"
    if band is not None and band.refused:
        return f"band refused: n_eff={band.n_eff:.1f} < min_neff={result.min_neff:.1f}"
    return None


def _row(asset: str, result: MechanismResult, band: BandRow | None) -> str:
    fan_row = result.rollout_fan[asset]
    mechanism = _triple(fan_row[_LOW], fan_row[_MEDIAN], fan_row[_HIGH])
    window = result.fitted["horizon"]
    return (
        f"{asset:<{_ASSET_WIDTH}}{window:<{_WINDOW_WIDTH}}"
        f"{mechanism:<{_MECHANISM_WIDTH}}{_precedent_cell(band):<{_PRECEDENT_WIDTH}}"
        f"{_agreement(fan_row, band)} n={result.fitted['n_obs']}/{band.n if band else 0}"
    )


def _reason_row(asset: str, result: MechanismResult, reason: str) -> str:
    window = result.fitted["horizon"]
    return f"{asset:<{_ASSET_WIDTH}}{window:<{_WINDOW_WIDTH}}{reason}"


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
    """The plain-text block: header, one row per graph asset, disagreeing edges, disclaimer.

    A row shows its fan only when the asset has observations and its Precedent band is not
    refused; a suppressed row prints the reason instead ("no data", or the refused band's
    `n_eff` against `min_neff`) — never a fabricated fan. The market row, whose band and fan
    are raw cumulative returns, is labelled `(raw)` and explained once.
    """
    fitted = result.fitted
    bands = {(b.asset, b.horizon): b for b in result.precedent}
    lines = [
        f"mechanism      {fitted['graph_version']} for {result.graph.event_key.event_type}, "
        f"fitted on {fitted['n_obs']} precedents as of "
        f"{result.asof_utc.astimezone(UTC):%Y-%m-%d}; "
        f"{len(fitted['underdetermined'])} nodes uninformative "
        "(own fit failed, or an unidentified latent upstream)",
        f"{'asset':<{_ASSET_WIDTH}}{'window':<{_WINDOW_WIDTH}}"
        f"{'mechanism p10/median/p90':<{_MECHANISM_WIDTH}}"
        f"{'precedent p10/median/p90':<{_PRECEDENT_WIDTH}}agree?",
    ]
    for asset in fitted["assets"]:
        band = bands.get((asset, fitted["horizon"]))
        reason = _suppressed_reason(asset, result, band)
        lines.append(
            _reason_row(asset, result, reason)
            if reason is not None
            else _row(asset, result, band)
        )
    raw = sorted(b.asset for b in result.precedent if b.basis == "raw_return")
    if raw:
        lines.append(
            f"{', '.join(raw)}: the market's own cumulative return (raw), not an abnormal "
            "return; every other row is abnormal against the market model."
        )
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
    raw = sorted({band.asset for band in result.precedent if band.basis == "raw_return"})
    raw_note = (
        f"{', '.join(raw)}: the market's own cumulative return (raw), not an abnormal "
        "return; every other row is abnormal against the market model."
        if raw
        else ""
    )
    values = {
        "TITLE": html.escape(title),
        "PAYLOAD_JSON": payload,
        "RAW_NOTE": html.escape(raw_note),
    }
    return re.sub(
        r"\{\{(TITLE|PAYLOAD_JSON|RAW_NOTE)\}\}", lambda match: values[match[1]], template
    )
