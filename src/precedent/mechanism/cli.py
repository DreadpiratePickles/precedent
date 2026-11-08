"""`precedent mechanism`: fit a causal graph for an event key and print or save the what-if.

This module needs only the spine. The data step (`_load_inputs`, `_fit_and_band` and the engine
asset panel behind the template vocabulary) is the one place that reaches for the Precedent
engine, and it imports it lazily, so a checkout without the engine still reports `not built yet`
instead of failing at import.
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from precedent.config import ConfigError, data_dir, references_dir
from precedent.mechanism.cockpit import (
    BandRow,
    MechanismResult,
    build_result,
    render_html,
    render_json,
    render_text,
)
from precedent.mechanism.fit_sem import FittedSEM, fit_sem
from precedent.mechanism.graph_schema import CausalGraph, NodeKind, load_params, node_vocabulary
from precedent.mechanism.simulate import Intervention
from precedent.mechanism.templates import load_templates, template_for
from precedent.spine.event_key import EventKey
from precedent.spine.event_schema import EventType
from precedent.spine.typers.select import typer_from_env

if TYPE_CHECKING:
    from precedent.engine.params import EngineParams

USAGE_ERROR = 2
DEFAULT_HORIZON = 5
_SEVERITY_LEVEL = {"low": 0.0, "mid": 1.0, "high": 2.0}
_MISSING_ENGINE = "not built yet: mechanism needs {module} from the Precedent track"
_PRECEDENT_PACKAGES = ("precedent.engine", "precedent.libraries")


class MechanismRefused(Exception):
    """The fit has too few effective precedents: below Precedent's `min_neff` nothing is shown."""

    def __init__(self, n_eff: float, min_neff: float) -> None:
        self.n_eff = n_eff
        self.min_neff = min_neff
        super().__init__(
            f"refused: the fit has n_eff={n_eff:.1f} below min_neff={min_neff:.1f}; "
            "no mechanism is shown"
        )


@dataclass(frozen=True)
class MechanismInputs:
    """What the data step reads from the Precedent library: the engine's own pair and axis.

    `returns` is indexed by the session axis of `session_cutoffs_utc` (the axis the events'
    `effective_position` indexes) and has the market's column.
    """

    events: pd.DataFrame
    rows: pd.DataFrame
    session_cutoffs_utc: pd.Series
    returns: pd.DataFrame


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="precedent mechanism",
        description="A fitted model of past co-movement under an assumed structure.",
    )
    parser.add_argument("text", nargs="?", help="event text; typed by PRECEDENT_TYPER (rules)")
    parser.add_argument("--event-type")
    parser.add_argument("--actor-kind")
    parser.add_argument("--severity-band", choices=("low", "mid", "high"), default="mid")
    schedule = parser.add_mutually_exclusive_group()
    schedule.add_argument("--scheduled", dest="scheduled", action="store_true")
    schedule.add_argument("--unscheduled", dest="scheduled", action="store_false")
    parser.set_defaults(scheduled=False)
    parser.add_argument("--direction", type=int, choices=(-1, 0, 1), default=0)
    parser.add_argument("--country-group", choices=("US", "DM", "EM", "none"), default="none")
    parser.add_argument("--impacted-kinds", default="", help="comma-separated impact kinds")
    parser.add_argument(
        "--asof",
        help="YYYY-MM-DD; default today (UTC). The command sees only what was available "
        "before that day began (midnight UTC).",
    )
    parser.add_argument("--horizon", type=int, default=DEFAULT_HORIZON)
    parser.add_argument("--do", action="append", default=[], metavar="NODE=VALUE")
    parser.add_argument("--html", type=Path, metavar="PATH")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-paths", type=int)
    return parser


def parse_do_flags(flags: list[str], graph: CausalGraph) -> tuple[Intervention, ...]:
    """`NODE=VALUE` strings to interventions on intermediate or asset nodes with a finite value."""
    node_ids = {node.id for node in graph.nodes}
    event_ids = set(graph.event_node_ids)
    parsed = []
    for flag in flags:
        node, separator, raw = flag.rpartition("=")
        if not separator or not node:
            raise ValueError(f"--do {flag!r}: expected NODE=VALUE")
        if node not in node_ids:
            raise ValueError(f"--do {flag!r}: {node!r} is not a node of the graph")
        if node in event_ids:
            raise ValueError(
                f"--do {flag!r}: {node!r} is an event variable, which is set by its key flag "
                "(for example --direction), not by do()"
            )
        try:
            value = float(raw)
        except ValueError as error:
            raise ValueError(f"--do {flag!r}: {raw!r} is not a number") from error
        if not math.isfinite(value):
            raise ValueError(f"--do {flag!r}: {raw!r} is not a finite number")
        parsed.append(Intervention(node, value))
    return tuple(parsed)


def _key_from_flags(args: argparse.Namespace) -> EventKey:
    kinds = tuple(sorted(k for k in args.impacted_kinds.split(",") if k))
    key = EventKey(
        event_type=args.event_type,
        actor_kind=args.actor_kind or "other",
        impacted_kinds=kinds,
        country_group=args.country_group,
        severity_band=args.severity_band,
        is_scheduled=args.scheduled,
        direction_sign=args.direction,
    )
    return EventKey.from_string(key.as_string())  # rejects an unknown type, actor or kind


def _key_from_text(text: str) -> EventKey | None:
    typed = typer_from_env().type_text(text)
    if typed.status == "typed" and typed.key is not None:
        return typed.key
    print(
        f"cannot type that text ({typed.status}: {typed.reason}).\n"
        "Pass the key yourself: precedent mechanism --event-type <type> --actor-kind <kind> "
        "--severity-band <band> --unscheduled --direction <sign>",
        file=sys.stderr,
    )
    return None


def event_values_for(key: EventKey, graph: CausalGraph) -> dict[str, float]:
    """A number for each event node of the graph, read off the key."""
    values: dict[str, float] = {}
    for node_id in graph.event_node_ids:
        field = node_id.rsplit(":", 1)[-1]
        if field == "direction_sign":
            values[node_id] = float(key.direction_sign)
        elif field == "severity_band":
            values[node_id] = _SEVERITY_LEVEL[key.severity_band]
        elif field == "is_scheduled":
            values[node_id] = 1.0 if key.is_scheduled else 0.0
        else:
            raise ValueError(f"event node {node_id!r} has no known value encoding")
    return values


def _vocabulary() -> frozenset[str]:
    """Every node id a template may use; the asset panel is the engine's (authoritative)."""
    from precedent.engine.params import load_engine_params

    return node_vocabulary(
        references_dir() / "mechanism" / "intermediates.json",
        sorted(load_engine_params().asset_panel),
        [member.value for member in EventType],
    )


def _load_inputs(directory: Path) -> MechanismInputs:
    """Read the library bundle and the market's open-to-open returns under `directory`.

    NO LIVE RUN: `precedent.libraries.bundle` is Precedent Task 10, not merged into this track
    yet, so this body is written against its interface (`load_bundle(<dir>/library)` with
    `events`, `rows`, `sessions`, `session_cutoffs_utc`) and has only run against a fake bundle.
    Until it merges the import below raises `ModuleNotFoundError`, which `run` reports as
    `not built yet`. Only the market's returns are loaded: the panel takes every asset's value
    from the library rows and the market's own from `returns`.
    """
    from precedent.engine.params import load_engine_params
    from precedent.libraries.bundle import load_bundle
    from precedent.spine.price_provider import ParquetPriceProvider, load_returns

    params = load_engine_params()
    bundle = load_bundle(directory / "library")
    sessions = bundle.sessions
    returns = load_returns(
        [params.market],
        sessions[0].strftime("%Y-%m-%d"),
        sessions[-1].strftime("%Y-%m-%d"),
        "open_to_open",
        ParquetPriceProvider(directory / "prices"),
    ).reindex(sessions)
    return MechanismInputs(
        events=bundle.events,
        rows=bundle.rows,
        session_cutoffs_utc=bundle.session_cutoffs_utc,
        returns=returns,
    )


def _band_rows(
    panel_values: pd.DataFrame,
    positions: pd.Series,
    assets: list[str],
    horizon: int,
    params: EngineParams,
) -> list[BandRow]:
    """Precedent's band per asset from one guarded panel: `summarize_responses` per column."""
    from precedent.engine.aggregate import summarize_responses

    rows = []
    for asset in assets:
        observed = panel_values[asset].dropna()
        summary = summarize_responses(
            observed.to_numpy(dtype=float),
            np.ones(len(observed)),
            positions.loc[observed.index].to_numpy(dtype="int64"),
            horizon,
            None,
            shrink_k=params.shrink_k,
            min_neff=params.min_neff,
            z=params.wilson_z,
        )
        quantiles = summary.quantiles or {}
        rows.append(
            BandRow(
                asset=asset,
                horizon=horizon,
                p10=quantiles.get(0.1),
                median=quantiles.get(0.5),
                p90=quantiles.get(0.9),
                n=summary.n,
                n_eff=summary.n_eff,
                refused=summary.refused,
            )
        )
    return rows


def _refused_rows(assets: list[str], horizon: int) -> list[BandRow]:
    return [
        BandRow(asset=a, horizon=horizon, p10=None, median=None, p90=None, n=0, n_eff=0.0,
                refused=True)
        for a in assets
    ]  # fmt: skip


def _fit_and_band(
    key: EventKey,
    graph: CausalGraph,
    asof_utc: datetime,
    horizon: int,
    inputs: MechanismInputs,
    params: EngineParams,
) -> tuple[FittedSEM, list[BandRow]]:
    """Fit the graph on the guarded panel and read Precedent's band for the same key.

    Pure given its inputs. `asof_utc` is the query time: the fit and the band see only what was
    available at that instant. The fit's query leaves `direction_sign` free (`mechanism_query`,
    so the direction column varies); Precedent's band keeps the exact key. Below `min_neff` the
    fit is refused (`MechanismRefused`); a band with no exact-key precedent is a refused band
    row, not a refusal of the command.
    """
    from precedent.engine.aggregate import effective_sample_size
    from precedent.engine.retrieval import PrecedentQuery
    from precedent.mechanism.panel import NoPrecedentError, build_guarded_panel, mechanism_query

    if horizon not in params.asset_horizons:
        listed = ", ".join(str(h) for h in params.asset_horizons)
        raise ValueError(f"horizon {horizon} is not in the library; use one of {listed}")
    available = pd.Timestamp(asof_utc)
    assets = [node.id for node in graph.nodes if node.kind is NodeKind.ASSET]
    shared = {
        "events": inputs.events,
        "rows": inputs.rows,
        "session_cutoffs_utc": inputs.session_cutoffs_utc,
        "returns": inputs.returns,
        "market": params.market,
        "assets": assets,
        "event_node_ids": graph.event_node_ids,
        "horizon": horizon,
        "top_k": params.top_k,
    }
    positions = inputs.events.set_index("event_id")["effective_position"]
    try:
        panel = build_guarded_panel(mechanism_query(key, available), **shared)
    except NoPrecedentError:
        raise MechanismRefused(0.0, params.min_neff) from None
    n_eff = effective_sample_size(
        np.ones(panel.n_obs), positions.loc[panel.event_values.index].to_numpy("int64"), horizon
    )
    if n_eff < params.min_neff:
        raise MechanismRefused(n_eff, params.min_neff)
    fitted = fit_sem(graph, panel)
    try:
        band_panel = build_guarded_panel(PrecedentQuery(key, available), **shared)
    except NoPrecedentError:
        return fitted, _refused_rows(assets, horizon)
    return fitted, _band_rows(band_panel.asset_returns, positions, assets, horizon, params)


def _code_commit() -> str | None:
    try:
        done = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
            cwd=Path(__file__).resolve().parent,
        )  # fmt: skip
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout.strip() or None


def _write_atomically(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(text)
        os.replace(temp_name, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp_name)
        raise


def _parse_asof(raw: str | None) -> datetime:
    day = date.fromisoformat(raw) if raw else datetime.now(UTC).date()
    return datetime(day.year, day.month, day.day, tzinfo=UTC)


def _resolve_graph(key: EventKey) -> CausalGraph | None:
    templates = load_templates(references_dir() / "mechanism" / "templates", _vocabulary())
    return template_for(key, templates)


def _usage_error(message: str) -> int:
    print(message, file=sys.stderr)
    return USAGE_ERROR


def _result_for(args: argparse.Namespace, key: EventKey, graph: CausalGraph) -> MechanismResult:
    interventions = parse_do_flags(args.do, graph)
    asof_utc = _parse_asof(args.asof)
    from precedent.engine.params import load_engine_params

    inputs = _load_inputs(args.data_dir or data_dir())
    fitted, bands = _fit_and_band(key, graph, asof_utc, args.horizon, inputs, load_engine_params())
    return build_result(
        graph=graph,
        fitted=fitted,
        event_values=event_values_for(key, graph),
        interventions=interventions,
        precedent=bands,
        seed=args.seed,
        n_paths=load_params().n_paths if args.n_paths is None else args.n_paths,
        asof_utc=asof_utc,
        code_commit=_code_commit(),
    )


def run(argv: list[str]) -> int:
    """The `precedent mechanism` entry point; returns the process exit code."""
    parser = _parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as stop:
        return int(stop.code or 0)
    if (args.text is None) == (args.event_type is None):
        return _usage_error(
            "precedent mechanism: give either the event text or --event-type, not both"
        )
    try:
        key = _key_from_text(args.text) if args.text is not None else _key_from_flags(args)
        if key is None:
            return USAGE_ERROR
        graph = _resolve_graph(key)
        if graph is None:
            return _usage_error(f"no template graph for event type {key.event_type}")
        parse_do_flags(args.do, graph)  # fail on a bad --do before any data is read
        result = _result_for(args, key, graph)
    except MechanismRefused as error:
        return _usage_error(str(error))
    except (ConfigError, FileNotFoundError) as error:
        return _usage_error(str(error))
    except ValueError as error:
        return _usage_error(f"precedent mechanism: {error}")
    except ModuleNotFoundError as error:
        if (error.name or "").startswith(_PRECEDENT_PACKAGES):
            return _usage_error(_MISSING_ENGINE.format(module=error.name))
        raise
    if args.html is not None:
        _write_atomically(args.html, render_html(result))
    print(render_json(result) if args.as_json else render_text(result))
    return 0
