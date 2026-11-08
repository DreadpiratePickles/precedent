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
_MISSING_ENGINE = "not built yet: mechanism needs {module} from the Precedent track"
_PRECEDENT_PACKAGES = ("precedent.engine", "precedent.libraries")


class MechanismRefused(Exception):
    """Too few effective precedents to show anything: for the fit or the exact-key band.

    With `band=False` the direction-free fit is below Precedent's `min_neff`; with
    `band=True` Precedent's exact-key band for the query is refused (no exact-key precedent,
    or a data-having asset's band below `min_neff`) — either way nothing is shown, because a
    mechanism result without its Precedent band is not reported.
    """

    def __init__(self, n_eff: float, min_neff: float, *, band: bool = False) -> None:
        self.n_eff = n_eff
        self.min_neff = min_neff
        self.band = band
        subject = "Precedent's exact-key band has" if band else "the fit has"
        super().__init__(
            f"refused: {subject} n_eff={n_eff:.1f} below min_neff={min_neff:.1f}; "
            "no mechanism is shown"
        )


class NoDataAssets(Exception):
    """Every asset of the graph lacks library rows at this horizon: there is nothing to show.

    Distinct from `MechanismRefused`, which needs precedents but too few of them; here the
    library may hold plenty of events and simply no rows for any of the graph's assets.
    """

    def __init__(self, assets: list[str], horizon: int) -> None:
        self.assets = list(assets)
        self.horizon = horizon
        names = ", ".join(self.assets)
        super().__init__(
            f"no data: none of the graph's assets has a library row at horizon {horizon} "
            f"({names}); no mechanism is shown"
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
    """`NODE=VALUE` strings to interventions on intermediate or asset nodes with a finite value.

    A repeated node is a usage error: two do() values for one node would silently keep only
    one of them.
    """
    node_ids = {node.id for node in graph.nodes}
    event_ids = set(graph.event_node_ids)
    parsed = []
    seen: set[str] = set()
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
        if node in seen:
            raise ValueError(f"--do {flag!r}: duplicate intervention on {node!r}")
        seen.add(node)
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


def _event_values(key: EventKey, graph: CausalGraph) -> dict[str, float]:
    """The graph's event-node values for `key`, in the panel's one encoding.

    `panel.event_values_of` is the single event-column encoding (the panel reads it off every
    precedent's key); a query's values are the same mapping, as floats. The import is lazy so a
    spine-only checkout still starts the command.
    """
    from precedent.mechanism.panel import event_values_of

    return {
        node_id: float(value)
        for node_id, value in event_values_of(key, graph.event_node_ids).items()
    }


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
    """Precedent's band per asset from one guarded panel: `summarize_responses` per column.

    Every band is abnormal returns against the market model except the market's own: the
    library has no market row, so its band is the raw cumulative return (`basis` says so).
    """
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
                basis="raw_return" if asset == params.market else "abnormal_return",
            )
        )
    return rows


def _fit_and_band(
    key: EventKey,
    graph: CausalGraph,
    asof_utc: datetime,
    horizon: int,
    inputs: MechanismInputs,
    params: EngineParams,
) -> tuple[FittedSEM, list[BandRow], list[str], list[str]]:
    """Fit the graph on the guarded panel and read Precedent's band for the same key.

    Pure given its inputs. `asof_utc` is the query time: the fit and the band see only what was
    available at that instant. The fit's query leaves `direction_sign` free (`mechanism_query`,
    so the direction column varies); Precedent's band keeps the exact key. Below `min_neff` the
    fit is refused (`MechanismRefused`).

    The band gates per asset row, not the command (ruling I-2, addendum of
    `docs/verification/mechanism-2026-11-08.md`): a row is suppressed — fan, band and sample
    sizes not shown — when the asset has no observations (an all-NaN panel column) or its
    exact-key band is refused; the renderers print the reason ("no data", or
    "band refused: n_eff=X < min_neff=Y"). `no_data_assets` and `refused_assets` name the two
    groups; every observed asset keeps its `BandRow` in `bands`, refused ones included, so the
    reason can quote `n_eff`. The command refuses (exit 2) only when no row survives: every
    asset with data has a refused band (`MechanismRefused`, band=True), or every asset lacks
    rows (`NoDataAssets`). Suppressed assets are still simulated — dropping them when the fan
    is read out keeps the counter-based PRNG draws identical.
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
    no_data_assets = [a for a in assets if panel.asset_returns[a].isna().all()]
    observed = [a for a in assets if a not in no_data_assets]
    if not observed:
        raise NoDataAssets(list(no_data_assets), horizon)
    band_shared = {**shared, "assets": observed}
    try:
        band_panel = build_guarded_panel(PrecedentQuery(key, available), **band_shared)
    except NoPrecedentError:
        raise MechanismRefused(0.0, params.min_neff, band=True) from None
    bands = _band_rows(band_panel.asset_returns, positions, observed, horizon, params)
    refused_assets = [b.asset for b in bands if b.refused]
    if len(no_data_assets) + len(refused_assets) == len(assets):
        raise MechanismRefused(
            max(b.n_eff for b in bands), params.min_neff, band=True
        )  # fmt: skip
    return fitted, bands, no_data_assets, refused_assets


def _code_commit() -> str | None:
    """The commit the run was made from, marked `-dirty` when the working tree has changes."""
    try:
        done = subprocess.run(
            ["git", "describe", "--always", "--dirty"],
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

    params = load_engine_params()
    inputs = _load_inputs(args.data_dir or data_dir())
    fitted, bands, no_data_assets, refused_assets = _fit_and_band(
        key, graph, asof_utc, args.horizon, inputs, params
    )
    return build_result(
        graph=graph,
        fitted=fitted,
        event_values=_event_values(key, graph),
        interventions=interventions,
        precedent=bands,
        no_data_assets=tuple(no_data_assets),
        refused_assets=tuple(refused_assets),
        min_neff=params.min_neff,
        seed=args.seed,
        n_paths=load_params().n_paths if args.n_paths is None else args.n_paths,
        asof_utc=asof_utc,
        code_commit=_code_commit(),
    )


def _library_integrity_errors() -> tuple[type[Exception], ...]:
    """The corrupt-library error, when the libraries package this loader needs is importable."""
    try:
        from precedent.libraries.manifest import LibraryIntegrityError
    except ImportError:
        return ()  # spine-only checkout: the loader reports "not built yet" instead
    return (LibraryIntegrityError,)


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
    except NoDataAssets as error:
        return _usage_error(str(error))
    except _library_integrity_errors() as error:
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
