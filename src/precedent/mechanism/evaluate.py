"""The mechanism ladder: template graph (M-T) and its degree-preserving placebo (M-P) scored
against Precedent's empirical band (P-B1) on identical units in time order.

A unit is an `(event, asset)` pair: a first-report event and one asset row of the library at the
scoring horizon (the market symbol has no row and is never a unit). Events are visited in the
order Precedent's own ladder uses, and every rung forecasts each unit from earlier events only:
P-B1 through `find_precedents`, a mechanism rung through `build_guarded_panel`, which goes through
the same `find_precedents` call and so the same two time guards. A mechanism rung scores a unit
only when P-B1 scored it, so the rungs are compared on the same units; a unit P-B1 refused is
counted nowhere for a mechanism rung.

The `mech_` rung files under `references/rungs/` belong to this ladder only. The spine
`run_ablation` never scores them (they add no factor panel), and any caller that runs
`load_rungs(references/rungs)` through the ablation filters the `mech_` rungs out.

This module imports the Precedent engine at module level; the mechanism CLI imports it lazily.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from precedent.engine.aggregate import summarize_responses
from precedent.engine.evaluate import pinball_loss, score_precedent_ladder
from precedent.engine.keys import EventKey
from precedent.engine.params import load_engine_params
from precedent.engine.retrieval import PrecedentQuery, find_precedents
from precedent.mechanism.fit_sem import fit_sem
from precedent.mechanism.graph_schema import CausalGraph, Edge, NodeKind, is_acyclic
from precedent.mechanism.panel import (
    NoPrecedentError,
    build_guarded_panel,
    event_values_of,
    mechanism_query,
)
from precedent.mechanism.simulate import fan, simulate

__all__ = [
    "LadderError",
    "PlaceboError",
    "precedent_scored_units",
    "score_mechanism_ladder",
    "shuffle_edges_preserving_degree",
]

PRECEDENT_RUNG = "P-B1"
OUTPUT_COLUMNS = (
    "rung",
    "event_type",
    "quantile",
    "pinball_loss",
    "coverage_80",
    "n_scored",
    "n_refused",
    "n_underdetermined",
)
_BAND_LOW = 0.1
_BAND_HIGH = 0.9
_SWAPS_PER_EDGE = 10


class PlaceboError(ValueError):
    """A graph has no degree-preserving rewiring that differs from it."""


def _new_edge(source: str, target: str, lag_band: str) -> Edge:
    """A bare edge; the final signs are drawn after all swaps (see the shuffler)."""
    return Edge.model_validate(
        {
            "source": source,
            "target": target,
            "sign": 1,
            "lag_band": lag_band,
            "source_doc_id": None,
            "evidence_span": None,
            "support": 1,
        }
    )


def _swapped(
    edges: list[Edge], i: int, j: int, kinds: dict[str, NodeKind], node_ids: list[str]
) -> list[Edge] | None:
    """`edges` with (a->b, c->d) at positions i, j replaced by (a->d, c->b), or None if invalid."""
    (a, b), (c, d) = (edges[i].source, edges[i].target), (edges[j].source, edges[j].target)
    present = {(e.source, e.target) for e in edges}
    if a == d or c == b or (a, d) in present or (c, b) in present:
        return None
    if NodeKind.EVENT in (kinds[d], kinds[b]):
        return None
    if any(
        kinds[s] is NodeKind.ASSET and kinds[t] is not NodeKind.ASSET for s, t in ((a, d), (c, b))
    ):
        return None
    proposal = list(edges)
    proposal[i] = _new_edge(a, d, edges[i].lag_band)
    proposal[j] = _new_edge(c, b, edges[j].lag_band)
    return proposal if is_acyclic(node_ids, proposal) else None


def shuffle_edges_preserving_degree(graph: CausalGraph, seed: int) -> CausalGraph:
    """The degree-preserving placebo of `graph`: same nodes, same in- and out-degree per node.

    Double-edge swaps: pick two edges (a->b, c->d) and propose (a->d, c->b), kept only when
    neither proposed edge duplicates an existing one or is a self-loop, no asset causes a
    non-asset, nothing points into an event node (event variables are exogenous), and the
    graph stays acyclic. Each new edge keeps the lag band of the edge it replaced. The placebo
    also randomises signs: every edge gets a sign drawn uniformly from {-1, +1}, and carries
    no evidence (`source_doc_id` and `evidence_span` are None, `support` is 1).

    If the random walk ends back on the original edge set, the first valid single swap (in a
    seeded order) is applied instead. Raises `PlaceboError` when no valid swap exists at all
    (a single edge, or a star-shaped graph); the original is never returned silently.
    Same `seed`, same placebo.
    """
    rng = np.random.default_rng(seed)
    node_ids = [node.id for node in graph.nodes]
    kinds = {node.id: node.kind for node in graph.nodes}
    original = [_new_edge(e.source, e.target, e.lag_band) for e in graph.edges]
    original_pairs = {(e.source, e.target) for e in original}
    edges = original
    if len(edges) >= 2:
        for _ in range(_SWAPS_PER_EDGE * len(edges)):
            i, j = (int(k) for k in rng.choice(len(edges), size=2, replace=False))
            edges = _swapped(edges, i, j, kinds, node_ids) or edges
        if {(e.source, e.target) for e in edges} == original_pairs:
            ordered = rng.permutation(len(original) * len(original))
            for k in ordered:
                i, j = divmod(int(k), len(original))
                if i != j and (proposal := _swapped(original, i, j, kinds, node_ids)):
                    edges = proposal
                    break
    if {(e.source, e.target) for e in edges} == original_pairs:
        raise PlaceboError(
            f"graph {graph.graph_id!r} has no degree-preserving rewiring that differs from it "
            f"({len(graph.edges)} edges)"
        )
    signs = rng.choice([-1, 1], size=len(edges))
    edges = [e.model_copy(update={"sign": int(s)}) for e, s in zip(edges, signs, strict=True)]
    return CausalGraph(
        graph_id=f"{graph.graph_id}:placebo-{seed}",
        graph_version=f"placebo-{seed}",
        event_key=graph.event_key,
        nodes=list(graph.nodes),
        edges=edges,
    )


class LadderError(RuntimeError):
    """The ladder's own P-B1 unit count disagrees with Precedent's ladder; nothing is scored."""


@dataclass(frozen=True)
class _Unit:
    """One scored P-B1 unit: the asset and the value realised for it."""

    asset: str
    value: float


def _scored_units_by_event(
    events: pd.DataFrame, rows: pd.DataFrame, session_cutoffs_utc: pd.Series, horizon: int
) -> list[tuple[pd.Series, list[_Unit]]]:
    """Per first-report event with asset rows at `horizon`, in the engine's order: the units
    P-B1 scored for it. The refusal rule is the engine's: exact-key precedents through
    `find_precedents`, equal weights, `summarize_responses` with the engine parameters."""
    params = load_engine_params()
    wanted = rows.loc[(rows["outcome_type"] == "asset") & (rows["horizon"] == horizon)]
    realised = {
        event_id: group
        for event_id, group in wanted.sort_values("outcome_key", kind="stable").groupby(
            "event_id", sort=False
        )
    }
    position_of = events.set_index("event_id")["effective_position"]
    ordered = events.sort_values(["available_utc", "event_id"], kind="stable")
    ordered = ordered.loc[ordered["is_first_report"].astype(bool)]
    visits: list[tuple[pd.Series, list[_Unit]]] = []
    for _, event in ordered.iterrows():
        event_rows = realised.get(event["event_id"])
        if event_rows is None:
            continue
        query = PrecedentQuery(
            key=EventKey.from_string(event["key"]),
            available_utc=event["available_utc"],
            story_cluster_id=event["story_cluster_id"],
        )
        found = find_precedents(
            query, events, rows, session_cutoffs_utc, horizon, "asset", top_k=params.top_k
        )
        scored: list[_Unit] = []
        for row in event_rows.itertuples(index=False):
            precedents = found.loc[found["outcome_key"] == row.outcome_key]
            summary = summarize_responses(
                precedents["value"].to_numpy(dtype=float),
                np.ones(len(precedents)),
                precedents["event_id"].map(position_of).to_numpy(),
                horizon,
                None,
                shrink_k=params.shrink_k,
                min_neff=params.min_neff,
                z=params.wilson_z,
            )
            if not summary.refused:
                scored.append(_Unit(str(row.outcome_key), float(row.value)))
        visits.append((event, scored))
    return visits


def precedent_scored_units(
    events: pd.DataFrame, rows: pd.DataFrame, session_cutoffs_utc: pd.Series, horizon: int = 5
) -> dict[str, list[str]]:
    """The assets P-B1 scored for each event, by `event_id`, in time order.

    An event with asset rows at `horizon` appears even when P-B1 refused all of them (empty
    list). These are the units every mechanism rung is scored on.
    """
    visits = _scored_units_by_event(events, rows, session_cutoffs_utc, horizon)
    return {event["event_id"]: [unit.asset for unit in units] for event, units in visits}


@dataclass
class _Tally:
    """What one `(rung, event_type)` group has scored so far."""

    realized: list[float] = field(default_factory=list)
    predicted: dict[float, list[float]] = field(default_factory=dict)
    band_low: list[float] = field(default_factory=list)
    band_high: list[float] = field(default_factory=list)
    n_refused: int = 0
    n_underdetermined: int = 0


def _forecast(
    graph: CausalGraph | None,
    key: EventKey,
    event: pd.Series,
    events: pd.DataFrame,
    rows: pd.DataFrame,
    session_cutoffs_utc: pd.Series,
    returns: pd.DataFrame,
    market: str,
    horizon: int,
    quantiles: tuple[float, ...],
    n_paths: int,
    seed: int,
    top_k: int,
    min_neff: float,
) -> tuple[pd.DataFrame, frozenset[str]] | None:
    """The graph's per-asset fan for the event and its underdetermined assets, or None when
    the graph is absent, no precedent passed the guard, or fewer than `min_neff` events did."""
    if graph is None:
        return None
    query = mechanism_query(key, event["available_utc"], event["story_cluster_id"])
    try:
        panel = build_guarded_panel(
            query,
            events,
            rows,
            session_cutoffs_utc,
            returns,
            market,
            [node.id for node in graph.nodes if node.kind is NodeKind.ASSET],
            graph.event_node_ids,
            horizon,
            top_k,
        )
    except NoPrecedentError:
        return None
    if panel.n_obs < min_neff:
        return None
    fitted = fit_sem(graph, panel)
    values = {node: float(v) for node, v in event_values_of(key, graph.event_node_ids).items()}
    rollout = simulate(fitted, graph, values, (), n_paths, seed)
    levels = tuple(sorted({*quantiles, _BAND_LOW, _BAND_HIGH}))
    return fan(rollout, levels), fitted.underdetermined


def _score_event(
    tally: _Tally,
    units: list[_Unit],
    forecast: tuple[pd.DataFrame, frozenset[str]] | None,
    quantiles: tuple[float, ...],
) -> None:
    """Add one event's P-B1-scored units to a rung's tally."""
    for unit in units:
        if forecast is None or unit.asset not in forecast[0].index:
            tally.n_refused += 1
            continue
        band = forecast[0].loc[unit.asset]
        if not np.isfinite(band.to_numpy(dtype=float)).all():
            raise LadderError(f"non-finite simulated quantiles for asset {unit.asset!r}")
        tally.realized.append(unit.value)
        for q in quantiles:
            tally.predicted.setdefault(q, []).append(float(band[q]))
        tally.band_low.append(float(band[_BAND_LOW]))
        tally.band_high.append(float(band[_BAND_HIGH]))
        tally.n_underdetermined += unit.asset in forecast[1]


def _mechanism_table(
    tallies: dict[tuple[str, str], _Tally], quantiles: tuple[float, ...]
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (rung, event_type), tally in tallies.items():
        realized = np.array(tally.realized, dtype=float)
        coverage = np.nan
        if tally.realized:
            inside = (realized >= np.array(tally.band_low)) & (
                realized <= np.array(tally.band_high)
            )
            coverage = float(np.mean(inside))
        for q in quantiles:
            loss = (
                pinball_loss(realized, np.array(tally.predicted[q]), q)
                if tally.realized
                else np.nan
            )
            records.append(
                {
                    "rung": rung,
                    "event_type": event_type,
                    "quantile": q,
                    "pinball_loss": loss,
                    "coverage_80": coverage,
                    "n_scored": len(tally.realized),
                    "n_refused": tally.n_refused,
                    "n_underdetermined": tally.n_underdetermined,
                }
            )
    return pd.DataFrame(records, columns=list(OUTPUT_COLUMNS))


def score_mechanism_ladder(
    events: pd.DataFrame,
    rows: pd.DataFrame,
    session_cutoffs_utc: pd.Series,
    returns: pd.DataFrame,
    market: str,
    graphs_by_rung: Mapping[str, Callable[[EventKey], CausalGraph | None]],
    horizon: int = 5,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
    n_paths: int = 2_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Score P-B1 and each mechanism rung of `graphs_by_rung` on the same units, in time order.

    `events` and `rows` are the engine's library pair, `returns` is indexed by the session axis
    of `session_cutoffs_utc` and holds the `market` column. Each key of `graphs_by_rung` is a
    rung name (for example `M-T`, `M-P`); its callable maps an event's key to the graph to fit
    for it, or None when the rung has no graph for that event (every unit is then refused: the
    placebo does not exist for the earnings and macro_data_release templates). The name `P-B1`
    is reserved for Precedent's own band and raises `ValueError`.

    For each event the graph is fitted on the guarded panel of `mechanism_query` (direction-free)
    and rolled out with the event's own key values; the forecast of an asset is the fan's
    quantile of its node. A unit is refused when the graph is None, the asset is not in the
    graph, no precedent passed the guard, or fewer than `min_neff` events did. Output columns
    are `OUTPUT_COLUMNS`, one row per `(rung, event_type, quantile)` sorted on those keys;
    `coverage_80` is the share of realised values inside the rung's own [q0.1, q0.9]; and
    `n_underdetermined` counts scored units whose asset is uninformative: its own equation
    had too little data to fit, or an unidentified latent sits upstream of it and the fit
    ran against a zero score (always 0 for P-B1). Quantiles must be among the engine's
    `QUANTILE_LEVELS`.
    """
    if PRECEDENT_RUNG in graphs_by_rung:
        raise ValueError(
            f"{PRECEDENT_RUNG!r} is Precedent's own rung and cannot be a mechanism rung"
        )
    precedent = score_precedent_ladder(
        events, rows, session_cutoffs_utc, "asset", rungs=("B1",), horizon=horizon,
        quantiles=quantiles,
    )  # fmt: skip
    visits = _scored_units_by_event(events, rows, session_cutoffs_utc, horizon)
    _check_agreement(precedent, visits)
    params = load_engine_params()
    tallies: dict[tuple[str, str], _Tally] = {}
    for event, units in visits:
        key = EventKey.from_string(event["key"])
        for rung, graph_for in graphs_by_rung.items():
            tally = tallies.setdefault((rung, event["event_type"]), _Tally())
            if not units:
                continue
            forecast = _forecast(
                graph_for(key), key, event, events, rows, session_cutoffs_utc, returns, market,
                horizon, quantiles, n_paths, seed, params.top_k, params.min_neff,
            )  # fmt: skip
            _score_event(tally, units, forecast, quantiles)
    baseline = precedent.assign(rung=PRECEDENT_RUNG, n_underdetermined=0)[list(OUTPUT_COLUMNS)]
    table = pd.concat([baseline, _mechanism_table(tallies, quantiles)], ignore_index=True)
    table = table.sort_values(["rung", "event_type", "quantile"], kind="stable")
    for column in ("quantile", "pinball_loss", "coverage_80"):
        table[column] = table[column].astype("float64")
    for column in ("n_scored", "n_refused", "n_underdetermined"):
        table[column] = table[column].astype("int64")
    return table.reset_index(drop=True)


def _check_agreement(precedent: pd.DataFrame, visits: list[tuple[pd.Series, list[_Unit]]]) -> None:
    """Fail loudly if this module's P-B1 units are not the ones Precedent's ladder scored."""
    ours: dict[str, int] = {}
    for event, units in visits:
        ours[event["event_type"]] = ours.get(event["event_type"], 0) + len(units)
    theirs = precedent.groupby("event_type")["n_scored"].first().astype(int).to_dict()
    if ours != theirs:
        raise LadderError(
            f"P-B1 units scored per event type disagree: this module {ours}, "
            f"precedent.engine.evaluate {theirs}"
        )
