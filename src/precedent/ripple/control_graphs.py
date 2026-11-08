"""Control graphs for the ripple ladder: sector membership, return correlation, and a placebo.

Every graph rung of the ripple ladder is reported beside its shuffled placebo. The sector and
correlation graphs are the cheap controls a filing-derived graph must beat; the placebo keeps
every node's degree and moves only who is linked to whom.

Core dependencies only (numpy, pandas).
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from precedent.ripple.link_store import Link, Relation

SECTOR_EXTRACTOR_VERSION = "rules-sector-v1"
CORRELATION_EXTRACTOR_VERSION = "rules-correlation-v1"
_MIN_OBSERVATIONS = 2
_ZERO_VARIANCE_TOLERANCE = 1e-12
_SWAP_ATTEMPTS_PER_EDGE = 10


def sector_links(
    sector_of: dict[str, str], valid_from_utc: datetime, valid_to_utc: datetime
) -> list[Link]:
    """One `peer_of` link for every ordered pair of distinct members of the same sector.

    Weight is 1.0 for every link (`Link` weights lie in [0, 1]); the factor stage normalises each
    receiver's incoming weights. A sector with one member yields no link. Order is deterministic:
    sectors sorted, then source members sorted, then receiver members sorted. Timestamps must be
    tz-aware; `Link` rejects naive ones.
    """
    members_by_sector: dict[str, list[str]] = {}
    for instrument, sector in sector_of.items():
        members_by_sector.setdefault(sector, []).append(instrument)
    links: list[Link] = []
    for sector in sorted(members_by_sector):
        members = sorted(members_by_sector[sector])
        for source in members:
            for receiver in members:
                if source == receiver:
                    continue
                links.append(
                    Link(
                        source_node=source,
                        receiver_node=receiver,
                        relation=Relation.PEER_OF,
                        weight=1.0,
                        valid_from_utc=valid_from_utc,
                        valid_to_utc=valid_to_utc,
                        source_doc_id=f"sector:{sector}",
                        evidence_span=None,
                        extractor_version=SECTOR_EXTRACTOR_VERSION,
                    )
                )
    return links


def _market_residuals(window_returns: pd.DataFrame, market: str) -> pd.DataFrame:
    """Each non-market column minus its OLS fit (intercept and slope) on the market column."""
    market_returns = window_returns[market].to_numpy(dtype=float)
    residuals: dict[str, np.ndarray] = {}
    for name in window_returns.columns:
        if name == market:
            continue
        instrument = window_returns[name].to_numpy(dtype=float)
        finite = np.isfinite(instrument) & np.isfinite(market_returns)
        residual = np.full(len(instrument), np.nan)
        if finite.sum() >= _MIN_OBSERVATIONS:
            x = market_returns[finite]
            y = instrument[finite]
            x_centred = x - x.mean()
            variance = float((x_centred**2).sum())
            slope = float((x_centred * (y - y.mean())).sum() / variance) if variance > 0 else 0.0
            intercept = y.mean() - slope * x.mean()
            residual[finite] = y - (intercept + slope * x)
            if np.nanstd(residual) <= _ZERO_VARIANCE_TOLERANCE:
                residual[:] = np.nan
        residuals[name] = residual
    return pd.DataFrame(residuals, index=window_returns.index)


def correlation_links(
    returns: pd.DataFrame,
    market: str,
    asof_pos: int,
    window: int = 250,
    top_k: int = 10,
) -> list[tuple[str, str, float]]:
    """Peer pairs from residual-return correlation over the `window` sessions ending at `asof_pos`.

    `returns` is a session-indexed frame of close-to-close returns, one column per instrument,
    including the `market` column. Only rows at integer positions `asof_pos - window + 1` to
    `asof_pos` inclusive are read, so nothing after the as-of session can leak in. Each
    instrument's residual is its return minus its OLS fit on the market over that window (fitted
    on that window only); the market column is excluded from the output. Pairwise Pearson
    correlation of residuals uses pairwise-complete observations; an instrument with fewer than
    two finite observations or zero variance gets no links.

    For each receiver the `top_k` other instruments with the highest strictly positive residual
    correlation are kept. Returns `(source, receiver, correlation)` tuples as Python floats,
    ordered by receiver, then descending correlation, then source name; empty when nothing
    qualifies. Raises ValueError when the window starts before position 0 or `asof_pos` is past
    the last row.

    Design note: the tuples become links with `relation="peer_of"`,
    `extractor_version="rules-correlation-v1"`, `weight=correlation` and
    `source_doc_id=f"corr:{asof session date}"`; `correlation_links_as_links` does that wrapping.
    """
    if window < 1 or top_k < 1:
        raise ValueError("window and top_k must be at least 1")
    if market not in returns.columns:
        raise ValueError(f"market column {market!r} is not in returns")
    if asof_pos >= len(returns) or asof_pos < 0:
        raise ValueError(f"asof_pos {asof_pos} is outside the {len(returns)} rows of returns")
    start = asof_pos - window + 1
    if start < 0:
        raise ValueError(f"window of {window} sessions does not fit before asof_pos {asof_pos}")
    residuals = _market_residuals(returns.iloc[start : asof_pos + 1], market)
    correlations = residuals.corr()
    pairs: list[tuple[str, str, float]] = []
    for receiver in sorted(correlations.columns):
        candidates = [
            (source, float(value))
            for source, value in correlations[receiver].items()
            if source != receiver and np.isfinite(value) and value > 0.0
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        pairs.extend((source, receiver, value) for source, value in candidates[:top_k])
    return pairs


def correlation_links_as_links(
    pairs: list[tuple[str, str, float]],
    asof_session: pd.Timestamp,
    valid_from_utc: datetime,
    valid_to_utc: datetime,
) -> list[Link]:
    """Wrap `correlation_links` output as `peer_of` links dated by the as-of session."""
    doc_id = f"corr:{asof_session.date().isoformat()}"
    return [
        Link(
            source_node=source,
            receiver_node=receiver,
            relation=Relation.PEER_OF,
            weight=correlation,
            valid_from_utc=valid_from_utc,
            valid_to_utc=valid_to_utc,
            source_doc_id=doc_id,
            evidence_span=None,
            extractor_version=CORRELATION_EXTRACTOR_VERSION,
        )
        for source, receiver, correlation in pairs
    ]


_NO_RELATION = "<no relation column>"


def _shuffled_receivers(
    sources: list[str], receivers: list[str], rng: np.random.Generator
) -> list[str] | None:
    """Receivers after degree-preserving edge swaps among one relation's edges, or None.

    None means no swap can change the edge set (a graph of one edge, one receiver, one source,
    or two nodes linked both ways). The random swaps can cancel out, so when they do a
    deterministic scan applies the first swap that really changes the edge set; None is returned
    only when no such swap exists.
    """
    n_edges = len(sources)
    original = set(zip(sources, receivers, strict=True))
    edges = set(original)
    swapped = list(receivers)
    picks = rng.integers(0, n_edges, size=(_SWAP_ATTEMPTS_PER_EDGE * n_edges, 2))
    for first, second in picks.tolist():
        if first == second:
            continue
        a, b = sources[first], swapped[first]
        c, d = sources[second], swapped[second]
        if a == d or c == b or (a, d) in edges or (c, b) in edges:
            continue
        edges.difference_update(((a, b), (c, d)))
        edges.update(((a, d), (c, b)))
        swapped[first], swapped[second] = d, b
    if edges != original:
        return swapped
    scanned = list(receivers)
    for first in range(n_edges):
        for second in range(first + 1, n_edges):
            a, b = sources[first], receivers[first]
            c, d = sources[second], receivers[second]
            if a in (c, d) or b in (c, d) or (a, d) in original or (c, b) in original:
                continue
            scanned[first], scanned[second] = d, b
            return scanned
    return None


def shuffle_by_relation(graph: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, int]:
    """A placebo graph and how many relation subgraphs could not be shuffled.

    Each relation's edges are shuffled on their own (the `relation` column; a graph without one
    is a single subgraph), so every node keeps its in-degree and out-degree within every relation
    and a customer edge never trades places with a supplier edge. Within a relation: pick two
    edges (a, b) and (c, d) and replace them with (a, d) and (c, b) unless that makes a
    self-link or duplicates an existing pair; `10 * n_edges` swaps are attempted, in the same
    order for a given `seed` (`numpy.random.default_rng([seed, relation position])`, relations
    in sorted order). A pair may appear under two relations; it is a duplicate only within one.

    A relation subgraph no swap can change (one edge, one receiver, one source, two nodes
    linked both ways) is returned unchanged and counted in the second element. Unchanged links
    make the placebo look more like the graph, which is conservative for the claim that a graph
    beats its placebo; callers must report the count. Raises ValueError for a self-link or a
    duplicate (source, receiver, relation).

    Non-node columns (weight, doc id, evidence, ...) stay attached to each original row's SOURCE
    side: a link's weight describes the source's influence, so only `receiver_node` moves. The
    result has the same columns and row count, a reset index, and is sorted by (source_node,
    receiver_node, relation) so it is reproducible. The input is not modified.
    """
    relation = graph["relation"] if "relation" in graph.columns else pd.Series(
        _NO_RELATION, index=graph.index
    )
    sources = graph["source_node"].tolist()
    receivers = graph["receiver_node"].tolist()
    triples = list(zip(sources, receivers, relation.tolist(), strict=True))
    if len(set(triples)) != len(triples):
        raise ValueError("graph has duplicate (source_node, receiver_node, relation) rows")
    if any(source == receiver for source, receiver, _ in triples):
        raise ValueError("graph has self-links")
    new_receivers = list(receivers)
    unchanged = 0
    positions_by_relation: dict[str, list[int]] = {}
    for position, name in enumerate(relation.tolist()):
        positions_by_relation.setdefault(str(name), []).append(position)
    for index, name in enumerate(sorted(positions_by_relation)):
        positions = positions_by_relation[name]
        rng = np.random.default_rng([seed, index])
        shuffled = _shuffled_receivers(
            [sources[p] for p in positions], [receivers[p] for p in positions], rng
        )
        if shuffled is None:
            unchanged += 1
            continue
        for position, receiver in zip(positions, shuffled, strict=True):
            new_receivers[position] = receiver
    result = graph.copy()
    result["receiver_node"] = new_receivers
    sort_columns = ["source_node", "receiver_node"] + (
        ["relation"] if "relation" in graph.columns else []
    )
    return result.sort_values(sort_columns, kind="stable").reset_index(drop=True), unchanged


def shuffle_preserving_degree(graph: pd.DataFrame, seed: int) -> pd.DataFrame:
    """The strict form of `shuffle_by_relation`: raises when nothing could be shuffled at all.

    A placebo graph with the same in-degree and out-degree for every node within every
    relation, and different links. Raises ValueError for a self-link, a duplicate (source,
    receiver, relation), or a graph none of whose relations a swap can change ("too small or too
    constrained to shuffle"). When only some relations cannot be shuffled they are left as they
    are; use `shuffle_by_relation` to learn how many.
    """
    shuffled, unchanged = shuffle_by_relation(graph, seed)
    n_relations = graph["relation"].nunique() if "relation" in graph.columns else 1
    if len(graph) == 0 or unchanged == n_relations:
        raise ValueError("graph is too small or too constrained to shuffle: no swap can change it")
    return shuffled
