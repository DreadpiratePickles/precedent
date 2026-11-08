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


def shuffle_preserving_degree(graph: pd.DataFrame, seed: int) -> pd.DataFrame:
    """A placebo graph: same out-degree and in-degree for every node, different links.

    Repeated edge swaps: pick two edges (a, b) and (c, d) and replace them with (a, d) and
    (c, b) unless that makes a self-link or duplicates an existing pair. Exactly
    `10 * n_edges` swaps are attempted, in the same order for a given `seed`
    (`numpy.random.default_rng(seed)`). Raises ValueError when the input has a self-link or a
    duplicate pair, or when the edge set is unchanged afterwards (graph too small to shuffle).

    Non-node columns (weight, doc id, evidence, ...) stay attached to each original row's
    SOURCE side: a link's weight describes the source's influence, so only `receiver_node`
    moves. The result has the same columns and row count, a reset index, and is sorted by
    (source_node, receiver_node) so it is reproducible. The input is not modified.
    """
    sources = graph["source_node"].tolist()
    receivers = graph["receiver_node"].tolist()
    original = set(zip(sources, receivers, strict=True))
    if len(original) != len(sources):
        raise ValueError("graph has duplicate (source_node, receiver_node) pairs")
    if any(source == receiver for source, receiver in original):
        raise ValueError("graph has self-links")
    n_edges = len(sources)
    if n_edges < 2:
        raise ValueError("graph is too small to shuffle")
    edges = set(original)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_edges, size=(_SWAP_ATTEMPTS_PER_EDGE * n_edges, 2))
    for first, second in picks.tolist():
        if first == second:
            continue
        a, b = sources[first], receivers[first]
        c, d = sources[second], receivers[second]
        if a == d or c == b or (a, d) in edges or (c, b) in edges:
            continue
        edges.difference_update(((a, b), (c, d)))
        edges.update(((a, d), (c, b)))
        receivers[first], receivers[second] = d, b
    if edges == original:
        raise ValueError("graph is too small to shuffle: no degree-preserving swap changed it")
    shuffled = graph.copy()
    shuffled["receiver_node"] = receivers
    return shuffled.sort_values(["source_node", "receiver_node"], kind="stable").reset_index(
        drop=True
    )
