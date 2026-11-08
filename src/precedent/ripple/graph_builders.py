"""Wiring pieces for the Ripple graph ladder: abnormal returns, rebuilt graphs, cached readers.

Everything here is deterministic and uses core dependencies only (numpy, pandas). The ladder
itself, which needs Qlib and LightGBM, lives in `precedent.ripple.evaluate`.

Links are still read only through `graph_asof`; the cached reader below calls it once per stretch
of cutoffs over which the store's visible set cannot change, instead of once per shock session.
"""

from __future__ import annotations

import hashlib
import json
import os
from bisect import bisect_right
from collections.abc import Callable, Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from precedent.ripple.control_graphs import (
    correlation_links,
    correlation_links_as_links,
    shuffle_by_relation,
)
from precedent.ripple.link_store import Link, LinkStore, graph_asof
from precedent.spine.abnormal_returns import abnormal_return_at

GraphForCutoff = Callable[[pd.Timestamp], pd.DataFrame]
_PLACEBO_KEY_COLUMNS = ["source_node", "receiver_node", "relation", "weight"]
_AFTER_LAST_CUTOFF = timedelta(days=1)
EVIDENCE_LABELS = ("LIVE-DATA", "SYNTHETIC")
LEDGER_MANIFEST_NAME = "MANIFEST.json"


def abnormal_return_frame(
    returns: pd.DataFrame, market: str, sessions: pd.DatetimeIndex
) -> pd.DataFrame:
    """Each instrument's market-model abnormal return on every session (the `market` is left out).

    Uses `abnormal_return_at` with the spine defaults, the same call `price_shocks` makes, so an
    event session's shock equals this frame's value there. NaN where the model cannot be fitted
    (too little history, no market variance). The result is indexed by `sessions`.
    """
    if sessions.tz is not None or not (sessions.is_unique and sessions.is_monotonic_increasing):
        raise ValueError("sessions must be a naive, strictly increasing DatetimeIndex")
    if not returns.index.equals(sessions):
        raise ValueError("returns.index must equal sessions exactly")
    if market not in returns.columns:
        raise ValueError(f"market column {market!r} is not in returns")
    market_returns = returns[market]
    columns = {
        name: [
            abnormal_return_at(returns[name], market_returns, position)
            for position in range(len(sessions))
        ]
        for name in returns.columns
        if name != market
    }
    return pd.DataFrame(columns, index=sessions, dtype="float64")


def rolling_correlation_links(
    returns: pd.DataFrame,
    market: str,
    session_cutoffs_utc: pd.Series,
    window: int,
    top_k: int,
    rebuild_every: int,
) -> tuple[list[Link], list[pd.Timestamp]]:
    """The G2 graph: residual-correlation peers, rebuilt every `rebuild_every` sessions.

    A snapshot is built at positions `window - 1`, `window - 1 + rebuild_every`, ... from the
    `window` sessions ending there (`correlation_links`), and is valid from that session's cutoff
    to the next snapshot's cutoff, so a shock on session `t` sees the snapshot built at or before
    `t` and nothing later. The last snapshot runs to one day past the last cutoff.

    Returns the links and the snapshot cutoffs (UTC), which are the instants at which the visible
    graph changes. Raises ValueError when `window` sessions never fit or `rebuild_every < 1`.
    """
    if rebuild_every < 1:
        raise ValueError(f"rebuild_every must be at least 1, got {rebuild_every}")
    if not returns.index.equals(session_cutoffs_utc.index):
        raise ValueError("session_cutoffs_utc.index must equal returns.index exactly")
    positions = list(range(window - 1, len(returns), rebuild_every))
    if not positions:
        raise ValueError(
            f"a correlation window of {window} sessions does not fit in {len(returns)} sessions"
        )
    links: list[Link] = []
    snapshot_cutoffs: list[pd.Timestamp] = []
    for index, position in enumerate(positions):
        valid_from = session_cutoffs_utc.iloc[position]
        valid_to = (
            session_cutoffs_utc.iloc[positions[index + 1]]
            if index + 1 < len(positions)
            else session_cutoffs_utc.iloc[-1] + _AFTER_LAST_CUTOFF
        )
        pairs = correlation_links(returns, market, position, window, top_k)
        links.extend(
            correlation_links_as_links(
                pairs, returns.index[position], valid_from.to_pydatetime(), valid_to.to_pydatetime()
            )
        )
        snapshot_cutoffs.append(valid_from)
    return links, snapshot_cutoffs


def snapshot_graph_for_cutoff(
    store: LinkStore, change_cutoffs_utc: Sequence[pd.Timestamp]
) -> GraphForCutoff:
    """A `graph_for_cutoff` over `store` that reads `graph_asof` once per stretch of cutoffs.

    `change_cutoffs_utc` must hold every instant at which the visible link set can change (each
    link's `valid_from_utc` and `valid_to_utc`), in increasing order. Two cutoffs with the same
    number of such instants at or before them see the same graph, so the second is served from
    the first's result. The returned frame is shared; callers must not modify it.
    """
    boundaries = list(change_cutoffs_utc)
    if boundaries != sorted(boundaries):
        raise ValueError("change_cutoffs_utc must be in increasing order")
    cache: dict[int, pd.DataFrame] = {}

    def graph_for_cutoff(cutoff_utc: pd.Timestamp) -> pd.DataFrame:
        stretch = bisect_right(boundaries, cutoff_utc)
        if stretch not in cache:
            cache[stretch] = graph_asof(store, cutoff_utc)
        return cache[stretch]

    return graph_for_cutoff


def _graph_digest(graph: pd.DataFrame) -> str:
    hashed = pd.util.hash_pandas_object(graph[_PLACEBO_KEY_COLUMNS], index=False)
    return hashlib.sha256(hashed.to_numpy().tobytes()).hexdigest()


class PlaceboGraphReader:
    """A `GraphForCutoff` returning the per-relation degree-preserving shuffle of another one.

    Each distinct graph is shuffled once (`shuffle_by_relation`) and the result is reused
    whenever the same graph comes back, so a snapshot keeps one placebo for its whole life. A
    relation subgraph no swap can change is left as it is rather than raising; the number of
    such subgraphs, over every distinct graph read so far, is `placebo_unshuffled` (read it after
    the ladder has run and report it: an unshuffled link makes the placebo look more like the
    graph, which is conservative). The returned frame is shared; callers must not modify it.
    """

    def __init__(self, graph_for_cutoff: GraphForCutoff, seed: int) -> None:
        self._graph_for_cutoff = graph_for_cutoff
        self._seed = seed
        self._cache: dict[str, pd.DataFrame] = {}
        self.placebo_unshuffled = 0

    def __call__(self, cutoff_utc: pd.Timestamp) -> pd.DataFrame:
        graph = self._graph_for_cutoff(cutoff_utc)
        if graph.empty:
            return graph
        key = _graph_digest(graph)
        if key not in self._cache:
            shuffled, unchanged = shuffle_by_relation(graph, self._seed)
            self.placebo_unshuffled += unchanged
            self._cache[key] = shuffled
        return self._cache[key]


def placebo_graph_for_cutoff(graph_for_cutoff: GraphForCutoff, seed: int) -> PlaceboGraphReader:
    """The degree-preserving, per-relation shuffle of whatever `graph_for_cutoff` returns.

    Never raises for a snapshot that cannot be shuffled (see `PlaceboGraphReader`).
    """
    return PlaceboGraphReader(graph_for_cutoff, seed)


def _write_atomically(path: Path, write: Callable[[Path], None]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        write(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_text_atomically(path: Path, text: str) -> None:
    """Write `text` through a temporary sibling and replace `path` in one step."""
    _write_atomically(path, lambda temporary: temporary.write_text(text, encoding="utf-8"))


def write_ripple_ledger(
    ripple_dir: Path,
    shocks: pd.Series,
    abnormal_returns: pd.DataFrame,
    manifest: Mapping[str, Any],
) -> Path:
    """Write `shocks.parquet`, `abnormal_returns.parquet` and `MANIFEST.json` under `ripple_dir`.

    The manifest is the caller's fields (`evidence_label`, which must be `LIVE-DATA` or
    `SYNTHETIC`, plus whatever identifies the run) and a `files` entry per parquet file with its
    SHA-256 and row count. Every file is written through a temporary sibling and replaced
    atomically, the manifest last. Returns the manifest path. Nothing is written when the
    label is invalid.
    """
    if manifest.get("evidence_label") not in EVIDENCE_LABELS:
        raise ValueError(f"manifest evidence_label must be one of {EVIDENCE_LABELS}")
    shock_table = shocks.rename("shock").reset_index()
    abnormal_table = abnormal_returns.rename_axis("datetime").reset_index()
    ripple_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, table in (
        ("shocks.parquet", shock_table),
        ("abnormal_returns.parquet", abnormal_table),
    ):
        path = ripple_dir / name
        _write_atomically(path, lambda temporary, frame=table: frame.to_parquet(temporary))
        files[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "rows": len(table)}
    manifest_path = ripple_dir / LEDGER_MANIFEST_NAME
    document = json.dumps({**manifest, "files": files}, indent=2, sort_keys=True) + "\n"
    write_text_atomically(manifest_path, document)
    return manifest_path
