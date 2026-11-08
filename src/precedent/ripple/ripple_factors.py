"""Research: the Ripple factor panel, a sparse (datetime, instrument) frame for Qlib.

A shock on session `t` travels along the links visible at `t`'s cutoff, one channel at a time
(where the shock comes from), then ages over later sessions. Links of relation
`country_exposure` are ignored here: they have no factor column in this milestone and are shown
only by the daily map.
"""

from collections.abc import Callable

import numpy as np
import pandas as pd

from precedent.ripple.link_store import Relation
from precedent.ripple.propagate import build_adjacency, propagate_shock

PANEL_COLUMNS = [
    "ripple_customer",
    "ripple_supplier",
    "ripple_peer",
    "ripple_commodity",
    "ripple_hop2",
    "ripple_unabsorbed",
]
CHANNEL_OF_RELATION = {
    Relation.CUSTOMER_OF.value: "ripple_customer",
    Relation.SUPPLIER_OF.value: "ripple_supplier",
    Relation.PEER_OF.value: "ripple_peer",
    Relation.INPUT_COMMODITY.value: "ripple_commodity",
    Relation.OUTPUT_COMMODITY.value: "ripple_commodity",
}
_SKIPPED_RELATIONS = {Relation.COUNTRY_EXPOSURE.value}  # no factor column in this milestone
_CHANNEL_COLUMNS = ["ripple_customer", "ripple_supplier", "ripple_peer", "ripple_commodity"]
_DECAYED_COLUMNS = [*_CHANNEL_COLUMNS, "ripple_hop2"]
_INDEX_NAMES = ["datetime", "instrument"]


def _validate_inputs(
    shocks: pd.Series,
    session_cutoffs_utc: pd.Series,
    sessions: pd.DatetimeIndex,
    abnormal_returns: pd.DataFrame,
    half_life_sessions: float,
    max_age_sessions: int,
) -> np.ndarray:
    """Check every input and return each shock's integer position in `sessions`."""
    if not half_life_sessions > 0:
        raise ValueError(f"half_life_sessions must be > 0, got {half_life_sessions}")
    if max_age_sessions < 0:
        raise ValueError(f"max_age_sessions must be >= 0, got {max_age_sessions}")
    if sessions.tz is not None or not (sessions.is_unique and sessions.is_monotonic_increasing):
        raise ValueError("sessions must be a naive, strictly increasing DatetimeIndex")
    if not session_cutoffs_utc.index.equals(sessions):
        raise ValueError("session_cutoffs_utc.index must equal sessions exactly")
    if not isinstance(session_cutoffs_utc.dtype, pd.DatetimeTZDtype):
        raise ValueError("session_cutoffs_utc must hold tz-aware timestamps")
    if not abnormal_returns.index.equals(sessions):
        raise ValueError("abnormal_returns.index must equal sessions exactly")
    if np.isinf(abnormal_returns.to_numpy(dtype=np.float64)).any():
        raise ValueError("abnormal_returns must be finite or NaN (NaN counts as zero)")
    if list(shocks.index.names) != _INDEX_NAMES:
        raise ValueError(
            f"shocks index must be named {_INDEX_NAMES}, got {list(shocks.index.names)}"
        )
    if getattr(shocks.index.get_level_values("datetime"), "tz", None) is not None:
        raise ValueError("shocks datetime level must be naive")
    if shocks.index.has_duplicates:
        raise ValueError("shocks has duplicate (datetime, instrument) rows")
    positions = sessions.as_unit("ns").get_indexer(
        pd.DatetimeIndex(shocks.index.get_level_values("datetime")).as_unit("ns")
    )
    if (positions < 0).any():
        bad = shocks.index.get_level_values("datetime")[positions < 0][0]
        raise ValueError(f"shock datetime {bad} is not in sessions")
    return positions


def _empty_panel() -> pd.DataFrame:
    empty_index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]).as_unit("ns"), pd.Index([], dtype=str)], names=_INDEX_NAMES
    )
    return pd.DataFrame({name: pd.Series(dtype="float64") for name in PANEL_COLUMNS}).set_axis(
        empty_index
    )


def _session_terms(
    graph: pd.DataFrame,
    instruments: list[str],
    shock_values: list[float],
    alpha: float,
    hops: int,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Per-receiver terms for one session's shocks on one graph.

    Returns the node list, an (n_nodes, 5) array in `_DECAYED_COLUMNS` order and each node's
    propagated total (every channel's hop 1 plus the summed hop 2).
    """
    nodes = sorted(set(instruments) | set(graph["source_node"]) | set(graph["receiver_node"]))
    position = {name: index for index, name in enumerate(nodes)}
    shock = np.zeros(len(nodes))
    for instrument, value in zip(instruments, shock_values, strict=True):
        shock[position[instrument]] = value
    terms = np.zeros((len(nodes), len(_DECAYED_COLUMNS)))
    unknown = sorted(set(graph["relation"]) - set(CHANNEL_OF_RELATION) - _SKIPPED_RELATIONS)
    if unknown:
        raise ValueError(f"graph has unknown relation values: {unknown}")
    channel_of_link = graph["relation"].map(CHANNEL_OF_RELATION)
    for column_index, channel in enumerate(_CHANNEL_COLUMNS):
        rows = graph[channel_of_link == channel]
        if rows.empty:
            continue
        adjacency, _ = build_adjacency(rows, nodes)
        hop1, hop2 = propagate_shock(adjacency, shock, alpha, hops)
        terms[:, column_index] = hop1
        terms[:, -1] += hop2
    return nodes, terms, terms.sum(axis=1)


def build_ripple_factor_panel(
    shocks: pd.Series,
    graph_for_cutoff: Callable[[pd.Timestamp], pd.DataFrame],
    session_cutoffs_utc: pd.Series,
    sessions: pd.DatetimeIndex,
    abnormal_returns: pd.DataFrame,
    alpha: float = 0.5,
    hops: int = 2,
    half_life_sessions: float = 3.0,
    max_age_sessions: int = 20,
) -> pd.DataFrame:
    """Propagate each session's shocks over that session's graph and age the result.

    `shocks` is indexed (`datetime`, `instrument`) with naive session labels. For the shocks of
    session `t`, `graph_for_cutoff(session_cutoffs_utc[t])` is the only graph used. Per channel
    (see `CHANNEL_OF_RELATION`) the hop-1 term goes to that channel's column; the hop-2 terms of
    all channels are summed into `ripple_hop2`. On session `s = t + age` (age 0 to
    `max_age_sessions`, never past the calendar) a receiver's five propagated columns are
    multiplied by `0.5 ** (age / half_life_sessions)`. `ripple_unabsorbed` is not decayed: it is
    the receiver's propagated total minus its own cumulative abnormal return over sessions
    after `t` up to and including `s`. A NaN abnormal return, or an instrument with no column in
    `abnormal_returns`, counts as zero in that cumulative sum. Shocks reaching the same
    (session, receiver) are summed column by column, so `ripple_unabsorbed` then subtracts the
    receiver's cumulative abnormal return once per shock, each measured from its own `t + 1`.
    `abnormal_returns` may hold NaN but not infinity (ValueError). A receiver whose terms are
    all zero gets no row, and the shocked source itself appears only if it also receives
    something.
    """
    positions = _validate_inputs(
        shocks,
        session_cutoffs_utc,
        sessions,
        abnormal_returns,
        half_life_sessions,
        max_age_sessions,
    )
    if shocks.empty:
        return _empty_panel()
    cumulative_abnormal = abnormal_returns.fillna(0.0).cumsum()
    calendar = sessions.as_unit("ns")
    ages = np.arange(max_age_sessions + 1)
    instruments = shocks.index.get_level_values("instrument").astype(str)
    pieces: list[pd.DataFrame] = []
    for shock_position in np.unique(positions):
        selected = positions == shock_position
        graph = graph_for_cutoff(session_cutoffs_utc.iloc[shock_position])
        nodes, terms, propagated_total = _session_terms(
            graph,
            list(instruments[selected]),
            shocks.to_numpy(dtype=np.float64)[selected].tolist(),
            alpha,
            hops,
        )
        active = np.flatnonzero((terms != 0.0).any(axis=1))
        if active.size == 0:
            continue
        receivers = [nodes[index] for index in active]
        ages_kept = ages[shock_position + ages < len(sessions)]  # never extrapolate past the end
        target = shock_position + ages_kept
        weight = 0.5 ** (ages_kept / half_life_sessions)
        base_cumulative = cumulative_abnormal.iloc[shock_position]
        for node_index, receiver in zip(active, receivers, strict=True):
            if receiver in cumulative_abnormal.columns:
                realised = (
                    cumulative_abnormal[receiver].to_numpy()[target] - base_cumulative[receiver]
                )
            else:
                realised = np.zeros(len(target))
            block = np.empty((len(target), len(PANEL_COLUMNS)))
            block[:, : len(_DECAYED_COLUMNS)] = np.outer(weight, terms[node_index])
            block[:, -1] = propagated_total[node_index] - realised
            pieces.append(
                pd.DataFrame(
                    block,
                    index=pd.MultiIndex.from_arrays(
                        [calendar[target], pd.Index([receiver] * len(target), dtype=str)],
                        names=_INDEX_NAMES,
                    ),
                    columns=PANEL_COLUMNS,
                )
            )
    if not pieces:
        return _empty_panel()
    panel = pd.concat(pieces).groupby(_INDEX_NAMES).sum()
    return panel[PANEL_COLUMNS].astype("float64").sort_index()
