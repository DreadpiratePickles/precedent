"""The time-guarded panel a structural fit reads: one row per past event the query may see.

`build_guarded_panel` asks the Precedent engine's `find_precedents` ONCE; that call applies both
time guards (the event was available by the query, and the session that completes its response
closed by then), so nothing here reads the library rows for selection. Everything the fit sees is
therefore knowable at `query.available_utc`.

This module imports the engine at module level. `cli.py` imports it lazily, so the mechanism CLI
still starts without the Precedent track installed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace

import numpy as np
import pandas as pd

from precedent.engine.retrieval import PrecedentQuery, find_precedents
from precedent.mechanism.fit_sem import GuardedPanel
from precedent.spine.event_key import EventKey

__all__ = ["PanelError", "build_guarded_panel", "mechanism_query"]

_EVENT_PREFIX = "event"
_DEFAULT_FIELD = "direction_sign"
_SEVERITY_ORDINAL = {"low": 0, "mid": 1, "high": 2}
_EVENT_FIELDS: dict[str, Callable[[EventKey], int]] = {
    "direction_sign": lambda key: key.direction_sign,
    "severity_band": lambda key: _SEVERITY_ORDINAL[key.severity_band],
    "is_scheduled": lambda key: int(key.is_scheduled),
}


class PanelError(ValueError):
    """A guarded panel that cannot be built; the message names what was missing or inconsistent."""


def mechanism_query(
    key: EventKey, available_utc: pd.Timestamp, story_cluster_id: str | None = None
) -> PrecedentQuery:
    """The query the structural fit is built from: `key` with `direction_sign` set to 0.

    The engine matches `direction_sign` exactly whenever the query's is non-zero, which would
    leave the direction column of the panel constant and its slope unidentifiable. With 0 the
    fit sees both signs; `severity_band` and `is_scheduled` stay matched, so their panel columns
    are constant (and, centred, carry a ridge slope of exactly zero). Precedent's own band keeps
    the exact query key; only the mechanism fit uses this query.

    `available_utc` must be timezone-aware (a naive time raises `ValueError`); it is converted
    to UTC.
    """
    if available_utc is not pd.NaT and available_utc.tzinfo is not None:
        available_utc = available_utc.tz_convert("UTC")
    return PrecedentQuery(
        key=replace(key, direction_sign=0),
        available_utc=available_utc,
        story_cluster_id=story_cluster_id,
    )


def build_guarded_panel(
    query: PrecedentQuery,
    events: pd.DataFrame,
    rows: pd.DataFrame,
    session_cutoffs_utc: pd.Series,
    returns: pd.DataFrame,
    market: str,
    assets: Sequence[str],
    event_node_ids: Sequence[str],
    horizon: int,
    top_k: int,
) -> GuardedPanel:
    """Pivot the precedents `find_precedents` lets `query` see into a `GuardedPanel`.

    `events` and `rows` are the engine's library pair; `assets` and `event_node_ids` are the
    graph's asset and event node ids, so the panel's columns are exactly the graph's. The
    panel's rows are the distinct `event_id`s `find_precedents` returned, in its order; a
    selected event with no completed asset row is absent (the engine drops it) and is not
    re-added.

    Asset columns: every asset other than `market` holds the library's value, a cumulative
    abnormal return against the market model (NaN where the library has no row, an all-NaN
    column for an asset with none). The library has no row for the market (its abnormal return
    against itself is zero), so when `market` is among `assets` its column is the market's OWN
    cumulative return: the sum of `returns[market]` over sessions `p + 2 .. p + 1 + horizon`
    for the event at `effective_position` p, the window the asset library uses. It is NaN if any
    of those returns is NaN or the window runs past the end of `returns`; never bridged, never
    zero. `returns` must be indexed by the session axis of `session_cutoffs_utc`, the axis
    `effective_position` indexes.

    Event columns: an id `event:<event_type>:<field>` for `direction_sign` (int), `severity_band`
    (ordinal low 0, mid 1, high 2) or `is_scheduled` (0/1), read from each precedent's key. A
    bare `event:<event_type>` id means `direction_sign`. Any other field, another event type, or
    a malformed id raises `PanelError`.

    No precedent passing the guard raises `PanelError` rather than returning an empty panel.
    """
    if horizon < 1:
        raise PanelError(f"horizon must be at least 1, got {horizon}")
    asset_list, event_list = _distinct(assets, "asset"), _distinct(event_node_ids, "event node")
    fields = [_event_field(node_id, query.key.event_type) for node_id in event_list]
    _check_returns(returns, session_cutoffs_utc, market)
    found = find_precedents(query, events, rows, session_cutoffs_utc, horizon, "asset", top_k=top_k)
    event_ids = list(dict.fromkeys(found["event_id"]))
    if not event_ids:
        raise PanelError(f"no precedent passed the time guard for {query.key.as_string()}")
    asset_returns = _asset_values(found, event_ids, asset_list)
    event_rows = events.set_index("event_id").loc[event_ids]
    if market in asset_list:
        asset_returns[market] = _market_returns(
            event_rows["effective_position"].astype("int64").to_numpy(),
            returns[market].to_numpy(dtype=float),
            session_cutoffs_utc,
            query.available_utc,
            horizon,
        )
    keys = [_parse_key(event_id, text) for event_id, text in event_rows["key"].items()]
    event_values = pd.DataFrame(
        {
            node_id: pd.Series([_EVENT_FIELDS[field](key) for key in keys], dtype="int64")
            for node_id, field in zip(event_list, fields, strict=True)
        }
    )
    event_values.index = pd.Index(event_ids, name="event_id", dtype=object)
    asset_returns.index = event_values.index
    return GuardedPanel(
        event_values=event_values,
        asset_returns=asset_returns,
        horizon=horizon,
        asof_utc=query.available_utc.tz_convert("UTC"),
    )


def _distinct(names: Sequence[str], what: str) -> list[str]:
    listed = list(names)
    repeated = sorted({n for n in listed if listed.count(n) > 1})
    if repeated:
        raise PanelError(f"duplicate {what} ids {repeated}")
    return listed


def _event_field(node_id: str, event_type: str) -> str:
    """The field an event column id names; the id must be this query's event type."""
    parts = node_id.split(":")
    if parts[0] != _EVENT_PREFIX or len(parts) not in (2, 3):
        raise PanelError(
            f"{node_id!r} is not an event column id (event:<event_type> or "
            "event:<event_type>:<field>)"
        )
    if parts[1] != event_type:
        raise PanelError(
            f"event column {node_id!r} is not for the query's event type {event_type!r}"
        )
    field = parts[2] if len(parts) == 3 else _DEFAULT_FIELD
    if field not in _EVENT_FIELDS:
        raise PanelError(
            f"event column {node_id!r} names unsupported field {field!r}; "
            f"supported: {sorted(_EVENT_FIELDS)}"
        )
    return field


def _check_returns(returns: pd.DataFrame, session_cutoffs_utc: pd.Series, market: str) -> None:
    if market not in returns.columns:
        raise PanelError(f"returns has no market column {market!r}")
    if not returns.index.equals(session_cutoffs_utc.index):
        raise PanelError(
            "returns must be indexed by the session axis of session_cutoffs_utc: returns has "
            f"{len(returns)} rows, session_cutoffs_utc has {len(session_cutoffs_utc)}"
        )


def _asset_values(found: pd.DataFrame, event_ids: list[str], assets: list[str]) -> pd.DataFrame:
    duplicated = found.duplicated(["event_id", "outcome_key"], keep=False)
    if duplicated.any():
        first = found.loc[duplicated].iloc[0]
        raise PanelError(
            f"library has more than one asset row for event {first['event_id']!r} and "
            f"asset {first['outcome_key']!r} at the panel horizon"
        )
    wide = found.pivot(index="event_id", columns="outcome_key", values="value")
    return wide.reindex(index=event_ids, columns=assets).astype("float64")


def _market_returns(
    positions: np.ndarray,
    market_returns: np.ndarray,
    session_cutoffs_utc: pd.Series,
    available_utc: pd.Timestamp,
    horizon: int,
) -> np.ndarray:
    """Sum of market returns over `p + 2 .. p + 1 + horizon` per event; NaN when unobserved."""
    out = np.full(len(positions), np.nan)
    for i, position in enumerate(positions):
        last = int(position) + 1 + horizon
        if last >= len(market_returns):
            continue  # the window runs past the end of returns
        if session_cutoffs_utc.iloc[last] > available_utc:
            raise PanelError(
                f"market window of the event at position {int(position)} completes after the "
                f"query: the session axis and the library disagree (session {last})"
            )
        window = market_returns[int(position) + 2 : last + 1]
        out[i] = np.nan if np.isnan(window).any() else float(window.sum())
    return out


def _parse_key(event_id: str, text: str) -> EventKey:
    try:
        return EventKey.from_string(text)
    except ValueError as error:
        raise PanelError(f"event {event_id!r} has an unreadable key: {error}") from error
