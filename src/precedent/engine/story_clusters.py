"""Story clusters: group repeat reports of one event so a story counts once, at its first report."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = ["assign_story_clusters"]

_REQUIRED_COLUMNS = ("event_id", "key", "primary_actor", "effective_position", "available_utc")
_NOT_NULL_COLUMNS = ("key", "primary_actor", "effective_position")
_SORT_ORDER = ["effective_position", "available_utc", "event_id"]


def assign_story_clusters(events: pd.DataFrame, window_sessions: int = 3) -> pd.DataFrame:
    """Add `story_cluster_id` and `is_first_report` to `events`, keeping row order and columns.

    A cluster is a run of rows with the same `key` and `primary_actor`, ordered by
    `effective_position`, then `available_utc`, then `event_id`, in which each row is at most
    `window_sessions` positions after the previous one. The cluster id is the `event_id` of its
    first row in that order, and only that row has `is_first_report` True.

    Works on row positions, so any index (including duplicate labels) is accepted and preserved.
    Raises ValueError for missing columns, a negative `window_sessions`, or a null in `key`,
    `primary_actor` or `effective_position`.
    """
    missing = sorted(set(_REQUIRED_COLUMNS) - set(events.columns))
    if missing:
        raise ValueError(f"events is missing required columns: {', '.join(missing)}")
    if window_sessions < 0:
        raise ValueError(f"window_sessions must be >= 0, got {window_sessions}")
    for column in _NOT_NULL_COLUMNS:
        if events[column].isna().any():
            raise ValueError(f"events column {column!r} contains null values")

    out = events.copy()
    if out.empty:
        out["story_cluster_id"] = pd.Series(dtype=object)
        out["is_first_report"] = pd.Series(dtype=bool)
        return out

    work = events[list(_REQUIRED_COLUMNS)].reset_index(drop=True)
    work["row_position"] = np.arange(len(work))
    ordered = work.sort_values(_SORT_ORDER, kind="stable")
    group_keys = [ordered["key"], ordered["primary_actor"]]
    gap = ordered.groupby(group_keys, sort=False)["effective_position"].diff()
    starts_cluster = gap.isna() | (gap > window_sessions)
    cluster_number = starts_cluster.groupby(group_keys, sort=False).cumsum()
    first_report_id = (
        ordered["event_id"]
        .where(starts_cluster)
        .groupby([*group_keys, cluster_number], sort=False)
        .ffill()
    )

    positions = ordered["row_position"].to_numpy()
    cluster_ids = np.empty(len(work), dtype=object)
    cluster_ids[positions] = first_report_id.to_numpy()
    is_first = np.zeros(len(work), dtype=bool)
    is_first[positions] = starts_cluster.to_numpy()

    out["story_cluster_id"] = cluster_ids
    out["is_first_report"] = is_first
    return out
