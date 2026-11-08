"""Story clusters: group repeat reports of one event so a story counts once, at its first report."""

from __future__ import annotations

import pandas as pd

__all__ = ["assign_story_clusters"]

_REQUIRED_COLUMNS = ("event_id", "key", "primary_actor", "effective_position", "available_utc")
_SORT_ORDER = ["effective_position", "available_utc", "event_id"]


def assign_story_clusters(events: pd.DataFrame, window_sessions: int = 3) -> pd.DataFrame:
    """Add `story_cluster_id` and `is_first_report` to `events`, keeping row order and columns.

    A cluster is a run of rows with the same `key` and `primary_actor`, ordered by
    `effective_position`, then `available_utc`, then `event_id`, in which each row is at most
    `window_sessions` positions after the previous one. The cluster id is the `event_id` of its
    first row in that order, and only that row has `is_first_report` True.
    """
    missing = sorted(set(_REQUIRED_COLUMNS) - set(events.columns))
    if missing:
        raise ValueError(f"events is missing required columns: {', '.join(missing)}")

    out = events.copy()
    if out.empty:
        out["story_cluster_id"] = pd.Series(dtype=object)
        out["is_first_report"] = pd.Series(dtype=bool)
        return out

    ordered = out.sort_values(_SORT_ORDER, kind="stable")
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

    out["story_cluster_id"] = first_report_id.reindex(out.index)
    out["is_first_report"] = starts_cluster.reindex(out.index).astype(bool)
    return out
