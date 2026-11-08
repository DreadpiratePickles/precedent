"""The `prec_` factor panel: what similar past events did, as (session, instrument) features.

For each first-report event and each instrument, the event's own query (its key, its
`available_utc`, its story) is run through `find_precedents` and the visible asset outcomes are
summarised. The cell lives for the event's own session and the next `max_age_sessions`.

Pure pandas and numpy: Qlib is not imported here, so this module is core-safe and its test needs
no `importorskip`. It belongs to the `research` extra only through its consumer, the M2 rung of
the Qlib ablation (`references/rungs/m2_precedent.yaml`).

Time: every value uses only the event's own query time, never anything from the session `s` that
carries it, so the panel is truncation-invariant by construction (design 4.5). Which instruments
get a cell is likewise read from the precedents the query may see, never from the event's own
outcome rows, which complete in the future.

`prec_placebo_z` is a stable column that this function fills with 0.0. The placebo rows live in
the bundle (`placebo_asset`) and are not an input here; a later caller that has them fills it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from precedent.engine.aggregate import summarize_responses
from precedent.engine.keys import EventKey
from precedent.engine.params import load_engine_params
from precedent.engine.retrieval import PrecedentQuery, find_precedents
from precedent.libraries.common import validate_library_rows

__all__ = ["PANEL_COLUMNS", "build_precedent_factor_panel"]

PANEL_COLUMNS = (
    "prec_car5_med",
    "prec_car5_iqr",
    "prec_neff",
    "prec_hit_rate",
    "prec_placebo_z",
)
_INDEX_NAMES = ["datetime", "instrument"]
_EVENT_COLUMNS = (
    "event_id",
    "key",
    "available_utc",
    "effective_position",
    "story_cluster_id",
    "is_first_report",
)
_CELL_COLUMNS = ("event_position", "event_id", "instrument", *PANEL_COLUMNS)


def build_precedent_factor_panel(
    events: pd.DataFrame,
    rows: pd.DataFrame,
    session_cutoffs_utc: pd.Series,
    sessions: pd.DatetimeIndex,
    horizon: int = 5,
    max_age_sessions: int = 5,
) -> pd.DataFrame:
    """Per-(session, instrument) precedent factors, from `asset` rows at `horizon` only.

    Columns, all float64: `prec_car5_med` (median of the precedents' responses),
    `prec_car5_iqr` (q0.75 - q0.25), `prec_neff`, `prec_hit_rate`, and `prec_placebo_z` (0.0
    here). A summary refused for too small an effective size gives `prec_neff` and 0.0 in the
    others. An instrument with no visible precedent gets no cell. A cell `(s, instrument)` exists
    for the sessions `s` with `0 <= position(s) - effective_position <= max_age_sessions`; when
    two events cover one cell the later `effective_position` wins, then the larger `event_id`.
    Returns an empty frame with the same index names and columns for empty inputs.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if max_age_sessions < 0:
        raise ValueError(f"max_age_sessions must be >= 0, got {max_age_sessions}")
    if sessions.tz is not None or not (sessions.is_unique and sessions.is_monotonic_increasing):
        raise ValueError("sessions must be a naive, strictly increasing DatetimeIndex")
    if events.empty or rows.empty:
        return _empty_panel()
    missing = [name for name in _EVENT_COLUMNS if name not in events.columns]
    if missing:
        raise ValueError(f"events is missing columns: {missing}")
    if events["is_first_report"].isna().any():
        raise ValueError("events has null is_first_report values")
    positions = events["effective_position"]
    if ((positions < 0) | (positions >= len(sessions))).any():
        raise ValueError(
            f"events.effective_position must lie in [0, {len(sessions)}) to index sessions"
        )
    checked_rows = validate_library_rows(rows)
    cells = _summarise_events(events, checked_rows, session_cutoffs_utc, horizon)
    if cells.empty:
        return _empty_panel()
    return _age_cells(cells, sessions, max_age_sessions)


def _summarise_events(
    events: pd.DataFrame, rows: pd.DataFrame, cutoffs: pd.Series, horizon: int
) -> pd.DataFrame:
    params = load_engine_params()
    position_of = events.set_index("event_id")["effective_position"]
    records: list[tuple] = []
    for event in events.loc[events["is_first_report"].astype(bool)].itertuples(index=False):
        story = None if pd.isna(event.story_cluster_id) else str(event.story_cluster_id)
        query = PrecedentQuery(
            key=EventKey.from_string(event.key),
            available_utc=event.available_utc,
            story_cluster_id=story,
        )
        found = find_precedents(query, events, rows, cutoffs, horizon, "asset", params.top_k)
        if found.empty:
            continue
        precedent_positions = found["event_id"].map(position_of).to_numpy()
        for instrument, group in found.groupby("outcome_key", sort=True):
            at = group.index.to_numpy()
            summary = summarize_responses(
                group["value"].to_numpy(),
                np.ones(len(group)),
                precedent_positions[at],
                horizon,
                None,
                is_binary=False,
                shrink_k=params.shrink_k,
                min_neff=params.min_neff,
                z=params.wilson_z,
            )
            if summary.refused:
                factors = (0.0, 0.0, summary.n_eff, 0.0, 0.0)
            else:
                spread = summary.quantiles[0.75] - summary.quantiles[0.25]
                factors = (summary.median, spread, summary.n_eff, summary.hit_rate, 0.0)
            records.append((int(event.effective_position), event.event_id, instrument, *factors))
    return pd.DataFrame(records, columns=list(_CELL_COLUMNS))


def _age_cells(
    cells: pd.DataFrame, sessions: pd.DatetimeIndex, max_age_sessions: int
) -> pd.DataFrame:
    ages = np.arange(max_age_sessions + 1)
    aged = cells.loc[cells.index.repeat(len(ages))].reset_index(drop=True)
    aged["position"] = aged["event_position"].to_numpy() + np.tile(ages, len(cells))
    aged = aged[aged["position"] < len(sessions)]  # the calendar ends; never extrapolate
    aged = aged.sort_values(["event_position", "event_id"], kind="stable")
    aged = aged.drop_duplicates(["position", "instrument"], keep="last")
    index = pd.MultiIndex.from_arrays(
        [
            sessions.as_unit("ns")[aged["position"].to_numpy()],
            pd.Index(aged["instrument"].to_numpy(), dtype=str),
        ],
        names=_INDEX_NAMES,
    )
    panel = aged[list(PANEL_COLUMNS)].astype("float64").set_axis(index)
    return panel.sort_index()


def _empty_panel() -> pd.DataFrame:
    index = pd.MultiIndex.from_arrays(
        [pd.DatetimeIndex([]).as_unit("ns"), pd.Index([], dtype=str)], names=_INDEX_NAMES
    )
    return pd.DataFrame({name: pd.Series(dtype="float64") for name in PANEL_COLUMNS}).set_axis(
        index
    )
