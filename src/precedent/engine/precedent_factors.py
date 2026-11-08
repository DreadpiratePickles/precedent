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

`prec_placebo_z` is (median - mean placebo) / sd placebo, per (instrument, horizon), over the
bundle's `placebo_asset` rows (the optional `placebo_asset` input). A placebo row counts for an
event only if the cutoff of its `response_complete_session` is at or before the event's
`available_utc`, the same guard `find_precedents` applies to precedents. It is 0.0 with no
placebo input, fewer than 2 usable values, or a zero spread; the sd is the sample sd (ddof=1).

Cost: one `find_precedents` call per first-report event, each of which scores every distinct key
string, so the build is O(first-report events x distinct keys). Not optimised; a precomputed
key-match cache is the obvious later step if a real library makes it slow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from precedent.engine.aggregate import summarize_responses
from precedent.engine.keys import EventKey
from precedent.engine.params import load_engine_params
from precedent.engine.retrieval import PrecedentQuery, find_precedents
from precedent.libraries.bundle import PLACEBO_COLUMNS
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
    *,
    placebo_asset: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-(session, instrument) precedent factors, from `asset` rows at `horizon` only.

    Columns, all float64: `prec_car5_med` (median of the precedents' responses),
    `prec_car5_iqr` (q0.75 - q0.25), `prec_neff`, `prec_hit_rate`, and `prec_placebo_z` (see
    the module docstring; `placebo_asset` has the bundle's `PLACEBO_COLUMNS`). A summary refused
    for too small an effective size gives `prec_neff` and 0.0 in the others. An instrument with
    no visible precedent gets no cell. A cell `(s, instrument)` exists for the sessions `s` with
    `0 <= position(s) - effective_position <= max_age_sessions`; when two events cover one cell
    the later `effective_position` wins, then the larger `event_id`. Returns an empty frame with
    the same index names and columns for empty inputs, after the same column checks.
    """
    if horizon < 1:
        raise ValueError(f"horizon must be at least 1, got {horizon}")
    if max_age_sessions < 0:
        raise ValueError(f"max_age_sessions must be >= 0, got {max_age_sessions}")
    if sessions.tz is not None or not (sessions.is_unique and sessions.is_monotonic_increasing):
        raise ValueError("sessions must be a naive, strictly increasing DatetimeIndex")
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
    placebo = _placebo_by_instrument(placebo_asset, session_cutoffs_utc, horizon)
    if events.empty or checked_rows.empty:
        return _empty_panel()
    cells = _summarise_events(events, checked_rows, session_cutoffs_utc, horizon, placebo)
    if cells.empty:
        return _empty_panel()
    return _age_cells(cells, sessions, max_age_sessions)


def _placebo_by_instrument(
    placebo_asset: pd.DataFrame | None, cutoffs: pd.Series, horizon: int
) -> dict[str, tuple[pd.DatetimeIndex, np.ndarray]]:
    """Placebo values at `horizon` per instrument, with the cutoff of each completing session."""
    if placebo_asset is None:
        return {}
    missing = [name for name in PLACEBO_COLUMNS if name not in placebo_asset.columns]
    if missing:
        raise ValueError(f"placebo_asset is missing columns: {missing}")
    if placebo_asset[list(PLACEBO_COLUMNS)].isna().any().any():
        raise ValueError("placebo_asset has null values")
    values = placebo_asset["value"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("placebo_asset has non-finite values")
    at_horizon = placebo_asset.loc[placebo_asset["horizon"] == horizon]
    if at_horizon.empty:
        return {}
    sessions = pd.to_datetime(at_horizon["response_complete_session"])
    unknown = sessions[~sessions.isin(cutoffs.index)]
    if not unknown.empty:
        raise KeyError(
            f"placebo_asset response_complete_session {unknown.iloc[0].date()} "
            "is not in session_cutoffs_utc"
        )
    completes = pd.DatetimeIndex(cutoffs.loc[sessions.to_numpy()])
    out: dict[str, tuple[pd.DatetimeIndex, np.ndarray]] = {}
    for instrument, at in at_horizon.groupby("outcome_key", sort=True).indices.items():
        out[str(instrument)] = (completes[at], at_horizon["value"].to_numpy(dtype=float)[at])
    return out


def _placebo_z(
    median: float,
    available_utc: pd.Timestamp,
    placebo: tuple[pd.DatetimeIndex, np.ndarray] | None,
) -> float:
    if placebo is None:
        return 0.0
    completes, values = placebo
    usable = values[np.asarray(completes <= available_utc)]
    if usable.size < 2:
        return 0.0
    spread = float(np.std(usable, ddof=1))
    if spread == 0.0:
        return 0.0
    return float((median - np.mean(usable)) / spread)


def _summarise_events(
    events: pd.DataFrame,
    rows: pd.DataFrame,
    cutoffs: pd.Series,
    horizon: int,
    placebo: dict[str, tuple[pd.DatetimeIndex, np.ndarray]],
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
                z_score = _placebo_z(
                    summary.median, event.available_utc, placebo.get(str(instrument))
                )
                factors = (summary.median, spread, summary.n_eff, summary.hit_rate, z_score)
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
