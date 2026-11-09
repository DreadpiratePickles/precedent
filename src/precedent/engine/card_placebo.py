"""The ordinary range behind the card's "vs ordinary" column, split out of `engine/card_build.py`.

For one card row, what happens regardless of the event, using only data known by the query's
`available_utc`: closed placebo sessions for assets, the unconditional rate for market
resolution and event-sequence rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from precedent.engine.retrieval import PrecedentQuery
from precedent.libraries.bundle import LibraryBundle
from precedent.libraries.event_sequence import unconditional_sequence_rate
from precedent.libraries.market_resolution import unconditional_market_rate
from precedent.spine.event_key import load_country_groups

__all__ = ["ordinary_range"]


def ordinary_range(
    kind: str, outcome_key: str, horizon: int, query: PrecedentQuery, bundle: LibraryBundle
) -> np.ndarray | float | None:
    """The ordinary range for a row: what happens regardless of the event, known by `asof`."""
    if kind == "asset":
        return _asset_ordinary_range(outcome_key, horizon, query.available_utc, bundle)
    if kind == "market_resolution":
        return unconditional_market_rate(
            bundle.markets, outcome_key, query.available_utc, bundle.session_cutoffs_utc
        )
    closed = int((bundle.session_cutoffs_utc <= query.available_utc).sum())
    if closed == 0:
        return None
    return unconditional_sequence_rate(
        bundle.events,
        outcome_key.split("/")[0],
        query.key.country_group,
        horizon,
        closed - 1,
        load_country_groups(),
    )


def _asset_ordinary_range(
    outcome_key: str, horizon: int, available_utc: pd.Timestamp, bundle: LibraryBundle
) -> np.ndarray | None:
    placebo = bundle.placebo_asset
    chosen = placebo.loc[(placebo["outcome_key"] == outcome_key) & (placebo["horizon"] == horizon)]
    if chosen.empty:
        return None
    sessions = pd.DatetimeIndex(chosen["response_complete_session"])
    closed = (bundle.session_cutoffs_utc.loc[sessions] <= available_utc).to_numpy()
    values = chosen["value"].to_numpy()[closed]
    return values if values.size else None
