"""Events to a sparse (datetime, instrument) factor panel.

A row at session d uses only events with effective_session <= d.
"""

import numpy as np
import pandas as pd

from precedent.spine.event_schema import Novelty

EVENT_COLUMNS = [
    "effective_session",
    "instrument",
    "severity",
    "direction",
    "confidence_weight",
    "novelty",
]
PANEL_COLUMNS = [
    "news_cnt_1d",
    "news_sev_max_1d",
    "news_signed_1d",
    "news_novel_share_1d",
    "news_signed_decay",
    "news_attention_decay",
]
_VALUE_RANGES = {"severity": (1, 10), "direction": (-1.0, 1.0), "confidence_weight": (0.0, 1.0)}


def _validate_events(events: pd.DataFrame, sessions: pd.DatetimeIndex) -> np.ndarray:
    """Check the events table and return each event's integer position in `sessions`."""
    missing_columns = [name for name in EVENT_COLUMNS if name not in events.columns]
    if missing_columns:
        raise ValueError(f"events is missing columns: {missing_columns}")
    n_missing = int(events[EVENT_COLUMNS].isna().sum().sum())
    if n_missing:
        raise ValueError(f"events has {n_missing} missing values in {EVENT_COLUMNS}")
    for name, (low, high) in _VALUE_RANGES.items():
        if not events[name].between(low, high).all():
            raise ValueError(f"events.{name} must lie in [{low}, {high}]")
    if not pd.api.types.is_integer_dtype(events["severity"]):
        raise ValueError(
            f"events.severity must be an integer column, got {events['severity'].dtype}"
        )
    unknown_novelty = set(events["novelty"]) - {member.value for member in Novelty}
    if unknown_novelty:
        raise ValueError(f"events.novelty has unknown values: {sorted(unknown_novelty)}")
    effective = events["effective_session"]
    if not pd.api.types.is_datetime64_dtype(effective) or effective.dt.tz is not None:
        raise ValueError("events.effective_session must be naive datetime64")
    position = sessions.as_unit("ns").get_indexer(pd.DatetimeIndex(effective).as_unit("ns"))
    n_bad = int((position < 0).sum())
    if n_bad:
        raise ValueError(
            f"{n_bad} of {len(events)} events have an effective_session that is not in sessions"
        )
    return position


def build_news_factor_panel(
    events: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    half_life_sessions: float = 3.0,
    max_age_sessions: int = 20,
) -> pd.DataFrame:
    """Aggregate events into per-(session, instrument) factors, ageing in session positions."""
    if not half_life_sessions > 0:
        raise ValueError(f"half_life_sessions must be > 0, got {half_life_sessions}")
    if max_age_sessions < 0:
        raise ValueError(f"max_age_sessions must be >= 0, got {max_age_sessions}")
    if sessions.tz is not None or not (sessions.is_unique and sessions.is_monotonic_increasing):
        raise ValueError("sessions must be a naive, strictly increasing DatetimeIndex")
    position = _validate_events(events, sessions)
    index_names = ["datetime", "instrument"]
    if events.empty:
        empty_index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex([]).as_unit("ns"), pd.Index([], dtype=str)], names=index_names
        )
        return pd.DataFrame({name: pd.Series(dtype="float64") for name in PANEL_COLUMNS}).set_axis(
            empty_index
        )

    attention = (events["severity"] / 10 * events["confidence_weight"]).to_numpy()
    per_event = pd.DataFrame(
        {
            "instrument": events["instrument"].to_numpy(),
            "position": position,
            "severity": events["severity"].to_numpy(),
            "signed": events["direction"].to_numpy() * attention,
            "attention": attention,
            "is_new": (events["novelty"] == Novelty.new.value).to_numpy(),
        }
    )
    ages = np.arange(max_age_sessions + 1)
    aged = per_event.loc[per_event.index.repeat(len(ages))].reset_index(drop=True)
    aged["age"] = np.tile(ages, len(per_event))
    aged["position"] = aged["position"] + aged["age"]
    aged = aged[
        aged["position"] < len(sessions)
    ].copy()  # the calendar ends; never extrapolate past it
    weight = 0.5 ** (aged["age"].to_numpy() / half_life_sessions)
    aged["signed_decay"] = aged["signed"] * weight
    aged["attention_decay"] = aged["attention"] * weight
    aged["datetime"] = sessions.as_unit("ns")[aged["position"].to_numpy()]

    decayed = aged.groupby(index_names).agg(
        news_signed_decay=("signed_decay", "sum"), news_attention_decay=("attention_decay", "sum")
    )
    same_day = (
        aged[aged["age"] == 0]
        .groupby(index_names)
        .agg(
            news_cnt_1d=("signed", "size"),
            news_sev_max_1d=("severity", "max"),
            news_signed_1d=("signed", "sum"),
            news_novel_share_1d=("is_new", "mean"),
        )
    )
    panel = decayed.join(same_day, how="left").fillna(0.0)
    return panel[PANEL_COLUMNS].astype("float64").sort_index()
