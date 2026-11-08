"""Macro factors from world events: a decayed direction signal and its beta-scaled form."""

import numpy as np
import pandas as pd

from precedent.spine.event_record import EventRecord

MACRO_DIRECTION_DECAY = "macro_direction_decay"
MACRO_X_BETA = "macro_x_beta"


def build_macro_factor_panel(
    macro_events: list[EventRecord],
    betas: pd.Series,
    sessions: pd.DatetimeIndex,
    half_life_sessions: float = 3.0,
) -> pd.DataFrame:
    """One row per (session, instrument in `betas.index`).

    An event effective at session s with macro direction d contributes
    `d * 0.5 ** ((pos(t) - pos(s)) / half_life_sessions)` on every session t >= s, where pos is
    the position in `sessions`; `macro_direction_decay` is the sum over events, identical for
    every instrument, and `macro_x_beta` multiplies it by the instrument's beta. Events with no
    effective session, or one that is not in `sessions`, are ignored. The value at session t uses
    only events effective on or before t.
    """
    if not half_life_sessions > 0:
        raise ValueError(f"half_life_sessions must be positive, got {half_life_sessions}")
    session_index = pd.DatetimeIndex(sessions).tz_localize(None).normalize()
    session_count = len(session_index)

    impulses = np.zeros(session_count)
    for record in macro_events:
        if record.effective_session is None:
            continue
        position = session_index.get_indexer([pd.Timestamp(record.effective_session)])[0]
        if position >= 0:
            impulses[position] += record.event.macro_direction

    offsets = np.arange(session_count)[:, None] - np.arange(session_count)[None, :]
    weights = np.where(offsets >= 0, 0.5 ** (np.maximum(offsets, 0) / half_life_sessions), 0.0)
    decay_by_session = weights @ impulses

    instruments = list(betas.index)
    beta_values = betas.to_numpy(dtype=float)
    decay = np.repeat(decay_by_session, len(instruments))
    x_beta = (decay_by_session[:, None] * beta_values[None, :]).ravel()
    index = pd.MultiIndex.from_product(
        [session_index.astype("datetime64[ns]"), instruments], names=["datetime", "instrument"]
    )
    return pd.DataFrame({MACRO_DIRECTION_DECAY: decay, MACRO_X_BETA: x_beta}, index=index)
