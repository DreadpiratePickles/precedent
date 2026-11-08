"""A planted-lead market for the research tests (research): provider bins and a news panel.

The return from the open of session s to the open of s+1 is `BETA * f[s] + noise`, where `f[s]`
is the signed news factor dated s. The label under convention C1 at session t is
`open[t+2]/open[t+1] - 1`, so it is driven by the factor dated t+1: a panel aligned to its own
session carries no signal, and the same panel shifted one session earlier leaks the label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fixtures.tiny_provider import write_tiny_provider

from precedent.spine.news_factors import build_news_factor_panel

N_SESSIONS = 260
N_INSTRUMENTS = 120
EVENT_PROBABILITY = 0.5
BETA = 0.02
NOISE_SD = 0.01
START_DAY = "2022-01-03"


@dataclass(frozen=True)
class PlantedMarket:
    sessions: pd.DatetimeIndex
    panel: pd.DataFrame


def write_planted_market(provider_dir: Path, seed: int) -> PlantedMarket:
    """Write the provider under `provider_dir` and return its sessions and aligned news panel."""
    rng = np.random.default_rng(seed)
    sessions = pd.bdate_range(START_DAY, periods=N_SESSIONS)
    names = [f"S{i:02d}" for i in range(N_INSTRUMENTS)]
    has_event = rng.random((N_SESSIONS, N_INSTRUMENTS)) < EVENT_PROBABILITY
    direction = rng.choice([-1.0, 1.0], size=(N_SESSIONS, N_INSTRUMENTS))
    rows, cols = np.nonzero(has_event)
    events = pd.DataFrame(
        {
            "effective_session": sessions[rows],
            "instrument": [names[col] for col in cols],
            "severity": 5,
            "direction": direction[rows, cols],
            "confidence_weight": 1.0,
            "novelty": "new",
        }
    )
    panel = build_news_factor_panel(events, sessions, max_age_sessions=0)
    signed = (
        panel["news_signed_1d"]
        .unstack("instrument")
        .reindex(index=sessions, columns=names)
        .fillna(0.0)
        .to_numpy()
    )
    next_return = BETA * signed + NOISE_SD * rng.standard_normal(signed.shape)
    log_open = np.log(100.0) + np.vstack(
        [np.zeros((1, N_INSTRUMENTS)), np.cumsum(np.log1p(next_return), axis=0)[:-1]]
    )
    open_price = np.exp(log_open)
    close_price = open_price * (1 + 0.005 * rng.standard_normal(signed.shape))
    write_tiny_provider(
        provider_dir,
        {
            name: pd.DataFrame(
                {"open": open_price[:, i], "close": close_price[:, i]}, index=sessions
            )
            for i, name in enumerate(names)
        },
    )
    return PlantedMarket(sessions=sessions, panel=panel)
