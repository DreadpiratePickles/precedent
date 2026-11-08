"""End-to-end on synthetic events: timestamps -> available time -> effective session -> factor panel."""
import datetime as dt

import pandas as pd

from news_factors import build_news_factor_panel
from pit_alignment import assign_effective_session, build_session_cutoffs, compute_available_utc

SESSIONS = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])


def ny_utc(text: str) -> pd.Timestamp:
    return pd.Timestamp(text, tz="America/New_York").tz_convert("UTC")


def test_event_seen_after_the_friday_cutoff_is_invisible_in_the_friday_row() -> None:
    seen = pd.Series(pd.DatetimeIndex([ny_utc("2024-01-05 15:59:00"), ny_utc("2024-01-05 16:00:01"), ny_utc("2024-01-06 11:00:00")]))
    published = pd.Series(pd.DatetimeIndex([ny_utc("2024-01-05 09:00:00")] * 3))  # an early publisher clock must not help
    available = compute_available_utc(published, seen, pd.Timedelta(0))
    effective = assign_effective_session(available, build_session_cutoffs(SESSIONS, dt.time(16, 0)))
    events = pd.DataFrame(
        {
            "effective_session": effective,
            "instrument": "AAA",
            "severity": 6,
            "direction": 1.0,
            "confidence_weight": 0.9,
            "novelty": "new",
        }
    )
    panel = build_news_factor_panel(events, SESSIONS)
    assert panel.loc[(pd.Timestamp("2024-01-05"), "AAA"), "news_cnt_1d"] == 1
    assert panel.loc[(pd.Timestamp("2024-01-08"), "AAA"), "news_cnt_1d"] == 2
    assert pd.Timestamp("2024-01-04") not in panel.index.get_level_values("datetime")
