"""Tests for point-in-time alignment of asynchronous events to market sessions."""
import datetime as dt

import numpy as np
import pandas as pd
import pytest

from pit_alignment import assign_effective_session, build_session_cutoffs, compute_available_utc

NY = "America/New_York"
CUTOFF_1600 = dt.time(16, 0)


def ny(text: str) -> pd.Timestamp:
    """New York wall-clock text -> UTC instant."""
    return pd.Timestamp(text, tz=NY).tz_convert("UTC")


def utc_series(*instants: pd.Timestamp | None) -> pd.Series:
    return pd.Series(pd.DatetimeIndex(list(instants), tz="UTC"))


# Fri 2024-01-05 and Mon 2024-01-08 are consecutive sessions; the weekend has none.
JAN_SESSIONS = pd.DatetimeIndex(["2024-01-04", "2024-01-05", "2024-01-08", "2024-01-09"])
JAN_CUTOFFS = build_session_cutoffs(JAN_SESSIONS, CUTOFF_1600)


def effective_for(*local_times: str) -> list:
    available = utc_series(*[ny(t) for t in local_times])
    return list(assign_effective_session(available, JAN_CUTOFFS))


def test_friday_before_and_at_the_cutoff_belong_to_friday() -> None:
    friday = pd.Timestamp("2024-01-05")
    assert effective_for("2024-01-05 15:59:00", "2024-01-05 16:00:00") == [friday, friday]


def test_one_second_after_the_cutoff_and_weekend_roll_to_monday() -> None:
    monday = pd.Timestamp("2024-01-08")
    assert effective_for("2024-01-05 16:00:01", "2024-01-06 12:00:00", "2024-01-07 23:59:59") == [monday] * 3


def test_cutoffs_are_dst_correct() -> None:
    sessions = pd.DatetimeIndex(["2024-03-08", "2024-03-11", "2024-11-01", "2024-11-04"])
    cutoffs = build_session_cutoffs(sessions, CUTOFF_1600)
    assert list(cutoffs.index) == list(sessions)
    assert cutoffs.dt.tz is not None and str(cutoffs.dt.tz) == "UTC"
    expected = ["2024-03-08 21:00", "2024-03-11 20:00", "2024-11-01 20:00", "2024-11-04 21:00"]
    assert [c.strftime("%Y-%m-%d %H:%M") for c in cutoffs] == expected


def test_nonexistent_local_cutoff_is_an_error_not_a_guess() -> None:
    with pytest.raises(ValueError):
        build_session_cutoffs(pd.DatetimeIndex(["2024-03-10"]), dt.time(2, 30))


def test_latency_can_push_an_event_across_the_cutoff() -> None:
    seen = utc_series(ny("2024-01-05 15:50:00"))
    available = compute_available_utc(seen, seen, pd.Timedelta(minutes=15))
    assert available.iloc[0] == ny("2024-01-05 16:05:00")
    assert assign_effective_session(available, JAN_CUTOFFS).iloc[0] == pd.Timestamp("2024-01-08")


def test_publisher_clock_later_than_first_seen_is_honoured() -> None:
    first_seen = utc_series(ny("2024-01-05 10:00:00"))
    published = utc_series(ny("2024-01-05 12:00:00"))
    assert compute_available_utc(published, first_seen, pd.Timedelta(0)).iloc[0] == ny("2024-01-05 12:00:00")


def test_publisher_clock_earlier_than_first_seen_does_not_move_the_event_earlier() -> None:
    first_seen = utc_series(ny("2024-01-05 17:00:00"))
    published = utc_series(ny("2024-01-05 09:00:00"))
    assert compute_available_utc(published, first_seen, pd.Timedelta(0)).iloc[0] == ny("2024-01-05 17:00:00")


def test_missing_published_time_uses_first_seen() -> None:
    first_seen = utc_series(ny("2024-01-05 17:00:00"))
    published = utc_series(None)
    assert compute_available_utc(published, first_seen, pd.Timedelta(0)).iloc[0] == ny("2024-01-05 17:00:00")


def test_missing_first_seen_raises() -> None:
    with pytest.raises(ValueError, match="first_seen"):
        compute_available_utc(utc_series(ny("2024-01-05 09:00:00")), utc_series(None), pd.Timedelta(0))


def test_naive_timestamps_raise() -> None:
    naive = pd.Series(pd.to_datetime(["2024-01-05 09:00:00"]))
    aware = utc_series(ny("2024-01-05 09:00:00"))
    with pytest.raises(ValueError, match="timezone"):
        compute_available_utc(naive, aware, pd.Timedelta(0))
    with pytest.raises(ValueError, match="timezone"):
        compute_available_utc(aware, naive, pd.Timedelta(0))
    with pytest.raises(ValueError, match="timezone"):
        assign_effective_session(naive, JAN_CUTOFFS)


def test_negative_latency_and_mismatched_indexes_raise() -> None:
    both = utc_series(ny("2024-01-05 09:00:00"))
    with pytest.raises(ValueError, match="latency"):
        compute_available_utc(both, both, pd.Timedelta(minutes=-1))
    with pytest.raises(ValueError, match="index"):
        compute_available_utc(both, both.rename_axis("x").set_axis([7]), pd.Timedelta(0))


def test_event_after_the_last_cutoff_gets_nat() -> None:
    result = effective_for("2024-01-09 16:00:00", "2024-01-09 16:00:01")
    assert result[0] == pd.Timestamp("2024-01-09")
    assert pd.isna(result[1])


def test_nat_available_time_raises_instead_of_landing_on_the_first_session() -> None:
    with pytest.raises(ValueError, match="missing"):
        assign_effective_session(utc_series(None), JAN_CUTOFFS)


def test_non_increasing_cutoffs_raise() -> None:
    bad = JAN_CUTOFFS.iloc[[0, 2, 1, 3]]
    with pytest.raises(ValueError, match="strictly increasing"):
        assign_effective_session(utc_series(ny("2024-01-05 10:00:00")), bad)
    with pytest.raises(ValueError, match="strictly increasing"):
        build_session_cutoffs(pd.DatetimeIndex(["2024-01-05", "2024-01-05"]), CUTOFF_1600)


def test_random_events_land_on_the_first_session_whose_cutoff_is_not_before_them() -> None:
    rng = np.random.default_rng(7)
    sessions = pd.bdate_range("2024-02-26", "2024-04-05")  # spans the 2024-03-10 DST change
    cutoffs = build_session_cutoffs(sessions, CUTOFF_1600)
    start = ny("2024-02-25 00:00:00")
    offsets_s = rng.integers(0, int(pd.Timedelta(days=45).total_seconds()), size=2000)
    random_instants = pd.Series(start + pd.to_timedelta(offsets_s, unit="s"))
    one_second = pd.Timedelta(seconds=1)
    boundary_instants = pd.concat([cutoffs - one_second, cutoffs, cutoffs + one_second], ignore_index=True)
    available = pd.concat([random_instants, boundary_instants], ignore_index=True)
    effective = assign_effective_session(available, cutoffs)
    for available_at, session in zip(available, effective, strict=True):
        if pd.isna(session):
            assert available_at > cutoffs.iloc[-1]
            continue
        position = cutoffs.index.get_loc(session)
        assert cutoffs.iloc[position] >= available_at
        if position > 0:
            assert cutoffs.iloc[position - 1] < available_at
