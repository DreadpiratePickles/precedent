"""Tests for the committed NYSE session calendar and its early-close cutoffs."""

import datetime as dt
from datetime import date

import pandas as pd
import pytest
from fakes import make_event_record

from precedent.spine.pit_alignment import align_records
from precedent.spine.session_calendar import (
    NyseCalendar,
    calendar_cutoffs,
    load_nyse_calendar,
    resolve_sessions,
    weekday_sessions,
)

CALENDAR = load_nyse_calendar()


def utc(text: str) -> dt.datetime:
    return pd.Timestamp(text, tz="UTC").to_pydatetime()


def test_thanksgiving_is_not_a_session() -> None:
    sessions = weekday_sessions(date(2024, 11, 25), date(2024, 11, 29), CALENDAR)
    assert list(sessions.strftime("%Y-%m-%d")) == [
        "2024-11-25",
        "2024-11-26",
        "2024-11-27",
        "2024-11-29",
    ]


def test_sessions_are_naive_midnight_dates() -> None:
    sessions = weekday_sessions(date(2024, 11, 25), date(2024, 11, 29), CALENDAR)
    assert isinstance(sessions, pd.DatetimeIndex)
    assert sessions.tz is None
    assert (sessions == sessions.normalize()).all()


def test_an_early_close_moves_the_cutoff() -> None:
    cutoffs = calendar_cutoffs(pd.DatetimeIndex(["2024-11-27", "2024-11-29"]), CALENDAR)
    assert list(cutoffs) == [
        pd.Timestamp(utc("2024-11-27 21:00")),
        pd.Timestamp(utc("2024-11-29 18:00")),
    ]
    assert str(cutoffs.dt.tz) == "UTC"
    assert list(cutoffs.index) == [pd.Timestamp("2024-11-27"), pd.Timestamp("2024-11-29")]


def test_news_after_a_half_day_close_rolls_to_monday() -> None:
    sessions = weekday_sessions(date(2024, 11, 25), date(2024, 12, 6), CALENDAR)
    cutoffs = calendar_cutoffs(sessions, CALENDAR)
    record = make_event_record(first_seen_utc=utc("2024-11-29 18:30"))
    aligned = align_records([record], cutoffs, latency=pd.Timedelta(0))[0]
    assert aligned.effective_session == date(2024, 12, 2)


def test_provider_sessions_win_when_present() -> None:
    provider = pd.DatetimeIndex(["2024-01-02", "2024-01-04"])
    resolved = resolve_sessions(
        date(2024, 1, 2), date(2024, 1, 4), CALENDAR, provider_sessions=provider
    )
    assert resolved.equals(provider)


def test_the_committed_calendar_is_used_when_the_provider_has_none() -> None:
    resolved = resolve_sessions(date(2024, 11, 27), date(2024, 11, 29), CALENDAR)
    assert list(resolved.strftime("%Y-%m-%d")) == ["2024-11-27", "2024-11-29"]


def test_the_calendar_refuses_years_it_does_not_cover() -> None:
    with pytest.raises(ValueError):
        weekday_sessions(date(2028, 1, 3), date(2028, 1, 7), CALENDAR)
    with pytest.raises(ValueError):
        weekday_sessions(date(2009, 12, 28), date(2010, 1, 5), CALENDAR)


def test_a_reversed_range_is_an_error() -> None:
    with pytest.raises(ValueError):
        weekday_sessions(date(2024, 1, 5), date(2024, 1, 2), CALENDAR)


def test_holiday_file_covers_2010_to_2027_on_weekdays() -> None:
    assert (CALENDAR.first_year, CALENDAR.last_year) == (2010, 2027)
    assert all(day.weekday() < 5 for day in CALENDAR.holidays)


@pytest.mark.parametrize(
    "closed",
    [
        date(2012, 10, 29),  # Hurricane Sandy
        date(2012, 10, 30),
        date(2017, 1, 2),  # New Year's Day observed
        date(2020, 7, 3),  # Independence Day observed
        date(2021, 12, 24),  # Christmas observed
        date(2022, 6, 20),  # Juneteenth observed, first year
        date(2024, 3, 29),  # Good Friday
        date(2025, 1, 9),  # national day of mourning
        date(2027, 6, 18),
    ],
)
def test_known_closures_are_holidays(closed: date) -> None:
    assert closed in CALENDAR.holidays


@pytest.mark.parametrize("open_day", [date(2021, 12, 31), date(2022, 12, 30), date(2027, 12, 31)])
def test_a_saturday_new_year_is_not_observed_on_the_friday(open_day: date) -> None:
    assert open_day not in CALENDAR.holidays


def test_juneteenth_is_not_a_holiday_before_2022() -> None:
    assert date(2021, 6, 18) not in CALENDAR.holidays
    assert date(2021, 6, 19) not in CALENDAR.holidays


def test_early_closes_are_open_weekdays_at_one_pm() -> None:
    assert CALENDAR.early_closes[date(2024, 11, 29)] == dt.time(13, 0)
    assert CALENDAR.early_closes[date(2024, 7, 3)] == dt.time(13, 0)
    assert CALENDAR.early_closes[date(2024, 12, 24)] == dt.time(13, 0)
    assert not set(CALENDAR.early_closes) & CALENDAR.holidays
    assert all(day.weekday() < 5 for day in CALENDAR.early_closes)


def test_a_calendar_is_frozen() -> None:
    with pytest.raises(ValueError):
        CALENDAR.first_year = 2000  # type: ignore[misc]


def test_a_calendar_rejects_an_early_close_on_a_holiday() -> None:
    with pytest.raises(ValueError):
        NyseCalendar(
            first_year=2024,
            last_year=2024,
            holidays=frozenset({date(2024, 11, 28)}),
            early_closes={date(2024, 11, 28): dt.time(13, 0)},
            source="test",
        )


def test_the_loader_reads_a_given_path_and_rejects_a_bad_file(tmp_path) -> None:
    good = tmp_path / "cal.json"
    good.write_text(
        '{"source": "s", "first_year": 2024, "last_year": 2024,'
        ' "holidays": ["2024-01-01"], "early_closes": {"2024-11-29": "13:00"}}'
    )
    loaded = load_nyse_calendar(good)
    assert loaded.holidays == frozenset({date(2024, 1, 1)})
    assert loaded.early_closes == {date(2024, 11, 29): dt.time(13, 0)}
    bad = tmp_path / "bad.json"
    bad.write_text(
        '{"source": "s", "first_year": 2024, "last_year": 2024, "holidays": ["2024-01-06"],'
        ' "early_closes": {}}'
    )
    with pytest.raises(ValueError):
        load_nyse_calendar(bad)


def test_cutoffs_refuse_a_session_outside_the_calendar_years() -> None:
    with pytest.raises(ValueError, match="2031-01-03"):
        calendar_cutoffs(pd.DatetimeIndex(["2031-01-03"]), CALENDAR)
    with pytest.raises(ValueError, match="2009-12-31"):
        calendar_cutoffs(pd.DatetimeIndex(["2009-12-31"]), CALENDAR)


def test_cutoffs_refuse_a_holiday_session() -> None:
    with pytest.raises(ValueError, match="2024-11-28"):
        calendar_cutoffs(pd.DatetimeIndex(["2024-11-27", "2024-11-28"]), CALENDAR)


def test_provider_sessions_are_validated_at_the_boundary() -> None:
    with pytest.raises(ValueError, match="2024-11-28"):
        resolve_sessions(
            date(2024, 11, 27),
            date(2024, 11, 29),
            CALENDAR,
            provider_sessions=pd.DatetimeIndex(["2024-11-27", "2024-11-28"]),
        )
    with pytest.raises(ValueError, match="2031-01-03"):
        resolve_sessions(
            date(2031, 1, 3),
            date(2031, 1, 3),
            CALENDAR,
            provider_sessions=pd.DatetimeIndex(["2031-01-03"]),
        )


def test_early_closes_cannot_be_mutated_through_the_calendar() -> None:
    with pytest.raises(TypeError):
        CALENDAR.early_closes[date(2024, 6, 3)] = dt.time(13, 0)  # type: ignore[index]
    assert date(2024, 6, 3) not in CALENDAR.early_closes
    source = {date(2024, 11, 29): dt.time(13, 0)}
    built = NyseCalendar(
        first_year=2024, last_year=2024, holidays=frozenset(), early_closes=source, source="t"
    )
    source[date(2024, 6, 3)] = dt.time(13, 0)
    assert date(2024, 6, 3) not in built.early_closes


def test_the_loader_names_the_file_for_a_malformed_shape(tmp_path) -> None:
    bad = tmp_path / "shape.json"
    bad.write_text(
        '{"source": "s", "first_year": 2024, "last_year": 2024, "holidays": [], "early_closes": []}'
    )
    with pytest.raises(ValueError, match="shape.json"):
        load_nyse_calendar(bad)


def test_an_early_close_never_moves_a_cutoff_later() -> None:
    cutoffs = calendar_cutoffs(
        pd.DatetimeIndex(["2024-11-29"]), CALENDAR, cutoff_local=dt.time(12, 0)
    )
    assert list(cutoffs) == [pd.Timestamp(utc("2024-11-29 17:00"))]


def test_a_weekend_session_is_refused_and_named():
    with pytest.raises(ValueError, match="weekend.*2024-01-06"):
        calendar_cutoffs(pd.DatetimeIndex(["2024-01-05", "2024-01-06"]), CALENDAR)
    with pytest.raises(ValueError, match="2024-01-07"):
        resolve_sessions(
            date(2024, 1, 5),
            date(2024, 1, 8),
            CALENDAR,
            provider_sessions=pd.DatetimeIndex(["2024-01-07", "2024-01-08"]),
        )


def test_default_cutoffs_for_caps_the_lookahead_at_the_calendars_last_year():
    from precedent.spine.session_calendar import default_cutoffs_for

    cutoffs = default_cutoffs_for([date(2027, 12, 20)], CALENDAR, 15)
    assert cutoffs.index.max() == pd.Timestamp("2027-12-31")
    inside = default_cutoffs_for([date(2024, 3, 4)], CALENDAR, 15)
    assert inside.index.max() == pd.Timestamp("2024-03-19")


def test_default_cutoffs_for_floors_the_start_at_the_calendars_first_year():
    from precedent.spine.session_calendar import default_cutoffs_for

    cutoffs = default_cutoffs_for([date(2009, 6, 1)], CALENDAR, 15)
    assert cutoffs.index.min() == pd.Timestamp("2010-01-04")
    assert cutoffs.index.max() < pd.Timestamp("2010-02-01")
    straddling = default_cutoffs_for([date(2009, 12, 30), date(2010, 1, 6)], CALENDAR, 15)
    assert straddling.index.min() == pd.Timestamp("2010-01-04")
