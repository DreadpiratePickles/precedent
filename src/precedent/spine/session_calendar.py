"""The committed NYSE session calendar: holidays, early closes and per-session cutoffs.

The data lives in `references/nyse_holidays.json`. A price provider's own session list wins when
there is one (`resolve_sessions`); the committed calendar is the fallback and the check.
"""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from precedent.config import references_dir
from precedent.spine.pit_alignment import build_session_cutoffs

_HOLIDAY_FILE = "nyse_holidays.json"


class NyseCalendar(BaseModel):
    """Holidays and early closes for first_year..last_year; immutable once loaded."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)
    first_year: int
    last_year: int
    holidays: frozenset[date]
    early_closes: Mapping[date, dt.time]
    source: str

    @field_validator("early_closes")
    @classmethod
    def _freeze_early_closes(cls, early_closes: Mapping[date, dt.time]) -> Mapping[date, dt.time]:
        return MappingProxyType(dict(early_closes))

    @model_validator(mode="after")
    def _check_consistency(self) -> NyseCalendar:
        if self.first_year > self.last_year:
            raise ValueError(f"first_year {self.first_year} is after last_year {self.last_year}")
        for name, days in (("holiday", self.holidays), ("early close", self.early_closes)):
            for day in days:
                if day.weekday() >= 5:
                    raise ValueError(f"{name} {day} is not a weekday")
                if not self.first_year <= day.year <= self.last_year:
                    raise ValueError(f"{name} {day} is outside {self.first_year}-{self.last_year}")
        clashes = sorted(set(self.early_closes) & self.holidays)
        if clashes:
            raise ValueError(f"early close on a holiday: {clashes[0]}")
        return self


def load_nyse_calendar(path: Path | None = None) -> NyseCalendar:
    """Read and validate the holiday file (default `references/nyse_holidays.json`)."""
    file_path = path if path is not None else references_dir() / _HOLIDAY_FILE
    try:
        raw: dict[str, Any] = json.loads(file_path.read_text(encoding="utf-8"))
        return NyseCalendar(
            first_year=raw["first_year"],
            last_year=raw["last_year"],
            holidays=frozenset(date.fromisoformat(day) for day in raw["holidays"]),
            early_closes={
                date.fromisoformat(day): dt.time.fromisoformat(close)
                for day, close in raw["early_closes"].items()
            },
            source=raw["source"],
        )
    except (OSError, KeyError, TypeError, AttributeError, ValueError) as error:
        raise ValueError(f"cannot load NYSE calendar from {file_path}: {error}") from error


def check_known_sessions(sessions: pd.DatetimeIndex, calendar: NyseCalendar) -> None:
    """Raise ValueError naming any session outside the calendar's years or on a holiday."""
    outside = [day for day in sessions if not calendar.first_year <= day.year <= calendar.last_year]
    if outside:
        raise ValueError(
            f"sessions outside the calendar's years {calendar.first_year}-{calendar.last_year}: "
            f"{_format_days(outside)}"
        )
    on_holiday = [day for day in sessions if day.date() in calendar.holidays]
    if on_holiday:
        raise ValueError(f"sessions that are NYSE holidays: {_format_days(on_holiday)}")


def _format_days(days: list[pd.Timestamp]) -> str:
    shown = ", ".join(day.strftime("%Y-%m-%d") for day in days[:5])
    return shown + (f" (and {len(days) - 5} more)" if len(days) > 5 else "")


def weekday_sessions(start: date, end: date, calendar: NyseCalendar) -> pd.DatetimeIndex:
    """Weekdays from start to end inclusive, minus holidays, as naive midnight timestamps."""
    if start > end:
        raise ValueError(f"start {start} is after end {end}")
    if start.year < calendar.first_year or end.year > calendar.last_year:
        raise ValueError(
            f"{start} to {end} is outside the calendar's years "
            f"{calendar.first_year}-{calendar.last_year}"
        )
    days = pd.bdate_range(start, end)
    return days[~days.isin([pd.Timestamp(day) for day in calendar.holidays])]


def calendar_cutoffs(
    sessions: pd.DatetimeIndex,
    calendar: NyseCalendar,
    cutoff_local: dt.time = dt.time(16, 0),
    tz: str = "America/New_York",
) -> pd.Series:
    """UTC cutoff per session; an early-close session uses its early close if that is earlier.

    Raises ValueError for a session outside the calendar's years or on a holiday, so an
    unknown day never silently gets a regular-close cutoff.
    """
    check_known_sessions(sessions, calendar)
    cutoffs = build_session_cutoffs(sessions, cutoff_local, tz)
    for day, early_close in calendar.early_closes.items():
        stamp = pd.Timestamp(day)
        if stamp in cutoffs.index:
            early = build_session_cutoffs(
                pd.DatetimeIndex([stamp]), min(early_close, cutoff_local), tz
            )
            cutoffs.loc[stamp] = early.iloc[0]
    return cutoffs


def resolve_sessions(
    start: date,
    end: date,
    calendar: NyseCalendar,
    provider_sessions: pd.DatetimeIndex | None = None,
) -> pd.DatetimeIndex:
    """The price provider's sessions when given, else the committed calendar's.

    Provider sessions are checked against the calendar's years and holidays (ValueError)."""
    if provider_sessions is not None:
        check_known_sessions(provider_sessions, calendar)
        return provider_sessions
    return weekday_sessions(start, end, calendar)
