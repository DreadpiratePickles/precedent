"""Point-in-time alignment: when could an event first be used, and for which decision session."""

import datetime as dt

import numpy as np
import pandas as pd

from precedent.spine.event_record import EventRecord


def _as_utc_ns(values: pd.Series, name: str) -> pd.Series:
    """Require timezone-aware timestamps and return them as UTC at nanosecond resolution."""
    if not isinstance(values.dtype, pd.DatetimeTZDtype):
        raise ValueError(f"{name} must hold timezone-aware timestamps, got dtype {values.dtype}")
    return values.dt.tz_convert("UTC").dt.as_unit("ns")


def compute_available_utc(
    published_utc: pd.Series, first_seen_utc: pd.Series, latency: pd.Timedelta
) -> pd.Series:
    """Return max(published, first_seen) + latency; a missing publisher time uses first_seen."""
    if latency < pd.Timedelta(0):
        raise ValueError(
            "latency must be >= 0 (a negative latency would use news before it was seen), "
            f"got {latency}"
        )
    published = _as_utc_ns(published_utc, "published_utc")
    first_seen = _as_utc_ns(first_seen_utc, "first_seen_utc")
    if not published.index.equals(first_seen.index):
        raise ValueError("published_utc and first_seen_utc must have the same index")
    n_missing_first_seen = int(first_seen.isna().sum())
    if n_missing_first_seen:
        raise ValueError(
            f"first_seen_utc has {n_missing_first_seen} missing values; "
            "a publisher clock alone is not trusted"
        )
    published = published.fillna(first_seen)
    latest_clock = published.where(published > first_seen, first_seen)
    return (latest_clock + latency).rename("available_utc")


def build_session_cutoffs(
    sessions: pd.DatetimeIndex, cutoff_local: dt.time, tz: str = "America/New_York"
) -> pd.Series:
    """Return, per naive session date, the UTC instant of the local cutoff (DST-correct)."""
    if sessions.tz is not None:
        raise ValueError("sessions must be naive session dates, got a timezone-aware index")
    if cutoff_local.tzinfo is not None:
        raise ValueError("cutoff_local must be a naive wall-clock time; the timezone comes from tz")
    if not (sessions == sessions.normalize()).all():
        raise ValueError("sessions must be dates at midnight, not timestamps with a time of day")
    if not (sessions.is_unique and sessions.is_monotonic_increasing):
        raise ValueError("sessions must be strictly increasing")
    time_of_day = pd.Timedelta(
        hours=cutoff_local.hour,
        minutes=cutoff_local.minute,
        seconds=cutoff_local.second,
        microseconds=cutoff_local.microsecond,
    )
    local_cutoffs = (sessions + time_of_day).tz_localize(tz, ambiguous="raise", nonexistent="raise")
    cutoffs_utc = local_cutoffs.tz_convert("UTC").as_unit("ns")
    return pd.Series(
        cutoffs_utc, index=pd.DatetimeIndex(sessions, name="session_date"), name="cutoff_utc"
    )


def assign_effective_session(available_utc: pd.Series, session_cutoffs_utc: pd.Series) -> pd.Series:
    """Return the first session whose cutoff is >= the available time; NaT after the last cutoff."""
    available = _as_utc_ns(available_utc, "available_utc")
    cutoffs = _as_utc_ns(session_cutoffs_utc, "session_cutoffs_utc")
    if len(cutoffs) == 0:
        raise ValueError("session_cutoffs_utc is empty")
    if not isinstance(cutoffs.index, pd.DatetimeIndex) or cutoffs.index.tz is not None:
        raise ValueError("session_cutoffs_utc must be indexed by naive session dates")
    if cutoffs.isna().any():
        raise ValueError("session_cutoffs_utc contains missing cutoffs")
    cutoff_ns = cutoffs.dt.tz_localize(None).to_numpy()
    if not (np.diff(cutoff_ns) > np.timedelta64(0)).all():
        raise ValueError("session_cutoffs_utc must be strictly increasing")
    n_missing = int(available.isna().sum())
    if n_missing:
        raise ValueError(
            f"available_utc has {n_missing} missing values; they cannot be assigned to a session"
        )
    position = np.searchsorted(cutoff_ns, available.dt.tz_localize(None).to_numpy(), side="left")
    session_ns = cutoffs.index.as_unit("ns").to_numpy()
    effective = np.full(len(position), np.datetime64("NaT"), dtype="datetime64[ns]")
    has_session = position < len(cutoff_ns)
    effective[has_session] = session_ns[position[has_session]]
    return pd.Series(effective, index=available.index, name="effective_session")


def align_records(
    records: list[EventRecord], session_cutoffs_utc: pd.Series, latency: pd.Timedelta
) -> list[EventRecord]:
    """Return copies of the records with `available_utc` and `effective_session` filled in.

    Inputs are not mutated and order is preserved. `effective_session` is None when the event
    became available after the last cutoff, so it belongs to no session in this calendar.
    """
    if not records:
        return []
    published = pd.Series(pd.to_datetime([r.published_utc for r in records], utc=True))
    first_seen = pd.Series(pd.to_datetime([r.first_seen_utc for r in records], utc=True))
    available = compute_available_utc(published, first_seen, latency)
    sessions = assign_effective_session(available, session_cutoffs_utc)
    return [
        record.model_copy(
            update={
                "available_utc": available_at.to_pydatetime(),
                "effective_session": None if pd.isna(session) else session.date(),
            }
        )
        for record, available_at, session in zip(records, available, sessions, strict=True)
    ]
