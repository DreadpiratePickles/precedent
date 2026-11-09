"""Deterministic silent-shift detectors: a rate, a direction or a set of links that moved.

No model runs here. `event_rate_shift` tests the last `window` sessions' event count against the
preceding `baseline` sessions, `direction_drift` runs a two-sided CUSUM over a subject's
`event_direction` claims, and `link_churn` compares a subject's link set now with an earlier one.
`run_shift_detectors` runs all three for one session as of its cutoff and appends nothing to any
store; a shift's id is one per kind and subject, so a later session re-fires it in place (when it
grew, a new episode began, or a drift changed sign).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from precedent.dream.claim_schema import Shift, ShiftKind
from precedent.dream.claim_store import claims_history_asof
from precedent.dream.detectors import _standalone_recorded_utc
from precedent.dream.inputs import DreamStores, documents_per_session
from precedent.ripple.link_store import GRAPH_COLUMNS, graph_asof

SHIFT_DETECTOR_VERSION = "dream-shift-v1"
_DIRECTION_STEP = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
SKIP_NO_LINK_STORE = "no_link_store"
SKIP_SHORT_HISTORY = "short_history"
SKIP_NO_BASELINE_EVENTS = "no_baseline_events"
SKIP_NO_LIVE_SESSIONS = "no_live_sessions"
SKIP_REASONS = (
    SKIP_NO_LINK_STORE,
    SKIP_SHORT_HISTORY,
    SKIP_NO_BASELINE_EVENTS,
    SKIP_NO_LIVE_SESSIONS,
)


def _position_of(session: date, sessions: pd.DatetimeIndex) -> int:
    here = int(sessions.searchsorted(pd.Timestamp(session), side="left"))
    if here >= len(sessions) or sessions[here] != pd.Timestamp(session):
        raise ValueError(f"session {session} is not on the sessions axis")
    return here


def _shift(
    kind: ShiftKind,
    subject: str,
    session: date,
    statistic: float,
    threshold: float,
    window_sessions: int,
    recorded_utc: datetime | None,
) -> Shift:
    return Shift(
        kind=kind,
        subject=subject,
        detected_session=session,
        statistic=statistic,
        threshold=threshold,
        window_sessions=window_sessions,
        detector_version=SHIFT_DETECTOR_VERSION,
        recorded_utc=recorded_utc
        if recorded_utc is not None
        else _standalone_recorded_utc(session),
    )


def _gap_positions(coverage: pd.Series | None, sessions: pd.DatetimeIndex) -> np.ndarray:
    """True at each position of `sessions` whose coverage count is zero (a feed gap)."""
    if coverage is None:
        return np.zeros(len(sessions), dtype=bool)
    return (coverage.reindex(sessions) == 0).to_numpy()


def _window_and_baseline(here: int, window: int, baseline: int) -> tuple[range, range]:
    """Positions (here-window, here], and the `baseline` positions before them, clipped at 0."""
    return range(here - window + 1, here + 1), range(
        max(0, here - window - baseline + 1), max(0, here - window + 1)
    )


def _event_counts(events: pd.DataFrame, subject: str, sessions: pd.DatetimeIndex) -> np.ndarray:
    """The subject's event count at each position of `sessions`."""
    mine = events.loc[events["subject"] == subject, "session"]
    positions = sessions.get_indexer(pd.DatetimeIndex(mine))
    return np.bincount(positions[positions >= 0], minlength=len(sessions))


def _rate_z(
    per_position: np.ndarray, here: int, window: int, baseline: int, gap: np.ndarray
) -> tuple[float | None, str | None]:
    """The Poisson z of the window against the baseline, or None and the reason it is undefined.

    Reasons are `short_history` (fewer than `window` positions end at `here`, none precede the
    window, or fewer than `baseline // 2` live baseline sessions exist — Opus final review M12 /
    D76: a clipped baseline of a few sessions fired noisily through the first year of any
    history), `no_live_sessions` (the window or the baseline is made only of feed gaps) and
    `no_baseline_events`.
    """
    if here < window:
        return None, SKIP_SHORT_HISTORY
    window_range, baseline_range = _window_and_baseline(here, window, baseline)
    window_live = [p for p in window_range if not gap[p]]
    baseline_live = [p for p in baseline_range if not gap[p]]
    if not window_live or not baseline_live:
        return None, SKIP_NO_LIVE_SESSIONS
    if len(baseline_live) < baseline // 2:
        return None, SKIP_SHORT_HISTORY
    baseline_events = int(per_position[baseline_live].sum())
    if baseline_events == 0:
        return None, SKIP_NO_BASELINE_EVENTS
    expected = baseline_events / len(baseline_live) * len(window_live)
    return (int(per_position[window_live].sum()) - expected) / float(np.sqrt(expected)), None


def event_rate_shift(
    events: pd.DataFrame,
    subject: str,
    session: date,
    sessions: pd.DatetimeIndex,
    window: int = 60,
    baseline: int = 250,
    threshold: float = 3.0,
    coverage: pd.Series | None = None,
    *,
    recorded_utc: datetime | None = None,
) -> Shift | None:
    """A Poisson rate test of the subject's events in the last `window` sessions against the
    `baseline` sessions before them.

    `events` has `subject` and `session` (naive session dates), one row per event. Sessions with
    `coverage == 0` are feed gaps and leave both windows; the rates use the sessions that remain.
    With `lambda_b` the baseline events per session and `w` the window's remaining sessions,
    `z = (n_window - lambda_b * w) / sqrt(lambda_b * w)`; the shift fires when `z > threshold`.
    None when fewer than `window` positions end at `session`, when either window has no usable
    session, when fewer than `baseline // 2` live baseline sessions exist (a clipped baseline
    is `short_history`, not a z on a few sessions — Opus final review M12 / D76), or when the
    baseline holds no event.
    """
    here = _position_of(session, sessions)
    z, _ = _rate_z(
        _event_counts(events, subject, sessions),
        here,
        window,
        baseline,
        _gap_positions(coverage, sessions),
    )
    if z is None or z <= threshold:
        return None
    return _shift(ShiftKind.event_rate, subject, session, z, threshold, window, recorded_utc)


def direction_drift(
    claims: pd.DataFrame,
    subject: str,
    session: date,
    sessions: pd.DatetimeIndex,
    k: float = 0.5,
    h: float = 4.0,
    *,
    recorded_utc: datetime | None = None,
) -> Shift | None:
    """A two-sided CUSUM over the subject's `event_direction` claims; fires when either sum
    exceeds `h`.

    `claims` is the `claims_history_asof` frame at the session's cutoff (D71), so the sums are
    recomputed from what was recorded by then and a backfill agrees with a live run. The sequence
    is ordered by `valid_from_utc` then `claim_id` and mapped positive +1, neutral 0, negative -1.
    The CUSUM runs on each value's deviation from the mean of the values before it (the first
    deviation is 0), so a steady direction never fires and a flip does: `S+ = max(0, S+ + d - k)`
    and `S- = max(0, S- - d - k)` start at zero. `statistic` is the larger sum, negative when it is
    `S-`; `window_sessions` is the sessions from the first claim in the sequence to `session`,
    inclusive.
    """
    here = _position_of(session, sessions)
    mine = claims[(claims["subject"] == subject) & (claims["attribute"] == "event_direction")]
    if mine.empty:
        return None
    ordered = mine.assign(_valid_from=pd.to_datetime(mine["valid_from_utc"], utc=True)).sort_values(
        ["_valid_from", "claim_id"], kind="stable"
    )
    upper = lower = earlier_total = 0.0
    for count, value in enumerate(ordered["value"]):
        if value not in _DIRECTION_STEP:
            raise ValueError(f"event_direction claim for {subject!r} has unknown value {value!r}")
        step = _DIRECTION_STEP[value]
        deviation = step - earlier_total / count if count else 0.0
        upper = max(0.0, upper + deviation - k)
        lower = max(0.0, lower - deviation - k)
        earlier_total += step
    if max(upper, lower) <= h:
        return None
    first = ordered["_valid_from"].iloc[0].tz_convert("UTC").tz_localize(None).normalize()
    started = int(sessions.searchsorted(first, side="left"))
    span = max(1, here - started + 1)
    statistic = upper if upper >= lower else -lower
    return _shift(ShiftKind.direction_drift, subject, session, statistic, h, span, recorded_utc)


def _link_set(graph: pd.DataFrame, subject: str) -> set[tuple[str, str]]:
    outgoing = graph[graph["source_node"] == subject]
    incoming = graph[graph["receiver_node"] == subject]
    return set(zip(outgoing["receiver_node"], outgoing["relation"], strict=True)) | set(
        zip(incoming["source_node"], incoming["relation"], strict=True)
    )


def link_churn(
    graph_now: pd.DataFrame,
    graph_then: pd.DataFrame,
    subject: str,
    session: date,
    threshold: float = 0.5,
    *,
    baseline_sessions: int = 250,
    recorded_utc: datetime | None = None,
) -> Shift | None:
    """The Jaccard distance between the subject's link set now and in `graph_then`.

    A link is `(receiver_node, relation)` for a row the subject sources, and
    `(source_node, relation)` for a row it receives. None when both sets are empty; fires when
    the distance exceeds `threshold`. `baseline_sessions` is the gap the caller used to pick
    `graph_then` and becomes the shift's `window_sessions`.
    """
    now, then = _link_set(graph_now, subject), _link_set(graph_then, subject)
    union = now | then
    if not union:
        return None
    distance = 1.0 - len(now & then) / len(union)
    if distance <= threshold:
        return None
    return _shift(
        ShiftKind.link_churn, subject, session, distance, threshold, baseline_sessions, recorded_utc
    )


@dataclass(frozen=True)
class ShiftReport:
    """What one shift run saw and set aside, so no silence is mistaken for a clean session.

    `skipped` maps each reason in `SKIP_REASONS` to a count (zero included): `no_link_store` and
    `short_history` count detectors not run (one each; the rate test and link churn can each lack
    history), `no_baseline_events` counts subjects whose baseline held no event, and
    `no_live_sessions` counts subjects whose window or baseline was made only of feed gaps, so no
    rate was tested.
    """

    by_kind: dict[str, int]
    subjects_checked: int
    feed_gap_sessions_excluded: int
    skipped: dict[str, int]


def _events_frame(stores: DreamStores, cutoffs: pd.Series) -> pd.DataFrame:
    """One event per record and resolved instrument, on the session its `available_utc` falls in."""
    cutoff_ns = cutoffs.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    rows: list[tuple[str, pd.Timestamp]] = []
    for record in stores.event_records:
        if record.available_utc is None:
            continue
        available = pd.Timestamp(record.available_utc).tz_convert("UTC").tz_localize(None)
        position = int(np.searchsorted(cutoff_ns, available.to_datetime64(), side="left"))
        if position >= len(cutoff_ns):
            continue
        for instrument in {impact.instrument for impact in record.resolved_impacts}:
            rows.append((instrument, cutoffs.index[position]))
    return pd.DataFrame(rows, columns=["subject", "session"])


def _latest_stored(stores: DreamStores, recorded_by: datetime) -> dict[str, Shift]:
    """The last stored record per shift id that was recorded by `recorded_by`."""
    latest: dict[str, Shift] = {}
    for shift in stores.shift_store.read_all():
        if shift.recorded_utc <= recorded_by:
            latest[shift.shift_id] = shift
    return latest


def _is_new_report(
    found: Shift, stored: Shift | None, sessions: pd.DatetimeIndex, here: int, growth: float
) -> bool:
    """Whether `found` is to be reported given the last report of the same id (D73).

    A shift never reported before is reported. One that was is reported again when its statistic
    grew by more than `growth`, or when the last report is more than `window_sessions` positions
    old (a new episode), or when a `direction_drift` changed sign. All three read the store as of
    the cutoff, so no state is carried between runs.
    """
    if stored is None:
        return True
    if abs(found.statistic) > abs(stored.statistic) * (1.0 + growth):
        return True
    last = int(sessions.searchsorted(pd.Timestamp(stored.detected_session), side="left"))
    if here - last > found.window_sessions:
        return True
    return found.kind is ShiftKind.direction_drift and found.statistic * stored.statistic < 0.0


def run_shift_detectors(
    stores: DreamStores, session: date, cutoffs: pd.Series, params: dict[str, Any]
) -> tuple[list[Shift], ShiftReport]:
    """All three shift detectors for `session`, read as of its cutoff; nothing is appended.

    Sessions with no documents are feed gaps and leave the rate windows. Subjects are the
    instruments with an event in the baseline and window, the instruments with an
    `event_direction` claim, and every node of the link graph now or `baseline` sessions ago;
    churn is skipped without a link store or when fewer than `baseline` sessions precede, and the
    rate test when fewer than `window` sessions precede; `ShiftReport.skipped` counts each.
    `cutoffs` must be indexed by `stores.sessions` (ValueError otherwise). A shift
    whose id already has a stored record, recorded by this cutoff, is returned only when its
    statistic grew by more than `shift_rereport_growth` of the stored one, when that record is
    more than `window_sessions` positions old (a new episode), or when a `direction_drift`
    changed sign (D73); the caller then uses `ShiftStore.refire`.
    """
    sessions = stores.sessions
    if not cutoffs.index.equals(sessions):
        raise ValueError("cutoffs must be indexed by the same sessions as stores.sessions")
    here = _position_of(session, sessions)
    cutoff = cutoffs[pd.Timestamp(session)]
    recorded = cutoff.to_pydatetime()
    rate, drift, churn = params["event_rate"], params["direction_drift"], params["link_churn"]
    window, baseline = rate["window_sessions"], rate["baseline_sessions"]

    coverage = documents_per_session(stores.event_records, cutoffs)
    window_range, baseline_range = _window_and_baseline(here, window, baseline)
    gaps = _gap_positions(coverage, sessions)
    skipped = dict.fromkeys(SKIP_REASONS, 0)
    enough_history = here >= window
    if not enough_history:
        skipped[SKIP_SHORT_HISTORY] += 1
    excluded = sum(bool(gaps[p]) for p in (*baseline_range, *window_range)) if enough_history else 0

    events = _events_frame(stores, cutoffs)
    span = set(sessions[[*baseline_range, *window_range]]) if enough_history else set()
    rate_subjects = sorted(set(events.loc[events["session"].isin(span), "subject"]))
    # The CUSUM needs the history, not the claims still valid (an event_direction claim lives a
    # few sessions): every claim that began in the baseline span, recorded by the cutoff (D71).
    lookback_start = cutoffs.iloc[max(0, here - baseline)].to_pydatetime()
    claims = claims_history_asof(
        stores.claim_store, recorded, lookback_start, attribute="event_direction"
    )
    drift_subjects = sorted(set(claims["subject"]))

    found: list[Shift] = []
    for subject in rate_subjects:
        z, reason = _rate_z(_event_counts(events, subject, sessions), here, window, baseline, gaps)
        if reason is not None:
            skipped[reason] += 1
        elif z is not None and z > rate["z_threshold"]:
            found.append(
                _shift(
                    ShiftKind.event_rate,
                    subject,
                    session,
                    z,
                    rate["z_threshold"],
                    window,
                    recorded,
                )
            )
    for subject in drift_subjects:
        shift = direction_drift(
            claims, subject, session, sessions, drift["k"], drift["h"], recorded_utc=recorded
        )
        if shift is not None:
            found.append(shift)

    churn_subjects: list[str] = []
    if stores.link_store is None:
        skipped[SKIP_NO_LINK_STORE] += 1
    elif here < baseline:
        skipped[SKIP_SHORT_HISTORY] += 1
    else:
        graph_now = graph_asof(stores.link_store, cutoff)
        graph_then = graph_asof(stores.link_store, cutoffs.iloc[here - baseline])
        nodes = pd.concat(
            [graph[column] for graph in (graph_now, graph_then) for column in GRAPH_COLUMNS[:2]]
        )
        churn_subjects = sorted(set(nodes))
        for subject in churn_subjects:
            shift = link_churn(
                graph_now,
                graph_then,
                subject,
                session,
                churn["jaccard_threshold"],
                baseline_sessions=baseline,
                recorded_utc=recorded,
            )
            if shift is not None:
                found.append(shift)

    stored = _latest_stored(stores, recorded)
    growth = params["shift_rereport_growth"]
    reported = [
        shift
        for shift in found
        if _is_new_report(shift, stored.get(shift.shift_id), sessions, here, growth)
    ]
    by_kind = {kind.value: sum(s.kind is kind for s in reported) for kind in ShiftKind}
    subjects = set(rate_subjects) | set(drift_subjects) | set(churn_subjects)
    return reported, ShiftReport(
        by_kind=by_kind,
        subjects_checked=len(subjects),
        feed_gap_sessions_excluded=excluded,
        skipped=skipped,
    )
