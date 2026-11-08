"""Deterministic tension detectors: claims that disagree, or sit against the price.

No model runs here. Each detector reads frames the stores already produce (`claims_asof`,
`graph_asof`, an abnormal-returns panel) and returns tensions for one session; `run_detectors`
stamps them with the session and its cutoff but appends nothing to any store. A tension's id is
a hash of its kind and the group it is about (D72), so the same tension found on two sessions is
one record, first opened by the earlier session, whichever claims make it up on each.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass
from datetime import date, datetime
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from precedent import config
from precedent.dream.claim_schema import Tension, TensionKind, claim_id_for, link_id_for
from precedent.dream.claim_store import claims_asof
from precedent.dream.inputs import DreamStores, documents_per_session
from precedent.ripple.link_store import GRAPH_COLUMNS, graph_asof
from precedent.spine.session_calendar import (
    NyseCalendar,
    default_cutoffs_for,
    load_nyse_calendar,
)

DETECTOR_VERSION = "dream-detect-v1"
ASYMMETRIC_LINK_NOTE = (
    "ASC 280 does not require naming the customer; weak evidence, scored separately"
)
_MIN_MARKET_VARIANCE = 1e-18  # the floor `abnormal_returns` applies to the market's variance
_DIRECTION_SIGN = {"positive": 1.0, "negative": -1.0}
_EXCLUDED_FROM_CONFLICT = "event_direction"
_LINK_CLAIM_ATTRIBUTE = {
    "supplier_of": "supplier_relationship",
    "customer_of": "customer_relationship",
}


@lru_cache(maxsize=1)
def _conflicting_pairs() -> frozenset[tuple[str, str]]:
    """Both orders of every pair in `claim_vocabulary.json["conflicting_value_pairs"]`."""
    path = config.references_dir() / "dream" / "claim_vocabulary.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))["conflicting_value_pairs"]
        pairs = {(str(a), str(b)) for a, b in raw}
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cannot read conflicting_value_pairs from {path}: {error!r}") from error
    return frozenset(pairs | {(b, a) for a, b in pairs})


@lru_cache(maxsize=1)
def _calendar() -> NyseCalendar:
    return load_nyse_calendar()


def _standalone_recorded_utc(session: date) -> datetime:
    """Stand-in `recorded_utc` for a detector called outside `run_detectors`: the session's
    cutoff on the committed NYSE calendar, so a tension is never recorded before its evidence.

    A date that is not a calendar session (a weekend, a holiday, a year the calendar does not
    cover) raises ValueError; the caller passes `recorded_utc=` for such a date.
    """
    day = pd.Timestamp(session).date()
    try:
        cutoffs = default_cutoffs_for([day], _calendar(), 0)
        return cutoffs.loc[pd.Timestamp(day)].to_pydatetime()
    except (KeyError, ValueError) as error:
        raise ValueError(
            f"{day} is not a session on the committed NYSE calendar; pass recorded_utc= "
            "for a date the calendar does not know"
        ) from error


def _tension(
    kind: TensionKind,
    group_key: str,
    claim_ids: list[str],
    subject: str,
    session: date,
    recorded_utc: datetime | None,
    details: dict[str, float | str],
) -> Tension:
    return Tension(
        kind=kind,
        group_key=group_key,
        claim_ids=sorted(claim_ids),
        subject=subject,
        opened_session=session,
        detector_version=DETECTOR_VERSION,
        details=details,
        recorded_utc=recorded_utc
        if recorded_utc is not None
        else _standalone_recorded_utc(session),
    )


def conflict_group_key(subject: str, attribute: str, period: str) -> str:
    return f"{subject}|{attribute}|{period}"


def asymmetric_group_key(source_node: str, receiver_node: str) -> str:
    return f"{source_node}|customer_of|{receiver_node}"


def detect_value_conflicts(
    claims: pd.DataFrame, session: date, *, recorded_utc: datetime | None = None
) -> list[Tension]:
    """One tension per (subject, attribute, period) holding claims from different documents
    whose values form a conflicting pair. A document never conflicts with itself, and a claim
    in another period or about another subject is not compared.

    `event_direction` takes no part (D72): the sign of a story is compared with the price by
    `text_vs_price` and followed over time by `direction_drift`, and opposite-sign news of
    unrelated event types about one name is not a conflict. The tension is identified by its
    group, so a later document joining the group changes its members, not its identity.
    """
    pairs = _conflicting_pairs()
    tensions: list[Tension] = []
    claims = claims[claims["attribute"] != _EXCLUDED_FROM_CONFLICT]
    for (subject, attribute, period), group in claims.groupby(
        ["subject", "attribute", "period"], sort=True
    ):
        members = list(zip(group["claim_id"], group["value"], group["source_doc_id"], strict=True))
        present = sorted({value for _, value, _ in members})
        involved: set[str] = set()
        values: set[str] = set()
        for value_a, value_b in itertools.combinations(present, 2):
            if (value_a, value_b) not in pairs:
                continue
            # Both values stand in the group; the pair conflicts when the claims that hold either
            # one come from at least two documents. Every such claim is on a conflicting side.
            sides = [(i, d) for i, value, d in members if value in (value_a, value_b)]
            if len({doc for _, doc in sides}) >= 2:
                involved.update(claim_id for claim_id, _ in sides)
                values.update((value_a, value_b))
        if involved:
            details: dict[str, float | str] = {
                "attribute": str(attribute),
                "period": str(period),
                "values": ",".join(sorted(values)),
            }
            tensions.append(
                _tension(
                    TensionKind.value_conflict,
                    conflict_group_key(str(subject), str(attribute), str(period)),
                    list(involved),
                    str(subject),
                    session,
                    recorded_utc,
                    details,
                )
            )
    return tensions


def detect_text_vs_price(
    claims: pd.DataFrame,
    abnormal_returns: pd.DataFrame,
    session: date,
    z_threshold: float = 2.0,
    *,
    recorded_utc: datetime | None = None,
) -> list[Tension]:
    """A positive or negative `event_direction` claim whose subject moved the other way.

    `abnormal_returns` is indexed (session, instrument) with `abnormal_return` and
    `residual_sd`; the claim fires when the return has the opposite sign and
    `abs(return / residual_sd) >= z_threshold`. A neutral claim, a missing row, a NaN and a
    zero residual deviation never fire.
    """
    if abnormal_returns.index.nlevels != 2:
        raise ValueError("abnormal_returns must be indexed by (session, instrument)")
    active = claims[
        (claims["attribute"] == "event_direction") & claims["value"].isin(list(_DIRECTION_SIGN))
    ]
    if active.empty:
        return []
    try:
        day = abnormal_returns.xs(pd.Timestamp(session), level=0)
    except KeyError:
        return []
    rows = day[["abnormal_return", "residual_sd"]].to_dict("index")
    tensions: list[Tension] = []
    for claim_id, subject, value in zip(
        active["claim_id"], active["subject"], active["value"], strict=True
    ):
        row = rows.get(subject)
        if row is None:
            continue
        abnormal, deviation = float(row["abnormal_return"]), float(row["residual_sd"])
        if not (np.isfinite(abnormal) and np.isfinite(deviation) and deviation > 0.0):
            continue
        z = abnormal / deviation
        if abnormal * _DIRECTION_SIGN[value] < 0.0 and abs(z) >= z_threshold:
            details: dict[str, float | str] = {
                "claim_value": str(value),
                "abnormal_return": abnormal,
                "z": z,
                "residual_sd": deviation,
            }
            tensions.append(
                _tension(
                    TensionKind.text_vs_price,
                    claim_id,
                    [claim_id],
                    str(subject),
                    session,
                    recorded_utc,
                    details,
                )
            )
    return sorted(tensions, key=lambda t: (t.subject, t.claim_ids))


def abnormal_returns_panel(
    returns: pd.DataFrame,
    market: str,
    sessions: pd.DatetimeIndex,
    est_window: tuple[int, int] = (-250, -21),
    min_obs: int = 120,
) -> pd.DataFrame:
    """Abnormal return and residual deviation for every session and non-market column.

    Alpha and beta are fitted by OLS over positions `pos + est_window[0]` to
    `pos + est_window[1]` (clipped at 0; the window `abnormal_return_at` uses), `abnormal_return`
    is `r - alpha - beta * market` at `pos`, and `residual_sd` is the window residuals' standard
    deviation (ddof=1). NaN where fewer than `min_obs` finite pairs or the market has no variance.
    The frame is indexed (session, instrument).
    """
    est_start, est_stop = est_window
    if not est_start <= est_stop < 0:
        raise ValueError(f"est_window must satisfy start <= stop < 0, got {est_window}")
    if min_obs < 2:
        raise ValueError(f"min_obs must be at least 2, got {min_obs}")
    if market not in returns.columns:
        raise ValueError(f"market column {market!r} is not in returns")
    if returns.index.intersection(sessions).empty:
        raise ValueError("returns and sessions have no session in common; the panel would be empty")
    aligned = returns.reindex(sessions).astype(float)
    x_all = aligned[market].to_numpy()
    assets = aligned.drop(columns=[market])
    y_all = assets.to_numpy()
    seen = np.isfinite(y_all) & np.isfinite(x_all)[:, None]
    x = np.where(seen, x_all[:, None], 0.0)
    y = np.where(seen, y_all, 0.0)
    width = est_stop - est_start + 1

    def window_sum(values: np.ndarray) -> np.ndarray:
        # Row `pos` holds the sum over positions pos+est_start..pos+est_stop (clipped at row 0).
        rolled = pd.DataFrame(values).rolling(width, min_periods=1).sum()
        return rolled.shift(-est_stop).to_numpy()

    count = window_sum(seen.astype(float))
    sum_x, sum_y = window_sum(x), window_sum(y)
    sum_xy, sum_xx, sum_yy = window_sum(x * y), window_sum(x * x), window_sum(y * y)
    with np.errstate(divide="ignore", invalid="ignore"):
        variance = sum_xx / count - (sum_x / count) ** 2
        beta = (sum_xy / count - sum_x * sum_y / count**2) / variance
        alpha = sum_y / count - beta * sum_x / count
        abnormal = y_all - (alpha + beta * x_all[:, None])
        residual_ss = (sum_yy - sum_y**2 / count) - beta * (sum_xy - sum_x * sum_y / count)
        deviation = np.sqrt(np.maximum(residual_ss, 0.0) / (count - 1))
    usable = (count >= min_obs) & (variance >= _MIN_MARKET_VARIANCE)
    index = pd.MultiIndex.from_product([sessions, assets.columns], names=["session", "instrument"])
    return pd.DataFrame(
        {
            "abnormal_return": np.where(usable, abnormal, np.nan).ravel(),
            "residual_sd": np.where(usable, deviation, np.nan).ravel(),
        },
        index=index,
    )


def detect_asymmetric_links(
    graph: pd.DataFrame,
    claims: pd.DataFrame,
    session: date,
    *,
    recorded_utc: datetime | None = None,
) -> tuple[list[Tension], int]:
    """A `customer_of` link A to B with no `supplier_of` link B to A in the graph.

    One tension per ordered pair (D72); its claim is the relationship claim of the link now in
    the graph for the pair, so a restating filing changes the members, not the tension. A link
    whose claim is not in `claims` is skipped, and the count of those skipped is returned beside
    the tensions.
    """
    reciprocal = {
        (row.source_node, row.receiver_node)
        for row in graph[graph["relation"] == "supplier_of"].itertuples()
    }
    known_claims = set(claims["claim_id"])
    tensions: list[Tension] = []
    skipped_without_claim = 0
    for row in graph[graph["relation"] == "customer_of"].itertuples():
        if (row.receiver_node, row.source_node) in reciprocal:
            continue
        link_subject = link_id_for(
            row.source_node, row.receiver_node, "customer_of", row.source_doc_id
        )
        claim_id = claim_id_for(
            link_subject, "customer_relationship", "unspecified", "ongoing", row.source_doc_id
        )
        if claim_id not in known_claims:
            skipped_without_claim += 1
            continue
        details: dict[str, float | str] = {
            "source_node": row.source_node,
            "receiver_node": row.receiver_node,
            "note": ASYMMETRIC_LINK_NOTE,
        }
        pair = asymmetric_group_key(row.source_node, row.receiver_node)
        tensions.append(
            _tension(
                TensionKind.asymmetric_link, pair, [claim_id], pair, session, recorded_utc, details
            )
        )
    return tensions, skipped_without_claim


def detect_stale_reliance(
    graph: pd.DataFrame,
    session: date,
    sessions: pd.DatetimeIndex,
    max_age_sessions: int = 300,
    *,
    recorded_utc: datetime | None = None,
) -> list[Tension]:
    """A supplier or customer link more than `max_age_sessions` sessions old whose source
    company has since stated another link in a different document.

    A restatement of the same link replaces it in `graph_asof`, so it never shows as stale.
    """
    here = int(sessions.searchsorted(pd.Timestamp(session), side="left"))
    if here >= len(sessions) or sessions[here] != pd.Timestamp(session):
        raise ValueError(f"session {session} is not on the sessions axis")
    tensions: list[Tension] = []
    for row in graph[graph["relation"].isin(list(_LINK_CLAIM_ATTRIBUTE))].itertuples():
        opened_on = pd.Timestamp(row.valid_from_utc).tz_convert("UTC").tz_localize(None)
        started = int(sessions.searchsorted(opened_on.normalize(), side="left"))
        age = here - started
        if started >= len(sessions) or age <= max_age_sessions:
            continue
        later = graph[
            (graph["source_node"] == row.source_node)
            & (graph["valid_from_utc"] > row.valid_from_utc)
            & (graph["source_doc_id"] != row.source_doc_id)
        ]
        if later.empty:
            continue
        newest = later.sort_values(["valid_from_utc", "source_doc_id"]).iloc[-1]
        subject = link_id_for(row.source_node, row.receiver_node, row.relation, row.source_doc_id)
        claim_id = claim_id_for(
            subject,
            _LINK_CLAIM_ATTRIBUTE[row.relation],
            "unspecified",
            "ongoing",
            row.source_doc_id,
        )
        details: dict[str, float | str] = {
            "age_sessions": age,
            "newer_doc_id": str(newest["source_doc_id"]),
        }
        tensions.append(
            _tension(
                TensionKind.stale_reliance,
                subject,
                [claim_id],
                subject,
                session,
                recorded_utc,
                details,
            )
        )
    return tensions


@dataclass(frozen=True)
class DetectReport:
    """What one detector run saw and set aside, so no silence is mistaken for a clean session."""

    by_kind: dict[str, int]
    claims_seen: int
    documents_in_session: int
    feed_gap: bool
    skipped_without_claim: int
    skipped_no_prices: int


def run_detectors(
    stores: DreamStores,
    session: date,
    cutoffs: pd.Series,
    params: dict[str, Any],
    *,
    abnormal_returns: pd.DataFrame | None = None,
) -> tuple[list[Tension], DetectReport]:
    """All four detectors for `session`, read as of its cutoff; nothing is appended to a store.

    A session with no documents is a feed gap and skips text-vs-price (a missing feed is not
    evidence of a quiet market). Without returns the detector is skipped and counted instead.
    `abnormal_returns`, a panel from `abnormal_returns_panel`, is used as it is when given, so a
    caller that runs many sessions builds it once; otherwise it is built from `stores.returns`.
    `cutoffs` must be indexed by `stores.sessions` (ValueError otherwise).
    """
    if not cutoffs.index.equals(stores.sessions):
        raise ValueError("cutoffs must be indexed by the same sessions as stores.sessions")
    cutoff = cutoffs[pd.Timestamp(session)]
    recorded = cutoff.to_pydatetime()
    claims = claims_asof(stores.claim_store, recorded)
    graph = (
        graph_asof(stores.link_store, cutoff)
        if stores.link_store is not None
        else pd.DataFrame(columns=GRAPH_COLUMNS)
    )
    documents = int(documents_per_session(stores.event_records, cutoffs)[pd.Timestamp(session)])
    feed_gap = documents == 0
    tensions = detect_value_conflicts(claims, session, recorded_utc=recorded)
    skipped_no_prices = 0
    if not feed_gap:
        panel = abnormal_returns
        if panel is None and stores.returns is not None and stores.market is not None:
            window = params["abnormal_returns"]
            panel = abnormal_returns_panel(
                stores.returns,
                stores.market,
                stores.sessions,
                tuple(window["est_window"]),
                window["min_obs"],
            )
        if panel is None:
            skipped_no_prices = 1
        else:
            tensions += detect_text_vs_price(
                claims,
                panel,
                session,
                params["text_vs_price"]["z_threshold"],
                recorded_utc=recorded,
            )
    asymmetric, skipped_without_claim = detect_asymmetric_links(
        graph, claims, session, recorded_utc=recorded
    )
    tensions += asymmetric
    tensions += detect_stale_reliance(
        graph,
        session,
        stores.sessions,
        params["stale_reliance"]["max_age_sessions"],
        recorded_utc=recorded,
    )
    by_kind = {kind.value: sum(t.kind is kind for t in tensions) for kind in TensionKind}
    return tensions, DetectReport(
        by_kind=by_kind,
        claims_seen=len(claims),
        documents_in_session=documents,
        feed_gap=feed_gap,
        skipped_without_claim=skipped_without_claim,
        skipped_no_prices=skipped_no_prices,
    )
