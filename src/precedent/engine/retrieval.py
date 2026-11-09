"""Retrieval behind a time guard: which past events, and which of their outcomes, a query may see.

`find_precedents` is the single guard. An outcome is returned only when the session that completes
its response closed, by its cutoff, at or before the moment the query became available. Every band
on a precedent card goes through this function.

Ruling P-42: the guard comes first and the cut second. Candidates are the similar past events
that have a completed row of the requested outcome type and horizon by the query time; slots are
counted per instrument for asset rows; and selection is window-aware (similarity, then recency;
an event whose outcome window overlaps one already accepted for that instrument is skipped) until
`top_k`. So `top_k` means "the `top_k` nearest completed, independent precedents".
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from precedent.engine.keys import EventKey, match_fields
from precedent.libraries.common import LIBRARY_COLUMNS, OUTCOME_TYPES, empty_library_rows

__all__ = [
    "PrecedentQuery",
    "exact_key_similarity",
    "find_precedents",
    "select_precedent_events",
]

SimilarityFunction = Callable[["PrecedentQuery", pd.DataFrame], pd.Series]


@dataclass(frozen=True)
class PrecedentQuery:
    """The new event: its typed key, the moment it became available, and what it is linked to."""

    key: EventKey
    available_utc: pd.Timestamp
    story_cluster_id: str | None = None
    canonical_text: str | None = None

    def __post_init__(self) -> None:
        if self.available_utc is pd.NaT or self.available_utc.tzinfo is None:
            raise ValueError(
                f"available_utc must be a timezone-aware timestamp, got {self.available_utc!r}"
            )


def exact_key_similarity(query: PrecedentQuery, events: pd.DataFrame) -> pd.Series:
    """1.0 where the five match fields are equal (and the direction sign, if the query sets one).

    Each distinct key string is parsed once. The result is aligned to `events.index`.
    """
    keys = events["key"]
    if keys.isna().any():
        raise ValueError("events have null keys")
    wanted = match_fields(query.key)
    sign = query.key.direction_sign
    score_of: dict[str, float] = {}
    for text in keys.unique():
        candidate = EventKey.from_string(text)
        same = match_fields(candidate) == wanted and (sign == 0 or candidate.direction_sign == sign)
        score_of[text] = 1.0 if same else 0.0
    return keys.map(score_of).astype("float64")


def select_precedent_events(
    query: PrecedentQuery,
    events: pd.DataFrame,
    top_k: int = 30,
    similarity: SimilarityFunction = exact_key_similarity,
) -> pd.DataFrame:
    """The `top_k` past events most like the query, as the events table plus `similarity`.

    Order of operations: events available at or before the query; first reports only; not the
    query's own story; similarity; drop zero similarity; sort by similarity, then latest
    `effective_position`, then `event_id`; keep `top_k`. The index of the result is fresh.

    This is the unguarded pool of look-alikes, not what a card shows: outcome bands come from
    `find_precedents`, which applies the completion guard before its own cut (P-42).
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    return _rank_candidates(query, events, similarity).head(top_k).reset_index(drop=True)


def _rank_candidates(
    query: PrecedentQuery, events: pd.DataFrame, similarity: SimilarityFunction
) -> pd.DataFrame:
    if "similarity" in events.columns:
        raise ValueError("events already has a similarity column")
    if not events["event_id"].is_unique:
        raise ValueError("events has duplicate event_id values")
    if events["is_first_report"].isna().any():
        raise ValueError("events has null is_first_report values")
    keep = events["available_utc"] <= query.available_utc
    keep &= events["is_first_report"].astype(bool)
    if query.story_cluster_id is not None:
        keep &= events["story_cluster_id"] != query.story_cluster_id
    candidates = events.loc[keep]
    scores = similarity(query, candidates)
    if not scores.index.equals(candidates.index):
        raise ValueError("similarity must return a Series aligned to the events it was given")
    if scores.isna().any():
        raise ValueError("similarity returned null scores")
    scored = candidates.assign(similarity=scores.astype("float64"))
    scored = scored.loc[scored["similarity"] != 0.0]
    ranked = scored.sort_values(
        ["similarity", "effective_position", "event_id"],
        ascending=[False, False, True],
        kind="stable",
    )
    return ranked


def find_precedents(
    query: PrecedentQuery,
    events: pd.DataFrame,
    rows: pd.DataFrame,
    session_cutoffs_utc: pd.Series,
    horizon: int,
    outcome_type: str,
    top_k: int = 30,
    similarity: SimilarityFunction = exact_key_similarity,
) -> pd.DataFrame:
    """Completed outcome rows of the `top_k` nearest independent precedents the query may see.

    Rows are of `outcome_type`, and their `response_complete_session` cutoff is at or before
    `query.available_utc`; a session missing from `session_cutoffs_utc` raises `KeyError`.
    Rows are filtered to `horizon` except for `market_resolution`, which carry their own.
    The guard is applied first; then, per instrument (asset rows; per market and horizon for
    `market_resolution`), events are taken in order of similarity, then recency, skipping any whose
    outcome window (`effective_position` .. + row horizon) overlaps one already taken, until
    `top_k`. Returns `LIBRARY_COLUMNS` plus `similarity`, in selection order.
    """
    if outcome_type not in OUTCOME_TYPES:
        raise ValueError(f"unknown outcome_type {outcome_type!r}; expected one of {OUTCOME_TYPES}")
    if not session_cutoffs_utc.index.is_unique:
        raise ValueError("session_cutoffs_utc has duplicate sessions")
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
    pool = _rank_candidates(query, events, similarity)
    of_type = rows.loc[rows["outcome_type"] == outcome_type, list(LIBRARY_COLUMNS)]
    found = pool[["event_id", "similarity", "effective_position"]].merge(
        of_type, on="event_id", how="inner"
    )
    if found.empty:
        return _empty_found()
    sessions = found["response_complete_session"]
    unknown = sessions[~sessions.isin(session_cutoffs_utc.index)]
    if not unknown.empty:
        raise KeyError(
            f"response_complete_session {unknown.iloc[0].date()} is not in session_cutoffs_utc"
        )
    cutoffs = session_cutoffs_utc.loc[sessions.to_numpy()].reset_index(drop=True)
    keep = cutoffs <= query.available_utc
    if outcome_type != "market_resolution":
        keep &= found["horizon"] == horizon
    found = found.loc[keep].reset_index(drop=True)
    if found.empty:
        return _empty_found()
    chosen = _accept_independent(found, outcome_type, top_k)
    result = found.loc[chosen, [*LIBRARY_COLUMNS, "similarity"]].reset_index(drop=True)
    return result if not result.empty else _empty_found()


def _accept_independent(found: pd.DataFrame, outcome_type: str, top_k: int) -> list[int]:
    """Row positions accepted in rank order: at most `top_k` per instrument, windows disjoint."""
    keys = found["outcome_key"].to_numpy()
    horizons = found["horizon"].to_numpy()
    starts = found["effective_position"].to_numpy()
    accepted: dict[object, list[int]] = {}
    chosen: list[int] = []
    for i in range(len(found)):
        slot = (keys[i], int(horizons[i])) if outcome_type == "market_resolution" else keys[i]
        taken = accepted.setdefault(slot, [])
        if len(taken) >= top_k:
            continue
        width = int(horizons[i])
        start = int(starts[i])
        if any(abs(start - other) < width for other in taken):
            continue
        taken.append(start)
        chosen.append(i)
    return chosen


def _empty_found() -> pd.DataFrame:
    return empty_library_rows().assign(similarity=pd.Series(dtype="float64"))
