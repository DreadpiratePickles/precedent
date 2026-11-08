"""Retrieval behind a time guard: which past events, and which of their outcomes, a query may see.

`find_precedents` is the single guard. An outcome is returned only when the session that completes
its response closed, by its cutoff, at or before the moment the query became available. Every band
on a precedent card goes through this function.
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
    """
    if top_k < 1:
        raise ValueError(f"top_k must be at least 1, got {top_k}")
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
    return ranked.head(top_k).reset_index(drop=True)


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
    """Outcome rows of the selected precedent events that the query is allowed to see.

    Rows are of `outcome_type`, and their `response_complete_session` cutoff is at or before
    `query.available_utc`; a session missing from `session_cutoffs_utc` raises `KeyError`.
    Rows are filtered to `horizon` except for `market_resolution`, which carry their own.
    Returns `LIBRARY_COLUMNS` plus `similarity`, in selection order.
    """
    if outcome_type not in OUTCOME_TYPES:
        raise ValueError(f"unknown outcome_type {outcome_type!r}; expected one of {OUTCOME_TYPES}")
    if not session_cutoffs_utc.index.is_unique:
        raise ValueError("session_cutoffs_utc has duplicate sessions")
    selected = select_precedent_events(query, events, top_k=top_k, similarity=similarity)
    of_type = rows.loc[rows["outcome_type"] == outcome_type, list(LIBRARY_COLUMNS)]
    found = selected[["event_id", "similarity"]].merge(of_type, on="event_id", how="inner")
    found = found[[*LIBRARY_COLUMNS, "similarity"]]
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
    result = found.loc[keep].reset_index(drop=True)
    return result if not result.empty else _empty_found()


def _empty_found() -> pd.DataFrame:
    return empty_library_rows().assign(similarity=pd.Series(dtype="float64"))
