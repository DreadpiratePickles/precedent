"""Baseline ladder scoring: how well history from earlier events predicts later ones.

Events are visited in time order and each is scored from earlier events only, through
`find_precedents`, so every band goes through the same time guard as a precedent card. Rung B0
pools every earlier first report (the unconditional distribution or rate); rung B1 keeps only
events with the same typed key. Continuous outcomes are scored with pinball loss per quantile
and the coverage of the 80% band; binary outcomes with the Brier score. A refused summary
(too few effective precedents) counts in `n_refused` and adds no loss.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from precedent import config
from precedent.engine.aggregate import QUANTILE_LEVELS, summarize_responses
from precedent.engine.keys import EventKey
from precedent.engine.params import load_engine_params
from precedent.engine.retrieval import (
    PrecedentQuery,
    exact_key_similarity,
    find_precedents,
)
from precedent.libraries.common import OUTCOME_TYPES

__all__ = ["RUNGS", "brier_score", "pinball_loss", "run", "score_precedent_ladder"]

RUNGS = ("B0", "B1")
OUTPUT_COLUMNS = (
    "rung",
    "outcome_type",
    "event_type",
    "quantile",
    "pinball_loss",
    "coverage_80",
    "brier",
    "n_scored",
    "n_refused",
)
_BAND_LOW = 0.1
_BAND_HIGH = 0.9
_NO_LIBRARY = "no library under {path}; run precedent build (or precedent fetch) first"

_Similarity = Callable[[PrecedentQuery, pd.DataFrame], pd.Series]


def pinball_loss(realized: np.ndarray, predicted_quantile: np.ndarray, q: float) -> float:
    """Mean over elements of ``max(q * (y - p), (q - 1) * (y - p))``."""
    if not 0.0 < q < 1.0:
        raise ValueError(f"q must lie in (0, 1), got {q}")
    error = np.asarray(realized, dtype=float) - np.asarray(predicted_quantile, dtype=float)
    return float(np.mean(np.maximum(q * error, (q - 1.0) * error)))


def brier_score(realized: np.ndarray, predicted_rate: np.ndarray) -> float:
    """Mean squared difference between the realised 0/1 outcome and the predicted rate."""
    difference = np.asarray(realized, dtype=float) - np.asarray(predicted_rate, dtype=float)
    return float(np.mean(difference * difference))


def _every_event(_query: PrecedentQuery, events: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=events.index, dtype="float64")


_SIMILARITY_OF_RUNG: dict[str, _Similarity] = {"B0": _every_event, "B1": exact_key_similarity}


class _Tally:
    """What one `(rung, event_type)` group has scored so far."""

    def __init__(self) -> None:
        self.realized: list[float] = []
        self.predicted: dict[float, list[float]] = {}
        self.rates: list[float] = []
        self.band_low: list[float] = []
        self.band_high: list[float] = []
        self.n_refused = 0


def score_precedent_ladder(
    events: pd.DataFrame,
    rows: pd.DataFrame,
    session_cutoffs_utc: pd.Series,
    outcome_type: str,
    rungs: tuple[str, ...] = RUNGS,
    horizon: int = 5,
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9),
) -> pd.DataFrame:
    """Score each rung on `outcome_type`, one unit per first-report event and realised row.

    A unit is an `(event, outcome_key)` row of `outcome_type` at `horizon` (any horizon for
    `market_resolution`). Its precedents are the rows of the same `outcome_key` that
    `find_precedents` returns for the event's key as of its own `available_utc`, summarised with
    the engine parameters. Output columns are `OUTPUT_COLUMNS`, one row per
    `(rung, event_type, quantile)` for continuous types and per `(rung, event_type)` for binary
    types; the loss columns that do not apply are NaN.
    """
    if outcome_type not in OUTCOME_TYPES:
        raise ValueError(f"unknown outcome_type {outcome_type!r}; expected one of {OUTCOME_TYPES}")
    unknown_rungs = [rung for rung in rungs if rung not in _SIMILARITY_OF_RUNG]
    if unknown_rungs:
        raise ValueError(f"unknown rung {unknown_rungs}; expected {tuple(_SIMILARITY_OF_RUNG)}")
    unsupported = [q for q in quantiles if q not in QUANTILE_LEVELS]
    if unsupported:
        raise ValueError(f"unsupported quantile {unsupported}; expected {QUANTILE_LEVELS}")
    params = load_engine_params()
    realised = _realised_rows(rows, outcome_type, horizon)
    position_of = events.set_index("event_id")["effective_position"]
    ordered = events.sort_values(["available_utc", "event_id"], kind="stable")
    ordered = ordered.loc[ordered["is_first_report"].astype(bool)]

    tallies: dict[tuple[str, str], _Tally] = {}
    is_binary = False
    for event in ordered.itertuples(index=False):
        event_rows = realised.get(event.event_id)
        if event_rows is None:
            continue
        query = PrecedentQuery(
            key=EventKey.from_string(event.key),
            available_utc=event.available_utc,
            story_cluster_id=event.story_cluster_id,
        )
        for rung in rungs:
            found = find_precedents(
                query,
                events,
                rows,
                session_cutoffs_utc,
                horizon,
                outcome_type,
                top_k=params.top_k,
                similarity=_SIMILARITY_OF_RUNG[rung],
            )
            tally = tallies.setdefault((rung, event.event_type), _Tally())
            for row in event_rows.itertuples(index=False):
                is_binary = bool(row.is_binary)
                precedents = found.loc[found["outcome_key"] == row.outcome_key]
                summary = summarize_responses(
                    precedents["value"].to_numpy(dtype=float),
                    np.ones(len(precedents)),
                    precedents["event_id"].map(position_of).to_numpy(),
                    horizon,
                    None,
                    is_binary=is_binary,
                    shrink_k=params.shrink_k,
                    min_neff=params.min_neff,
                    z=params.wilson_z,
                )
                if summary.refused:
                    tally.n_refused += 1
                    continue
                tally.realized.append(float(row.value))
                if is_binary:
                    tally.rates.append(float(summary.rate))
                    continue
                for q in quantiles:
                    tally.predicted.setdefault(q, []).append(summary.quantiles[q])
                tally.band_low.append(summary.quantiles[_BAND_LOW])
                tally.band_high.append(summary.quantiles[_BAND_HIGH])
    return _table(tallies, outcome_type, is_binary, quantiles)


def _realised_rows(rows: pd.DataFrame, outcome_type: str, horizon: int) -> dict[str, pd.DataFrame]:
    """The rows to be scored, by event id, sorted by outcome key and horizon."""
    wanted = rows.loc[rows["outcome_type"] == outcome_type]
    if outcome_type != "market_resolution":
        wanted = wanted.loc[wanted["horizon"] == horizon]
    ordered = wanted.sort_values(["outcome_key", "horizon"], kind="stable")
    return {event_id: group for event_id, group in ordered.groupby("event_id", sort=False)}


def _table(
    tallies: dict[tuple[str, str], _Tally],
    outcome_type: str,
    is_binary: bool,
    quantiles: tuple[float, ...],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for (rung, event_type), tally in sorted(tallies.items()):
        base: dict[str, object] = {
            "rung": rung,
            "outcome_type": outcome_type,
            "event_type": event_type,
            "n_scored": len(tally.realized),
            "n_refused": tally.n_refused,
        }
        realized = np.array(tally.realized, dtype=float)
        if is_binary:
            brier = brier_score(realized, np.array(tally.rates)) if tally.realized else np.nan
            records.append(
                {
                    **base,
                    "quantile": np.nan,
                    "pinball_loss": np.nan,
                    "coverage_80": np.nan,
                    "brier": brier,
                }
            )
            continue
        coverage = np.nan
        if tally.realized:
            inside = (realized >= np.array(tally.band_low)) & (
                realized <= np.array(tally.band_high)
            )
            coverage = float(np.mean(inside))
        for q in quantiles:
            loss = (
                pinball_loss(realized, np.array(tally.predicted[q]), q)
                if tally.realized
                else np.nan
            )
            records.append(
                {
                    **base,
                    "quantile": q,
                    "pinball_loss": loss,
                    "coverage_80": coverage,
                    "brier": np.nan,
                }
            )
    frame = pd.DataFrame(records, columns=list(OUTPUT_COLUMNS))
    for column in ("quantile", "pinball_loss", "coverage_80", "brier"):
        frame[column] = frame[column].astype("float64")
    for column in ("n_scored", "n_refused"):
        frame[column] = frame[column].astype("int64")
    return frame


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="precedent evaluate",
        description="Score the baseline ladder (B0 pooled, B1 exact key) on the built library.",
    )
    parser.add_argument("--data-dir", type=Path, default=None, help="default: PRECEDENT_DATA_DIR")
    parser.add_argument("--horizon", type=int, default=5, help="sessions; default 5")
    return parser.parse_args(argv)


def run(argv: list[str]) -> int:
    """`precedent evaluate`: 0 on success, 2 when no library is present under the data dir."""
    args = _parse(argv)
    data_dir = args.data_dir if args.data_dir is not None else config.data_dir()
    library_dir = data_dir / "library"
    try:
        from precedent.libraries.bundle import LibraryMissingError, load_bundle
    except ModuleNotFoundError as error:
        if error.name != "precedent.libraries.bundle":
            raise
        print(_NO_LIBRARY.format(path=library_dir), file=sys.stderr)
        return 2
    try:
        bundle = load_bundle(library_dir)
    except LibraryMissingError:
        print(_NO_LIBRARY.format(path=library_dir), file=sys.stderr)
        return 2
    tables = [
        score_precedent_ladder(
            bundle.events,
            bundle.rows,
            bundle.session_cutoffs_utc,
            outcome_type,
            rungs=RUNGS,
            horizon=args.horizon,
        )
        for outcome_type in OUTCOME_TYPES
        if (bundle.rows["outcome_type"] == outcome_type).any()
    ]
    empty = pd.DataFrame(columns=list(OUTPUT_COLUMNS))
    result = pd.concat(tables, ignore_index=True) if tables else empty
    print(result.to_string(index=False))
    return 0
