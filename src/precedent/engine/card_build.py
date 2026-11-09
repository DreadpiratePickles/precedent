"""Card assembly: `run_card` and the functions behind it, split out of `engine/ask.py`.

`run_card` loads the library bundle, types the text (if it is text), asks `find_precedents` for the
completed, independent precedents of the key, and builds a `CardModel` from their outcomes. Nothing
here forecasts: every number is a summary of what happened after similar events, behind the time
guard of `find_precedents`. The command line lives in `engine/ask.py`, which re-exports
`run_card`, `refused_card_without_library` and `UNTYPED` for its callers.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from precedent.engine.aggregate import PrecedentSummary, effective_sample_size, summarize_responses
from precedent.engine.card import (
    KEY_FLAGS_HINT,
    CardModel,
    CardRow,
    NearestPrecedent,
    SourceCoverage,
)
from precedent.engine.card_placebo import ordinary_range as _placebo
from precedent.engine.keys import typed_as_line
from precedent.engine.params import EngineParams, load_engine_params
from precedent.engine.retrieval import PrecedentQuery, find_precedents
from precedent.libraries.build import LIBRARY_DIR_NAME
from precedent.libraries.bundle import LibraryBundle, load_bundle
from precedent.libraries.common import OUTCOME_TYPES
from precedent.spine.typers.base import Typer

__all__ = ["UNTYPED", "refused_card_without_library", "run_card"]

UNTYPED = "UNTYPED"
ALL_OUTCOME_TYPES = "all"
_NO_PRICES = "no prices"
_NO_LIBRARY_ROWS = "no library rows"


# --- the card -----------------------------------------------------------------------------------


def _check_request(outcome_type: str, horizon: int, params: EngineParams) -> None:
    if outcome_type != ALL_OUTCOME_TYPES and outcome_type not in OUTCOME_TYPES:
        raise ValueError(
            f"unknown outcome_type {outcome_type!r}; expected one of "
            f"{(ALL_OUTCOME_TYPES, *OUTCOME_TYPES)}"
        )
    union = sorted(set(params.asset_horizons) | set(params.sequence_horizons))
    allowed = {
        "asset": sorted(params.asset_horizons),
        "event_sequence": sorted(params.sequence_horizons),
    }.get(outcome_type, union)  # market rows carry their own horizons; `all` takes the union
    if horizon not in allowed:
        raise ValueError(
            f"unsupported horizon {horizon!r} for outcome type {outcome_type!r}; "
            f"expected one of {allowed}"
        )


def _as_date(value: object) -> dt.date:
    return pd.Timestamp(value).date()  # type: ignore[arg-type]


def _coverage(bundle: LibraryBundle) -> list[SourceCoverage]:
    return [
        SourceCoverage(name=s.name, first_date=s.first_date, last_date=s.last_date, rows=s.rows)
        for s in bundle.manifest.sources
    ]


def _card(
    *,
    params: EngineParams,
    headline: str,
    typed_as: str,
    typed_reason: str | None,
    key: str | None,
    asof_utc: pd.Timestamp,
    horizon: int,
    outcome_type: str,
    n: int = 0,
    n_eff: float = 0.0,
    earliest: dt.date | None = None,
    latest: dt.date | None = None,
    rows: list[CardRow] | None = None,
    not_built: dict[str, str] | None = None,
    nearest: list[NearestPrecedent] | None = None,
    covers: list[SourceCoverage] | None = None,
    hint: str | None = None,
) -> CardModel:
    return CardModel(
        headline=headline,
        typed_as=typed_as,
        typed_reason=typed_reason,
        key=key,
        asof_utc=asof_utc.to_pydatetime(),
        horizon=horizon,
        outcome_type=outcome_type,
        n=n,
        n_eff=n_eff,
        earliest=earliest,
        latest=latest,
        refused=not rows,
        min_neff=params.min_neff,
        rows=rows or [],
        not_built=not_built or {},
        nearest=nearest or [],
        covers=covers or [],
        hint=hint,
    )


def _untyped_card(
    text: str, reason: str, asof: pd.Timestamp, horizon: int, outcome_type: str, covers: list
) -> CardModel:
    return _card(
        params=load_engine_params(),
        headline=text,
        typed_as=UNTYPED,
        typed_reason=reason,
        key=None,
        asof_utc=asof,
        horizon=horizon,
        outcome_type=outcome_type,
        covers=covers,
        hint=KEY_FLAGS_HINT,
    )


def _resolve_query(
    query: PrecedentQuery | str, typer: Typer, asof_utc: pd.Timestamp | None
) -> tuple[PrecedentQuery | None, str, str | None, str | None, pd.Timestamp]:
    """(query or None when untyped, headline, typed_reason, untyped reason, asof)."""
    if isinstance(query, PrecedentQuery):
        # a caller may hand over a plain `datetime` (the card's own `asof_utc`); use a Timestamp
        available = pd.Timestamp(query.available_utc)
        query = dataclasses.replace(query, available_utc=available)
        return query, typed_as_line(query.key), None, None, available
    asof = asof_utc if asof_utc is not None else pd.Timestamp.now(tz="UTC")
    typed = typer.type_text(query)
    if typed.status != "typed" or typed.key is None:
        return None, query, typed.reason, typed.reason, asof
    return PrecedentQuery(typed.key, asof), query, typed.reason, None, asof


def _check_asof(asof_utc: pd.Timestamp | None) -> None:
    if asof_utc is not None and (asof_utc is pd.NaT or asof_utc.tzinfo is None):
        raise ValueError(f"asof_utc must be a timezone-aware timestamp, got {asof_utc!r}")


def run_card(
    query: PrecedentQuery | str,
    horizon: int,
    outcome_type: str,
    data_dir: Path,
    typer: Typer,
    *,
    asof_utc: pd.Timestamp | None = None,
) -> CardModel:
    """The precedent card for `query` at `horizon` from the library under `data_dir / library`.

    A `str` is typed with `typer`; text the typer cannot type gives the untyped card (refused,
    with the key-flags hint). `asof_utc` is the moment a text query became available (default
    now); a `PrecedentQuery` carries its own. Raises `ValueError` for an unknown `outcome_type`
    or an unsupported `horizon`, `LibraryMissingError` or `LibraryIntegrityError` for a library
    that is absent or altered (the library is read before any text is typed).
    """
    params = load_engine_params()
    _check_request(outcome_type, horizon, params)
    _check_asof(asof_utc)
    bundle = load_bundle(data_dir / LIBRARY_DIR_NAME)
    precedent_query, headline, typed_reason, untyped_reason, asof = _resolve_query(
        query, typer, asof_utc
    )
    if precedent_query is None:
        assert untyped_reason is not None
        return _untyped_card(
            headline, untyped_reason, asof, horizon, outcome_type, _coverage(bundle)
        )
    return _precedent_card(
        precedent_query, headline, typed_reason, horizon, outcome_type, bundle, params
    )


def refused_card_without_library(
    query: PrecedentQuery | str,
    horizon: int,
    outcome_type: str,
    typer: Typer,
    *,
    asof_utc: pd.Timestamp | None = None,
) -> CardModel:
    """The refused card for a query when there is no library to search (`covers` is empty)."""
    params = load_engine_params()
    _check_request(outcome_type, horizon, params)
    _check_asof(asof_utc)
    precedent_query, headline, typed_reason, untyped_reason, asof = _resolve_query(
        query, typer, asof_utc
    )
    if precedent_query is None:
        assert untyped_reason is not None
        return _untyped_card(headline, untyped_reason, asof, horizon, outcome_type, [])
    return _card(
        params=params,
        headline=headline,
        typed_as=typed_as_line(precedent_query.key),
        typed_reason=typed_reason,
        key=precedent_query.key.as_string(),
        asof_utc=asof,
        horizon=horizon,
        outcome_type=outcome_type,
    )


def _precedent_card(
    query: PrecedentQuery,
    headline: str,
    typed_reason: str | None,
    horizon: int,
    outcome_type: str,
    bundle: LibraryBundle,
    params: EngineParams,
) -> CardModel:
    found_by_kind, not_built = _guarded_rows(query, horizon, outcome_type, bundle, params)
    # P-43: n, independent, the REFUSED decision and the nearest list describe the guarded rows
    # shown below, never the unguarded selection (whose outcomes may still be pending).
    guarded = _guarded_events(found_by_kind, bundle.events)
    n = len(guarded)
    positions = guarded["effective_position"].to_numpy()
    n_eff = effective_sample_size(np.ones(n), positions, horizon)
    refused = n_eff < params.min_neff
    sessions = [_as_date(value) for value in guarded["effective_session"]]
    nearest = [
        NearestPrecedent(
            date=_as_date(event.effective_session),
            one_line=event.one_line,
            similarity=float(event.similarity),
            source_ref=event.source_ref,
        )
        for event in guarded.head(params.nearest_shown).itertuples(index=False)
    ]
    rows: list[CardRow] = []
    if not refused:
        rows = _outcome_rows(found_by_kind, horizon, query, bundle, params)
    card = _card(
        params=params,
        headline=headline,
        typed_as=typed_as_line(query.key),
        typed_reason=typed_reason,
        key=query.key.as_string(),
        asof_utc=query.available_utc,
        horizon=horizon,
        outcome_type=outcome_type,
        n=n,
        n_eff=n_eff,
        earliest=min(sessions) if sessions else None,
        latest=max(sessions) if sessions else None,
        rows=rows,
        not_built=not_built if not refused else {},
        nearest=nearest,
        covers=_coverage(bundle),
    )
    # `_card` marks a card refused when it has no rows; an unrefused card with no row to show
    # (every outcome of the type still pending) stays unrefused.
    return card.model_copy(update={"refused": bool(refused)})


# --- outcome rows -------------------------------------------------------------------------------


def _guarded_rows(
    query: PrecedentQuery,
    horizon: int,
    outcome_type: str,
    bundle: LibraryBundle,
    params: EngineParams,
) -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    """The completed, independent precedent rows per asked type, and the types not built."""
    asked = OUTCOME_TYPES if outcome_type == ALL_OUTCOME_TYPES else (outcome_type,)
    found_by_kind: dict[str, pd.DataFrame] = {}
    not_built: dict[str, str] = {}
    for kind in asked:
        if not (bundle.rows["outcome_type"] == kind).any():
            not_built[kind] = _NO_PRICES if kind == "asset" else _NO_LIBRARY_ROWS
            continue
        found_by_kind[kind] = find_precedents(
            query, bundle.events, bundle.rows, bundle.session_cutoffs_utc, horizon, kind,
            top_k=params.top_k,
        )
    return found_by_kind, not_built


def _guarded_events(found_by_kind: dict[str, pd.DataFrame], events: pd.DataFrame) -> pd.DataFrame:
    """The distinct events behind the guarded rows, nearest first, with their `similarity`."""
    parts = [found[["event_id", "similarity"]] for found in found_by_kind.values()]
    if not parts:
        return events.iloc[0:0].assign(similarity=pd.Series(dtype="float64"))
    ids = pd.concat(parts).groupby("event_id", as_index=False)["similarity"].max()
    merged = ids.merge(events, on="event_id", how="inner")
    return merged.sort_values(
        ["similarity", "effective_position", "event_id"],
        ascending=[False, False, True],
        kind="stable",
    ).reset_index(drop=True)


def _outcome_rows(
    found_by_kind: dict[str, pd.DataFrame],
    horizon: int,
    query: PrecedentQuery,
    bundle: LibraryBundle,
    params: EngineParams,
) -> list[CardRow]:
    # P-42: find_precedents picks its own events (guard first), so positions come from the library
    position_of = bundle.events.set_index("event_id")["effective_position"]
    rows: list[CardRow] = []
    for kind, found in found_by_kind.items():
        found = found.assign(window_start=found["event_id"].map(position_of))
        of_type = [
            _card_row(kind, key, group, horizon, query, bundle, params)
            for key, group in found.groupby("outcome_key", sort=True)
        ]
        of_type.sort(key=lambda row: (-row.n, row.outcome_key))
        rows.extend(of_type[: params.card_rows_per_type])
    return rows


def _card_row(
    kind: str,
    outcome_key: str,
    group: pd.DataFrame,
    horizon: int,
    query: PrecedentQuery,
    bundle: LibraryBundle,
    params: EngineParams,
) -> CardRow:
    is_binary = kind != "asset"
    summary = summarize_responses(
        group["value"].to_numpy(),
        np.ones(len(group)),
        group["window_start"].to_numpy(),
        horizon,
        _placebo(kind, outcome_key, horizon, query, bundle),
        is_binary=is_binary,
        shrink_k=params.shrink_k,
        min_neff=params.min_neff,
        z=params.wilson_z,
    )
    row_horizon = None if kind == "market_resolution" else horizon
    common = {
        "outcome_type": kind,
        "outcome_key": outcome_key,
        "horizon": row_horizon,
        "is_binary": is_binary,
        "n": summary.n,
        "n_eff": summary.n_eff,
        "refused": summary.refused,
    }
    if summary.refused:
        return CardRow(
            **common, k=summary.k, rate=None, interval=None, median=None, centre_shrunk=None,
            band=None, vs_ordinary=None,
        )
    return _summary_row(common, summary)


def _summary_row(common: dict, summary: PrecedentSummary) -> CardRow:
    vs_ordinary = None
    if summary.inside_ordinary_range is not None:
        vs_ordinary = "inside" if summary.inside_ordinary_range else "outside"
    if summary.mode == "binary":
        return CardRow(
            **common, k=summary.k, rate=summary.rate, interval=summary.interval, median=None,
            centre_shrunk=None, band=None, vs_ordinary=vs_ordinary,
        )
    assert summary.quantiles is not None
    return CardRow(
        **common, k=None, rate=None, interval=None, median=summary.median,
        centre_shrunk=summary.centre_shrunk, band=(summary.quantiles[0.1], summary.quantiles[0.9]),
        vs_ordinary=vs_ordinary,
    )
