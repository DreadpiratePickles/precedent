"""What the Dream detectors read: the stores, the price panel, the session axis, and the
detector parameters, validated at the boundary.

The spine keeps no per-session count of how many documents arrived, so `documents_per_session`
computes it here; a session with no documents is a feed gap, not a quiet market.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from precedent import config
from precedent.dream.claim_store import ClaimStore, ResolutionStore, ShiftStore, TensionStore
from precedent.ripple.link_store import LinkStore
from precedent.spine.event_record import EventRecord

DETECTOR_PARAMS_FILE = Path("dream") / "detector_params.json"


@dataclass(frozen=True)
class DreamStores:
    """Everything one detector run reads; no detector writes to any store.

    `returns` is indexed by naive session dates, one column per instrument (simple returns),
    with `market` naming the market column. `session_cutoffs_utc` is indexed by `sessions`.
    """

    claim_store: ClaimStore
    tension_store: TensionStore
    shift_store: ShiftStore
    resolution_store: ResolutionStore
    event_records: list[EventRecord]
    link_store: LinkStore | None
    returns: pd.DataFrame | None
    market: str | None
    sessions: pd.DatetimeIndex
    session_cutoffs_utc: pd.Series

    def __post_init__(self) -> None:
        if self.returns is not None:
            if self.market is None:
                raise ValueError("returns were given without the name of the market column")
            if self.market not in self.returns.columns:
                raise ValueError(f"market column {self.market!r} is not in returns")
        if not self.sessions.equals(self.session_cutoffs_utc.index):
            raise ValueError("session_cutoffs_utc must be indexed by sessions")


def documents_per_session(records: list[EventRecord], session_cutoffs_utc: pd.Series) -> pd.Series:
    """Records per session over the whole cutoffs index, zero where none.

    A record belongs to the first session whose cutoff is at or after its `available_utc`;
    a record with no `available_utc`, or one past the last cutoff, is not counted.
    """
    cutoff_ns = session_cutoffs_utc.dt.tz_convert("UTC").dt.tz_localize(None).to_numpy()
    counts = np.zeros(len(cutoff_ns), dtype=int)
    for record in records:
        if record.available_utc is None:
            continue
        available = pd.Timestamp(record.available_utc).tz_convert("UTC").tz_localize(None)
        position = int(np.searchsorted(cutoff_ns, available.to_datetime64(), side="left"))
        if position < len(cutoff_ns):
            counts[position] += 1
    return pd.Series(counts, index=session_cutoffs_utc.index, name="documents")


class TextVsPriceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    z_threshold: float = Field(gt=0.0)


class AbnormalReturnsParams(BaseModel):
    """The market-model window; the defaults are those of `abnormal_return_at`."""

    model_config = ConfigDict(extra="forbid")
    est_window: tuple[int, int] = (-250, -21)
    min_obs: int = Field(default=120, ge=2)


class AsymmetricLinkParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    lookback_months: int = Field(ge=1)


class StaleRelianceParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    max_age_sessions: int = Field(ge=0)


class EventRateParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_sessions: int = Field(ge=1)
    baseline_sessions: int = Field(ge=1)
    z_threshold: float = Field(gt=0.0)


class DirectionDriftParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    k: float = Field(ge=0.0)
    h: float = Field(gt=0.0)


class LinkChurnParams(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jaccard_threshold: float = Field(ge=0.0, le=1.0)


class DetectorParams(BaseModel):
    """The whole `detector_params.json`; every value is a proposal until a result supports it."""

    model_config = ConfigDict(extra="forbid")
    proposed: bool
    story_window_sessions: int = Field(ge=1)
    annual_claim_expiry_months: int = Field(ge=1)
    text_vs_price: TextVsPriceParams
    abnormal_returns: AbnormalReturnsParams = Field(default_factory=AbnormalReturnsParams)
    asymmetric_link: AsymmetricLinkParams
    stale_reliance: StaleRelianceParams
    event_rate: EventRateParams
    direction_drift: DirectionDriftParams
    link_churn: LinkChurnParams
    shift_rereport_growth: float
    review_sample_per_kind: int = Field(ge=1)
    detector_keep_precision: float = Field(ge=0.0, le=1.0)


def load_detector_params(path: Path | None = None) -> dict[str, Any]:
    """The validated detector parameters as a plain dict; an unknown or missing key raises."""
    file_path = path if path is not None else config.references_dir() / DETECTOR_PARAMS_FILE
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError(f"cannot read detector parameters from {file_path}: {error!r}") from error
    return DetectorParams.model_validate(raw).model_dump()
