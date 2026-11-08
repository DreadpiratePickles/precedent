"""Market records: a resolved binary market, typed by a typer, plus the ingest counters."""

from __future__ import annotations

import re
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from precedent.spine.typers.base import TypedText, Typer

MAX_CALLS_REASON_PREFIX = "max_calls"
DRY_RUN_REASON = "dry-run"
_TRANSIENT_REFUSAL_PREFIXES = ("max_calls", "http")
_OFFSET_SUFFIX = re.compile(r"(Z|[+-]\d{2}(:?\d{2})?)$", re.IGNORECASE)


class ResolvedMarket(BaseModel):
    """A binary market that has resolved, before it is typed."""

    model_config = ConfigDict(extra="forbid")

    market_id: str  # "polymarket:<conditionId>" or "kalshi:<ticker>"
    venue: Literal["polymarket", "kalshi"]
    question: str
    created_utc: datetime
    close_utc: datetime | None
    resolved_utc: datetime
    resolution: Literal["yes", "no"]
    category: str | None
    source_ref: str


class MarketRecord(ResolvedMarket):
    """A resolved market with the typer's answer for its question."""

    typed: TypedText


@dataclass
class MarketIngestReport:
    """Counters for one ingest run; every raw item lands in exactly one bucket."""

    requests: int = 0
    seen: int = 0
    kept: int = 0
    skipped_not_binary: int = 0
    skipped_unresolved: int = 0
    ambiguous: int = 0
    skipped_missing_time: int = 0
    skipped_malformed: int = 0
    skipped_category: int = 0
    untyped: int = 0
    refused_invalid: int = 0
    max_calls_reached: int = 0
    not_stored_transient: int = 0
    resolved_by_prices_only: int = 0
    stopped_by_budget: bool = False
    stopped_by_transient: bool = False

    def absorb(self, other: MarketIngestReport) -> None:
        """Add another report's counters into this one (flags are left alone)."""
        for field in fields(self):
            value = getattr(other, field.name)
            if type(value) is int:
                setattr(self, field.name, getattr(self, field.name) + value)


def parse_utc_timestamp(value: object) -> datetime | None:
    """A timezone-aware UTC datetime from an ISO-8601 string, or None.

    Accepts `Z`, `+HH`, `+HH:MM` offsets and a space separator (`2024-01-03 14:15:00+00`).
    A missing, unparseable or offset-less value is None: a time is never guessed.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        parsed = _parse_with_pandas(text)
    if parsed is None or parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _parse_with_pandas(text: str) -> datetime | None:
    """The fallback for shapes `fromisoformat` rejects; it must carry an explicit offset."""
    if not _OFFSET_SUFFIX.search(text):
        return None
    try:
        stamp = pd.to_datetime(text, utc=True)
    except (ValueError, TypeError):
        return None
    if pd.isna(stamp):
        return None
    return stamp.to_pydatetime()


def is_transient_outcome(typed: TypedText) -> bool:
    """True when a typing outcome says nothing about the question and must not be stored.

    A spent call budget (`max_calls...`), a transport failure (`http ...`) and a dry run
    (`untyped` with reason "dry-run") would otherwise be kept under the real typer version,
    so a later run could never type the market (and re-typing would raise a conflict).
    """
    if typed.status == "refused_invalid":
        return (typed.reason or "").startswith(_TRANSIENT_REFUSAL_PREFIXES)
    return typed.status == "untyped" and typed.reason == DRY_RUN_REASON


def type_markets(
    markets: list[ResolvedMarket], typer: Typer, report: MarketIngestReport
) -> list[MarketRecord]:
    """Type every market's question; untyped answers are kept and counted, transient ones are not.

    An outcome that `is_transient_outcome` is counted (`not_stored_transient`, and its own
    counter: `max_calls_reached`, `refused_invalid` or `untyped`) and left out of the result, so
    a later run types that market. `kept` counts the returned records only.
    """
    records: list[MarketRecord] = []
    for market in markets:
        typed = typer.type_text(market.question)
        if typed.status == "untyped":
            report.untyped += 1
        elif typed.status == "refused_invalid":
            if (typed.reason or "").startswith(MAX_CALLS_REASON_PREFIX):
                report.max_calls_reached += 1
            else:
                report.refused_invalid += 1
        if is_transient_outcome(typed):
            report.not_stored_transient += 1
            continue
        records.append(MarketRecord(**market.model_dump(), typed=typed))
        report.kept += 1
    return records
