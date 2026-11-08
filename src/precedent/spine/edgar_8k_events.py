"""Rule-typed events from an EDGAR submissions document: one event per 8-K item, no model."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from precedent.config import references_dir
from precedent.spine.event_record import EventRecord, ResolvedImpact
from precedent.spine.event_schema import EventType, Impact, NewsEvent
from precedent.spine.pit_alignment import align_records, unalign_before
from precedent.spine.session_calendar import (
    calendar_start_utc,
    default_cutoffs_for,
    load_nyse_calendar,
)

PARSER_VERSION_PREFIX = "rules-8k-v1"
SOURCE = "edgar"
EIGHT_K_FORMS = frozenset({"8-K", "8-K/A"})
AMENDMENT_FORM = "8-K/A"
DEFAULT_LATENCY = pd.Timedelta(minutes=5)
CUTOFF_LOOKAHEAD = timedelta(days=15)
DEFAULT_SEVERITY = 5
MAX_NAME_CHARS = 80
_ITEM_RULES_FILE = "item_rules_8k.json"
_STATE_CODES_FILE = "us_state_codes.json"
_REQUIRED_COLUMNS = (
    "accessionNumber",
    "acceptanceDateTime",
    "form",
    "items",
)
_TICKER_PATTERN = re.compile(r"^[A-Z][A-Z0-9.\-]{0,5}$")


def edgar_parser_version(path: Path | None = None) -> str:
    """`rules-8k-v1:<12 hex of the item rules file's bytes>`: a rule edit starts a new vintage."""
    file_path = path or references_dir() / _ITEM_RULES_FILE
    try:
        digest = hashlib.sha256(file_path.read_bytes()).hexdigest()
    except OSError as error:
        raise ValueError(f"cannot read the 8-K item rules {file_path}: {error}") from error
    return f"{PARSER_VERSION_PREFIX}:{digest[:12]}"


class ItemRule(BaseModel):
    """How one 8-K item code is typed. Every entry in the committed file is an owner proposal."""

    model_config = ConfigDict(extra="forbid")
    event_type: EventType
    description: str = Field(min_length=1, max_length=120)
    severity: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = DEFAULT_SEVERITY
    is_scheduled: bool = False


class ItemRules(BaseModel):
    model_config = ConfigDict(extra="forbid")
    proposed: bool
    source: str
    items: dict[str, ItemRule]


@dataclass(frozen=True)
class EdgarIngestReport:
    """Counts for one submissions document: filings seen, records made, and what was set aside.

    `skipped_filings_without_items` counts 8-K filings whose `items` is empty, separator-only
    or null."""

    filings_read: int
    records: int
    skipped_items: int
    skipped_forms: int
    skipped_filings_without_items: int


def load_item_rules(path: Path | None = None) -> ItemRules:
    """Read and validate the item rules file (default `references/item_rules_8k.json`)."""
    file_path = path if path is not None else references_dir() / _ITEM_RULES_FILE
    try:
        return ItemRules.model_validate(json.loads(file_path.read_text(encoding="utf-8")))
    except (OSError, ValueError, ValidationError) as error:
        raise ValueError(f"cannot load 8-K item rules from {file_path}: {error}") from error


def load_us_state_codes(path: Path | None = None) -> frozenset[str]:
    """The EDGAR codes that mean a US state, DC or territory; any other code is non-US."""
    file_path = path if path is not None else references_dir() / _STATE_CODES_FILE
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))
        codes = raw["codes"]
        if not isinstance(codes, list) or not all(isinstance(code, str) for code in codes):
            raise TypeError("codes must be a list of strings")
        return frozenset(codes)
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cannot load US state codes from {file_path}: {error}") from error


def _malformed(reason: str) -> ValueError:
    return ValueError(f"malformed EDGAR submissions document: {reason}")


def _recent_columns(submissions: dict[str, Any]) -> dict[str, list[Any]]:
    try:
        recent = submissions["filings"]["recent"]
        columns = {name: recent[name] for name in _REQUIRED_COLUMNS}
    except (KeyError, TypeError) as error:
        raise _malformed(f"missing filings.recent field {error}") from error
    if not all(isinstance(column, list) for column in columns.values()):
        raise _malformed("filings.recent fields must be arrays")
    if len({len(column) for column in columns.values()}) > 1:
        raise _malformed("filings.recent arrays differ in length")
    return columns


def _company_name(submissions: dict[str, Any]) -> str:
    name = submissions.get("name")
    if not isinstance(name, str) or not name.strip():
        raise _malformed("name is missing or empty")
    return name.strip()[:MAX_NAME_CHARS]


def _business_state_or_country(submissions: dict[str, Any]) -> str | None:
    business = (submissions.get("addresses") or {}).get("business") or {}
    code = business.get("stateOrCountry")
    return code if isinstance(code, str) else None


def _accepted_utc(text: Any, accession: str) -> datetime:
    try:
        accepted = datetime.fromisoformat(text)
    except (TypeError, ValueError) as error:
        raise _malformed(f"acceptanceDateTime {text!r} of {accession} is not ISO-8601") from error
    if accepted.tzinfo is None:
        raise _malformed(f"acceptanceDateTime {text!r} of {accession} has no timezone")
    return accepted.astimezone(UTC)


def _item_codes(items: Any, accession: str) -> list[str]:
    """Item codes of one filing; a null `items` is no codes (the caller counts it)."""
    if items is None:
        return []
    if not isinstance(items, str):
        raise _malformed(f"items of {accession} is not a string")
    codes = [code.strip() for code in items.split(",") if code.strip()]
    return list(dict.fromkeys(codes))


def _default_cutoffs(first_seen: list[datetime]) -> pd.Series:
    return default_cutoffs_for(
        [stamp.date() for stamp in first_seen], load_nyse_calendar(), CUTOFF_LOOKAHEAD.days
    )


def events_from_submissions(
    submissions: dict[str, Any],
    instruments: list[str],
    *,
    ingested_utc: datetime,
    session_cutoffs_utc: pd.Series | None = None,
    latency: pd.Timedelta = DEFAULT_LATENCY,
) -> tuple[list[EventRecord], EdgarIngestReport]:
    """One aligned EventRecord per 8-K item, with one resolved impact per instrument.

    `instruments` are every symbol that trades on the filer's CIK (share classes); the record's
    `article_id` is `<accession>:<item>` and carries no instrument, so one filing item is one
    record however many symbols share the CIK. `ticker_hint` is the first instrument that looks
    like a ticker.

    Only forms 8-K and 8-K/A produce records. An item code with no rule is skipped and counted.
    A malformed document raises ValueError rather than yielding an empty result.
    """
    columns = _recent_columns(submissions)
    company = _company_name(submissions)
    rules = load_item_rules().items
    is_us_filer = _business_state_or_country(submissions) in load_us_state_codes()
    if not instruments or len(set(instruments)) != len(instruments):
        raise ValueError(f"instruments must be a non-empty list without repeats, got {instruments}")
    ticker_hint = next((name for name in instruments if _TICKER_PATTERN.match(name)), None)
    parser_version = edgar_parser_version()
    filings_read = len(columns["form"])
    skipped_items = skipped_forms = skipped_without_items = 0
    records: list[EventRecord] = []
    for accession, accepted_text, form, items in zip(
        columns["accessionNumber"],
        columns["acceptanceDateTime"],
        columns["form"],
        columns["items"],
        strict=True,
    ):
        if form not in EIGHT_K_FORMS:
            skipped_forms += 1
            continue
        accepted = _accepted_utc(accepted_text, accession)
        codes = _item_codes(items, accession)
        if not codes:
            skipped_without_items += 1
            continue
        for item in codes:
            rule = rules.get(item)
            if rule is None:
                skipped_items += 1
                continue
            records.append(
                _record(
                    accession,
                    item,
                    rule,
                    form,
                    accepted,
                    company,
                    instruments,
                    ticker_hint,
                    is_us_filer,
                    ingested_utc,
                    parser_version,
                )
            )
    if records:
        cutoffs = (
            session_cutoffs_utc
            if session_cutoffs_utc is not None
            else _default_cutoffs([r.first_seen_utc for r in records])
        )
        records = align_records(records, cutoffs, latency)
        if session_cutoffs_utc is None:  # the default cutoffs start with the calendar's years
            records = unalign_before(records, calendar_start_utc(load_nyse_calendar()))
    return records, EdgarIngestReport(
        filings_read, len(records), skipped_items, skipped_forms, skipped_without_items
    )


def _record(
    accession: str,
    item: str,
    rule: ItemRule,
    form: str,
    accepted: datetime,
    company: str,
    instruments: list[str],
    ticker_hint: str | None,
    is_us_filer: bool,
    ingested_utc: datetime,
    parser_version: str,
) -> EventRecord:
    article_id = f"{accession}:{item}"
    country = "USA" if is_us_filer else None
    event = NewsEvent(
        schema_version="1.0",
        event_type=rule.event_type,
        actor={"name": company, "kind": "company", "country_iso3": country},
        action=rule.description,
        impacted=[Impact(name=company, kind="company", ticker_hint=ticker_hint, direction_step=0)],
        severity=rule.severity,
        macro_direction_step=0,
        confidence="high",
        horizon="weeks",
        novelty="update" if form == AMENDMENT_FORM else "new",
        is_scheduled=rule.is_scheduled,
        evidence_span=f"8-K item {item}",
    )
    return EventRecord(
        article_id=article_id,
        body_sha256=hashlib.sha256(article_id.encode("utf-8")).hexdigest(),
        source=SOURCE,
        published_utc=accepted,
        first_seen_utc=accepted,
        ingested_utc=ingested_utc,
        available_utc=None,
        effective_session=None,
        parser_version=parser_version,
        event=event,
        resolved_impacts=[
            ResolvedImpact(
                instrument=instrument, kind="company", direction=0.0, match_method="cik_map"
            )
            for instrument in instruments
        ],
        linked_countries=["USA"] if is_us_filer else [],
    )
