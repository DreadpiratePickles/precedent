"""Claims built by rule from stores that already exist: event records and dated links. No model.

An event record gives one `event_direction` claim per resolved impact, at the story's first
report only, valid for `story_window` sessions. An 8-K record also gives a typed claim where the
item map in `references/dream/claim_vocabulary.json` names one. A supplier or customer link gives
a relationship claim valid over the link's own dates.
"""

from __future__ import annotations

import json
from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from precedent.config import references_dir
from precedent.dream.claim_schema import Attribute, Claim, Period, Value, link_id_for
from precedent.engine.story_clusters import assign_story_clusters
from precedent.ripple.link_store import Link, Relation
from precedent.spine.event_record import EventRecord, ResolvedImpact
from precedent.spine.session_calendar import default_cutoffs_for, load_nyse_calendar

EVENTS_EXTRACTOR_VERSION = "rules-events-v1"
LINKS_EXTRACTOR_VERSION = "rules-links-v1"
EDGAR_SOURCE = "edgar"
FILER_MATCH_METHOD = "cik_map"  # how edgar_8k_events resolves the filer's own instruments
RELATIONSHIP_ATTRIBUTE = "relationship"
_VOCABULARY_FILE = Path("dream") / "claim_vocabulary.json"
_LINK_ATTRIBUTES = {
    Relation.SUPPLIER_OF: Attribute.supplier_relationship,
    Relation.CUSTOMER_OF: Attribute.customer_relationship,
}


@dataclass(frozen=True)
class RuleClaimReport:
    """What one run set aside or emitted, so no record disappears without a count."""

    records: int
    unaligned_skipped: int
    event_direction_claims: int
    typed_claims: int
    repeat_reports_skipped: int
    relationship_without_counterparty: int


@dataclass(frozen=True)
class _ItemRule:
    attribute: Attribute | None  # None: the item names a relationship (needs a counterparty)
    value: Value


def _load_item_map(path: Path | None) -> dict[str, _ItemRule | None]:
    """The 8-K item map, validated: each entry is null or an attribute and a value of the enums."""
    file_path = path if path is not None else references_dir() / _VOCABULARY_FILE
    try:
        raw = json.loads(file_path.read_text(encoding="utf-8"))["item_map_8k"]
        if not isinstance(raw, dict):
            raise TypeError("item_map_8k must be an object")
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"cannot read item_map_8k from {file_path}: {error!r}") from error
    rules: dict[str, _ItemRule | None] = {}
    for item, entry in raw.items():
        if item in ("proposed", "note"):
            continue
        rules[item] = None if entry is None else _parse_item_rule(item, entry, file_path)
    return rules


def _parse_item_rule(item: str, entry: Any, file_path: Path) -> _ItemRule:
    try:
        attribute_name, value_name = entry["attribute"], entry["value"]
        attribute = None if attribute_name == RELATIONSHIP_ATTRIBUTE else Attribute(attribute_name)
        return _ItemRule(attribute, Value(value_name))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"item {item!r} in {file_path} is malformed: {error!r}") from error


def _position_of(available_utc: datetime, cutoffs: list[pd.Timestamp]) -> int:
    """Index of the first cutoff at or after `available_utc` (the effective-session rule)."""
    return bisect_left(cutoffs, pd.Timestamp(available_utc))


def _default_cutoffs(aligned: list[EventRecord], story_window: int) -> pd.Series:
    dates = sorted(record.available_utc.date() for record in aligned if record.available_utc)
    return default_cutoffs_for(dates, load_nyse_calendar(), story_window * 3 + 14)


def _filer_instruments(record: EventRecord) -> set[str]:
    return {
        impact.instrument
        for impact in record.resolved_impacts
        if impact.kind == "company" and impact.match_method == FILER_MATCH_METHOD
    }


def _position_frame(
    aligned: list[EventRecord], cutoffs: list[pd.Timestamp]
) -> tuple[pd.DataFrame, list[tuple[EventRecord, ResolvedImpact, int]]]:
    rows: list[tuple[EventRecord, ResolvedImpact, int]] = []
    for record in aligned:
        assert record.available_utc is not None
        position = _position_of(record.available_utc, cutoffs)
        rows.extend((record, impact, position) for impact in record.resolved_impacts)
    frame = pd.DataFrame(
        {
            "event_id": [record.article_id for record, _, _ in rows],
            "key": [f"{r.event.event_type}:{i.instrument}" for r, i, _ in rows],
            "primary_actor": [record.event.actor.name for record, _, _ in rows],
            "effective_position": [position for _, _, position in rows],
            "available_utc": [record.available_utc for record, _, _ in rows],
        }
    )
    return frame, rows


def _direction_value(direction: float) -> Value:
    if direction > 0:
        return Value.positive
    return Value.negative if direction < 0 else Value.neutral


def _event_claim(
    record: EventRecord,
    impact: ResolvedImpact,
    attribute: Attribute,
    value: Value,
    valid_to_utc: datetime | None,
) -> Claim:
    assert record.available_utc is not None
    valid_from = record.available_utc
    return Claim(
        subject=impact.instrument,
        attribute=attribute,
        period=Period.unspecified,
        value=value,
        valid_from_utc=valid_from,
        valid_to_utc=valid_to_utc,
        recorded_utc=valid_from,  # D69: knowledge time is available_utc, not the ingest clock
        source_doc_id=record.article_id,
        evidence_span=record.event.evidence_span,
        extractor_version=EVENTS_EXTRACTOR_VERSION,
    )


def _valid_to(
    record: EventRecord, position: int, story_window: int, cutoffs: list[pd.Timestamp]
) -> datetime:
    end = position + story_window
    if end >= len(cutoffs):
        raise ValueError(
            f"session cutoffs end at position {len(cutoffs) - 1}, before {end} needed "
            f"for article {record.article_id!r}"
        )
    return cutoffs[end].to_pydatetime()


def _typed_claims(
    record: EventRecord, item_map: dict[str, _ItemRule | None]
) -> tuple[list[Claim], bool]:
    """The 8-K typed claims of one record, and whether a relationship item had no counterparty."""
    if record.source != EDGAR_SOURCE or ":" not in record.article_id:
        return [], False
    rule = item_map.get(record.article_id.rsplit(":", 1)[1])
    if rule is None:
        return [], False
    filer = _filer_instruments(record)
    if rule.attribute is None:
        has_counterparty = any(i.instrument not in filer for i in record.resolved_impacts)
        # A counterparty alone does not say supplier or customer; the rules parser never
        # resolves one, so no relationship claim is built from an item.
        return [], not has_counterparty
    # M9 (D76): a typed item describes the filer; a resolved counterparty must not inherit it.
    return [
        _event_claim(record, impact, rule.attribute, rule.value, None)
        for impact in record.resolved_impacts
        if impact.instrument in filer
    ], False


def claims_from_events_with_report(
    records: list[EventRecord],
    story_window: int = 3,
    session_cutoffs_utc: pd.Series | None = None,
    *,
    vocabulary_path: Path | None = None,
) -> tuple[list[Claim], RuleClaimReport]:
    """Claims from event records, with counts of what was emitted and what was set aside.

    A record with no `available_utc` is skipped and counted. Raises ValueError when the session
    cutoffs end before a record's story window does, or when the item map is malformed.
    """
    item_map = _load_item_map(vocabulary_path)
    aligned = [record for record in records if record.available_utc is not None]
    unaligned = len(records) - len(aligned)
    if not aligned:
        return [], RuleClaimReport(len(records), unaligned, 0, 0, 0, 0)
    cutoff_series = (
        session_cutoffs_utc
        if session_cutoffs_utc is not None
        else _default_cutoffs(aligned, story_window)
    )
    cutoffs = [pd.Timestamp(stamp) for stamp in cutoff_series]
    frame, rows = _position_frame(aligned, cutoffs)
    first_report = assign_story_clusters(frame, window_sessions=story_window)["is_first_report"]
    typed_claims: list[Claim] = []
    repeats = no_counterparty = 0
    for record in aligned:
        typed, missing = _typed_claims(record, item_map)
        typed_claims.extend(typed)
        no_counterparty += missing
    event_claims: list[Claim] = []
    for (record, impact, position), is_first in zip(rows, first_report, strict=True):
        if not is_first:
            repeats += 1
            continue
        valid_to = _valid_to(record, position, story_window, cutoffs)
        value = _direction_value(impact.direction)
        event_claims.append(
            _event_claim(record, impact, Attribute.event_direction, value, valid_to)
        )
    return event_claims + typed_claims, RuleClaimReport(
        records=len(records),
        unaligned_skipped=unaligned,
        event_direction_claims=len(event_claims),
        typed_claims=len(typed_claims),
        repeat_reports_skipped=repeats,
        relationship_without_counterparty=no_counterparty,
    )


def claims_from_events(
    records: list[EventRecord],
    story_window: int = 3,
    session_cutoffs_utc: pd.Series | None = None,
    *,
    vocabulary_path: Path | None = None,
) -> list[Claim]:
    """The claims of `claims_from_events_with_report`, without the report."""
    return claims_from_events_with_report(
        records, story_window, session_cutoffs_utc, vocabulary_path=vocabulary_path
    )[0]


def claims_from_links(links: list[Link]) -> list[Claim]:
    """One relationship claim per supplier or customer link; every other relation is excluded."""
    claims: list[Claim] = []
    for link in links:
        attribute = _LINK_ATTRIBUTES.get(link.relation)
        if attribute is None:
            continue
        claims.append(
            Claim(
                subject=link_id_for(
                    link.source_node, link.receiver_node, link.relation, link.source_doc_id
                ),
                attribute=attribute,
                period=Period.unspecified,
                value=Value.ongoing,
                valid_from_utc=link.valid_from_utc,
                valid_to_utc=link.valid_to_utc,
                recorded_utc=link.valid_from_utc,
                source_doc_id=link.source_doc_id,
                evidence_span=link.evidence_span,
                extractor_version=LINKS_EXTRACTOR_VERSION,
            )
        )
    return claims
