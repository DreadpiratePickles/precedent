"""The event key: a discrete description of an event, used to find similar past events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Literal, get_args

from precedent.config import references_dir
from precedent.spine.event_record import EventRecord

CountryGroup = Literal["US", "DM", "EM", "none"]
SeverityBand = Literal["low", "mid", "high"]

_NO_KINDS = "none"
_SCHEDULED = "scheduled"
_UNSCHEDULED = "unscheduled"
_KEY_PARTS = 7


@dataclass(frozen=True)
class EventKey:
    """Seven discrete fields; the string form is stable and is what the store indexes."""

    event_type: str
    actor_kind: str
    impacted_kinds: tuple[str, ...]
    country_group: CountryGroup
    severity_band: SeverityBand
    is_scheduled: bool
    direction_sign: int

    def as_string(self) -> str:
        kinds = "+".join(self.impacted_kinds) or _NO_KINDS
        schedule = _SCHEDULED if self.is_scheduled else _UNSCHEDULED
        return "/".join(
            [
                self.event_type,
                self.actor_kind,
                kinds,
                self.country_group,
                self.severity_band,
                schedule,
                str(self.direction_sign),
            ]
        )

    @classmethod
    def from_string(cls, text: str) -> EventKey:
        parts = text.split("/")
        if len(parts) != _KEY_PARTS:
            raise ValueError(f"event key needs {_KEY_PARTS} parts, got {len(parts)}: {text!r}")
        event_type, actor_kind, kinds, group, band, schedule, sign = parts
        if not event_type or not actor_kind or not kinds:
            raise ValueError(f"event key has an empty part: {text!r}")
        if group not in get_args(CountryGroup):
            raise ValueError(f"unknown country group {group!r} in {text!r}")
        if band not in get_args(SeverityBand):
            raise ValueError(f"unknown severity band {band!r} in {text!r}")
        if schedule not in (_SCHEDULED, _UNSCHEDULED):
            raise ValueError(f"unknown schedule flag {schedule!r} in {text!r}")
        if sign not in ("-1", "0", "1"):
            raise ValueError(f"direction sign must be -1, 0 or 1, got {sign!r} in {text!r}")
        return cls(
            event_type=event_type,
            actor_kind=actor_kind,
            impacted_kinds=() if kinds == _NO_KINDS else tuple(kinds.split("+")),
            country_group=group,  # type: ignore[arg-type]
            severity_band=band,  # type: ignore[arg-type]
            is_scheduled=schedule == _SCHEDULED,
            direction_sign=int(sign),
        )


def severity_band(severity: int) -> SeverityBand:
    """Map severity 1-10 to low (1-3), mid (4-6) or high (7-10)."""
    if not 1 <= severity <= 10:
        raise ValueError(f"severity must be 1 to 10, got {severity}")
    if severity <= 3:
        return "low"
    return "mid" if severity <= 6 else "high"


@cache
def _load_country_groups(path: Path) -> dict[str, CountryGroup]:
    groups = json.loads(path.read_text(encoding="utf-8"))["groups"]
    mapping: dict[str, CountryGroup] = {}
    for group, members in groups.items():
        for iso3 in members:
            if iso3 in mapping:
                raise ValueError(f"{iso3} appears in more than one country group in {path}")
            mapping[iso3] = group
    return mapping


def load_country_groups(path: Path | None = None) -> dict[str, CountryGroup]:
    """Return ISO3 -> group from references/country_groups.json; cached per path."""
    return _load_country_groups(path or references_dir() / "country_groups.json")


def country_group_of(iso3: str | None) -> CountryGroup:
    """Group of a country code; `none` for no country; an unlisted code is an error."""
    if iso3 is None:
        return "none"
    try:
        return load_country_groups()[iso3]
    except KeyError:
        raise ValueError(f"country code {iso3!r} is not in country_groups.json") from None


def _sign(value: float) -> int:
    return (value > 0) - (value < 0)


def build_event_key(record: EventRecord) -> EventKey:
    """Derive the key from the record's parsed event only; time and ingest fields play no part."""
    event = record.event
    macro_sign = _sign(event.macro_direction_step)
    direction_sign = macro_sign or _sign(sum(i.direction_step for i in event.impacted))
    return EventKey(
        event_type=event.event_type.value,
        actor_kind=event.actor.kind.value,
        impacted_kinds=tuple(sorted({i.kind.value for i in event.impacted})),
        country_group=country_group_of(event.actor.country_iso3),
        severity_band=severity_band(event.severity),
        is_scheduled=event.is_scheduled,
        direction_sign=direction_sign,
    )
