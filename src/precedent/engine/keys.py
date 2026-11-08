"""Event-key helpers for the engine: build a key from flags or a dict, and show how it was typed."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import fields
from typing import Any, get_args

from precedent.spine.event_key import (
    CountryGroup,
    EventKey,
    SeverityBand,
    build_event_key,
    severity_band,
)
from precedent.spine.event_schema import ActorKind, EventType, ImpactKind

__all__ = [
    "EventKey",
    "build_event_key",
    "key_from_dict",
    "key_from_flags",
    "match_fields",
    "severity_band",
    "typed_as_line",
]

_DIRECTION_SIGNS = (-1, 0, 1)


def _require_member(name: str, value: object, allowed: Iterable[object]) -> None:
    allowed_values = tuple(allowed)
    if value not in allowed_values:
        raise ValueError(f"unknown {name}: {value!r}; expected one of {allowed_values}")


def key_from_flags(
    event_type: str,
    actor_kind: str,
    severity_band: str,
    is_scheduled: bool,
    country_group: str = "none",
    impacted_kinds: Iterable[str] = (),
    direction_sign: int = 0,
) -> EventKey:
    """Build a validated key; every enum-like field is checked against its source of truth."""
    _require_member("event_type", event_type, (member.value for member in EventType))
    _require_member("actor_kind", actor_kind, (member.value for member in ActorKind))
    _require_member("severity_band", severity_band, get_args(SeverityBand))
    _require_member("country_group", country_group, get_args(CountryGroup))
    if isinstance(impacted_kinds, str):
        raise ValueError(
            f"impacted_kinds must be a collection of kinds, not a single string: {impacted_kinds!r}"
        )
    kinds = tuple(impacted_kinds)
    for kind in kinds:
        _require_member("impacted_kinds", kind, (member.value for member in ImpactKind))
    if isinstance(direction_sign, bool):
        raise ValueError(f"unknown direction_sign: {direction_sign!r}; expected one of -1, 0, 1")
    _require_member("direction_sign", direction_sign, _DIRECTION_SIGNS)
    if not isinstance(is_scheduled, bool):
        raise ValueError(f"unknown is_scheduled: {is_scheduled!r}; expected a bool")
    return EventKey(
        event_type=event_type,
        actor_kind=actor_kind,
        impacted_kinds=tuple(sorted(set(kinds))),
        country_group=country_group,  # type: ignore[arg-type]
        severity_band=severity_band,  # type: ignore[arg-type]
        is_scheduled=is_scheduled,
        direction_sign=direction_sign,
    )


def key_from_dict(d: Mapping[str, Any]) -> EventKey:
    """Build a key from a mapping with exactly the `EventKey` field names."""
    known = {f.name for f in fields(EventKey)}
    unknown = sorted(set(d) - known)
    if unknown:
        raise ValueError(f"unknown event key field(s): {', '.join(unknown)}")
    required = {"event_type", "actor_kind", "severity_band", "is_scheduled"}
    missing = sorted(required - set(d))
    if missing:
        raise ValueError(f"missing event key field(s): {', '.join(missing)}")
    return key_from_flags(**d)


def match_fields(key: EventKey) -> tuple[str, str, str, str, bool]:
    """The five fields a precedent must share with the new event to match."""
    return (
        key.event_type,
        key.actor_kind,
        key.country_group,
        key.severity_band,
        key.is_scheduled,
    )


def typed_as_line(key: EventKey) -> str:
    """The card line that says how the event was typed."""
    schedule = "scheduled" if key.is_scheduled else "unscheduled"
    return (
        f"{key.event_type} / {key.actor_kind} / {key.country_group}"
        f" / severity {key.severity_band} / {schedule}"
    )
