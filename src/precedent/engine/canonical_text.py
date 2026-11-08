"""One line of text that describes a typed event by its structure alone.

The line is built only from enum values and role tokens: the event type, the actor's kind, the
country group, the severity band, whether the event was scheduled, the kinds of what it impacts,
the sign of its direction, its horizon and its novelty. No entity name, action or evidence span
reaches it, so a text embedder sees what kind of event this is, never who was named.
"""

from __future__ import annotations

from precedent.spine.event_key import build_event_key
from precedent.spine.event_record import EventRecord

__all__ = ["canonical_text"]

_NO_KINDS = "none"


def _role_token(kind: str) -> str:
    return f"[{kind.upper()}]"


def canonical_text(record: EventRecord) -> str:
    """The canonical one-line description of `record`'s parsed event.

    Shape: ``{event_type} by [ACTOR_KIND] in {country_group} severity {band}
    {scheduled|unscheduled} impacted {kinds} direction {sign} horizon {horizon} novelty
    {novelty}``. `kinds` are the impacted kinds as bracketed upper-case tokens, sorted,
    de-duplicated and joined by `+` (`none` when there are none); `sign` is the event key's
    direction sign (-1, 0 or 1). The key fields come from `build_event_key`, so the text can
    never disagree with the key the record is filed under.
    """
    key = build_event_key(record)
    event = record.event
    kinds = "+".join(_role_token(kind) for kind in key.impacted_kinds) or _NO_KINDS
    schedule = "scheduled" if key.is_scheduled else "unscheduled"
    return (
        f"{key.event_type} by {_role_token(key.actor_kind)} in {key.country_group} "
        f"severity {key.severity_band} {schedule} impacted {kinds} "
        f"direction {key.direction_sign} horizon {event.horizon.value} "
        f"novelty {event.novelty.value}"
    )
