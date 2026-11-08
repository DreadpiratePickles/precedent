"""Dream's records: claims about a subject, and the tensions, shifts and resolutions found in them.

Every timestamp is timezone-aware and held in UTC; a naive one is rejected. Identifiers are
content hashes, so the same claim read twice has the same id.
"""

import hashlib
from datetime import UTC, date, datetime
from enum import Enum, StrEnum
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


def _to_utc(moment: datetime) -> datetime:
    return moment.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]


class Attribute(StrEnum):
    """What a claim is about. A proposed first list, extended by the owner's approval."""

    guidance_direction = "guidance_direction"
    capex_direction = "capex_direction"
    headcount_direction = "headcount_direction"
    capacity_direction = "capacity_direction"
    demand_direction = "demand_direction"
    pricing_direction = "pricing_direction"
    credit_outlook = "credit_outlook"
    supplier_relationship = "supplier_relationship"
    customer_relationship = "customer_relationship"
    regulatory_status = "regulatory_status"
    litigation_status = "litigation_status"
    event_direction = "event_direction"  # sign of an event record's direction_step


class Period(StrEnum):
    current_quarter = "current_quarter"
    next_quarter = "next_quarter"
    fiscal_year = "fiscal_year"
    unspecified = "unspecified"


class Value(StrEnum):
    up = "up"
    down = "down"
    flat = "flat"
    started = "started"
    ended = "ended"
    ongoing = "ongoing"
    positive = "positive"
    negative = "negative"
    neutral = "neutral"


class TensionKind(StrEnum):
    value_conflict = "value_conflict"
    text_vs_price = "text_vs_price"
    asymmetric_link = "asymmetric_link"
    stale_reliance = "stale_reliance"


class ShiftKind(StrEnum):
    event_rate = "event_rate"
    direction_drift = "direction_drift"
    link_churn = "link_churn"


class Hypothesis(StrEnum):
    one_is_stale = "one_is_stale"
    different_scope = "different_scope"
    genuine_conflict = "genuine_conflict"
    extraction_error = "extraction_error"


def _digest(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _text(part: Any) -> str:
    return part.value if isinstance(part, Enum) else str(part)


def claim_id_for(
    subject: str,
    attribute: Attribute | str,
    period: Period | str,
    value: Value | str,
    source_doc_id: str,
) -> str:
    """The claim id: sha256 of the five identifying parts joined with `|`."""
    return _digest(subject, _text(attribute), _text(period), _text(value), source_doc_id)


def tension_id_for(kind: TensionKind | str, claim_ids: list[str], subject: str) -> str:
    """The tension id; the claim ids are sorted first so their order does not matter."""
    return _digest(_text(kind), *sorted(claim_ids), subject)


def shift_id_for(kind: ShiftKind | str, subject: str) -> str:
    """The shift id: one shift per kind and subject, re-fired in place by later sessions."""
    return _digest(_text(kind), subject)


def _check_or_fill_id(record: BaseModel, field: str, expected: str) -> None:
    supplied = getattr(record, field)
    if not supplied:
        setattr(record, field, expected)
    elif supplied != expected:
        raise ValueError(f"{field} {supplied!r} does not match its content hash {expected!r}")


class Claim(BaseModel):
    """One statement a document makes about a subject, valid from when it became known."""

    model_config = ConfigDict(extra="forbid")
    claim_id: str = ""
    subject: str  # an instrument id or a link id
    attribute: Attribute
    period: Period
    value: Value
    valid_from_utc: UtcDatetime
    valid_to_utc: UtcDatetime | None = None
    recorded_utc: UtcDatetime
    source_doc_id: str
    evidence_span: str | None = None
    extractor_version: str
    superseded_by: str | None = None
    expiry_reason: str | None = None

    @model_validator(mode="after")
    def _check_claim(self) -> "Claim":
        if not self.extractor_version.startswith("rules-") and not (
            self.evidence_span and self.evidence_span.strip()
        ):
            raise ValueError("a claim from a model extractor needs a non-empty evidence_span")
        if self.valid_to_utc is not None and self.valid_to_utc < self.valid_from_utc:
            raise ValueError("valid_to_utc is before valid_from_utc")
        _check_or_fill_id(
            self,
            "claim_id",
            claim_id_for(self.subject, self.attribute, self.period, self.value, self.source_doc_id),
        )
        return self


class Tension(BaseModel):
    """Claims, or a claim and a price, that do not sit together as of a session."""

    model_config = ConfigDict(extra="forbid")
    tension_id: str = ""
    kind: TensionKind
    claim_ids: list[str] = Field(min_length=1)
    subject: str
    opened_session: date
    detector_version: str
    details: dict[str, float | str] = Field(default_factory=dict)
    recorded_utc: UtcDatetime
    status: Literal["open", "closed", "acknowledged"] = "open"
    closed_session: date | None = None

    @model_validator(mode="after")
    def _check_tension(self) -> "Tension":
        if self.kind is TensionKind.value_conflict and len(self.claim_ids) < 2:
            raise ValueError("a value_conflict tension needs at least two claim ids")
        if (self.status == "closed") != (self.closed_session is not None):
            raise ValueError("closed_session is set exactly when status is 'closed'")
        _check_or_fill_id(
            self, "tension_id", tension_id_for(self.kind, self.claim_ids, self.subject)
        )
        return self


class Shift(BaseModel):
    """A silent change in a rate, a direction or a set of links, past a stated threshold."""

    model_config = ConfigDict(extra="forbid")
    shift_id: str = ""
    kind: ShiftKind
    subject: str
    detected_session: date
    statistic: float
    threshold: float
    window_sessions: int = Field(gt=0)
    detector_version: str
    recorded_utc: UtcDatetime

    @model_validator(mode="after")
    def _check_shift(self) -> "Shift":
        _check_or_fill_id(self, "shift_id", shift_id_for(self.kind, self.subject))
        return self


class Resolution(BaseModel):
    """A proposed explanation of a tension, accepted or rejected by the owner."""

    model_config = ConfigDict(extra="forbid")
    resolution_id: str
    tension_id: str
    hypothesis: Hypothesis
    evidence_span: str | None = None
    proposer_version: str
    status: Literal["proposed", "accepted", "rejected"]
    proposed_utc: UtcDatetime
    decided_utc: UtcDatetime | None = None

    @field_validator("resolution_id", "tension_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("identifier must not be blank")
        return value

    @model_validator(mode="after")
    def _check_resolution(self) -> "Resolution":
        if (self.status == "proposed") != (self.decided_utc is None):
            raise ValueError("decided_utc is set exactly when status is accepted or rejected")
        return self


def claim_json_schema() -> dict[str, Any]:
    """The JSON Schema of a stored claim; `references/dream/claim_schema_v1.json` equals it."""
    return Claim.model_json_schema()
