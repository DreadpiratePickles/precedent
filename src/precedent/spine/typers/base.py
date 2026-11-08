"""The typer protocol: text in, a typed result out. Every typer returns the same value object."""

from __future__ import annotations

from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, model_validator

from precedent.spine.event_key import EventKey

TYPING_FIELDS = (
    "event_type",
    "actor_kind",
    "impacted_kinds",
    "country_group",
    "severity_band",
    "is_scheduled",
    "direction_sign",
)


class TypedText(BaseModel):
    """A typer's answer. `typed` carries a key and the evidence span; the others carry neither."""

    model_config = ConfigDict(frozen=True, extra="forbid", arbitrary_types_allowed=True)

    status: Literal["typed", "untyped", "refused_invalid"]
    key: EventKey | None
    reason: str
    evidence_span: str | None
    typer_version: str

    @model_validator(mode="after")
    def _key_and_span_exactly_when_typed(self) -> Self:
        if self.status == "typed":
            if self.key is None or not self.evidence_span:
                raise ValueError("a typed result needs a key and an evidence_span")
        elif self.key is not None or self.evidence_span is not None:
            raise ValueError(f"a {self.status} result carries neither a key nor an evidence_span")
        return self


class Typer(Protocol):
    version: str

    def type_text(self, text: str) -> TypedText: ...
