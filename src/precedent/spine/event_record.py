"""The stored event record: the model's NewsEvent wrapped in a deterministic envelope."""

from datetime import date
from typing import Annotated

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, StringConstraints

from precedent.spine.event_schema import NewsEvent

Iso3Code = Annotated[str, StringConstraints(pattern=r"^[A-Z]{3}$")]


class ResolvedImpact(BaseModel):
    """One impacted name matched to an instrument id, with how it was matched."""

    model_config = ConfigDict(extra="forbid")
    instrument: str
    kind: str
    direction: float
    match_method: str


class EventRecord(BaseModel):
    """One parsed article, keyed by (article_id, parser_version). Timestamps are tz-aware."""

    model_config = ConfigDict(extra="forbid")
    article_id: str
    body_sha256: str
    source: str
    published_utc: AwareDatetime | None
    first_seen_utc: AwareDatetime
    ingested_utc: AwareDatetime
    available_utc: AwareDatetime | None
    effective_session: date | None
    parser_version: str
    event: NewsEvent
    resolved_impacts: list[ResolvedImpact]
    linked_countries: list[Iso3Code] = Field(
        description="Actor1 and Actor2 countries (GDELT), the filer's country (EDGAR), or empty."
    )
