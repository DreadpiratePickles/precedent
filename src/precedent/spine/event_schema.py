"""Closed-vocabulary contract for what the LLM news parser may emit (schema version 1.0)."""
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EventType(StrEnum):
    """What kind of event the article reports."""

    earnings = "earnings"
    guidance = "guidance"
    mna = "mna"
    capital_return = "capital_return"
    management_change = "management_change"
    product = "product"
    legal_regulatory = "legal_regulatory"
    labor = "labor"
    supply_disruption = "supply_disruption"
    cyber_incident = "cyber_incident"
    credit_rating = "credit_rating"
    macro_data_release = "macro_data_release"
    central_bank = "central_bank"
    fiscal_policy = "fiscal_policy"
    trade_or_sanctions = "trade_or_sanctions"
    armed_conflict = "armed_conflict"
    election_or_political = "election_or_political"
    natural_disaster = "natural_disaster"
    commodity_supply = "commodity_supply"
    other = "other"
    no_market_event = "no_market_event"


class ActorKind(StrEnum):
    """Who performed the action in the article."""

    company = "company"
    government = "government"
    central_bank = "central_bank"
    regulator = "regulator"
    international_org = "international_org"
    person = "person"
    other = "other"


class ImpactKind(StrEnum):
    """What kind of thing the event touches."""

    company = "company"
    sector = "sector"
    commodity = "commodity"
    country = "country"
    index = "index"
    currency = "currency"


class Horizon(StrEnum):
    """How long the event is expected to matter, as described by the article."""

    intraday = "intraday"
    days = "days"
    weeks = "weeks"
    months = "months"
    structural = "structural"


class Novelty(StrEnum):
    """Whether the article reports something new."""

    new = "new"
    update = "update"
    recap = "recap"
    opinion = "opinion"


class Confidence(StrEnum):
    """The parser's own confidence in its reading of the article."""

    low = "low"
    medium = "medium"
    high = "high"


CONFIDENCE_WEIGHTS: dict[Confidence, float] = {
    Confidence.low: 0.3,
    Confidence.medium: 0.6,
    Confidence.high: 0.9,
}

_NO_EXTRA_KEYS = ConfigDict(extra="forbid")


class Entity(BaseModel):
    """The actor named in the article."""

    model_config = _NO_EXTRA_KEYS
    name: str = Field(min_length=1, max_length=80)
    kind: ActorKind
    country_iso3: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")


class Impact(BaseModel):
    """One thing the event is said to help or hurt, in five steps from -2 to +2."""

    model_config = _NO_EXTRA_KEYS
    name: str = Field(min_length=1, max_length=80)
    kind: ImpactKind
    ticker_hint: str | None = Field(default=None, pattern=r"^[A-Z][A-Z0-9.\-]{0,5}$")
    direction_step: Literal[-2, -1, 0, 1, 2]

    @property
    def direction(self) -> float:
        """Direction scaled to -1.0 .. 1.0."""
        return self.direction_step / 2


class NewsEvent(BaseModel):
    """One news article reduced to one structured event; every scored field is a closed enum."""

    model_config = _NO_EXTRA_KEYS
    schema_version: Literal["1.0"]
    event_type: EventType
    actor: Entity
    action: str = Field(min_length=1, max_length=120)
    impacted: list[Impact] = Field(max_length=8)
    severity: Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    macro_direction_step: Literal[-2, -1, 0, 1, 2]
    confidence: Confidence
    horizon: Horizon
    novelty: Novelty
    is_scheduled: bool
    evidence_span: str = Field(min_length=1, max_length=200)

    @property
    def macro_direction(self) -> float:
        """Risk-appetite direction scaled to -1.0 .. 1.0."""
        return self.macro_direction_step / 2

    @property
    def confidence_weight(self) -> float:
        """Numeric weight for the parser's confidence label."""
        return CONFIDENCE_WEIGHTS[self.confidence]


def news_event_json_schema() -> dict:
    """Return the JSON Schema that a grammar-constrained decoder should enforce."""
    return NewsEvent.model_json_schema()
