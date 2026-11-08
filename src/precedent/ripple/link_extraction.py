"""Model-extracted links (graph G4): the model reads a passage, code decides what is kept.

The model emits structure only: who the filer's counterparty is, which relation, and a quote.
It never supplies a weight and never forecasts. A link is kept only if its quoted evidence is
in the passage the model was shown and its counterparty resolves to a known node. Nothing here
calls a model: the extractor is injected (`EventExtractor`), so tests use a fake.

Core dependencies only (pydantic, pandas).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

import pandas as pd
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)

from precedent import config
from precedent.ripple.link_store import RULES_EXTRACTOR_PREFIX, Link, Relation
from precedent.ripple.params import load_ripple_params
from precedent.spine.event_parser import EventExtractor

LINK_PROMPT_FILE = "link_prompt_v1.txt"
MAX_LINKS_PER_PASSAGE = 20
_MAX_ERROR_CHARS = 300


class CounterpartyKind(StrEnum):
    """What the named counterparty is."""

    COMPANY = "company"
    COMMODITY = "commodity"
    COUNTRY = "country"


_KIND_OF_RELATION = {
    Relation.CUSTOMER_OF: CounterpartyKind.COMPANY,
    Relation.SUPPLIER_OF: CounterpartyKind.COMPANY,
    Relation.PEER_OF: CounterpartyKind.COMPANY,
    Relation.INPUT_COMMODITY: CounterpartyKind.COMMODITY,
    Relation.OUTPUT_COMMODITY: CounterpartyKind.COMMODITY,
    Relation.COUNTRY_EXPOSURE: CounterpartyKind.COUNTRY,
}
_Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ExtractedLink(BaseModel):
    """One link as the model proposes it, from the counterparty's side (see `parse_links`)."""

    model_config = ConfigDict(extra="forbid")
    relation: Relation
    counterparty_name: _Text
    counterparty_kind: CounterpartyKind
    evidence_span: _Text

    @model_validator(mode="after")
    def _check_kind_matches_relation(self) -> Self:
        expected = _KIND_OF_RELATION[self.relation]
        if self.counterparty_kind != expected:
            raise ValueError(
                f"relation {self.relation.value} needs counterparty_kind {expected.value}"
            )
        return self


class ExtractedLinks(BaseModel):
    """The whole model answer for one passage: at most 20 links, empty when none is named."""

    model_config = ConfigDict(extra="forbid")
    links: list[ExtractedLink] = Field(max_length=MAX_LINKS_PER_PASSAGE)


@dataclass(frozen=True)
class LinkParseReport:
    """What happened to one passage's proposals.

    `proposed` is the number of links the model returned; `evidence_missing` those whose quote
    is not in the passage; `unresolved` those whose counterparty did not resolve (a counterparty
    that resolves to the filer itself, a self-link, is counted here); `kept` the links returned
    (a repeat of the same source and relation is dropped, so `kept` can be below the rest).
    When the whole output is unusable (bad JSON or a schema violation), `invalid_output` is
    True, `error` says why without echoing the output, and every count is 0.
    """

    proposed: int
    evidence_missing: int
    unresolved: int
    kept: int
    invalid_output: bool = False
    error: str | None = None


def load_link_prompt() -> str:
    """The prompt template, `references/link_prompt_v1.txt`, with `{filer}` and `{passage}`."""
    return (config.references_dir() / LINK_PROMPT_FILE).read_text(encoding="utf-8")


def _normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def _require_text(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


def _failure_text(exc: ValidationError) -> str:
    problems = exc.errors(include_url=False, include_context=False, include_input=False)
    text = "; ".join(f"{'.'.join(map(str, p['loc'])) or '<root>'}: {p['msg']}" for p in problems)
    return text[:_MAX_ERROR_CHARS]


def parse_links(
    filer: str,
    passage: str,
    accepted_utc: datetime,
    source_doc_id: str,
    extractor: EventExtractor,
    extractor_version: str,
    resolve: Callable[[str], str | None],
) -> tuple[list[Link], LinkParseReport]:
    """Ask the extractor once about `passage` and keep only the links code can vouch for.

    Direction: the counterparty is the link's `source_node` and `filer` its `receiver_node`,
    because a ripple channel is named for where the shock comes from (`ripple_customer` is a
    shock arriving from a company's customers). So `customer_of` means the counterparty is a
    customer of the filer. Every kept link starts at `accepted_utc` (the filing's acceptance
    time), expires after `annual_link_expiry_months` and has weight 1.0; the model never gives
    a weight. `resolve` maps a counterparty name to a node name, or None when it is unknown.

    Never raises on bad model output (see `LinkParseReport.invalid_output`). Raises ValueError
    for bad caller input: a naive `accepted_utc`, an empty `filer`, `passage`, `source_doc_id`
    or `extractor_version`, or an `extractor_version` starting with `rules-`.
    """
    if accepted_utc.tzinfo is None or accepted_utc.utcoffset() is None:
        raise ValueError("accepted_utc must be tz-aware")
    for name, value in (
        ("filer", filer),
        ("passage", passage),
        ("source_doc_id", source_doc_id),
        ("extractor_version", extractor_version),
    ):
        _require_text(name, value)
    if extractor_version.startswith(RULES_EXTRACTOR_PREFIX):
        raise ValueError(
            f"extractor_version must not start with {RULES_EXTRACTOR_PREFIX!r}: "
            "model-extracted links are not rule-built"
        )
    valid_from = pd.Timestamp(accepted_utc).tz_convert("UTC")
    months = load_ripple_params()["annual_link_expiry_months"]
    valid_to = valid_from + pd.DateOffset(months=months)

    prompt = load_link_prompt().format(filer=filer, passage=passage)
    raw = extractor.generate(prompt, ExtractedLinks.model_json_schema())
    try:
        proposed = ExtractedLinks.model_validate_json(raw, strict=True).links
    except ValidationError as exc:
        return [], LinkParseReport(0, 0, 0, 0, invalid_output=True, error=_failure_text(exc))

    seen_text = _normalise_whitespace(passage)
    evidence_missing = unresolved = 0
    kept: dict[tuple[str, Relation], Link] = {}
    for proposal in proposed:
        if _normalise_whitespace(proposal.evidence_span) not in seen_text:
            evidence_missing += 1
            continue
        source = resolve(proposal.counterparty_name)
        if source is None or source == filer:
            unresolved += 1
            continue
        key = (source, proposal.relation)
        if key in kept:
            continue
        kept[key] = Link(
            source_node=source,
            receiver_node=filer,
            relation=proposal.relation,
            weight=1.0,
            valid_from_utc=valid_from.to_pydatetime(),
            valid_to_utc=valid_to.to_pydatetime(),
            source_doc_id=source_doc_id,
            evidence_span=proposal.evidence_span,
            extractor_version=extractor_version,
        )
    links = list(kept.values())
    return links, LinkParseReport(len(proposed), evidence_missing, unresolved, len(links))
