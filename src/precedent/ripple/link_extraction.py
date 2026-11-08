"""Model-extracted links (graph G4): the model reads a passage, code decides what is kept.

The model emits structure only: who the filer's counterparty is, which relation, and a quote.
It never supplies a weight and never forecasts. A link is kept only if its quoted evidence is
in the passage the model was shown, its counterparty resolves to a known node and the
counterparty is named inside that quote. Nothing here calls a model: the extractor is injected
(`EventExtractor`), so tests use a fake.

Core dependencies only (pydantic, pandas).
"""

from __future__ import annotations

import re
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
_PROMPT_FILE_PATTERN = re.compile(r"link_prompt_v[0-9]+\.txt")
MAX_LINKS_PER_PASSAGE = 20
_MAX_ERROR_CHARS = 300
_EXTRA_LOCATION = "<extra>"


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
    that resolves to the filer itself, a self-link, is counted here); `name_not_in_evidence`
    those whose counterparty name does not occur inside the quoted span (a true sentence that
    does not name the counterparty is no evidence for the link); `kept` the links returned
    (a repeat of the same source and relation is dropped and counted in `duplicates`), so
    `proposed == evidence_missing + name_not_in_evidence + unresolved + duplicates + kept`.
    When the whole output is unusable (bad JSON or a schema violation), `invalid_output` is
    True, `error` says why without echoing the output, and every count is 0.
    """

    proposed: int
    evidence_missing: int
    unresolved: int
    kept: int
    invalid_output: bool = False
    error: str | None = None
    duplicates: int = 0
    name_not_in_evidence: int = 0


def _check_prompt_file(prompt_file: str) -> None:
    if not _PROMPT_FILE_PATTERN.fullmatch(prompt_file):
        raise ValueError("prompt_file must be a bare file name like link_prompt_v1.txt")


def load_link_prompt(*, prompt_file: str = LINK_PROMPT_FILE) -> str:
    """A prompt template under `references/` with `{filer}` and `{passage}`.

    `prompt_file` is a bare name matching `link_prompt_v<digits>.txt` (the one file the
    prompt-revision loop may add to); anything else raises ValueError.
    """
    _check_prompt_file(prompt_file)
    return (config.references_dir() / prompt_file).read_text(encoding="utf-8")


def _normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def _require_text(name: str, value: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{name} must be non-empty")


_KNOWN_LOCATIONS = frozenset(ExtractedLinks.model_fields) | frozenset(ExtractedLink.model_fields)


def _safe_location(loc: tuple[int | str, ...]) -> str:
    """Indices and the schema's own field names only; a model-chosen key becomes `<extra>`."""
    parts = [
        str(part) if isinstance(part, int) or part in _KNOWN_LOCATIONS else _EXTRA_LOCATION
        for part in loc
    ]
    return ".".join(parts) or "<root>"


def _failure_text(exc: ValidationError) -> str:
    problems = exc.errors(include_url=False, include_context=False, include_input=False)
    text = "; ".join(f"{_safe_location(p['loc'])}: {p['msg']}" for p in problems)
    return text[:_MAX_ERROR_CHARS]


def _quote_is_in(span: str, passage: str, *, ignore_case: bool = False) -> bool:
    """True if the whitespace-normalised span occurs in the normalised passage on word edges.

    A span edge that is a letter or digit must not sit inside a longer word, so a quote cut
    mid-word ("... for 1" out of "... for 14%") does not count. `ignore_case` compares
    case-insensitively (used for the counterparty name inside the span).
    """
    quote, text = _normalise_whitespace(span), _normalise_whitespace(passage)
    if ignore_case:
        quote, text = quote.casefold(), text.casefold()
    if not quote:
        return False
    start = text.find(quote)
    while start != -1:
        end = start + len(quote)
        starts_clean = not (quote[0].isalnum() and start > 0 and text[start - 1].isalnum())
        ends_clean = not (quote[-1].isalnum() and end < len(text) and text[end].isalnum())
        if starts_clean and ends_clean:
            return True
        start = text.find(quote, start + 1)
    return False


def parse_links(
    filer: str,
    passage: str,
    accepted_utc: datetime,
    source_doc_id: str,
    extractor: EventExtractor,
    extractor_version: str,
    resolve: Callable[[str], str | None],
    *,
    prompt_file: str = LINK_PROMPT_FILE,
) -> tuple[list[Link], LinkParseReport]:
    """Ask the extractor once about `passage` and keep only the links code can vouch for.

    Direction: the counterparty is the link's `source_node` and `filer` its `receiver_node`,
    because a ripple channel is named for where the shock comes from (`ripple_customer` is a
    shock arriving from a company's customers). So `customer_of` means the counterparty is a
    customer of the filer. Every kept link starts at `accepted_utc` (the filing's acceptance
    time), expires after `annual_link_expiry_months` and has weight 1.0; the model never gives
    a weight. `resolve` maps a counterparty name to a node name, or None (or an empty name) when
    it is unknown. A link is kept only if the counterparty name occurs inside its evidence span
    (whitespace-normalised, case-insensitive, on word edges): a model must not attach a true
    sentence that never names the counterparty (`LinkParseReport.name_not_in_evidence`).
    `prompt_file` names the prompt version under `references/` (see
    `load_link_prompt`); the revision loop's runner passes it.

    Never raises on bad model output (see `LinkParseReport.invalid_output`). Raises ValueError
    for bad caller input: a naive `accepted_utc`, an empty `filer`, `passage`, `source_doc_id`
    or `extractor_version`, an `extractor_version` starting with `rules-`, or a bad `prompt_file`.
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
    _check_prompt_file(prompt_file)
    if extractor_version.startswith(RULES_EXTRACTOR_PREFIX):
        raise ValueError(
            f"extractor_version must not start with {RULES_EXTRACTOR_PREFIX!r}: "
            "model-extracted links are not rule-built"
        )
    valid_from = pd.Timestamp(accepted_utc).tz_convert("UTC")
    months = load_ripple_params()["annual_link_expiry_months"]
    valid_to = valid_from + pd.DateOffset(months=months)

    prompt = load_link_prompt(prompt_file=prompt_file).format(filer=filer, passage=passage)
    raw = extractor.generate(prompt, ExtractedLinks.model_json_schema())
    try:
        proposed = ExtractedLinks.model_validate_json(raw, strict=True).links
    except ValidationError as exc:
        return [], LinkParseReport(0, 0, 0, 0, invalid_output=True, error=_failure_text(exc))

    evidence_missing = unresolved = duplicates = name_not_in_evidence = 0
    kept: dict[tuple[str, Relation], Link] = {}
    for proposal in proposed:
        if not _quote_is_in(proposal.evidence_span, passage):
            evidence_missing += 1
            continue
        source = resolve(proposal.counterparty_name)
        if source is None or not source.strip() or source == filer:
            unresolved += 1
            continue
        if not _quote_is_in(proposal.counterparty_name, proposal.evidence_span, ignore_case=True):
            name_not_in_evidence += 1
            continue
        key = (source, proposal.relation)
        if key in kept:
            duplicates += 1
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
    report = LinkParseReport(
        len(proposed),
        evidence_missing,
        unresolved,
        len(links),
        duplicates=duplicates,
        name_not_in_evidence=name_not_in_evidence,
    )
    assert report.proposed == (
        evidence_missing + name_not_in_evidence + unresolved + duplicates + len(links)
    )
    return links, report
