"""Article -> NewsEvent parsing step; the model sits behind the EventExtractor interface."""
import hashlib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from pydantic import ValidationError

from event_schema import NewsEvent, news_event_json_schema

DEFAULT_MAX_BODY_CHARS = 6000

PROMPT_TEMPLATE = """You convert one news article into one JSON object that follows the provided schema.
Rules:
1. Use only what the article states. Do not use outside knowledge about what happened later.
2. Do not predict prices or returns. direction_step and macro_direction_step describe whether the event, as described, is unfavourable (-2, -1), neutral (0) or favourable (1, 2) for the named entity or for risk appetite in general.
3. severity scale: 1-2 routine or expected; 3-4 notable for one company; 5-6 material for one company or notable for a sector; 7-8 material for a sector or a national economy; 9-10 systemic or global.
4. evidence_span is copied word for word from the article (at most 200 characters) and supports event_type and action.
5. novelty: "new" for a first report, "update" for new facts on a known story, "recap" for a summary of facts already reported, "opinion" for commentary.
6. If the article reports no event that could matter to markets, set event_type to "no_market_event", severity to 1 and impacted to [].
Return the JSON object only.

Title: {title}
Article:
{body}"""


class EventExtractor(Protocol):
    """Anything that turns a prompt into model text; transport errors belong to the adapter."""

    def generate(self, prompt: str, json_schema: dict) -> str: ...


class ParseStatus(str, Enum):
    """Outcome of parsing one article."""

    ok = "ok"
    no_market_event = "no_market_event"
    invalid_json = "invalid_json"
    schema_violation = "schema_violation"
    evidence_not_in_article = "evidence_not_in_article"


@dataclass(frozen=True)
class ParseResult:
    """Parse outcome plus what is needed to cache it and trace it."""

    status: ParseStatus
    event: NewsEvent | None
    error: str | None
    parser_version: str
    body_sha256: str


def _visible_body(body: str, max_body_chars: int) -> str:
    """Return the head of the body that the model is allowed to see."""
    if max_body_chars < 1:
        raise ValueError(f"max_body_chars must be >= 1, got {max_body_chars}")
    return body[:max_body_chars]


def build_prompt(title: str, body: str, max_body_chars: int = DEFAULT_MAX_BODY_CHARS) -> str:
    """Fill the prompt template, keeping only the head of the body."""
    return PROMPT_TEMPLATE.format(title=title, body=_visible_body(body, max_body_chars))


def _normalise_whitespace(text: str) -> str:
    return " ".join(text.split())


def parse_article(
    title: str,
    body: str,
    extractor: EventExtractor,
    parser_version: str,
    max_body_chars: int = DEFAULT_MAX_BODY_CHARS,
) -> ParseResult:
    """Call the extractor once and classify its output; never raises on bad model output."""
    body_sha256 = hashlib.sha256(body.encode("utf-8")).hexdigest()

    def outcome(status: ParseStatus, event: NewsEvent | None = None, error: str | None = None) -> ParseResult:
        return ParseResult(status, event, error, parser_version, body_sha256)

    raw = extractor.generate(build_prompt(title, body, max_body_chars), news_event_json_schema())
    try:
        event = NewsEvent.model_validate_json(raw, strict=True)
    except ValidationError as exc:
        problems = exc.errors(include_url=False, include_context=False, include_input=False)
        text = "; ".join(f"{'.'.join(map(str, p['loc'])) or '<root>'}: {p['msg']}" for p in problems)
        is_bad_json = any(p["type"] == "json_invalid" for p in problems)
        return outcome(ParseStatus.invalid_json if is_bad_json else ParseStatus.schema_violation, error=text)
    seen_text = _normalise_whitespace(f"{title}\n{_visible_body(body, max_body_chars)}")
    evidence = _normalise_whitespace(event.evidence_span)
    if not evidence or evidence not in seen_text:
        return outcome(ParseStatus.evidence_not_in_article, error="evidence_span is not a substring of the article text")
    if event.event_type.value == "no_market_event":
        return outcome(ParseStatus.no_market_event, event)
    return outcome(ParseStatus.ok, event)


def cache_key(body_sha256: str, parser_version: str) -> str:
    """Return the cache key for one (article body, parser version) pair."""
    if not re.fullmatch(r"[0-9a-f]{64}", body_sha256):
        raise ValueError("body_sha256 must be 64 lowercase hex characters")
    if not parser_version:
        raise ValueError("parser_version must be non-empty")
    return f"{parser_version}:{body_sha256}"
