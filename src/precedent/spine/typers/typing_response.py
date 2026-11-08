"""What a model typer must answer, how that answer is checked, and the HTTP loop both share."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from precedent.config import ConfigError, references_dir
from precedent.spine.event_key import EventKey
from precedent.spine.event_parser import _normalise_whitespace  # the parser's evidence rule
from precedent.spine.event_schema import ActorKind, EventType, ImpactKind
from precedent.spine.http import (
    Clock,
    HttpError,
    HttpRequest,
    HttpResponse,
    Transport,
    send_with_retries,
)
from precedent.spine.typers.base import TypedText

PROMPT_FILE = "typing_prompt_v1.txt"
MAX_RETRIES = 2
_UNSUPPORTED_KEYWORDS = frozenset(
    {"minLength", "maxLength", "minimum", "maximum", "maxItems", "minItems"}
)


class TypingResponse(BaseModel):
    """The seven key fields plus the evidence span; the model answers this and nothing else."""

    model_config = ConfigDict(extra="forbid", strict=True)

    event_type: EventType
    actor_kind: ActorKind
    impacted_kinds: list[ImpactKind]
    country_group: Literal["US", "DM", "EM", "none"]
    severity_band: Literal["low", "mid", "high"]
    is_scheduled: bool
    direction_sign: Literal[-1, 0, 1]
    evidence_span: str


def _reduce_to_structured_output_subset(node: Any) -> Any:
    if isinstance(node, dict):
        reduced = {
            key: _reduce_to_structured_output_subset(value)
            for key, value in node.items()
            if key not in _UNSUPPORTED_KEYWORDS
        }
        if reduced.get("type") == "object":
            reduced["additionalProperties"] = False
        return reduced
    if isinstance(node, list):
        return [_reduce_to_structured_output_subset(value) for value in node]
    return node


def typing_response_json_schema() -> dict:
    """The response schema in the subset both structured-output decoders accept."""
    return _reduce_to_structured_output_subset(TypingResponse.model_json_schema())


def _refused(reason: str, typer_version: str) -> TypedText:
    return TypedText(
        status="refused_invalid",
        key=None,
        reason=reason,
        evidence_span=None,
        typer_version=typer_version,
    )


def untyped(reason: str, typer_version: str) -> TypedText:
    return TypedText(
        status="untyped", key=None, reason=reason, evidence_span=None, typer_version=typer_version
    )


def to_typed_text(raw_json: str, text: str, typer_version: str) -> TypedText:
    """Validate model output against the schema and the source text; never raises on bad output."""
    try:
        response = TypingResponse.model_validate_json(raw_json, strict=True)
    except ValidationError as error:
        problems = error.errors(include_url=False, include_context=False, include_input=False)
        detail = "; ".join(
            f"{'.'.join(map(str, p['loc'])) or '<root>'}: {p['msg']}" for p in problems
        )
        return _refused(f"schema: {detail}", typer_version)
    evidence = _normalise_whitespace(response.evidence_span)
    if not evidence or evidence not in _normalise_whitespace(text):
        return _refused("evidence_span not in text", typer_version)
    if response.event_type == EventType.no_market_event:
        return untyped("model: no_market_event", typer_version)
    key = EventKey(
        event_type=response.event_type.value,
        actor_kind=response.actor_kind.value,
        impacted_kinds=tuple(sorted({kind.value for kind in response.impacted_kinds})),
        country_group=response.country_group,
        severity_band=response.severity_band,
        is_scheduled=response.is_scheduled,
        direction_sign=response.direction_sign,
    )
    return TypedText(
        status="typed",
        key=key,
        reason="model",
        evidence_span=response.evidence_span,
        typer_version=typer_version,
    )


def load_typing_prompt() -> str:
    path = references_dir() / PROMPT_FILE
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise ConfigError(f"cannot read the typing prompt at {path}: {error}") from error


def prompt_digest(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]


class ReplyRefused(Exception):
    """A reply that cannot be read as an answer; carries the refusal reason."""


class _MaxCallsReached(Exception):
    pass


class _CountingTransport:
    """Counts every attempt, retries included, and refuses to send past the limit."""

    def __init__(self, inner: Transport, max_calls: int) -> None:
        self._inner = inner
        self._max_calls = max_calls
        self.calls_made = 0

    def send(self, request: HttpRequest) -> HttpResponse:
        if self.calls_made >= self._max_calls:
            raise _MaxCallsReached
        self.calls_made += 1
        return self._inner.send(request)


class ModelTyper:
    """Shared flow of the model typers: dry run, call limit, bounded retries, reply parsing.

    A subclass sets `version`, builds the request, and turns the reply body into the model's
    raw JSON text (raising `ReplyRefused` for a refusal, truncation or odd shape)."""

    version: str

    def __init__(self, transport: Transport, clock: Clock, max_calls: int, dry_run: bool) -> None:
        self._counter = _CountingTransport(transport, max_calls)
        self._clock = clock
        self._dry_run = dry_run

    @property
    def calls_made(self) -> int:
        return self._counter.calls_made

    def _build_request(self, text: str) -> HttpRequest:
        raise NotImplementedError

    def _model_json(self, body: Any) -> str:
        raise NotImplementedError

    def type_text(self, text: str) -> TypedText:
        if self._dry_run:
            return untyped("dry-run", self.version)
        try:
            response = send_with_retries(
                self._counter, self._clock, self._build_request(text), max_retries=MAX_RETRIES
            )
        except _MaxCallsReached:
            return _refused("max_calls reached", self.version)
        except HttpError as error:
            return _refused(f"http {error.status}", self.version)
        try:
            raw_json = self._model_json(response.json())
        except ReplyRefused as refusal:
            return _refused(str(refusal), self.version)
        except ValueError as error:  # the body is not JSON (JSONDecodeError, UnicodeDecodeError)
            return _refused(f"schema: reply body is not JSON: {error}", self.version)
        return to_typed_text(raw_json, text, self.version)

