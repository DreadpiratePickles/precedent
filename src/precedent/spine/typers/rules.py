"""The rules typer: a keyword table, no network, the default and the CI path."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from precedent.config import ConfigError, references_dir
from precedent.spine.event_key import EventKey, SeverityBand, load_country_groups
from precedent.spine.event_schema import ActorKind, EventType, ImpactKind
from precedent.spine.typers.base import TypedText


class _Direction(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    positive: list[str] = Field(default_factory=list, alias="1")
    negative: list[str] = Field(default_factory=list, alias="-1")


class _EventTypeRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    include: list[str]
    exclude: list[str] = Field(default_factory=list)
    actor_kind: ActorKind
    impacted_kinds: list[ImpactKind]
    direction: _Direction = Field(default_factory=_Direction)


class _SeverityRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    high: list[str]
    low: list[str]
    default: Literal["low", "mid", "high"]


class _ScheduleRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scheduled: list[str]
    unscheduled: list[str]
    default: Literal["scheduled", "unscheduled"]


class _RulesFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    event_types: dict[EventType, _EventTypeRule]
    severity: _SeverityRule
    scheduled: _ScheduleRule
    countries: dict[str, str] = Field(default_factory=dict)
    countries_case_sensitive: dict[str, str] = Field(default_factory=dict)
    event_type_countries: dict[EventType, dict[str, str]] = Field(default_factory=dict)


def _strip_comments(value: object) -> object:
    """Drop every key that starts with an underscore (documentation fields), recursively."""
    if isinstance(value, dict):
        return {k: _strip_comments(v) for k, v in value.items() if not k.startswith("_")}
    return value


def compile_phrase(phrase: str, *, case_sensitive: bool = False) -> re.Pattern[str]:
    """Whole-word pattern, case-insensitive unless asked. Spaces match any whitespace run; a
    trailing `*` lets the last word continue (`recall*` matches recalls and recalled)."""
    stem = phrase[:-1] if phrase.endswith("*") else phrase
    body = r"\s+".join(re.escape(word) for word in stem.split())
    tail = r"\w*" if phrase.endswith("*") else ""
    flags = 0 if case_sensitive else re.IGNORECASE
    return re.compile(rf"(?<!\w){body}{tail}(?!\w)", flags)


def _first_match(patterns: list[re.Pattern[str]], text: str) -> re.Match[str] | None:
    found = [m for p in patterns if (m := p.search(text))]
    return min(found, key=lambda m: m.start()) if found else None


def _any_match(patterns: list[re.Pattern[str]], text: str) -> bool:
    return any(p.search(text) for p in patterns)


@dataclass(frozen=True)
class _CompiledType:
    rule: _EventTypeRule
    include: list[re.Pattern[str]]
    exclude: list[re.Pattern[str]]
    positive: list[re.Pattern[str]]
    negative: list[re.Pattern[str]]
    countries: list[tuple[re.Pattern[str], str]]


def _compile_countries(
    table: dict[str, str], *, case_sensitive: bool = False
) -> list[tuple[re.Pattern[str], str]]:
    return [
        (compile_phrase(name, case_sensitive=case_sensitive), iso3) for name, iso3 in table.items()
    ]


class RulesTyper:
    def __init__(
        self, rules_path: Path | None = None, country_groups_path: Path | None = None
    ) -> None:
        path = rules_path or references_dir() / "typing_rules.json"
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ConfigError(f"cannot read typing rules at {path}: {error}") from error
        try:
            rules = _RulesFile.model_validate(_strip_comments(json.loads(raw)))
        except (ValueError, ValidationError) as error:
            raise ConfigError(f"typing rules in {path} are invalid: {error}") from error
        self.version = f"rules-typing-v1:{hashlib.sha256(raw).hexdigest()[:12]}"
        self._rules = rules
        self._groups = load_country_groups(country_groups_path)
        self._types = {
            event_type.value: _CompiledType(
                rule=rule,
                include=[compile_phrase(p) for p in rule.include],
                exclude=[compile_phrase(p) for p in rule.exclude],
                positive=[compile_phrase(p) for p in rule.direction.positive],
                negative=[compile_phrase(p) for p in rule.direction.negative],
                countries=_compile_countries(rules.event_type_countries.get(event_type, {})),
            )
            for event_type, rule in rules.event_types.items()
        }
        self._countries = [
            *_compile_countries(rules.countries),
            *_compile_countries(rules.countries_case_sensitive, case_sensitive=True),
        ]
        self._high = [compile_phrase(p) for p in rules.severity.high]
        self._low = [compile_phrase(p) for p in rules.severity.low]
        self._scheduled = [compile_phrase(p) for p in rules.scheduled.scheduled]
        self._unscheduled = [compile_phrase(p) for p in rules.scheduled.unscheduled]

    def type_text(self, text: str) -> TypedText:
        scores = {
            name: sum(1 for p in compiled.include if p.search(text))
            for name, compiled in self._types.items()
            if not _any_match(compiled.exclude, text)
        }
        scores = {name: n for name, n in scores.items() if n > 0}
        if not scores:
            return self._untyped("no event_type pattern matched")
        best = max(scores.values())
        winners = sorted(name for name, n in scores.items() if n == best)
        if len(winners) > 1:
            return self._untyped(f"ambiguous: {', '.join(winners)}")
        event_type = winners[0]
        compiled = self._types[event_type]
        span = _first_match(compiled.include, text)
        assert span is not None  # the winner has at least one include match
        iso3 = self._country(text, compiled)
        key = EventKey(
            event_type=event_type,
            actor_kind=compiled.rule.actor_kind.value,
            impacted_kinds=tuple(sorted(k.value for k in compiled.rule.impacted_kinds)),
            country_group=self._groups.get(iso3, "none") if iso3 else "none",
            severity_band=self._severity(text),
            is_scheduled=self._is_scheduled(text),
            direction_sign=int(_any_match(compiled.positive, text))
            - int(_any_match(compiled.negative, text)),
        )
        return TypedText(
            status="typed",
            key=key,
            reason=f"matched {event_type} on {span.group(0)!r}",
            evidence_span=span.group(0),
            typer_version=self.version,
        )

    def _untyped(self, reason: str) -> TypedText:
        return TypedText(
            status="untyped", key=None, reason=reason, evidence_span=None,
            typer_version=self.version,
        )

    def _country(self, text: str, compiled: _CompiledType) -> str | None:
        candidates = [
            (m.start(), iso3)
            for pattern, iso3 in [*self._countries, *compiled.countries]
            if (m := pattern.search(text))
        ]
        return min(candidates)[1] if candidates else None

    def _severity(self, text: str) -> SeverityBand:
        if _any_match(self._high, text):
            return "high"
        if _any_match(self._low, text):
            return "low"
        return self._rules.severity.default

    def _is_scheduled(self, text: str) -> bool:
        if _any_match(self._unscheduled, text):
            return False
        if _any_match(self._scheduled, text):
            return True
        return self._rules.scheduled.default == "scheduled"
