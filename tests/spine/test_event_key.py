"""Tests for the event key: the discrete description used to find similar past events."""
import json
from datetime import UTC, datetime

import pytest

from precedent.config import references_dir
from precedent.spine.event_key import (
    EventKey,
    build_event_key,
    country_group_of,
    load_country_groups,
    severity_band,
)
from precedent.spine.event_record import EventRecord
from precedent.spine.event_schema import NewsEvent


def record_with(
    event_type: str = "earnings",
    actor_kind: str = "company",
    actor_country: str | None = "USA",
    impacted: list[tuple[str, int]] | None = None,
    severity: int = 5,
    is_scheduled: bool = True,
    macro_direction_step: int = 0,
) -> EventRecord:
    impacted = [("company", 1)] if impacted is None else impacted
    event = NewsEvent.model_validate(
        {
            "schema_version": "1.0",
            "event_type": event_type,
            "actor": {"name": "Actor", "kind": actor_kind, "country_iso3": actor_country},
            "action": "did something",
            "impacted": [
                {"name": f"X{i}", "kind": kind, "ticker_hint": None, "direction_step": step}
                for i, (kind, step) in enumerate(impacted)
            ],
            "severity": severity,
            "macro_direction_step": macro_direction_step,
            "confidence": "high",
            "horizon": "days",
            "novelty": "new",
            "is_scheduled": is_scheduled,
            "evidence_span": "span",
        }
    )
    return EventRecord(
        article_id="a1",
        body_sha256="0" * 64,
        source="test",
        published_utc=None,
        first_seen_utc=datetime(2024, 1, 5, 14, 0, tzinfo=UTC),
        ingested_utc=datetime(2024, 1, 5, 14, 5, tzinfo=UTC),
        available_utc=None,
        effective_session=None,
        parser_version="v1",
        event=event,
        resolved_impacts=[],
        linked_countries=[],
    )


@pytest.mark.parametrize(
    ("severity", "band"),
    [(1, "low"), (3, "low"), (4, "mid"), (6, "mid"), (7, "high"), (10, "high")],
)
def test_severity_bands(severity: int, band: str) -> None:
    assert severity_band(severity) == band


@pytest.mark.parametrize("severity", [0, 11, -1])
def test_severity_outside_one_to_ten_is_an_error(severity: int) -> None:
    with pytest.raises(ValueError, match="severity"):
        severity_band(severity)


def test_key_string_is_stable_and_round_trips() -> None:
    key = build_event_key(
        record_with(
            event_type="central_bank",
            actor_kind="central_bank",
            actor_country="USA",
            impacted=[("country", -1)],
            severity=8,
            is_scheduled=False,
            macro_direction_step=-1,
        )
    )
    assert key.as_string() == "central_bank/central_bank/country/US/high/unscheduled/-1"
    assert EventKey.from_string(key.as_string()) == key


def test_key_with_no_impacts_uses_none_and_round_trips() -> None:
    key = build_event_key(record_with(impacted=[], actor_country=None))
    assert key.as_string() == "earnings/company/none/none/mid/scheduled/0"
    assert EventKey.from_string(key.as_string()) == key


def test_several_kinds_are_sorted_unique_and_joined_with_plus() -> None:
    key = build_event_key(record_with(impacted=[("sector", 1), ("company", 1), ("sector", -1)]))
    assert key.impacted_kinds == ("company", "sector")
    assert "/company+sector/" in key.as_string()
    assert EventKey.from_string(key.as_string()) == key


def test_direction_falls_back_to_the_impacts() -> None:
    key = build_event_key(record_with(macro_direction_step=0, impacted=[("sector", 2), ("company", 1)]))
    assert (key.direction_sign, key.impacted_kinds) == (1, ("company", "sector"))


def test_macro_direction_wins_over_the_impacts() -> None:
    key = build_event_key(record_with(macro_direction_step=-1, impacted=[("company", 2)]))
    assert key.direction_sign == -1


def test_cancelling_impacts_give_direction_zero() -> None:
    key = build_event_key(record_with(impacted=[("company", 1), ("sector", -1)]))
    assert key.direction_sign == 0


def test_no_actor_country_is_group_none() -> None:
    assert build_event_key(record_with(actor_country=None)).country_group == "none"


def test_country_groups_by_actor_country() -> None:
    assert country_group_of("USA") == "US"
    assert country_group_of("DEU") == "DM"
    assert country_group_of("BRA") == "EM"
    assert country_group_of(None) == "none"


def test_an_unknown_code_is_an_error_not_a_default() -> None:
    with pytest.raises(ValueError, match="ZZZ"):
        country_group_of("ZZZ")


@pytest.mark.parametrize(
    "text",
    [
        "",
        "a/b/c",
        "earnings/company/none/XX/mid/scheduled/0",
        "earnings/company/none/US/medium/scheduled/0",
        "earnings/company/none/US/mid/maybe/0",
        "earnings/company/none/US/mid/scheduled/2",
        "earnings/company/none/US/mid/scheduled/x",
    ],
)
def test_a_malformed_key_string_is_a_value_error(text: str) -> None:
    with pytest.raises(ValueError):
        EventKey.from_string(text)


def test_every_iso3_code_is_in_exactly_one_group() -> None:
    groups = json.loads((references_dir() / "country_groups.json").read_text())["groups"]
    codes = [code for members in groups.values() for code in members]
    assert len(codes) == len(set(codes)) >= 240 and groups["US"] == ["USA"]


def test_country_groups_file_is_marked_proposed_and_sourced() -> None:
    data = json.loads((references_dir() / "country_groups.json").read_text())
    assert data["proposed"] is True and data["source"] and data["checked_utc"]
    assert len(data["groups"]["DM"]) == 22  # MSCI's 23 developed markets, less the US group


def test_load_country_groups_is_cached() -> None:
    assert load_country_groups() is load_country_groups()
