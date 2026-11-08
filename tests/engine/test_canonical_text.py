from fakes import make_event_record

from precedent.engine.canonical_text import canonical_text
from precedent.spine.event_schema import NewsEvent


def _impact(item):
    name, kind, step = (item, "company", 2) if isinstance(item, str) else item
    return {"name": name, "kind": kind, "ticker_hint": None, "direction_step": step}


def record_about(actor_name, impacted=(), **event_fields):
    """A record whose actor is `actor_name`; `impacted` holds names or (name, kind, step)."""
    event = {
        "schema_version": "1.0",
        "event_type": "earnings",
        "actor": {"name": actor_name, "kind": "company", "country_iso3": "USA"},
        "action": "reported revenue above guidance",
        "impacted": [_impact(item) for item in impacted],
        "severity": 5,
        "macro_direction_step": 0,
        "confidence": "high",
        "horizon": "days",
        "novelty": "new",
        "is_scheduled": True,
        "evidence_span": "Acme reported revenue of 12 billion dollars",
    }
    event.update(event_fields)
    return make_event_record(event=NewsEvent.model_validate(event))


def test_canonical_text_contains_no_entity_names():
    text = canonical_text(record_about("Acme Corp", impacted=["Globex Inc"]))
    assert "Acme" not in text and "Globex" not in text and "[COMPANY]" in text


def test_canonical_text_is_one_exact_line_of_enum_values():
    text = canonical_text(record_about("Acme Corp", impacted=["Globex Inc"]))
    assert text == (
        "earnings by [COMPANY] in US severity mid scheduled "
        "impacted [COMPANY] direction 1 horizon days novelty new"
    )


def test_action_and_evidence_words_never_appear():
    text = canonical_text(record_about("Acme Corp", impacted=["Globex Inc"]))
    for word in ("revenue", "guidance", "billion", "dollars", "reported", "above"):
        assert word not in text
    assert "\n" not in text


def test_impacted_kinds_are_sorted_deduplicated_and_none_when_empty():
    impacted = [("A", "sector", -1), ("B", "commodity", -1), ("C", "sector", -1)]
    text = canonical_text(record_about("Fed", impacted=impacted, event_type="central_bank"))
    assert "impacted [COMMODITY]+[SECTOR] direction -1" in text
    bare = canonical_text(record_about("Fed", impacted=[]))
    assert "impacted none direction 0" in bare


def test_actor_kind_scheduled_flag_and_severity_band_follow_the_record():
    text = canonical_text(
        record_about(
            "Fed",
            event_type="central_bank",
            actor={"name": "Fed", "kind": "central_bank", "country_iso3": None},
            severity=9,
            is_scheduled=False,
            macro_direction_step=-2,
            horizon="weeks",
            novelty="update",
        )
    )
    assert text == (
        "central_bank by [CENTRAL_BANK] in none severity high unscheduled "
        "impacted none direction -1 horizon weeks novelty update"
    )


def test_entity_names_do_not_change_the_text():
    first = canonical_text(record_about("Acme Corp", impacted=["Globex Inc"]))
    second = canonical_text(record_about("Initech", impacted=["Umbrella"]))
    assert first == second
