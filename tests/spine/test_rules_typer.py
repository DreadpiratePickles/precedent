"""Tests for the deterministic rules typer."""
import json

import pytest

from precedent.config import ConfigError
from precedent.spine.typers.rules import RulesTyper

RULES = RulesTyper()


@pytest.mark.parametrize(
    "text, key",
    [
        (
            "Fed unexpectedly cuts rates in emergency move",
            "central_bank/central_bank/country/US/high/unscheduled/1",
        ),
        (
            "Will the Federal Reserve cut rates at its December meeting?",
            "central_bank/central_bank/country/US/mid/scheduled/1",
        ),
        (
            "Will Russia and Ukraine agree to a ceasefire before July?",
            "armed_conflict/government/country/EM/mid/unscheduled/1",
        ),
    ],
)
def test_rules_type_known_phrasings(text, key):
    typed = RULES.type_text(text)
    assert typed.status == "typed" and typed.key.as_string() == key and typed.evidence_span in text


def test_sports_are_untyped():
    typed = RULES.type_text("Will the Lakers win the NBA championship?")
    assert (typed.status, typed.key, typed.reason) == ("untyped", None, "no event_type pattern matched")


def test_a_tie_between_event_types_is_untyped_not_guessed():
    typed = RULES.type_text("Will the US impose tariffs on Chinese steel after the election?")
    assert typed.status == "untyped" and typed.reason.startswith("ambiguous:")
    assert typed.reason == "ambiguous: election_or_political, trade_or_sanctions"


def test_the_same_text_gets_the_same_answer_and_version_names_the_rules_hash():
    text = "Moody's downgrades Italy's sovereign rating"
    assert RULES.type_text(text) == RULES.type_text(text)
    assert RULES.version.startswith("rules-typing-v1:") and len(RULES.version.split(":")[1]) == 12


def test_matching_is_whole_word_and_case_insensitive(tmp_path):
    rules = _rules({"earnings": {"include": ["earnings"]}})
    typer = RulesTyper(_write(tmp_path, rules), None)
    assert typer.type_text("Big EARNINGS day").status == "typed"
    assert typer.type_text("Preearnings chatter").status == "untyped"


def test_evidence_span_is_copied_from_the_original_text_across_whitespace(tmp_path):
    rules = _rules({"central_bank": {"include": ["federal reserve"]}})
    typed = RulesTyper(_write(tmp_path, rules), None).type_text("The Federal \n  Reserve acts")
    assert typed.status == "typed" and typed.evidence_span == "Federal \n  Reserve"


def test_an_exclude_match_disqualifies_the_event_type(tmp_path):
    rules = _rules({"labor": {"include": ["strike"], "exclude": ["missile"]}})
    typer = RulesTyper(_write(tmp_path, rules), None)
    assert typer.type_text("Missile strike at dawn").reason == "no event_type pattern matched"
    assert typer.type_text("Dockworkers strike").status == "typed"


def test_unscheduled_cue_beats_scheduled_cue_and_both_direction_cues_give_zero(tmp_path):
    entry = {"include": ["rates"], "direction": {"1": ["cut"], "-1": ["hike"]}}
    rules = _rules({"central_bank": entry})
    rules["scheduled"] = {"scheduled": ["meeting"], "unscheduled": ["emergency"], "default": "unscheduled"}
    typer = RulesTyper(_write(tmp_path, rules), None)
    both = typer.type_text("Emergency meeting on rates: cut or hike")
    assert both.key.is_scheduled is False and both.key.direction_sign == 0
    assert typer.type_text("Meeting on rates, cut").key.is_scheduled is True
    assert typer.type_text("Meeting on rates, cut").key.direction_sign == 1


def test_comment_keys_are_ignored_and_unknown_event_types_are_a_config_error(tmp_path):
    rules = _rules({"earnings": {"include": ["earnings"]}})
    rules["_comment"] = "ignored"
    assert RulesTyper(_write(tmp_path, rules), None).type_text("earnings").status == "typed"
    with pytest.raises(ConfigError):
        RulesTyper(_write(tmp_path, _rules({"not_a_type": {"include": ["x"]}})), None)


def test_the_pronoun_us_is_not_a_country_but_the_abbreviation_is():
    pronoun = RULES.type_text("Will Tesla beat earnings estimates this quarter? Tell us")
    assert pronoun.status == "typed" and pronoun.key.country_group == "none"
    for text in ("U.S. tariffs on steel hit exports", "Will the US cut rates?", "USA tariffs"):
        typed = RULES.type_text(text)
        assert typed.status == "typed" and typed.key.country_group == "US", text


def test_a_missing_rules_file_is_a_config_error_naming_the_path(tmp_path):
    missing = tmp_path / "nope.json"
    with pytest.raises(ConfigError, match="nope.json"):
        RulesTyper(missing, None)


def _rules(event_types):
    full = {
        t: {"include": [], "exclude": [], "actor_kind": "company", "impacted_kinds": ["company"],
            "direction": {"1": [], "-1": []}, **entry}
        for t, entry in event_types.items()
    }
    return {
        "version": "rules-typing-v1",
        "event_types": full,
        "severity": {"high": [], "low": [], "default": "mid"},
        "scheduled": {"scheduled": [], "unscheduled": [], "default": "unscheduled"},
        "countries": {},
    }


def _write(tmp_path, rules):
    path = tmp_path / "rules.json"
    path.write_text(json.dumps(rules), encoding="utf-8")
    return path
