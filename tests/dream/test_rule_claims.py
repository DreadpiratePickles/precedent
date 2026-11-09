import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from fakes import make_event_record

from precedent.dream.claim_schema import link_id_for
from precedent.dream.rule_claims import (
    RuleClaimReport,
    claims_from_events,
    claims_from_events_with_report,
    claims_from_links,
)
from precedent.ripple.link_store import Link, Relation
from precedent.spine.event_record import ResolvedImpact


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def record_with_impact(
    instrument: str,
    direction: float,
    available: str | None,
    article_id: str = "gdelt:1",
    ingested: str = "2024-01-05T14:05:00Z",
    event_type: str = "earnings",
    actor: str = "Acme Corp",
):
    base = make_event_record()
    event = base.event.model_copy(
        update={
            "event_type": event_type,
            "actor": base.event.actor.model_copy(update={"name": actor}),
        }
    )
    return make_event_record(
        article_id=article_id,
        event=event,
        available_utc=None if available is None else utc(available),
        ingested_utc=utc(ingested),
        resolved_impacts=[
            {
                "instrument": instrument,
                "kind": "company",
                "direction": direction,
                "match_method": "ticker_hint",
            }
        ],
    )


def record_for_8k_item(
    item: str,
    instrument: str,
    available: str = "2024-02-01T21:35:00Z",
    accession: str = "0000320193-24-000001",
    source: str = "edgar",
):
    base = make_event_record()
    event = base.event.model_copy(update={"evidence_span": f"8-K item {item}"})
    return make_event_record(
        article_id=f"{accession}:{item}",
        source=source,
        event=event,
        available_utc=utc(available),
        resolved_impacts=[
            {
                "instrument": instrument,
                "kind": "company",
                "direction": 0.0,
                "match_method": "cik_map",
            }
        ],
    )


def session_cutoffs(count: int, start: str = "2024-02-01") -> pd.Series:
    sessions = pd.bdate_range(start, periods=count)
    stamps = (sessions + pd.Timedelta(hours=21)).tz_localize("UTC")
    return pd.Series(stamps, index=pd.DatetimeIndex(sessions, name="session_date"))


def make_link(relation: Relation, source: str = "SUPP", receiver: str = "ACME") -> Link:
    return Link(
        source_node=source,
        receiver_node=receiver,
        relation=relation,
        weight=0.5,
        valid_from_utc=utc("2024-02-01T21:35:00Z"),
        valid_to_utc=utc("2025-05-01T21:35:00Z"),
        source_doc_id="doc-9",
        evidence_span="Supplier of cells",
        extractor_version="rules-links-v1",
    )


def test_event_direction_claim_is_short_lived():
    [c] = claims_from_events(
        [record_with_impact("AAPL", 2.0, "2024-02-01T21:35:00Z")], story_window=3
    )
    assert (c.attribute, c.value, c.subject) == ("event_direction", "positive", "AAPL")
    assert c.valid_to_utc is not None and c.valid_to_utc > c.valid_from_utc
    assert c.valid_from_utc == utc("2024-02-01T21:35:00Z")
    # first cutoff at or after the 21:35Z arrival is 2 Feb; three sessions later is 7 Feb
    assert c.valid_to_utc == utc("2024-02-07T21:00:00Z")
    assert (c.period, c.source_doc_id, c.extractor_version) == (
        "unspecified",
        "gdelt:1",
        "rules-events-v1",
    )
    assert c.evidence_span == "Acme reported revenue of 12 billion dollars"


@pytest.mark.parametrize(
    ("direction", "expected"), [(1.5, "positive"), (-0.5, "negative"), (0.0, "neutral")]
)
def test_event_direction_value_follows_sign(direction, expected):
    [c] = claims_from_events([record_with_impact("AAPL", direction, "2024-02-01T21:35:00Z")])
    assert c.value == expected


def test_exit_costs_item_becomes_a_headcount_claim():
    claims = claims_from_events([record_for_8k_item("2.05", "AAPL")])
    typed = [c for c in claims if c.attribute == "headcount_direction"]
    assert len(typed) == 1
    assert (typed[0].value, typed[0].subject, typed[0].period) == ("down", "AAPL", "unspecified")
    assert typed[0].valid_to_utc is None
    assert typed[0].valid_from_utc == utc("2024-02-01T21:35:00Z")
    assert typed[0].evidence_span == "8-K item 2.05"
    assert typed[0].extractor_version == "rules-events-v1"
    assert [c.attribute for c in claims].count("event_direction") == 1


def test_typed_8k_claim_is_emitted_only_for_the_filer():
    """M9: a typed 8-K item describes the filer; a resolved counterparty must not inherit it."""
    base = make_event_record()
    event = base.event.model_copy(update={"evidence_span": "8-K item 2.05"})
    record = make_event_record(
        article_id="0000320193-24-000001:2.05",
        source="edgar",
        event=event,
        available_utc=utc("2024-02-01T21:35:00Z"),
        resolved_impacts=[
            {"instrument": "AAPL", "kind": "company", "direction": 0.0, "match_method": "cik_map"},
            {
                "instrument": "SUP",
                "kind": "company",
                "direction": 0.0,
                "match_method": "ticker_hint",
            },
        ],
    )
    typed = [c for c in claims_from_events([record]) if c.attribute == "headcount_direction"]
    assert [c.subject for c in typed] == ["AAPL"]


def test_other_events_item_produces_no_typed_claim():
    claims = claims_from_events(
        [record_for_8k_item("8.01", "AAPL"), record_for_8k_item("7.01", "AAPL", accession="x-2")]
    )
    assert claims and all(c.attribute == "event_direction" for c in claims)


def test_unlisted_item_produces_only_the_event_direction_claim():
    claims = claims_from_events([record_for_8k_item("5.02", "AAPL")])
    assert [c.attribute for c in claims] == ["event_direction"]


def test_relationship_item_without_counterparty_emits_no_typed_claim_and_is_counted():
    claims, report = claims_from_events_with_report(
        [record_for_8k_item("1.01", "AAPL"), record_for_8k_item("1.02", "AAPL", accession="x-2")]
    )
    assert all(c.attribute == "event_direction" for c in claims)
    assert report.relationship_without_counterparty == 2
    assert report.typed_claims == 0


def test_relationship_item_with_an_unclassified_counterparty_emits_no_typed_claim():
    record = record_for_8k_item("1.01", "AAPL")
    other = {"instrument": "MSFT", "kind": "company", "direction": 0.0, "match_method": "alias"}
    record = record.model_copy(
        update={"resolved_impacts": [*record.resolved_impacts, ResolvedImpact(**other)]}
    )
    claims, report = claims_from_events_with_report([record])
    assert all(c.attribute == "event_direction" for c in claims)
    assert report.relationship_without_counterparty == 0
    assert report.typed_claims == 0


def test_non_edgar_record_with_item_like_article_id_has_no_typed_claim():
    claims = claims_from_events([record_for_8k_item("2.05", "AAPL", source="gdelt")])
    assert [c.attribute for c in claims] == ["event_direction"]


def test_repeat_report_in_window_emits_no_second_claim_and_is_counted():
    first = record_with_impact("AAPL", 1.0, "2024-02-01T21:35:00Z", article_id="gdelt:1")
    repeat = record_with_impact("AAPL", 1.0, "2024-02-02T21:35:00Z", article_id="gdelt:2")
    claims, report = claims_from_events_with_report([repeat, first], story_window=3)
    assert [c.source_doc_id for c in claims] == ["gdelt:1"]
    assert report.event_direction_claims == 1
    assert report.repeat_reports_skipped == 1


def test_reports_further_apart_than_the_window_emit_two_claims():
    early = record_with_impact("AAPL", 1.0, "2024-02-01T21:35:00Z", article_id="gdelt:1")
    late = record_with_impact("AAPL", 1.0, "2024-02-12T21:35:00Z", article_id="gdelt:2")
    claims, report = claims_from_events_with_report([early, late], story_window=3)
    assert sorted(c.source_doc_id for c in claims) == ["gdelt:1", "gdelt:2"]
    assert report.repeat_reports_skipped == 0


def test_recorded_utc_is_valid_from_whatever_the_ingest_time():
    # D69: ingest wall-clock time (audit-only in the spine) is no knowledge time
    late_ingest = record_with_impact(
        "AAPL", 1.0, "2024-02-01T21:35:00Z", ingested="2024-03-01T10:00:00Z"
    )
    [c] = claims_from_events([late_ingest])
    assert c.recorded_utc == c.valid_from_utc == utc("2024-02-01T21:35:00Z")
    early_ingest = record_with_impact(
        "AAPL", 1.0, "2024-02-01T21:35:00Z", article_id="gdelt:2", ingested="2024-01-01T00:00:00Z"
    )
    [d] = claims_from_events([early_ingest])
    assert d.recorded_utc == d.valid_from_utc


def test_unaligned_record_is_skipped_and_counted():
    unaligned = record_with_impact("AAPL", 1.0, None)
    aligned = record_with_impact("MSFT", -1.0, "2024-02-01T21:35:00Z", article_id="gdelt:2")
    claims, report = claims_from_events_with_report([unaligned, aligned])
    assert [c.subject for c in claims] == ["MSFT"]
    assert report == RuleClaimReport(
        records=2,
        unaligned_skipped=1,
        event_direction_claims=1,
        typed_claims=0,
        repeat_reports_skipped=0,
        relationship_without_counterparty=0,
    )


def test_only_unaligned_records_give_no_claims():
    assert claims_from_events([record_with_impact("AAPL", 1.0, None)]) == []
    assert claims_from_events([]) == []


def test_valid_to_is_the_cutoff_story_window_sessions_later():
    cutoffs = session_cutoffs(8)
    record = record_with_impact("AAPL", 1.0, "2024-02-01T20:00:00Z")  # before the 1 Feb cutoff
    [c] = claims_from_events([record], story_window=3, session_cutoffs_utc=cutoffs)
    assert c.valid_to_utc == cutoffs.iloc[3].to_pydatetime()
    [d] = claims_from_events([record], story_window=0, session_cutoffs_utc=cutoffs)
    assert d.valid_to_utc == cutoffs.iloc[0].to_pydatetime()


def test_cutoffs_that_end_too_early_raise_naming_the_article():
    record = record_with_impact("AAPL", 1.0, "2024-02-01T20:00:00Z", article_id="gdelt:77")
    with pytest.raises(ValueError, match="gdelt:77"):
        claims_from_events([record], story_window=3, session_cutoffs_utc=session_cutoffs(3))


def test_record_after_the_last_cutoff_raises_naming_the_article():
    record = record_with_impact("AAPL", 1.0, "2024-03-01T20:00:00Z", article_id="gdelt:78")
    with pytest.raises(ValueError, match="gdelt:78"):
        claims_from_events([record], session_cutoffs_utc=session_cutoffs(8))


def _vocabulary(tmp_path: Path, item_map: dict) -> Path:
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps({"item_map_8k": {"proposed": True, **item_map, "note": "x"}}))
    return path


def test_malformed_vocabulary_attribute_raises(tmp_path):
    path = _vocabulary(tmp_path, {"2.05": {"attribute": "nonsense", "value": "down"}})
    with pytest.raises(ValueError, match="2.05"):
        claims_from_events([record_for_8k_item("2.05", "AAPL")], vocabulary_path=path)


def test_malformed_vocabulary_value_raises(tmp_path):
    path = _vocabulary(
        tmp_path, {"2.05": {"attribute": "headcount_direction", "value": "sideways"}}
    )
    with pytest.raises(ValueError, match="2.05"):
        claims_from_events([record_for_8k_item("2.05", "AAPL")], vocabulary_path=path)


def test_vocabulary_without_item_map_raises(tmp_path):
    path = tmp_path / "vocabulary.json"
    path.write_text(json.dumps({"attributes": []}))
    with pytest.raises(ValueError, match="item_map_8k"):
        claims_from_events([record_for_8k_item("2.05", "AAPL")], vocabulary_path=path)


def test_custom_vocabulary_drives_typed_claims(tmp_path):
    path = _vocabulary(tmp_path, {"5.02": {"attribute": "capex_direction", "value": "up"}})
    claims = claims_from_events([record_for_8k_item("5.02", "AAPL")], vocabulary_path=path)
    assert sorted(c.attribute for c in claims) == ["capex_direction", "event_direction"]


def test_link_claim_copies_validity_and_skips_peers():
    supplier = make_link(Relation.SUPPLIER_OF)
    peer = make_link(Relation.PEER_OF, source="PEER")
    claims = claims_from_links([supplier, peer])
    assert len(claims) == 1
    c = claims[0]
    assert c.valid_from_utc == supplier.valid_from_utc
    assert c.valid_to_utc == supplier.valid_to_utc
    assert c.value == "ongoing"
    assert c.attribute == "supplier_relationship"
    assert c.period == "unspecified"
    assert c.subject == link_id_for("SUPP", "ACME", Relation.SUPPLIER_OF, "doc-9")
    assert c.source_doc_id == "doc-9"
    assert c.evidence_span == "Supplier of cells"
    assert c.extractor_version == "rules-links-v1"
    assert c.recorded_utc == c.valid_from_utc


def test_customer_link_becomes_a_customer_relationship_and_other_relations_are_excluded():
    others = [
        make_link(relation)
        for relation in (
            Relation.INPUT_COMMODITY,
            Relation.OUTPUT_COMMODITY,
            Relation.COUNTRY_EXPOSURE,
        )
    ]
    claims = claims_from_links([make_link(Relation.CUSTOMER_OF), *others])
    assert [c.attribute for c in claims] == ["customer_relationship"]


def test_link_id_is_stable_and_depends_on_every_part():
    base = link_id_for("A", "B", Relation.SUPPLIER_OF, "d")
    assert base == link_id_for("A", "B", "supplier_of", "d")
    assert (
        len(
            {
                base,
                link_id_for("B", "A", "supplier_of", "d"),
                link_id_for("A", "B", "customer_of", "d"),
                link_id_for("A", "B", "supplier_of", "e"),
            }
        )
        == 4
    )
