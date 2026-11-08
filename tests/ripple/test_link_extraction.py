import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from precedent import config
from precedent.ripple.link_extraction import (
    CounterpartyKind,
    ExtractedLink,
    ExtractedLinks,
    LinkParseReport,
    load_link_prompt,
    parse_links,
)
from precedent.ripple.link_store import LinkStore, Relation, graph_asof

PASSAGE = (
    "Item 1. Business. Our largest customer, Globex Inc accounted for 14% of revenue in 2019. "
    "We buy copper from several mines."
)
UNNAMED_PASSAGE = "One customer accounted for 12% of our revenue. We did not name it."
ACCEPTED = datetime(2020, 3, 2, 21, 30, tzinfo=UTC)
GLOBEX_SPAN = "Globex Inc accounted for 14% of revenue"


def resolve_known(name: str) -> str | None:
    return {"Globex Inc": "GLBX"}.get(name)


RESOLVE = resolve_known


class FakeExtractor:
    """Returns one canned string and records every call."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.calls: list[tuple[str, dict]] = []

    def generate(self, prompt: str, json_schema: dict) -> str:
        self.calls.append((prompt, json_schema))
        return self.reply


def json_links(rows: list[tuple[str, str, str, str]]) -> str:
    return json.dumps(
        {
            "links": [
                {
                    "relation": relation,
                    "counterparty_name": name,
                    "counterparty_kind": kind,
                    "evidence_span": span,
                }
                for relation, name, kind, span in rows
            ]
        }
    )


GLOBEX_ROW = ("customer_of", "Globex Inc", "company", GLOBEX_SPAN)


def run(reply: str, passage: str = PASSAGE, filer: str = "ACME", resolve=RESOLVE, **overrides):
    arguments = {
        "filer": filer,
        "passage": passage,
        "accepted_utc": ACCEPTED,
        "source_doc_id": "10K-2020",
        "extractor": FakeExtractor(reply),
        "extractor_version": "llm-links-v1",
        "resolve": resolve,
    }
    arguments.update(overrides)
    return parse_links(**arguments)


def test_link_without_evidence_in_the_passage_is_dropped():
    extractor = FakeExtractor(
        json_links(
            [("customer_of", "Globex Inc", "company", "this sentence is not in the passage")]
        )
    )
    links, report = parse_links(
        "ACME", PASSAGE, ACCEPTED, "10K-2020", extractor, "llm-links-v1", resolve=RESOLVE
    )
    assert links == [] and report.evidence_missing == 1


def test_kept_link_carries_the_filing_time_and_span():
    extractor = FakeExtractor(json_links([GLOBEX_ROW]))
    links, _ = parse_links(
        "ACME", PASSAGE, ACCEPTED, "10K-2020", extractor, "llm-links-v1", resolve=RESOLVE
    )
    assert links[0].valid_from_utc == ACCEPTED
    assert links[0].evidence_span.startswith("Globex Inc accounted")


def test_unnamed_customer_is_not_invented():
    extractor = FakeExtractor(json_links([]))
    links, report = parse_links(
        "ACME", UNNAMED_PASSAGE, ACCEPTED, "10K-2020", extractor, "llm-links-v1", resolve=RESOLVE
    )
    assert links == [] and report.proposed == 0


def test_kept_link_fields_follow_the_decisions():
    links, report = run(json_links([GLOBEX_ROW]))
    link = links[0]
    assert link.source_node == "GLBX" and link.receiver_node == "ACME"  # counterparty is source
    assert link.relation == Relation.CUSTOMER_OF
    assert link.weight == 1.0
    assert link.source_doc_id == "10K-2020" and link.extractor_version == "llm-links-v1"
    assert link.valid_to_utc == datetime(2021, 6, 2, 21, 30, tzinfo=UTC)  # 15 months
    assert report == LinkParseReport(
        proposed=1, evidence_missing=0, unresolved=0, kept=1, invalid_output=False, error=None
    )


def test_expiry_follows_the_parameter_file():
    from precedent.ripple.params import load_ripple_params

    months = load_ripple_params()["annual_link_expiry_months"]
    links, _ = run(json_links([GLOBEX_ROW]))
    assert pd.Timestamp(links[0].valid_to_utc) == pd.Timestamp(ACCEPTED) + pd.DateOffset(
        months=months
    )


def test_non_utc_accepted_time_is_normalised():
    from datetime import timedelta, timezone

    eastern = datetime(2020, 3, 2, 16, 30, tzinfo=timezone(timedelta(hours=-5)))
    links, _ = run(json_links([GLOBEX_ROW]), accepted_utc=eastern)
    assert links[0].valid_from_utc == ACCEPTED


def test_whitespace_different_evidence_is_accepted_and_stripped():
    spaced = "  Globex   Inc accounted\nfor 14%  of revenue "
    links, report = run(json_links([("customer_of", "Globex Inc", "company", spaced)]))
    assert report.kept == 1 and report.evidence_missing == 0
    assert links[0].evidence_span == spaced.strip()


def test_evidence_with_a_changed_word_is_dropped():
    reply = json_links([("customer_of", "Globex Inc", "company", "Globex Inc accounted for 15%")])
    links, report = run(reply)
    assert links == [] and report.evidence_missing == 1 and report.kept == 0


def test_unresolved_counterparty_is_counted():
    reply = json_links([("supplier_of", "Initech LLC", "company", "We buy copper")])
    links, report = run(reply)
    assert links == [] and report.unresolved == 1 and report.proposed == 1


def test_evidence_is_checked_before_resolution():
    seen: list[str] = []

    def recording(name: str) -> str | None:
        seen.append(name)
        return "GLBX"

    reply = json_links([("customer_of", "Globex Inc", "company", "not in the passage")])
    _, report = run(reply, resolve=recording)
    assert report.evidence_missing == 1 and report.unresolved == 0 and seen == []


def test_self_link_counts_as_unresolved():
    links, report = run(json_links([GLOBEX_ROW]), filer="GLBX")
    assert links == [] and report.unresolved == 1 and report.kept == 0


def test_commodity_link_uses_the_resolved_commodity_node():
    reply = json_links([("input_commodity", "copper", "commodity", "We buy copper")])
    links, _ = run(reply, resolve=lambda name: "COPPER" if name == "copper" else None)
    assert links[0].source_node == "COPPER" and links[0].relation == Relation.INPUT_COMMODITY


def test_duplicate_proposals_are_kept_once_with_the_first_span():
    reply = json_links(
        [
            GLOBEX_ROW,
            ("customer_of", "Globex Inc", "company", "Our largest customer, Globex Inc"),
            ("supplier_of", "Globex Inc", "company", GLOBEX_SPAN),
        ]
    )
    links, report = run(reply)
    assert report.proposed == 3 and report.kept == 2 == len(links)
    first = next(link for link in links if link.relation == Relation.CUSTOMER_OF)
    assert first.evidence_span == GLOBEX_SPAN


@pytest.mark.parametrize(
    "reply",
    [
        "this is not json",
        "",
        json.dumps({"links": [{"relation": "owns", "counterparty_name": "X",
                               "counterparty_kind": "company", "evidence_span": "a"}]}),
        json.dumps({"links": [
            {"relation": "customer_of", "counterparty_name": "Globex Inc",
             "counterparty_kind": "country", "evidence_span": GLOBEX_SPAN}]}),
        json.dumps({"links": [
            {"relation": "input_commodity", "counterparty_name": "copper",
             "counterparty_kind": "company", "evidence_span": "We buy copper"}]}),
        json.dumps({"links": [
            {"relation": "customer_of", "counterparty_name": "Globex Inc",
             "counterparty_kind": "company", "evidence_span": GLOBEX_SPAN, "weight": 0.9}]}),
        json.dumps({"links": [
            {"relation": "customer_of", "counterparty_name": "Globex Inc",
             "counterparty_kind": "company", "evidence_span": ""}]}),
        json.dumps({"links": [
            {"relation": "customer_of", "counterparty_name": "   ",
             "counterparty_kind": "company", "evidence_span": GLOBEX_SPAN}]}),
        json.dumps({"links": "none"}),
        json.dumps({"links": [], "extra": 1}),
        json.dumps([]),
    ],
)
def test_invalid_model_output_gives_no_links_and_never_raises(reply):
    links, report = run(reply)
    assert links == []
    assert report.invalid_output is True
    assert report.error
    assert report.proposed == report.kept == report.unresolved == report.evidence_missing == 0


def test_twenty_one_links_are_a_schema_violation_and_twenty_are_fine():
    row = ("customer_of", "Globex Inc", "company", GLOBEX_SPAN)
    _, too_many = run(json_links([row] * 21))
    assert too_many.invalid_output is True
    _, at_limit = run(json_links([row] * 20))
    assert at_limit.invalid_output is False and at_limit.proposed == 20


def test_error_text_does_not_echo_the_model_output():
    secret = "ZXQ-UNIQUE-MARKER-9137"
    row = {
        "relation": "customer_of",
        "counterparty_name": "Globex Inc",
        "counterparty_kind": "company",
        "evidence_span": GLOBEX_SPAN,
    }
    replies = [
        json.dumps({"links": [{"relation": secret}]}),
        secret,
        json.dumps({"links": [], secret: 1}),
        json.dumps({"links": [{**row, secret: 1}]}),
        json.dumps({secret: {secret: [secret]}}),
        json.dumps({"links": [{**row, "relation": secret}]}),
    ]
    for reply in replies:
        _, report = run(reply)
        assert report.invalid_output and report.error
        assert secret not in report.error
    _, report = run(json.dumps({"links": [{**row, secret: 1}]}))
    assert report.error.startswith("links.0.<extra>:")


def test_extractor_is_called_once_with_the_passage_and_the_schema():
    extractor = FakeExtractor(json_links([GLOBEX_ROW]))
    run(extractor.reply, extractor=extractor)
    assert len(extractor.calls) == 1
    prompt, schema = extractor.calls[0]
    assert PASSAGE in prompt and "ACME" in prompt
    assert schema == ExtractedLinks.model_json_schema()
    assert "links" in schema["properties"]


def test_long_passage_is_shown_whole():
    long_passage = "filler words. " * 5000 + "Globex Inc accounted for 14% of revenue"
    extractor = FakeExtractor(json_links([GLOBEX_ROW]))
    _, report = run(extractor.reply, passage=long_passage, extractor=extractor)
    assert long_passage in extractor.calls[0][0] and report.kept == 1


def test_braces_in_the_passage_do_not_break_the_prompt():
    passage = PASSAGE + " {filer} {passage} {0}"
    extractor = FakeExtractor(json_links([GLOBEX_ROW]))
    _, report = run(extractor.reply, passage=passage, extractor=extractor)
    assert report.kept == 1 and passage in extractor.calls[0][0]


def test_rules_extractor_version_is_rejected():
    with pytest.raises(ValueError, match="rules-"):
        run(json_links([GLOBEX_ROW]), extractor_version="rules-10k-v1")


def test_naive_accepted_time_is_rejected():
    with pytest.raises(ValueError, match="tz-aware"):
        run(json_links([GLOBEX_ROW]), accepted_utc=datetime(2020, 3, 2, 21, 30))


@pytest.mark.parametrize("field", ["filer", "passage", "source_doc_id", "extractor_version"])
def test_empty_caller_input_is_rejected(field):
    with pytest.raises(ValueError, match=field):
        run(json_links([GLOBEX_ROW]), **{field: "  "})


def test_bad_caller_input_never_reaches_the_model():
    extractor = FakeExtractor(json_links([GLOBEX_ROW]))
    with pytest.raises(ValueError):
        run(extractor.reply, extractor=extractor, filer="")
    assert extractor.calls == []


def test_kept_links_round_trip_through_the_store_and_the_asof_read(tmp_path):
    links, _ = run(json_links([GLOBEX_ROW]))
    store = LinkStore(tmp_path)
    assert store.append(links).appended == 1
    before = graph_asof(store, pd.Timestamp(ACCEPTED) - pd.Timedelta(minutes=1))
    at = graph_asof(store, pd.Timestamp(ACCEPTED))
    assert before.empty
    assert at[["source_node", "receiver_node", "relation"]].values.tolist() == [
        ["GLBX", "ACME", "customer_of"]
    ]
    assert at.loc[0, "evidence_span"] == GLOBEX_SPAN


def test_extracted_link_closed_enums_and_pairing():
    link = ExtractedLink(
        relation=Relation.COUNTRY_EXPOSURE,
        counterparty_name=" Chile ",
        counterparty_kind=CounterpartyKind.COUNTRY,
        evidence_span="sales in Chile",
    )
    assert link.counterparty_name == "Chile"
    with pytest.raises(ValueError):
        ExtractedLink(
            relation=Relation.PEER_OF,
            counterparty_name="X",
            counterparty_kind=CounterpartyKind.COMMODITY,
            evidence_span="x",
        )


def test_prompt_file_has_both_placeholders_and_every_closed_value():
    text = (config.references_dir() / "link_prompt_v1.txt").read_text(encoding="utf-8")
    assert "{filer}" in text and "{passage}" in text
    for relation in Relation:
        assert relation.value in text
    for kind in CounterpartyKind:
        assert kind.value in text
    assert text.format(filer="F", passage="P")  # no stray braces


def test_rung_file_matches_the_g3_shape():
    root = config.references_dir() / "rungs"
    g3 = (root / "m3_ripple_g3.yaml").read_text(encoding="utf-8").splitlines()
    g4 = (root / "m3_ripple_g4.yaml").read_text(encoding="utf-8").splitlines()
    assert g4[0] == "name: m3_ripple_g4"
    assert g4[-1] == g3[-1] == 'panel_prefixes: ["news_", "ripple_"]'


def test_mid_word_cut_of_the_evidence_is_dropped():
    for cut in ("Globex Inc accounted for 1", "lobex Inc accounted for 14%", "Globex Inc accou"):
        _, report = run(json_links([("customer_of", "Globex Inc", "company", cut)]))
        assert report.evidence_missing == 1 and report.kept == 0, cut


def test_evidence_ending_in_punctuation_or_at_the_passage_edges_is_accepted():
    for span in ("Globex Inc accounted for 14% of revenue in 2019.", "Item 1. Business.",
                 "We buy copper from several mines."):
        _, report = run(json_links([("customer_of", "Globex Inc", "company", span)]))
        assert report.kept == 1, span


def test_report_counts_add_up():
    reply = json_links(
        [
            GLOBEX_ROW,  # kept
            GLOBEX_ROW,  # duplicate
            ("customer_of", "Globex Inc", "company", "not in the passage"),  # evidence missing
            ("supplier_of", "Initech LLC", "company", "We buy copper"),  # unresolved
        ]
    )
    links, report = run(reply)
    assert (report.proposed, report.evidence_missing, report.unresolved) == (4, 1, 1)
    assert (report.duplicates, report.kept) == (1, 1) == (1, len(links))
    assert report.proposed == (
        report.evidence_missing + report.unresolved + report.duplicates + report.kept
    )


@pytest.mark.parametrize("node", ["", "   "])
def test_blank_resolved_node_is_unresolved(node):
    links, report = run(json_links([GLOBEX_ROW]), resolve=lambda name: node)
    assert links == [] and report.unresolved == 1 and report.kept == 0


def test_span_copied_only_from_the_prompt_template_is_dropped():
    extractor = FakeExtractor(
        json_links([("customer_of", "Globex Inc", "company", "=== PASSAGE START ===")])
    )
    _, report = run(extractor.reply, extractor=extractor)
    assert "=== PASSAGE START ===" in extractor.calls[0][0]  # it is in the prompt, not the passage
    assert report.evidence_missing == 1 and report.kept == 0
    reply = json_links([("customer_of", "Globex Inc", "company", "closed lists")])
    _, report = run(reply)
    assert report.evidence_missing == 1


def test_a_later_prompt_version_is_used(tmp_path, monkeypatch):
    references = tmp_path / "references"
    references.mkdir()
    source = config.references_dir()
    (references / "ripple_params.json").write_bytes((source / "ripple_params.json").read_bytes())
    (references / "link_prompt_v2.txt").write_text(
        "VERSION TWO for {filer}\n{passage}\n", encoding="utf-8"
    )
    monkeypatch.setattr(config, "references_dir", lambda: references)
    extractor = FakeExtractor(json_links([GLOBEX_ROW]))
    _, report = run(extractor.reply, extractor=extractor, prompt_file="link_prompt_v2.txt")
    assert report.kept == 1
    assert extractor.calls[0][0].startswith("VERSION TWO for ACME")
    assert load_link_prompt(prompt_file="link_prompt_v2.txt").startswith("VERSION TWO")


@pytest.mark.parametrize(
    "name",
    [
        "../link_prompt_v1.txt",
        "sub/link_prompt_v1.txt",
        "/etc/passwd",
        "link_prompt_v.txt",
        "link_prompt_v1.md",
        "link_prompt_v1.txt\n",
        "typing_prompt_v1.txt",
        "",
    ],
)
def test_bad_prompt_file_name_is_rejected_before_the_model_is_called(name):
    extractor = FakeExtractor(json_links([GLOBEX_ROW]))
    with pytest.raises(ValueError, match="prompt_file"):
        run(extractor.reply, extractor=extractor, prompt_file=name)
    assert extractor.calls == []
    with pytest.raises(ValueError, match="prompt_file"):
        load_link_prompt(prompt_file=name)
