"""D75: the note says which detectors did not run and why; M11: the count is documents in the
session, not documents ingested."""

import json
from datetime import date

from dream_synthetic import gdelt_record, write_events, write_prices

from precedent.dream.report import run
from precedent.dream.report_render import not_run_items


def dream(out, day, *extra):
    return run(["--date", day, "--out", str(out), *extra])


def not_run_lines(out, day):
    text = (out / "dreams" / f"dream_{day}.md").read_text(encoding="utf-8")
    section = text.split("## Not run")[1].split("\n\n")[0]
    return [line for line in section.splitlines() if line.startswith("- [not_run]")]


def test_without_prices_or_links_the_note_names_both(tmp_path, stores_with_gap):
    assert dream(tmp_path, "2024-06-03") == 0
    lines = not_run_lines(tmp_path, "2024-06-03")
    assert "- [not_run] text_vs_price: no prices given" in lines
    assert "- [not_run] asymmetric_link/stale_reliance/link_churn: no link store" in lines
    assert any("short_history" in line for line in lines)
    text = (tmp_path / "dreams" / "dream_2024-06-03.md").read_text()
    assert f"## Not run ({len(lines)})" in text


def test_the_jsonl_twin_carries_the_same_items_and_the_count(tmp_path, stores_with_gap):
    dream(tmp_path, "2024-06-03")
    rows = [
        json.loads(line)
        for line in (tmp_path / "dreams" / "dream_2024-06-03.jsonl").read_text().splitlines()
    ]
    items = [r for r in rows if r["type"] == "not_run"]
    assert items and {"detector", "reason", "text"} <= set(items[0])
    assert rows[0]["counts"]["not_run"] == len(items)
    assert [r["text"] for r in items] == not_run_lines(tmp_path, "2024-06-03")


def test_prices_given_and_found_leave_text_vs_price_out_of_the_section(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    write_events(
        tmp_path / "data" / "events",
        [
            gdelt_record(1, "2023-09-01T15:00:00Z", instrument=None),
            gdelt_record(2, "2024-06-03T14:00:00Z"),
        ],
    )
    write_prices(tmp_path / "prices", "2023-06-01", "2024-06-07")
    flags = ["--prices", str(tmp_path / "prices"), "--market", "MKT"]
    assert dream(tmp_path / "out", "2024-06-03", *flags) == 0
    assert not any("text_vs_price" in x for x in not_run_lines(tmp_path / "out", "2024-06-03"))


def test_a_feed_gap_with_prices_says_text_vs_price_was_skipped_for_it():
    items = not_run_items(
        session=date(2024, 6, 4),
        prices_given=True,
        prices_found=True,
        feed_gap=True,
        has_link_store=True,
        skipped={},
    )
    assert [i["text"] for i in items] == [
        "- [not_run] text_vs_price: feed gap: no documents in the session"
    ]


def test_every_skip_reason_has_a_line_and_a_clean_run_has_none():
    clean = dict(
        session=date(2024, 6, 4),
        prices_given=True,
        prices_found=True,
        feed_gap=False,
        has_link_store=True,
    )
    assert not_run_items(**clean, skipped={}) == []
    items = not_run_items(
        **clean,
        skipped={
            "link_without_claim": 2,
            "shift_short_history": 1,
            "shift_no_live_sessions": 3,
            "shift_no_baseline_events": 4,
            "shift_no_link_store": 1,
        },
    )
    assert [i["text"] for i in items] == [
        "- [not_run] asymmetric_link: link_without_claim (2 links)",
        "- [not_run] event_rate/link_churn: short_history (1 not run)",
        "- [not_run] event_rate: no_live_sessions (3 subjects)",
        "- [not_run] event_rate: no_baseline_events (4 subjects)",
    ]


def test_the_title_counts_documents_in_the_session_not_documents_ingested(
    tmp_path, populated_stores
):
    dream(tmp_path, "2024-06-03")
    title = (tmp_path / "dreams" / "dream_2024-06-03.md").read_text().splitlines()[0]
    assert title == "# Dream for 2024-06-03   (documents in session: 3; feed gap: no)"
    assert "ingested" not in title
