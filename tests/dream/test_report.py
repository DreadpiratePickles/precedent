import json
from datetime import date

import pandas as pd
import pytest
from dream_synthetic import (
    at,
    count_lines,
    edgar_record,
    gdelt_record,
    hash_tree,
    session_cutoffs,
    write_events,
    write_prices,
)

from precedent.dream import report as report_module
from precedent.dream.claim_schema import Claim, Shift, ShiftKind, claim_id_for
from precedent.dream.claim_store import ClaimStore, ShiftStore, TensionStore
from precedent.dream.report import diff_between, run
from precedent.dream.report_render import DISCLAIMER

SESSION = "2024-06-03"
SECTIONS = [
    "Opened tensions",
    "Closed tensions",
    "Silent shifts",
    "Expired claims",
    "New claims",
    "Proposed resolutions awaiting review",
]


def dream(out, day=SESSION, *extra):
    return run(["--date", day, "--out", str(out), *extra])


def markdown(out, day=SESSION) -> str:
    return (out / "dreams" / f"dream_{day}.md").read_text(encoding="utf-8")


def item_lines(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("- [")]


# --- the brief's tests -------------------------------------------------------------------


def test_dream_is_idempotent(tmp_path, populated_stores):
    assert dream(tmp_path) == 0
    first = (tmp_path / "dreams" / f"dream_{SESSION}.md").read_bytes()
    first_jsonl = (tmp_path / "dreams" / f"dream_{SESSION}.jsonl").read_bytes()
    lines = count_lines(tmp_path)
    assert dream(tmp_path) == 0
    assert (tmp_path / "dreams" / f"dream_{SESSION}.md").read_bytes() == first
    assert (tmp_path / "dreams" / f"dream_{SESSION}.jsonl").read_bytes() == first_jsonl
    assert count_lines(tmp_path) == lines  # the stores gained no line either


def test_backfilled_dream_equals_live_dream(tmp_path, populated_stores_with_late_backfill):
    # a document about 2024-06-03 was only ingested in September; the June dream must not see it
    assert dream(tmp_path) == 0
    june = markdown(tmp_path)
    assert "LATE-DOC" not in june
    # the late record is not counted as a document of the session either (it was not ingested yet)
    assert "documents ingested: 3;" in june.splitlines()[0]
    assert "LATE-DOC" not in (tmp_path / "claims" / "claims.jsonl").read_text()


def test_a_september_run_that_stores_the_late_claim_leaves_the_june_dream_unchanged(
    tmp_path, populated_stores_with_late_backfill
):
    assert dream(tmp_path) == 0
    june = (tmp_path / "dreams" / "dream_2024-06-03.md").read_bytes()
    june_jsonl = (tmp_path / "dreams" / "dream_2024-06-03.jsonl").read_bytes()
    assert dream(tmp_path, "2024-09-03") == 0
    assert "LATE-DOC" in markdown(tmp_path, "2024-09-03")  # new in September's window
    assert "LATE-DOC" in (tmp_path / "claims" / "claims.jsonl").read_text()
    assert dream(tmp_path) == 0  # re-run June with the late claim now in the claim store
    assert (tmp_path / "dreams" / "dream_2024-06-03.md").read_bytes() == june
    assert (tmp_path / "dreams" / "dream_2024-06-03.jsonl").read_bytes() == june_jsonl


def test_every_tension_line_names_its_documents(tmp_path, populated_stores):
    dream(tmp_path)
    lines = item_lines(markdown(tmp_path))
    assert lines and all("(" in line and ")" in line for line in lines)
    conflict = next(line for line in lines if line.startswith("- [value_conflict]"))
    assert conflict == (
        "- [value_conflict] ACME event_direction unspecified: "
        'negative (gdelt:102, "Acme announced a round of layoffs") vs '
        'positive (gdelt:101, "Acme raised its full-year outlook")'
    )
    link = next(line for line in lines if line.startswith("- [asymmetric_link]"))
    assert "ACME -> BIGCO" in link and '(10K-ACME-2023, "Customer A accounted' in link


def test_inputs_are_untouched(tmp_path, populated_stores):
    before = (hash_tree(populated_stores.event_root), hash_tree(populated_stores.link_root))
    dream(tmp_path)
    assert (hash_tree(populated_stores.event_root), hash_tree(populated_stores.link_root)) == before


def test_feed_gap_session_says_so(tmp_path, stores_with_gap):
    assert dream(tmp_path, "2024-06-04") == 0
    text = markdown(tmp_path, "2024-06-04")
    assert text.splitlines()[0] == (
        "# Dream for 2024-06-04   (documents ingested: 0; feed gap: yes)"
    )
    assert "feed gap: yes" in text


# --- layout and the structured twin ------------------------------------------------------------


def test_markdown_layout_and_counts(tmp_path, populated_stores):
    dream(tmp_path)
    lines = markdown(tmp_path).splitlines()
    assert lines[0] == "# Dream for 2024-06-03   (documents ingested: 3; feed gap: no)"
    counts = {"Opened tensions": 2, "New claims": 4}  # a value conflict and an asymmetric link
    assert [h for h in lines if h.startswith("## ")] == [
        f"## {name} ({counts.get(name, 0)})" for name in SECTIONS
    ]
    assert (
        lines[-1]
        == DISCLAIMER
        == ("Tensions and shifts are observations about documents and prices, not forecasts.")
    )
    assert markdown(tmp_path).endswith("\n") and not markdown(tmp_path).endswith("\n\n")


def test_new_claims_are_those_recorded_in_the_session_window(tmp_path, populated_stores):
    dream(tmp_path)
    claims = markdown(tmp_path).split("## New claims")[1].split("## Proposed")[0]
    assert claims.count("- [claim]") == 4
    assert "gdelt:100" not in claims  # recorded on the previous session
    assert "ACME event_direction unspecified: positive (gdelt:101" in claims


def test_jsonl_has_header_items_in_markdown_order_and_footer(tmp_path, populated_stores):
    dream(tmp_path)
    rows = [
        json.loads(line)
        for line in (tmp_path / "dreams" / f"dream_{SESSION}.jsonl").read_text().splitlines()
    ]
    header, items, footer = rows[0], rows[1:-1], rows[-1]
    assert header["type"] == "header" and header["session"] == SESSION
    assert header["documents_ingested"] == 3 and header["feed_gap"] is False
    assert header["cutoff_utc"] == "2024-06-03T20:00:00+00:00"
    assert header["counts"]["opened_tensions"] == 2 and header["counts"]["new_claims"] == 4
    assert footer == {"type": "footer", "disclaimer": DISCLAIMER}
    assert [r["text"] for r in items] == item_lines(markdown(tmp_path))  # same items, same order
    assert [r["type"] for r in items] == ["opened_tension"] * 2 + ["new_claim"] * 4
    tension = next(r for r in items if r["type"] == "opened_tension")
    assert tension["tension_id"] and tension["claim_ids"]


def test_no_wall_clock_reaches_the_outputs(tmp_path, populated_stores):
    dream(tmp_path)
    jsonl = (tmp_path / "dreams" / f"dream_{SESSION}.jsonl").read_text()
    stamps = {w for w in jsonl.replace('"', " ").split() if w.startswith("20") and "T" in w}
    assert stamps <= {"2024-06-03T20:00:00+00:00"}  # the cutoff, nothing taken from a clock


# --- steps of the night --------------------------------------------------------------------------


def test_only_records_visible_at_the_cutoff_become_claims(tmp_path, stores_with_gap):
    dream(tmp_path)  # the 2024-06-05 record is still in the future
    stored = (tmp_path / "claims" / "claims.jsonl").read_text()
    assert "gdelt:200" in stored and "gdelt:201" not in stored


def test_two_gdelt_reports_of_one_event_in_one_session_become_one_claim(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    day = date(2024, 6, 3)
    write_events(
        tmp_path / "data" / "events",
        [
            gdelt_record(300, "2024-06-03T14:00:00Z", effective_session=day),
            gdelt_record(301, "2024-06-03T15:00:00Z", effective_session=day),  # a duplicate
        ],
    )
    assert dream(tmp_path) == 0
    stored = (tmp_path / "claims" / "claims.jsonl").read_text().splitlines()
    assert len(stored) == 1 and "gdelt:300" in stored[0]


def test_an_annual_claim_expires_after_fifteen_months_and_its_tension_closes(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    # an 8-K item 2.05 claims headcount down from 2023-02-01; a later claim says up
    write_events(
        tmp_path / "data" / "events", [edgar_record("0001", "2.05", "2023-02-01T21:30:00Z")]
    )
    rival = Claim(
        subject="ACME",
        attribute="headcount_direction",
        period="unspecified",
        value="up",
        valid_from_utc=at("2024-02-01T15:00:00Z"),
        recorded_utc=at("2024-02-01T15:00:00Z"),
        source_doc_id="PRESEED-1",
        evidence_span="Acme plans to hire",
        extractor_version="rules-test-v1",
    )
    ClaimStore(tmp_path / "claims").append([rival])

    assert dream(tmp_path, "2024-03-04") == 0
    opened = markdown(tmp_path, "2024-03-04")
    assert "## Opened tensions (1)" in opened and "## Expired claims (0)" in opened
    assert (
        "[value_conflict] ACME headcount_direction unspecified: down (edgar:0001:2.05, " in opened
    )

    assert dream(tmp_path, "2024-05-06") == 0  # 15 months after 2023-02-01 has passed
    later = markdown(tmp_path, "2024-05-06")
    assert "## Expired claims (1)" in later and "## Closed tensions (1)" in later
    expired = next(line for line in item_lines(later) if line.startswith("- [expired]"))
    assert "edgar:0001:2.05" in expired and "rule:annual_15_months" in expired
    closed = later.split("## Closed tensions")[1].split("## Silent")[0]
    assert "[value_conflict] ACME headcount_direction" in closed
    [tension] = TensionStore(tmp_path / "tensions").latest()
    assert tension.status == "closed" and tension.closed_session == date(2024, 5, 6)
    assert tension.recorded_utc == at("2024-05-06T20:00:00Z")  # the cutoff, never a clock
    # the expiry is an appended record; the original line stays
    lines = (tmp_path / "claims" / "claims.jsonl").read_text().count("edgar:0001:2.05")
    assert lines >= 2


def test_a_tension_closes_when_its_story_window_claim_expires(tmp_path, populated_stores):
    dream(tmp_path)  # opens the ACME positive-versus-negative conflict on 2024-06-03
    dream(tmp_path, "2024-06-06")  # three sessions later both event claims have ended
    closed = markdown(tmp_path, "2024-06-06").split("## Closed tensions")[1].split("## Silent")[0]
    assert "[value_conflict] ACME event_direction" in closed
    statuses = {t.kind.value: t.status for t in TensionStore(tmp_path / "tensions").latest()}
    assert statuses["value_conflict"] == "closed" and statuses["asymmetric_link"] == "open"


def test_re_running_an_earlier_session_does_not_close_a_later_tension(tmp_path, stores_with_gap):
    assert dream(tmp_path, "2024-06-05") == 0  # 200 positive and 201 negative conflict on 06-05
    [opened] = TensionStore(tmp_path / "tensions").latest()
    assert opened.opened_session == date(2024, 6, 5) and opened.status == "open"
    assert dream(tmp_path, "2024-06-03") == 0  # claim 201 is not yet visible at that cutoff
    [again] = TensionStore(tmp_path / "tensions").latest()
    assert again.status == "open"


# --- silent shifts -----------------------------------------------------------------------------------


def planted_history(data_dir) -> str:
    """0.1 ACME events per session for 250 sessions, then 0.3 for 60, a filler every session."""
    sessions = session_cutoffs("2023-03-01", "2024-12-31")
    positions = list(range(0, 250, 10)) + list(range(250, 310, 3))
    records = []
    for position in range(311):
        available = (sessions.iloc[position] - pd.Timedelta(hours=1)).isoformat()
        records.append(gdelt_record(1000 + position, available, instrument=None))
        if position in positions:
            records.append(gdelt_record(5000 + position, available, instrument="ACME"))
    write_events(data_dir / "events", records)
    return sessions.index[310].strftime("%Y-%m-%d")


def test_a_silent_shift_is_reported_and_stays_byte_identical_on_a_re_run(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    day = planted_history(tmp_path / "data")
    assert dream(tmp_path, day) == 0
    text = markdown(tmp_path, day)
    assert "## Silent shifts (1)" in text
    [line] = [x for x in item_lines(text) if x.startswith("- [event_rate]")]
    assert line.startswith("- [event_rate] ACME: event rate z ") and "(threshold 3.0)" in line
    first = (tmp_path / "dreams" / f"dream_{day}.md").read_bytes()
    lines = count_lines(tmp_path / "shifts")
    assert dream(tmp_path, day) == 0
    assert (tmp_path / "dreams" / f"dream_{day}.md").read_bytes() == first
    assert count_lines(tmp_path / "shifts") == lines  # not re-reported: it did not grow


def test_a_shift_that_grew_past_the_re_report_fraction_is_re_fired_in_place(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    day = planted_history(tmp_path / "data")
    dream(tmp_path / "first", day)
    [fresh] = ShiftStore(tmp_path / "first" / "shifts").latest()
    earlier = session_cutoffs("2023-03-01", "2024-12-31")
    previous = earlier.index[309]
    older = Shift(
        kind=ShiftKind.event_rate,
        subject="ACME",
        detected_session=previous.date(),
        statistic=fresh.statistic / 2,
        threshold=fresh.threshold,
        window_sessions=fresh.window_sessions,
        detector_version=fresh.detector_version,
        recorded_utc=earlier.iloc[309].to_pydatetime(),
    )
    ShiftStore(tmp_path / "second" / "shifts").append([older])
    assert dream(tmp_path / "second", day) == 0
    stored = ShiftStore(tmp_path / "second" / "shifts")
    assert len(stored.read_all()) == 2 and len(stored.latest()) == 1
    [again] = stored.latest()
    assert again.shift_id == fresh.shift_id and again.statistic == fresh.statistic
    assert again.detected_session.isoformat() == day
    assert "## Silent shifts (1)" in markdown(tmp_path / "second", day)


# --- text versus price ----------------------------------------------------------------------------------


def priced_stores(
    tmp_path, monkeypatch, shock_day=SESSION, first_price="2023-06-01", last_price="2024-06-07"
):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    from dream_synthetic import gdelt_record as rec

    write_events(
        tmp_path / "data" / "events",
        [
            rec(1, "2023-09-01T15:00:00Z", instrument=None),  # history before the session
            rec(2, "2024-06-03T14:00:00Z", span="Acme raised its full-year outlook"),
        ],
    )
    prices = tmp_path / "prices"
    write_prices(prices, first_price, last_price, shocks={("ACME", shock_day): -0.08})
    return ["--prices", str(prices), "--market", "MKT"]


def test_text_vs_price_uses_a_panel_built_once_per_invocation(tmp_path, monkeypatch):
    flags = priced_stores(tmp_path, monkeypatch)
    calls = []
    real = report_module.abnormal_returns_panel

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(report_module, "abnormal_returns_panel", counting)
    assert dream(tmp_path, SESSION, *flags) == 0
    assert len(calls) == 1
    [line] = [x for x in item_lines(markdown(tmp_path)) if x.startswith("- [text_vs_price]")]
    assert line.startswith(
        '- [text_vs_price] ACME: event_direction positive (gdelt:2, "Acme raised its full-year outlook"), '
        "abnormal return -"
    )
    assert "(z -" in line


def test_prices_with_no_row_for_the_session_skip_the_detector_and_say_so(tmp_path, monkeypatch):
    flags = priced_stores(
        tmp_path, monkeypatch, shock_day="2024-05-30", last_price="2024-05-31"
    )  # nothing for 06-03
    assert dream(tmp_path, SESSION, *flags) == 0
    text = markdown(tmp_path)
    assert text.splitlines()[0] == (
        "# Dream for 2024-06-03   (documents ingested: 1; feed gap: no; prices: none for 2024-06-03)"
    )
    assert "[text_vs_price]" not in text
    header = json.loads(
        (tmp_path / "dreams" / f"dream_{SESSION}.jsonl").read_text().splitlines()[0]
    )
    assert header["skipped"]["no_prices"] == 1 and header["prices_none_for"] == SESSION


def test_prices_for_the_session_leave_the_header_line_unchanged(tmp_path, monkeypatch):
    flags = priced_stores(tmp_path, monkeypatch)
    assert dream(tmp_path, SESSION, *flags) == 0
    assert "prices:" not in markdown(tmp_path).splitlines()[0]


def test_a_dream_without_prices_says_the_price_detector_was_skipped(tmp_path, populated_stores):
    dream(tmp_path)
    header = json.loads(
        (tmp_path / "dreams" / f"dream_{SESSION}.jsonl").read_text().splitlines()[0]
    )
    assert header["skipped"]["no_prices"] == 1


def test_returns_that_share_no_session_with_the_axis_exit_one(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    write_events(tmp_path / "data" / "events", [gdelt_record(1, "2024-06-03T14:00:00Z")])
    write_prices(tmp_path / "old", "2019-01-02", "2019-06-28")
    code = dream(tmp_path, SESSION, "--prices", str(tmp_path / "old"), "--market", "MKT")
    assert code == 1
    assert "no session in common" in capsys.readouterr().err
    assert not (tmp_path / "dreams").exists()


# --- exit codes -------------------------------------------------------------------------------------------


def test_a_missing_date_is_a_usage_error(tmp_path, populated_stores):
    assert run(["--out", str(tmp_path)]) == 2
    assert not (tmp_path / "dreams").exists()


@pytest.mark.parametrize("day", ["2024-06-01", "2024-07-04", "2031-01-06", "not-a-date"])
def test_a_date_that_is_not_a_session_is_a_usage_error(tmp_path, populated_stores, day, capsys):
    assert run(["--date", day, "--out", str(tmp_path)]) == 2
    assert not (tmp_path / "dreams").exists()
    capsys.readouterr()


def test_an_empty_or_missing_events_directory_is_a_usage_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    assert dream(tmp_path) == 2  # no data directory at all
    (tmp_path / "data" / "events").mkdir(parents=True)
    assert dream(tmp_path) == 2  # a directory with no event file
    assert "events" in capsys.readouterr().err
    assert not (tmp_path / "dreams").exists()


def test_prices_need_a_market_that_the_directory_holds(tmp_path, populated_stores, capsys):
    write_prices(tmp_path / "prices", "2024-01-02", "2024-06-05")
    assert dream(tmp_path, SESSION, "--prices", str(tmp_path / "prices")) == 2
    assert dream(tmp_path, SESSION, "--prices", str(tmp_path / "prices"), "--market", "NOPE") == 2
    assert not (tmp_path / "dreams").exists()
    capsys.readouterr()


def test_a_store_conflict_exits_one_and_writes_no_dream(tmp_path, populated_stores, capsys):
    clash = Claim(
        subject="ACME",
        attribute="event_direction",
        period="unspecified",
        value="positive",
        valid_from_utc=at("2024-06-03T14:00:00Z"),
        recorded_utc=at("2024-06-03T14:05:00Z"),
        source_doc_id="gdelt:101",
        evidence_span="a different span for the same claim id",
        extractor_version="rules-events-v1",
    )
    ClaimStore(tmp_path / "claims").append([clash])
    assert dream(tmp_path) == 1
    error = capsys.readouterr().err
    assert "claims.jsonl" in error
    assert claim_id_for("ACME", "event_direction", "unspecified", "positive", "gdelt:101") in error
    assert not (tmp_path / "dreams").exists()


def test_a_malformed_store_line_exits_one_and_names_the_file(tmp_path, populated_stores, capsys):
    (tmp_path / "tensions").mkdir()
    (tmp_path / "tensions" / "tensions.jsonl").write_text("{not json}\n")
    assert dream(tmp_path) == 1
    assert "tensions.jsonl line 1" in capsys.readouterr().err
    assert not (tmp_path / "dreams").exists()


def test_json_flag_prints_one_machine_readable_summary(tmp_path, populated_stores, capsys):
    assert dream(tmp_path, SESSION, "--json") == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["session"] == SESSION and summary["counts"]["opened_tensions"] == 2
    assert summary["markdown"].endswith("dream_2024-06-03.md")


# --- the morning view -----------------------------------------------------------------------------------------


def test_diff_between_shows_what_changed_across_sessions(tmp_path, populated_stores):
    dream(tmp_path)
    dream(tmp_path, "2024-06-06")
    a = tmp_path / "dreams" / "dream_2024-06-03.md"
    b = tmp_path / "dreams" / "dream_2024-06-06.md"
    text = diff_between(a, b)
    assert text.startswith("--- dream_2024-06-03.md\n+++ dream_2024-06-06.md\n")
    assert "-## Opened tensions (2)" in text and "+## Closed tensions (1)" in text
    assert diff_between(a, a) == ""


# --- parser versions (D53) -----------------------------------------------------------------------------


def two_parser_versions(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    first = gdelt_record(1, "2024-06-03T14:00:00Z")
    second = gdelt_record(2, "2024-06-03T15:00:00Z", event_type="labor", direction=-1.0)
    write_events(
        tmp_path / "data" / "events",
        [
            first.model_copy(update={"parser_version": "alpha"}),
            second.model_copy(update={"parser_version": "beta"}),
        ],
    )


def test_several_parser_versions_without_a_flag_exit_two_and_name_the_versions(
    tmp_path, monkeypatch, capsys
):
    two_parser_versions(tmp_path, monkeypatch)
    assert dream(tmp_path) == 2
    error = capsys.readouterr().err
    assert "alpha" in error and "beta" in error and "--parser-version" in error
    assert not (tmp_path / "dreams").exists() and not (tmp_path / "claims").exists()


def test_the_parser_version_flag_selects_one_events_file(tmp_path, monkeypatch):
    two_parser_versions(tmp_path, monkeypatch)
    assert dream(tmp_path, SESSION, "--parser-version", "beta") == 0
    stored = (tmp_path / "claims" / "claims.jsonl").read_text()
    assert "gdelt:2" in stored and "gdelt:1" not in stored


def test_an_unknown_parser_version_is_a_usage_error(tmp_path, populated_stores, capsys):
    assert dream(tmp_path, SESSION, "--parser-version", "nope") == 2
    assert "v1" in capsys.readouterr().err  # the versions that do exist
    assert not (tmp_path / "dreams").exists()


def test_a_single_events_file_is_used_without_a_flag(tmp_path, populated_stores):
    assert dream(tmp_path) == 0
    assert dream(tmp_path, SESSION, "--parser-version", "v1") == 0
