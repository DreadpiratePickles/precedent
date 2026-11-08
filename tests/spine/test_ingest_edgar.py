import json
from pathlib import Path

import pytest
from fakes import FakeClock, FakeTransport, ok, status

from precedent.spine.edgar_8k_events import edgar_parser_version
from precedent.spine.event_store import EventStore
from precedent.spine.ingest.edgar import read_ciks, run

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE = json.loads((FIXTURES / "submissions_sample.json").read_text())
FOREIGN = json.loads((FIXTURES / "submissions_foreign_sample.json").read_text())
URL_ACME = "https://data.sec.gov/submissions/CIK0001234567.json"
URL_FOREIGN = "https://data.sec.gov/submissions/CIK0007654321.json"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_SEC_USER_AGENT", "precedent-research contact@example.org")
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    ciks = tmp_path / "ciks.csv"
    ciks.write_text("cik,symbol\n1234567,ACME\n\n# a comment\n7654321,XYZ\n")
    return ciks


def stored_ids(tmp_path):
    store = EventStore(tmp_path / "data" / "events")
    path = store.path_for(edgar_parser_version())
    return [json.loads(line)["article_id"] for line in path.read_text().splitlines()]


def last_json_line(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_run_appends_the_8k_events_and_prints_one_json_line(workspace, tmp_path, capsys):
    transport = FakeTransport({URL_ACME: ok(SAMPLE), URL_FOREIGN: ok(FOREIGN)})
    code = run(["--ciks", str(workspace)], transport=transport, clock=FakeClock())
    assert code == 0
    assert sorted(stored_ids(tmp_path)) == sorted(
        [
            "0001234567-24-000001:2.02",
            "0001234567-24-000001:5.02",
            "0001234567-24-000002:5.02",
            "0007654321-24-000001:1.01",
        ]
    )
    summary = last_json_line(capsys)
    assert summary["ciks_fetched"] == 2
    assert summary["records"] == 4
    assert summary["ciks_failed"] == []
    assert summary["stopped_by_block"] is False
    assert all(r.headers["User-Agent"].startswith("precedent-research") for r in transport.requests)


def test_a_second_run_appends_nothing_new(workspace, tmp_path):
    for _ in range(2):
        transport = FakeTransport({URL_ACME: ok(SAMPLE), URL_FOREIGN: ok(FOREIGN)})
        assert run(["--ciks", str(workspace)], transport=transport, clock=FakeClock()) == 0
    assert len(stored_ids(tmp_path)) == 4


def test_max_requests_bounds_the_fetches(workspace, tmp_path, capsys):
    transport = FakeTransport({URL_ACME: ok(SAMPLE)})
    code = run(
        ["--ciks", str(workspace), "--max-requests", "1"], transport=transport, clock=FakeClock()
    )
    assert code == 0
    assert [r.url for r in transport.requests] == [URL_ACME]
    summary = last_json_line(capsys)
    assert (summary["ciks_fetched"], summary["ciks_not_attempted"]) == (1, 1)


def test_a_failed_cik_is_reported_and_the_rest_are_kept(workspace, tmp_path, capsys):
    transport = FakeTransport({URL_ACME: status(404), URL_FOREIGN: ok(FOREIGN)})
    code = run(["--ciks", str(workspace)], transport=transport, clock=FakeClock())
    assert code == 1
    assert stored_ids(tmp_path) == ["0007654321-24-000001:1.01"]
    summary = last_json_line(capsys)
    assert [failure["cik"] for failure in summary["ciks_failed"]] == [1234567]
    assert summary["ciks_fetched"] == 1


def test_a_rate_limit_stops_the_run(workspace, capsys):
    transport = FakeTransport({URL_ACME: [status(429)] * 4, URL_FOREIGN: ok(FOREIGN)})
    code = run(["--ciks", str(workspace)], transport=transport, clock=FakeClock())
    assert code == 1
    assert len(transport.requests) == 4
    summary = last_json_line(capsys)
    assert summary["stopped_by_rate_limit"] is True
    assert summary["ciks_not_attempted"] == 1


def test_a_403_stops_the_run_after_one_request(workspace, tmp_path, capsys):
    transport = FakeTransport({URL_ACME: status(403), URL_FOREIGN: ok(FOREIGN)})
    code = run(["--ciks", str(workspace)], transport=transport, clock=FakeClock())
    assert code == 1
    assert len(transport.requests) == 1
    summary = last_json_line(capsys)
    assert summary["stopped_by_block"] is True
    assert summary["stopped_by_rate_limit"] is False
    assert summary["ciks_not_attempted"] == 1
    assert summary["ciks_fetched"] == 0


def test_help_exits_with_argparse_code_zero(capsys):
    assert run(["--help"]) == 0
    assert "--ciks" in capsys.readouterr().out


def test_the_summary_counts_filings_without_items(workspace, capsys):
    document = json.loads(json.dumps(SAMPLE))
    document["filings"]["recent"]["items"][3] = None
    transport = FakeTransport({URL_ACME: ok(document), URL_FOREIGN: ok(FOREIGN)})
    assert run(["--ciks", str(workspace)], transport=transport, clock=FakeClock()) == 0
    assert last_json_line(capsys)["skipped_filings_without_items"] == 1


def test_a_missing_user_agent_is_exit_2_and_no_request_is_made(workspace, monkeypatch, capsys):
    monkeypatch.delenv("PRECEDENT_SEC_USER_AGENT")
    transport = FakeTransport([])
    assert run(["--ciks", str(workspace)], transport=transport, clock=FakeClock()) == 2
    assert "PRECEDENT_SEC_USER_AGENT" in capsys.readouterr().err
    assert transport.requests == []


def test_a_bad_ciks_file_is_exit_2(workspace, capsys):
    workspace.write_text("not-a-cik,ACME\n")
    assert run(["--ciks", str(workspace)], transport=FakeTransport([]), clock=FakeClock()) == 2
    assert "expected 'cik,symbol'" in capsys.readouterr().err


def test_a_missing_ciks_file_is_exit_2(workspace, tmp_path):
    missing = tmp_path / "nope.csv"
    assert run(["--ciks", str(missing)], transport=FakeTransport([]), clock=FakeClock()) == 2


def test_missing_arguments_are_exit_2(capsys):
    assert run([]) == 2


def test_a_non_positive_max_requests_is_exit_2(workspace):
    argv = ["--ciks", str(workspace), "--max-requests", "0"]
    assert run(argv, transport=FakeTransport([]), clock=FakeClock()) == 2


def test_read_ciks_skips_header_blank_and_comment_lines(tmp_path):
    path = tmp_path / "ciks.csv"
    path.write_text("cik,symbol\n# note\n\n1234567, ACME \n")
    entries = read_ciks(path)
    assert [(e.cik, e.symbols) for e in entries] == [(1234567, ("ACME",))]


def test_read_ciks_groups_share_classes_under_one_cik(tmp_path):
    path = tmp_path / "ciks.csv"
    path.write_text("1652044,GOOGL\n1234567,ACME\n1652044,GOOG\n1652044,GOOG\n")
    assert [(e.cik, e.symbols) for e in read_ciks(path)] == [
        (1652044, ("GOOG", "GOOGL")),
        (1234567, ("ACME",)),
    ]


def test_read_ciks_sorts_symbols_so_line_order_does_not_matter(tmp_path):
    one, other = tmp_path / "one.csv", tmp_path / "other.csv"
    one.write_text("1,GOOGL\n1,GOOG\n")
    other.write_text("1,GOOG\n1,GOOGL\n")
    assert [e.symbols for e in read_ciks(one)] == [("GOOG", "GOOGL")]
    assert read_ciks(one) == read_ciks(other)


def test_two_symbols_on_one_cik_make_one_fetch_and_one_record_per_item(workspace, tmp_path, capsys):
    workspace.write_text("1234567,ACME\n1234567,ACMEB\n")
    transport = FakeTransport({URL_ACME: ok(SAMPLE)})
    assert run(["--ciks", str(workspace)], transport=transport, clock=FakeClock()) == 0
    assert [r.url for r in transport.requests] == [URL_ACME]
    ids = stored_ids(tmp_path)
    assert len(ids) == len(set(ids)) == 3
    stored = EventStore(tmp_path / "data" / "events").read(edgar_parser_version())
    assert all([i.instrument for i in r.resolved_impacts] == ["ACME", "ACMEB"] for r in stored)
    assert last_json_line(capsys)["ciks_listed"] == 1


def test_a_store_conflict_is_reported_per_cik_and_the_summary_still_prints(
    workspace, tmp_path, capsys
):
    first = FakeTransport({URL_ACME: ok(SAMPLE), URL_FOREIGN: ok(FOREIGN)})
    assert run(["--ciks", str(workspace)], transport=first, clock=FakeClock()) == 0
    capsys.readouterr()
    changed = json.loads(json.dumps(SAMPLE))
    changed["name"] = "Renamed Devices Inc."
    second = FakeTransport({URL_ACME: ok(changed), URL_FOREIGN: ok(FOREIGN)})
    code = run(["--ciks", str(workspace)], transport=second, clock=FakeClock())
    assert code == 1
    summary = last_json_line(capsys)
    assert [failure["cik"] for failure in summary["ciks_failed"]] == [1234567]
    assert summary["ciks_fetched"] == 1
    assert len(second.requests) == 2  # the conflict did not stop the next CIK
