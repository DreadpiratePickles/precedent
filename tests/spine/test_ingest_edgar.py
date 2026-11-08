import json
from pathlib import Path

import pytest
from fakes import FakeClock, FakeTransport, ok, status

from precedent.spine.event_store import EventStore
from precedent.spine.ingest.edgar import read_ciks, run

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SAMPLE = json.loads((FIXTURES / "submissions_sample.json").read_text())
FOREIGN = json.loads((FIXTURES / "submissions_foreign_sample.json").read_text())
URL_APPLE = "https://data.sec.gov/submissions/CIK0000320193.json"
URL_FOREIGN = "https://data.sec.gov/submissions/CIK0001234567.json"


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_SEC_USER_AGENT", "precedent-research contact@example.org")
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    ciks = tmp_path / "ciks.csv"
    ciks.write_text("cik,symbol\n320193,AAPL\n\n# a comment\n1234567,XYZ\n")
    return ciks


def stored_ids(tmp_path):
    store = EventStore(tmp_path / "data" / "events")
    path = store.path_for("rules-8k-v1")
    return [json.loads(line)["article_id"] for line in path.read_text().splitlines()]


def last_json_line(capsys):
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def test_run_appends_the_8k_events_and_prints_one_json_line(workspace, tmp_path, capsys):
    transport = FakeTransport({URL_APPLE: ok(SAMPLE), URL_FOREIGN: ok(FOREIGN)})
    code = run(["--ciks", str(workspace)], transport=transport, clock=FakeClock())
    assert code == 0
    assert sorted(stored_ids(tmp_path)) == sorted(
        [
            "0000320193-24-000001:2.02",
            "0000320193-24-000001:5.02",
            "0000320193-24-000002:5.02",
            "0001234567-24-000001:1.01",
        ]
    )
    summary = last_json_line(capsys)
    assert summary["ciks_fetched"] == 2
    assert summary["records"] == 4
    assert summary["ciks_failed"] == []
    assert all(r.headers["User-Agent"].startswith("precedent-research") for r in transport.requests)


def test_a_second_run_appends_nothing_new(workspace, tmp_path):
    for _ in range(2):
        transport = FakeTransport({URL_APPLE: ok(SAMPLE), URL_FOREIGN: ok(FOREIGN)})
        assert run(["--ciks", str(workspace)], transport=transport, clock=FakeClock()) == 0
    assert len(stored_ids(tmp_path)) == 4


def test_max_requests_bounds_the_fetches(workspace, tmp_path, capsys):
    transport = FakeTransport({URL_APPLE: ok(SAMPLE)})
    code = run(
        ["--ciks", str(workspace), "--max-requests", "1"], transport=transport, clock=FakeClock()
    )
    assert code == 0
    assert [r.url for r in transport.requests] == [URL_APPLE]
    summary = last_json_line(capsys)
    assert (summary["ciks_fetched"], summary["ciks_not_attempted"]) == (1, 1)


def test_a_failed_cik_is_reported_and_the_rest_are_kept(workspace, tmp_path, capsys):
    transport = FakeTransport({URL_APPLE: status(404), URL_FOREIGN: ok(FOREIGN)})
    code = run(["--ciks", str(workspace)], transport=transport, clock=FakeClock())
    assert code == 1
    assert stored_ids(tmp_path) == ["0001234567-24-000001:1.01"]
    summary = last_json_line(capsys)
    assert [failure["cik"] for failure in summary["ciks_failed"]] == [320193]
    assert summary["ciks_fetched"] == 1


def test_a_rate_limit_stops_the_run(workspace, capsys):
    transport = FakeTransport({URL_APPLE: [status(429)] * 4, URL_FOREIGN: ok(FOREIGN)})
    code = run(["--ciks", str(workspace)], transport=transport, clock=FakeClock())
    assert code == 1
    assert len(transport.requests) == 4
    summary = last_json_line(capsys)
    assert summary["stopped_by_rate_limit"] is True
    assert summary["ciks_not_attempted"] == 1


def test_a_missing_user_agent_is_exit_2_and_no_request_is_made(workspace, monkeypatch, capsys):
    monkeypatch.delenv("PRECEDENT_SEC_USER_AGENT")
    transport = FakeTransport([])
    assert run(["--ciks", str(workspace)], transport=transport, clock=FakeClock()) == 2
    assert "PRECEDENT_SEC_USER_AGENT" in capsys.readouterr().err
    assert transport.requests == []


def test_a_bad_ciks_file_is_exit_2(workspace, capsys):
    workspace.write_text("not-a-cik,AAPL\n")
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
    path.write_text("cik,symbol\n# note\n\n320193, AAPL \n")
    entries = read_ciks(path)
    assert [(e.cik, e.symbol) for e in entries] == [(320193, "AAPL")]
