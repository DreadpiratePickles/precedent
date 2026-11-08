"""Tests for the resumable GDELT 2.0 fetcher and the ingest command."""

import hashlib
import json
from datetime import UTC, date, datetime

import pytest
from fakes import FakeClock, FakeTransport, ok_bytes, ok_text, status
from fixtures.gdelt_rows import gdelt_row, tsv, zip_bytes

from precedent.config import GDELT_MASTERFILELIST_URL
from precedent.spine.event_store import EventStore
from precedent.spine.gdelt_events import GdeltFormatError, gdelt_parser_version
from precedent.spine.gdelt_fetch import GdeltFetcher, parse_masterfilelist
from precedent.spine.http import HttpError
from precedent.spine.ingest import gdelt as ingest_gdelt

BASE = "http://files.example.test/gdeltv2/"
D = date(2024, 1, 3)


def name_of(stamp: str) -> str:
    return f"{stamp}.export.CSV.zip"


def body_of(stamp: str) -> bytes:
    return zip_bytes(tsv(gdelt_row(GLOBALEVENTID=stamp[-6:], DATEADDED=stamp)), stamp + ".export.CSV")


def master_line(stamp: str, kind: str = "export.CSV.zip", body: bytes | None = None) -> str:
    data = body if body is not None else body_of(stamp)
    return f"{len(data)} {hashlib.md5(data).hexdigest()} {BASE}{stamp}.{kind}"


def transport_for(
    stamps: list[str], corrupted: tuple[str, ...] = (), missing: tuple[str, ...] = ()
) -> FakeTransport:
    master = "\n".join(master_line(s) for s in stamps) + "\n"
    responses = {GDELT_MASTERFILELIST_URL: ok_text(master)}
    for stamp in stamps:
        served = b"corrupted bytes" if stamp in corrupted else body_of(stamp)
        responses[f"{BASE}{name_of(stamp)}"] = status(404) if stamp in missing else ok_bytes(served)
    return FakeTransport(responses)


def downloads(transport: FakeTransport) -> list[str]:
    return [r.url.rsplit("/", 1)[1] for r in transport.requests if r.url.startswith(BASE)]


MASTER_WITH_MENTIONS_AND_GKG = "\n".join(
    [
        master_line("20240103141500"),
        master_line("20240103141500", "mentions.CSV.zip"),
        master_line("20240103141500", "gkg.csv.zip"),
        master_line("20240103143000"),
        master_line("20240103143000", "translation.export.CSV.zip"),
        "",
    ]
)
THREE_DAYS = ["20240102120000", "20240103141500", "20240103143000", "20240104120000"]
TWO_FILES = ["20240103141500", "20240103143000"]


def test_master_list_keeps_export_files_only():
    assert [
        e.url.rsplit("/", 1)[1] for e in parse_masterfilelist(MASTER_WITH_MENTIONS_AND_GKG)
    ] == ["20240103141500.export.CSV.zip", "20240103143000.export.CSV.zip"]


def test_master_list_entries_carry_size_md5_and_stamp():
    entry = parse_masterfilelist("1234 0123456789abcdef0123456789abcdef " + BASE + name_of("20240103141500"))[0]
    assert (entry.size_bytes, entry.md5) == (1234, "0123456789abcdef0123456789abcdef")
    assert entry.stamp_utc == datetime(2024, 1, 3, 14, 15, tzinfo=UTC)


@pytest.mark.parametrize(
    "bad_line",
    ["only two fields", "abc 0123456789abcdef0123456789abcdef " + BASE + "x.export.CSV.zip",
     "12 nothex " + BASE + "20240103141500.export.CSV.zip",
     "12 0123456789abcdef0123456789abcdef " + BASE + "20241399250000.export.CSV.zip"],
)
def test_a_malformed_master_line_names_its_line_number(bad_line):
    text = master_line("20240103141500") + "\n" + bad_line + "\n"
    with pytest.raises(GdeltFormatError, match="line 2"):
        parse_masterfilelist(text)


def test_fetch_downloads_only_the_window_and_verifies_md5(tmp_path):
    transport = transport_for(THREE_DAYS)
    report = GdeltFetcher(transport, tmp_path / "state.json", tmp_path, max_files=10).fetch(D, D)
    assert (report.listed, report.in_window, report.downloaded, report.md5_failed) == (4, 2, 2, 0)
    assert sorted(downloads(transport)) == [name_of(s) for s in TWO_FILES]
    assert sorted(p.name for p in (tmp_path / "raw" / "gdelt").iterdir()) == [
        name_of(s) for s in TWO_FILES
    ]
    done = json.loads((tmp_path / "state.json").read_text())["done"]
    assert set(done) == {name_of(s) for s in TWO_FILES}


def test_the_window_is_inclusive_by_utc_date(tmp_path):
    stamps = ["20240102235959", "20240103000000", "20240103235959", "20240104000000"]
    report = GdeltFetcher(transport_for(stamps), tmp_path / "s.json", tmp_path, max_files=10).fetch(D, D)
    assert (report.in_window, report.downloaded) == (2, 2)


def test_a_bad_md5_is_not_marked_done(tmp_path):
    corrupted = "20240103143000"
    report = GdeltFetcher(
        transport_for(TWO_FILES, corrupted=(corrupted,)), tmp_path / "state.json", tmp_path, max_files=10
    ).fetch(D, D)
    assert report.md5_failed == 1 and report.downloaded == 1
    assert name_of(corrupted) not in json.loads((tmp_path / "state.json").read_text())["done"]
    assert not (tmp_path / "raw" / "gdelt" / name_of(corrupted)).exists()


def test_a_corrupted_file_is_fetched_again_on_the_next_run(tmp_path):
    state = tmp_path / "state.json"
    GdeltFetcher(transport_for(TWO_FILES, corrupted=("20240103143000",)), state, tmp_path, 10).fetch(D, D)
    second = GdeltFetcher(transport_for(TWO_FILES), state, tmp_path, max_files=10).fetch(D, D)
    assert (second.already_done, second.downloaded, second.md5_failed) == (1, 1, 0)


def test_fetch_resumes_and_respects_max_files(tmp_path):
    state = tmp_path / "state.json"
    first = GdeltFetcher(transport_for(TWO_FILES), state, tmp_path, max_files=1).fetch(D, D)
    assert (first.downloaded, first.remaining) == (1, 1)
    second_transport = transport_for(TWO_FILES)
    second = GdeltFetcher(second_transport, state, tmp_path, max_files=10).fetch(D, D)
    assert (second.already_done, second.downloaded, second.remaining) == (1, 1, 0)
    assert downloads(second_transport) == [name_of("20240103143000")]


def test_a_done_file_that_vanished_from_disk_is_fetched_again(tmp_path):
    state = tmp_path / "state.json"
    GdeltFetcher(transport_for(TWO_FILES), state, tmp_path, max_files=10).fetch(D, D)
    (tmp_path / "raw" / "gdelt" / name_of("20240103141500")).unlink()
    again = GdeltFetcher(transport_for(TWO_FILES), state, tmp_path, max_files=10).fetch(D, D)
    assert (again.already_done, again.downloaded) == (1, 1)


def test_a_download_that_fails_keeps_earlier_progress(tmp_path):
    state = tmp_path / "state.json"
    transport = transport_for(TWO_FILES, missing=("20240103143000",))
    with pytest.raises(HttpError):
        GdeltFetcher(transport, state, tmp_path, max_files=10, clock=FakeClock()).fetch(D, D)
    assert set(json.loads(state.read_text())["done"]) == {name_of("20240103141500")}


def test_done_paths_lists_verified_files_in_the_window(tmp_path):
    fetcher = GdeltFetcher(transport_for(THREE_DAYS), tmp_path / "s.json", tmp_path, max_files=10)
    fetcher.fetch(D, D)
    assert [p.name for p in fetcher.done_paths(D, D)] == [name_of(s) for s in TWO_FILES]
    assert fetcher.done_paths(date(2024, 1, 5), date(2024, 1, 5)) == []


def test_run_fetches_parses_and_appends_events(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    code = ingest_gdelt.run(
        ["--since", "2024-01-03", "--until", "2024-01-03", "--max-files", "5"],
        transport=transport_for(TWO_FILES),
        clock=FakeClock(datetime(2024, 1, 4, tzinfo=UTC)),
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 0
    assert (report["fetch"]["downloaded"], report["ingest"]["rows_read"]) == (2, 2)
    assert report["ingest"]["kept"] + report["ingest"]["dropped_duplicate"] == 2
    stored = EventStore(tmp_path / "events").read(gdelt_parser_version())
    assert len(stored) == report["ingest"]["kept"] == report["store"]["added"]


def test_run_exits_1_when_an_md5_fails(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    code = ingest_gdelt.run(
        ["--since", "2024-01-03", "--until", "2024-01-03"],
        transport=transport_for(TWO_FILES, corrupted=("20240103143000",)),
        clock=FakeClock(datetime(2024, 1, 4, tzinfo=UTC)),
    )
    assert code == 1 and json.loads(capsys.readouterr().out)["fetch"]["md5_failed"] == 1


def test_run_refuses_an_inverted_window(tmp_path, monkeypatch):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as stop:
        ingest_gdelt.run(["--since", "2024-01-04", "--until", "2024-01-03"])
    assert stop.value.code == 2


def test_file_requests_carry_the_timeout_in_its_own_slot(tmp_path):
    """Regression: the timeout was once passed positionally into the params slot."""
    transport = transport_for(TWO_FILES)
    GdeltFetcher(transport, tmp_path / "state.json", tmp_path, max_files=10).fetch(D, D)
    file_requests = [r for r in transport.requests if r.url.startswith(BASE)]
    assert file_requests
    for request in file_requests:
        assert request.params is None
        assert isinstance(request.timeout_s, float)
