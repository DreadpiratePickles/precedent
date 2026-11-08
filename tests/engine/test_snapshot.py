"""Tests for building a reproducible library snapshot and installing a verified download."""

from __future__ import annotations

import datetime as dt
import io
import json
import zipfile
from dataclasses import dataclass, field

import httpx
import pandas as pd
import pytest
from engine_synthetic import CUTOFFS, SESSIONS

from precedent import config
from precedent.engine.snapshot import (
    SnapshotFetchError,
    build_snapshot,
    fetch_snapshot,
    run,
    snapshot_name,
)
from precedent.libraries.bundle import load_bundle, write_bundle
from precedent.libraries.manifest import LibraryIntegrityError, ManifestSource
from precedent.libraries.market_resolution import MARKETS_FRAME_COLUMNS

BASE = "https://releases.invalid/precedent"
TAG = "v0.1.0"
KEY = "armed_conflict/government/none/EM/high/unscheduled/0"


def _events() -> pd.DataFrame:
    rows = []
    for n, position in enumerate((10, 20)):
        rows.append(
            {
                "event_id": f"gdelt:{n}",
                "key": KEY,
                "event_type": "armed_conflict",
                "country_group": "EM",
                "available_utc": CUTOFFS.iloc[position] - pd.Timedelta(hours=1),
                "effective_session": SESSIONS[position].date(),
                "effective_position": position,
                "primary_actor": "MLI",
                "linked_countries": ["MLI"],
                "story_cluster_id": f"gdelt:{n}",
                "is_first_report": True,
                "one_line": "MLI: armed_conflict",
                "source": "gdelt",
                "source_ref": f"gdelt:{n}",
            }
        )
    frame = pd.DataFrame(rows)
    frame["available_utc"] = frame["available_utc"].astype("datetime64[ns, UTC]")
    return frame


def _row(outcome_type: str, key: str, horizon: int, value: float) -> dict[str, object]:
    return {
        "event_id": "gdelt:0",
        "outcome_type": outcome_type,
        "outcome_key": key,
        "horizon": horizon,
        "value": value,
        "is_binary": outcome_type != "asset",
        "effective_session": SESSIONS[10].date(),
        "response_complete_session": SESSIONS[10 + horizon].date(),
        "source_ref": "gdelt:0",
    }


def _markets() -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "market_id": pd.Series(["kalshi:KX-1"], dtype="str"),
            "outcome_key": pd.Series([KEY], dtype="str"),
            "created_utc": pd.Series(
                pd.to_datetime(["2020-01-02T10:00:00Z"], utc=True).as_unit("ns"),
                dtype="datetime64[ns, UTC]",
            ),
            "resolved_utc": pd.Series(
                pd.to_datetime(["2020-02-10T15:00:00Z"], utc=True).as_unit("ns"),
                dtype="datetime64[ns, UTC]",
            ),
            "value": pd.Series([1.0], dtype="float64"),
            "response_complete_session": pd.Series(
                [SESSIONS[27].as_unit("ns")], dtype="datetime64[ns]"
            ),
        }
    )
    return frame[list(MARKETS_FRAME_COLUMNS)]


@pytest.fixture
def library_root(tmp_path_factory):
    """A data dir whose `library/` is a small valid bundle written by `write_bundle`.

    It lives beside the test's own `tmp_path`, not in it: several tests install into `tmp_path`
    and assert that nothing else is there.
    """
    rows = pd.DataFrame(
        [
            _row("event_sequence", KEY, 5, 1.0),
            _row("market_resolution", KEY, 7, 1.0),
            _row("asset", "QQQ", 1, 0.012),
        ]
    )
    placebo = pd.DataFrame(
        {
            "outcome_key": pd.Series(["QQQ"], dtype="str"),
            "horizon": pd.Series([1], dtype="int64"),
            "value": pd.Series([0.01], dtype="float64"),
            "response_complete_session": pd.Series(
                SESSIONS[[21]].as_unit("ns"), dtype="datetime64[ns]"
            ),
        }
    )
    source = ManifestSource(
        name="gdelt2",
        first_date=dt.date(2020, 1, 16),
        last_date=dt.date(2020, 2, 3),
        rows=2,
        citation="synthetic citation",
    )
    root = tmp_path_factory.mktemp("library_root")
    write_bundle(
        root / "library",
        events=_events(),
        rows=rows,
        placebo_asset=placebo,
        markets=_markets(),
        sessions=SESSIONS,
        session_cutoffs_utc=CUTOFFS,
        build_report={"note": "synthetic"},
        sources=[source],
        commit="0" * 40,
    )
    return root


@dataclass
class SnapshotServer:
    """Serves one built snapshot through `httpx.MockTransport`; helpers rewrite the bytes."""

    manifest_bytes: bytes
    entries: dict[str, bytes]
    outer_manifest: bytes = field(init=False)

    def __post_init__(self) -> None:
        self.outer_manifest = self.manifest_bytes

    def _zip(self) -> bytes:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
            for name, data in self.entries.items():
                archive.writestr(zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0)), data)
        return buffer.getvalue()

    def flip_byte(self, name: str) -> None:
        data = bytearray(self.entries[name])
        data[len(data) // 2] ^= 0xFF
        self.entries[name] = bytes(data)

    def add_entry(self, name: str, data: bytes) -> None:
        self.entries[name] = data

    def replace_outer_manifest(self, rows_changed: bool) -> None:
        document = json.loads(self.manifest_bytes)
        if rows_changed:
            document["files"][0]["rows"] = (document["files"][0]["rows"] or 0) + 1
        self.outer_manifest = json.dumps(document).encode()

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        prefix = f"/precedent/{TAG}/"
        if path == prefix + "MANIFEST.json":
            return httpx.Response(200, content=self.outer_manifest)
        if path == prefix + snapshot_name(TAG):
            return httpx.Response(200, content=self._zip())
        return httpx.Response(404)

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)


@pytest.fixture
def snapshot_server(tmp_path_factory, library_root):
    served = tmp_path_factory.mktemp("served")
    zip_path = build_snapshot(library_root / "library", served, TAG)
    with zipfile.ZipFile(zip_path) as archive:
        entries = {info.filename: archive.read(info) for info in archive.infolist()}
    return SnapshotServer((served / "MANIFEST.json").read_bytes(), entries)


def test_fetch_installs_a_verified_snapshot(tmp_path, snapshot_server):
    manifest = fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)
    assert load_bundle(tmp_path / "library").manifest == manifest


def test_a_tampered_byte_is_refused_and_nothing_is_installed(tmp_path, snapshot_server):
    snapshot_server.flip_byte("rows_event_sequence.parquet")
    with pytest.raises(LibraryIntegrityError, match="rows_event_sequence.parquet"):
        fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_a_path_escaping_entry_is_refused(tmp_path, snapshot_server):
    snapshot_server.add_entry("../evil.txt", b"x")
    with pytest.raises(LibraryIntegrityError):
        fetch_snapshot("v0.1.0", tmp_path / "d", snapshot_server.transport, BASE)
    assert not (tmp_path / "evil.txt").exists()
    assert list((tmp_path / "d").iterdir()) == []


def test_an_absolute_entry_is_refused(tmp_path, snapshot_server):
    snapshot_server.add_entry("/abs.txt", b"x")
    with pytest.raises(LibraryIntegrityError):
        fetch_snapshot("v0.1.0", tmp_path / "d", snapshot_server.transport, BASE)


def test_an_unlisted_entry_is_refused(tmp_path, snapshot_server):
    snapshot_server.add_entry("extra.parquet", b"x")
    with pytest.raises(LibraryIntegrityError, match="extra.parquet"):
        fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_inner_and_outer_manifests_must_agree(tmp_path, snapshot_server):
    snapshot_server.replace_outer_manifest(rows_changed=True)
    with pytest.raises(LibraryIntegrityError, match="MANIFEST"):
        fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)


def test_a_failed_fetch_leaves_an_installed_library_untouched(tmp_path, snapshot_server):
    fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)
    before = (tmp_path / "library" / "MANIFEST.json").read_bytes()
    snapshot_server.flip_byte("events.parquet")
    with pytest.raises(LibraryIntegrityError):
        fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)
    assert (tmp_path / "library" / "MANIFEST.json").read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["library"]


def test_a_second_fetch_replaces_the_library(tmp_path, snapshot_server):
    fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)
    manifest = fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)
    assert load_bundle(tmp_path / "library").manifest == manifest
    assert sorted(p.name for p in tmp_path.iterdir()) == ["library"]


def test_missing_release_is_a_fetch_error(tmp_path, snapshot_server):
    with pytest.raises(SnapshotFetchError):
        fetch_snapshot("v9.9.9", tmp_path, snapshot_server.transport, BASE)
    assert sorted(p.name for p in tmp_path.iterdir()) == []


def test_a_timeout_is_a_fetch_error(tmp_path):
    def timing_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(SnapshotFetchError, match="timed out"):
        fetch_snapshot("v0.1.0", tmp_path, httpx.MockTransport(timing_out), BASE)


def test_a_tag_that_could_escape_the_url_is_refused(tmp_path, snapshot_server):
    with pytest.raises(ValueError, match="release tag"):
        fetch_snapshot("../x", tmp_path, snapshot_server.transport, BASE)


def test_build_snapshot_is_byte_identical_across_runs(tmp_path, library_root):
    a = build_snapshot(library_root / "library", tmp_path / "a", "v0.1.0").read_bytes()
    b = build_snapshot(library_root / "library", tmp_path / "b", "v0.1.0").read_bytes()
    assert a == b


def test_build_snapshot_lists_sorted_entries_with_fixed_timestamps(tmp_path, library_root):
    zip_path = build_snapshot(library_root / "library", tmp_path / "out", "v0.1.0")
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
    names = [info.filename for info in infos]
    assert names == sorted(names) and "MANIFEST.json" in names
    assert {info.date_time for info in infos} == {(1980, 1, 1, 0, 0, 0)}
    assert (tmp_path / "out" / "MANIFEST.json").read_bytes() == (
        library_root / "library" / "MANIFEST.json"
    ).read_bytes()


def test_build_snapshot_refuses_a_bundle_that_does_not_verify(tmp_path, library_root):
    (library_root / "library" / "extra.parquet").write_bytes(b"x")
    with pytest.raises(LibraryIntegrityError, match="extra.parquet"):
        build_snapshot(library_root / "library", tmp_path / "out", "v0.1.0")
    assert not (tmp_path / "out").exists()


def test_fetch_without_a_configured_release_exits_2(capsys, monkeypatch):
    monkeypatch.setattr(config, "release_base_url", lambda: None)
    assert (
        run(["--version", "v0.1.0"]) == 2
        and "no release repository configured" in capsys.readouterr().err
    )


def test_fetch_run_prints_sources_and_installs(tmp_path, snapshot_server, capsys, monkeypatch):
    monkeypatch.setattr(config, "release_base_url", lambda: BASE)
    monkeypatch.setattr(
        "precedent.engine.snapshot.httpx.HTTPTransport", lambda: snapshot_server.transport
    )
    code = run(["--version", TAG, "--data-dir", str(tmp_path / "dd")])
    assert code == 0
    assert "gdelt2: 2020-01-16 to 2020-02-03, 2 rows. synthetic citation" in capsys.readouterr().out
    assert (tmp_path / "dd" / "library" / "MANIFEST.json").is_file()


def test_fetch_run_exits_2_on_a_failed_download(tmp_path, snapshot_server, capsys, monkeypatch):
    monkeypatch.setattr(config, "release_base_url", lambda: BASE)
    monkeypatch.setattr(
        "precedent.engine.snapshot.httpx.HTTPTransport", lambda: snapshot_server.transport
    )
    assert run(["--version", "v9.9.9", "--data-dir", str(tmp_path / "dd")]) == 2
    assert "error:" in capsys.readouterr().err
