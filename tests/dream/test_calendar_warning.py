"""M7 (Opus final review / D76): the dream header warns within 60 days of the calendar's end,
so the nightly failure when the story window runs past the last cutoff is announced, not
surprising. The end is derived from references/nyse_holidays.json `last_year`, never hardcoded.
"""

import json
from datetime import date, timedelta

import pytest
from dream_synthetic import gdelt_record, write_events

from precedent.config import references_dir
from precedent.dream.report import run
from precedent.spine.session_calendar import load_nyse_calendar, weekday_sessions

WARNING_DAYS = 60


def _calendar_end() -> date:
    raw = json.loads((references_dir() / "nyse_holidays.json").read_text(encoding="utf-8"))
    calendar = load_nyse_calendar()
    days = weekday_sessions(
        date(raw["last_year"], 12, 1), date(raw["last_year"], 12, 31), calendar
    )
    return days[-1].date()


def _dream(tmp_path, monkeypatch, session: date) -> None:
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    write_events(
        tmp_path / "data" / "events",
        [gdelt_record(1, f"{(session - timedelta(days=7)).isoformat()}T14:00:00Z")],
    )
    assert run(["--date", session.isoformat(), "--out", str(tmp_path / "out")]) == 0


def _header(out: date, tmp_path) -> dict:
    path = tmp_path / "out" / "dreams" / f"dream_{out.isoformat()}.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def _title(out: date, tmp_path) -> str:
    path = tmp_path / "out" / "dreams" / f"dream_{out.isoformat()}.md"
    return path.read_text(encoding="utf-8").splitlines()[0]


def test_a_session_within_sixty_days_of_the_calendar_end_warns_in_the_header(tmp_path, monkeypatch):
    end = _calendar_end()
    session = weekday_sessions(end - timedelta(days=WARNING_DAYS), end, load_nyse_calendar())[0]
    session = session.date()
    _dream(tmp_path, monkeypatch, session)
    warning = _header(session, tmp_path).get("calendar_warning")
    assert warning is not None
    assert end.isoformat() in warning and "nyse_holidays.json" in warning
    lines = (tmp_path / "out" / "dreams" / f"dream_{session}.md").read_text(
        encoding="utf-8"
    ).splitlines()
    assert "calendar warning" in lines[1] and end.isoformat() in lines[1]


def test_a_session_further_out_has_no_warning(tmp_path, monkeypatch):
    end = _calendar_end()
    session = weekday_sessions(
        end - timedelta(days=120), end - timedelta(days=WARNING_DAYS + 1), load_nyse_calendar()
    )[0]
    session = session.date()
    _dream(tmp_path, monkeypatch, session)
    assert "calendar_warning" not in _header(session, tmp_path)
    assert _title(session, tmp_path).startswith("# Dream for ")
