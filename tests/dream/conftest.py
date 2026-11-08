"""Fixtures for the Dream report tests: a data directory the command reads by default."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
from dream_synthetic import (
    customer_link,
    gdelt_record,
    write_events,
    write_links,
)


@dataclass(frozen=True)
class DreamInputs:
    """Where the generated inputs live; `PRECEDENT_DATA_DIR` points at `data_dir`."""

    data_dir: Path
    event_root: Path
    link_root: Path


def _inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> DreamInputs:
    data_dir = tmp_path / "data"
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(data_dir))
    return DreamInputs(data_dir, data_dir / "events", data_dir / "ripple" / "links")


def _june_records():
    return [
        # the session before: one document, so 2024-06-03 is not the first session
        gdelt_record(100, "2024-05-31T15:00:00Z", instrument="BETA", event_type="product"),
        # 2024-06-03: ACME's earnings read positive and its labor news reads negative
        gdelt_record(
            101,
            "2024-06-03T14:00:00Z",
            span="Acme raised its full-year outlook",
            event_type="earnings",
            direction=1.0,
        ),
        gdelt_record(
            102,
            "2024-06-03T15:30:00Z",
            span="Acme announced a round of layoffs",
            event_type="labor",
            direction=-1.0,
        ),
        gdelt_record(103, "2024-06-03T16:00:00Z", instrument="BETA", event_type="mna"),
    ]


@pytest.fixture
def populated_stores(tmp_path, monkeypatch) -> DreamInputs:
    """Four documents around 2024-06-03 and one one-sided customer link first stated that day."""
    inputs = _inputs(tmp_path, monkeypatch)
    write_events(inputs.event_root, _june_records())
    write_links(inputs.link_root, [customer_link()])
    return inputs


@pytest.fixture
def populated_stores_with_late_backfill(tmp_path, monkeypatch) -> DreamInputs:
    """`populated_stores` plus a June document that was only ingested in September."""
    inputs = _inputs(tmp_path, monkeypatch)
    late = gdelt_record(
        9001,
        "2024-06-03T13:00:00Z",
        instrument="ACME",
        event_type="legal_regulatory",
        direction=-1.0,
        span="LATE-DOC says Acme faces a new probe",
        ingested="2024-09-02T10:00:00Z",
    )
    write_events(inputs.event_root, [*_june_records(), late])
    write_links(inputs.link_root, [customer_link()])
    return inputs


@pytest.fixture
def stores_with_gap(tmp_path, monkeypatch) -> DreamInputs:
    """Documents on 2024-06-03 and 2024-06-05 and none on 2024-06-04."""
    inputs = _inputs(tmp_path, monkeypatch)
    write_events(
        inputs.event_root,
        [
            gdelt_record(200, "2024-06-03T14:00:00Z"),
            gdelt_record(201, "2024-06-05T14:00:00Z", event_type="labor", direction=-1.0),
        ],
    )
    return inputs
