"""Generated inputs for the Dream report tests: event files, a link file, parquet prices.

Everything is synthetic and written under a test's `tmp_path`; nothing here touches a network.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from fakes import make_event_record

from precedent.ripple.link_store import Link, LinkStore
from precedent.spine.event_record import EventRecord, ResolvedImpact
from precedent.spine.event_schema import EventType
from precedent.spine.event_store import EventStore
from precedent.spine.session_calendar import (
    calendar_cutoffs,
    load_nyse_calendar,
    weekday_sessions,
)

GDELT = "gdelt2"  # the source name the spine's GDELT converter writes


def at(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def gdelt_record(
    number: int,
    available: str,
    *,
    instrument: str | None = "ACME",
    direction: float = 1.0,
    event_type: str = "earnings",
    span: str = "Acme reported revenue of 12 billion dollars",
    ingested: str | None = None,
    effective_session: date | None = None,
) -> EventRecord:
    """One GDELT-style record available at `available`; no `instrument` makes a filler record."""
    moment = at(available)
    record = make_event_record(
        article_id=f"gdelt:{number}",
        source=GDELT,
        first_seen_utc=moment,
        ingested_utc=at(ingested) if ingested else moment + timedelta(minutes=5),
        available_utc=moment,
        effective_session=effective_session,
        resolved_impacts=(
            []
            if instrument is None
            else [
                ResolvedImpact(
                    instrument=instrument, kind="company", direction=direction, match_method="alias"
                )
            ]
        ),
    )
    event = record.event.model_copy(
        update={"event_type": EventType(event_type), "evidence_span": span}
    )
    return record.model_copy(update={"event": event})


def edgar_record(
    accession: str, item: str, available: str, instrument: str = "ACME"
) -> EventRecord:
    """An 8-K record for `item`; the filer's own instrument resolves through the CIK map."""
    moment = at(available)
    return make_event_record(
        article_id=f"edgar:{accession}:{item}",
        source="edgar",
        first_seen_utc=moment,
        ingested_utc=moment + timedelta(minutes=5),
        available_utc=moment,
        resolved_impacts=[
            ResolvedImpact(
                instrument=instrument, kind="company", direction=-1.0, match_method="cik_map"
            )
        ],
    )


def plant_capex_conflict(
    out: Path,
    first: str = "2024-06-03T14:30:00Z",
    second: str = "2024-06-03T15:00:00Z",
    valid_to: str | None = "2024-06-06T20:00:00Z",
) -> None:
    """Two documents that disagree about ACME's capex, stored as claims under `out`.

    Dream's rules parser builds no typed conflict (event_direction is excluded from
    value_conflict, D72), so a value_conflict test plants the claims a model parser would.
    """
    from precedent.dream.claim_schema import Claim
    from precedent.dream.claim_store import ClaimStore

    def claim(value: str, doc: str, when: str, span: str) -> Claim:
        return Claim(
            subject="ACME",
            attribute="capex_direction",
            period="current_quarter",
            value=value,
            valid_from_utc=at(when),
            valid_to_utc=at(valid_to) if valid_to else None,
            recorded_utc=at(when),
            source_doc_id=doc,
            evidence_span=span,
            extractor_version="rules-test-v1",
        )

    ClaimStore(out / "claims").append(
        [
            claim("up", "8K-ACME-A", first, "Acme will raise capex"),
            claim("down", "8K-ACME-B", second, "Acme will cut capex"),
        ]
    )


def write_events(root: Path, records: list[EventRecord]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    EventStore(root).append(records)


def customer_link(
    source: str = "ACME",
    receiver: str = "BIGCO",
    valid_from: str = "2024-06-03T12:00:00Z",
    valid_to: str = "2025-06-03T12:00:00Z",
    doc: str = "10K-ACME-2023",
) -> Link:
    return Link(
        source_node=source,
        receiver_node=receiver,
        relation="customer_of",
        weight=0.5,
        valid_from_utc=at(valid_from),
        valid_to_utc=at(valid_to),
        source_doc_id=doc,
        evidence_span="Customer A accounted for 22 percent of revenue",
        extractor_version="rules-links-v1",
    )


def write_links(root: Path, links: list[Link]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    LinkStore(root).append(links)


def write_prices(
    prices_dir: Path,
    first: str,
    last: str,
    shocks: dict[tuple[str, str], float] | None = None,
    instruments: tuple[str, ...] = ("ACME",),
    market: str = "MKT",
    seed: int = 3,
) -> None:
    """Parquet files like `precedent prices import` writes: beta-one instruments, 1% noise.

    `shocks` maps (symbol, session date) to an extra return on that session.
    """
    prices_dir.mkdir(parents=True, exist_ok=True)
    days = weekday_sessions(
        date.fromisoformat(first), date.fromisoformat(last), load_nyse_calendar()
    )
    rng = np.random.default_rng(seed)
    market_returns = rng.normal(0.0, 0.01, len(days))
    series = {market: market_returns}
    for name in instruments:
        own = 0.0001 + market_returns + rng.normal(0.0, 0.01, len(days))
        for (symbol, day), shock in (shocks or {}).items():
            if symbol == name:
                own[days.get_loc(pd.Timestamp(day))] += shock
        series[name] = own
    for symbol, returns in series.items():
        close = 100.0 * np.cumprod(1.0 + returns)
        frame = pd.DataFrame({"date": days, "open": close, "close": close})
        frame.to_parquet(prices_dir / f"{symbol}.parquet")


def session_cutoffs(first: str, last: str) -> pd.Series:
    days = weekday_sessions(
        date.fromisoformat(first), date.fromisoformat(last), load_nyse_calendar()
    )
    return calendar_cutoffs(days, load_nyse_calendar())


def hash_tree(root: Path) -> str:
    """One digest over every file's relative path and bytes under `root`."""
    digest = hashlib.sha256()
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def count_lines(root: Path) -> int:
    """Lines in every file under `root`."""
    return sum(len(p.read_bytes().splitlines()) for p in root.rglob("*") if p.is_file())
