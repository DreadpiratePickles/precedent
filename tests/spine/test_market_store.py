"""Tests for market records, typing and the market store."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from precedent.spine.event_key import EventKey
from precedent.spine.market_record import (
    MarketIngestReport,
    MarketRecord,
    ResolvedMarket,
    parse_utc_timestamp,
    type_markets,
)
from precedent.spine.market_store import MarketConflictError, MarketStore
from precedent.spine.typers.base import TypedText
from precedent.spine.typers.fake import FakeTyper

FED_KEY = EventKey(
    event_type="central_bank",
    actor_kind="central_bank",
    impacted_kinds=("bond",),
    country_group="US",
    severity_band="high",
    is_scheduled=True,
    direction_sign=-1,
)


def typed(key: EventKey = FED_KEY, version: str = "fake-v1") -> TypedText:
    return TypedText(
        status="typed",
        key=key,
        reason="rule",
        evidence_span="Fed",
        typer_version=version,
    )


def resolved(market_id: str, question: str, venue: str = "polymarket") -> ResolvedMarket:
    return ResolvedMarket(
        market_id=market_id,
        venue=venue,
        question=question,
        created_utc=datetime(2024, 1, 1, 9, tzinfo=UTC),
        close_utc=None,
        resolved_utc=datetime(2024, 1, 3, 14, 15, tzinfo=UTC),
        resolution="yes",
        category=None,
        source_ref="slug",
    )


FED_MARKET = resolved("polymarket:0xaaa", "Will the Fed cut rates?")
SPORTS_MARKET = resolved("polymarket:0xbbb", "Will the home team win on Sunday?")
MARKET = MarketRecord(**FED_MARKET.model_dump(), typed=typed())


def test_store_is_keyed_by_market_and_typer_version(tmp_path: Path):
    store = MarketStore(tmp_path)
    assert store.append([MARKET]).added == 1
    assert store.append([MARKET]).identical_skipped == 1
    with pytest.raises(MarketConflictError):
        store.append([MARKET.model_copy(update={"resolution": "no"})])
    other_version = MARKET.model_copy(
        update={"typed": MARKET.typed.model_copy(update={"typer_version": "rules-typing-v2"})}
    )
    store.append([other_version])
    assert len(store.read("polymarket", MARKET.typed.typer_version)) == 1
    assert len(store.read("polymarket", "rules-typing-v2")) == 1


def test_a_record_round_trips_through_the_file_with_its_event_key(tmp_path: Path):
    store = MarketStore(tmp_path)
    store.append([MARKET])
    assert store.read("polymarket", "fake-v1") == [MARKET]
    assert (tmp_path / "markets_polymarket.jsonl").read_text().count("\n") == 1


def test_a_conflicting_append_writes_nothing(tmp_path: Path):
    store = MarketStore(tmp_path)
    store.append([MARKET])
    fresh = MARKET.model_copy(update={"market_id": "polymarket:0xnew"})
    with pytest.raises(MarketConflictError):
        store.append([fresh, MARKET.model_copy(update={"resolution": "no"})])
    assert [r.market_id for r in store.read("polymarket", "fake-v1")] == [MARKET.market_id]


def test_each_venue_has_its_own_file_and_a_missing_file_reads_empty(tmp_path: Path):
    store = MarketStore(tmp_path)
    kalshi = MARKET.model_copy(update={"market_id": "kalshi:T1", "venue": "kalshi"})
    store.append([MARKET, kalshi])
    assert sorted(p.name for p in tmp_path.glob("*.jsonl")) == [
        "markets_kalshi.jsonl",
        "markets_polymarket.jsonl",
    ]
    assert MarketStore(tmp_path / "elsewhere").read("kalshi", "fake-v1") == []


def test_a_malformed_line_names_the_file_and_line(tmp_path: Path):
    store = MarketStore(tmp_path)
    store.append([MARKET])
    path = tmp_path / "markets_polymarket.jsonl"
    path.write_text(path.read_text() + '{"market_id": "polymarket:bad"}\n')
    with pytest.raises(ValueError, match=r"markets_polymarket\.jsonl line 2"):
        store.read("polymarket", "fake-v1")


def test_untyped_markets_are_kept_counted_and_marked(tmp_path: Path):
    report = MarketIngestReport()
    records = type_markets(
        [SPORTS_MARKET, FED_MARKET], FakeTyper({FED_MARKET.question: typed()}), report
    )
    assert len(records) == 2 and report.untyped == 1 and report.kept == 2
    MarketStore(tmp_path).append(records)
    frame = MarketStore(tmp_path).to_frame("fake-v1").set_index("market_id")
    assert frame.loc[SPORTS_MARKET.market_id, "event_key"] == "UNTYPED"
    assert frame.loc[FED_MARKET.market_id, "event_key"] == FED_KEY.as_string()
    assert list(frame.columns) == [
        "venue",
        "created_utc",
        "close_utc",
        "resolved_utc",
        "resolution",
        "event_key",
        "typed_status",
        "category",
        "source_ref",
    ]
    assert str(frame["resolved_utc"].dtype) == "datetime64[ns, UTC]"


def test_to_frame_of_an_empty_store_has_the_columns(tmp_path: Path):
    frame = MarketStore(tmp_path).to_frame("fake-v1")
    assert len(frame) == 0 and "event_key" in frame.reset_index().columns


class _BudgetTyper:
    version = "budget-v1"

    def __init__(self, answers: list[TypedText]) -> None:
        self._answers = answers

    def type_text(self, text: str) -> TypedText:
        return self._answers.pop(0)


def test_refusals_and_an_exhausted_budget_are_counted_separately():
    refused = TypedText(
        status="refused_invalid", key=None, reason="bad json", evidence_span=None,
        typer_version="budget-v1",
    )
    exhausted = TypedText(
        status="refused_invalid", key=None, reason="max_calls reached", evidence_span=None,
        typer_version="budget-v1",
    )
    report = MarketIngestReport()
    records = type_markets(
        [FED_MARKET, SPORTS_MARKET, resolved("polymarket:0xccc", "Third?")],
        _BudgetTyper([typed(version="budget-v1"), refused, exhausted]),
        report,
    )
    # Contract change (final review F4): an exhausted budget is transient and is not stored.
    assert [r.typed.status for r in records] == ["typed", "refused_invalid"]
    assert (report.refused_invalid, report.max_calls_reached, report.untyped) == (1, 1, 0)
    assert report.not_stored_transient == 1
    assert report.kept == 2


@pytest.mark.parametrize(
    ("status", "reason", "stored"),
    [
        ("refused_invalid", "max_calls reached", False),
        ("refused_invalid", "http 503", False),
        ("refused_invalid", "http 0", False),
        ("untyped", "dry-run", False),
        ("refused_invalid", "schema: reply body is not JSON", True),
        ("refused_invalid", "evidence span not in text", True),
        ("untyped", "no_market_event", True),
    ],
)
def test_only_transient_typing_outcomes_are_left_out(status, reason, stored):
    answer = TypedText(
        status=status, key=None, reason=reason, evidence_span=None, typer_version="budget-v1"
    )
    report = MarketIngestReport()
    records = type_markets([FED_MARKET], _BudgetTyper([answer]), report)
    assert len(records) == (1 if stored else 0)
    assert (report.kept, report.not_stored_transient) == ((1, 0) if stored else (0, 1))


@pytest.mark.parametrize(
    "value",
    [
        "2024-01-03 14:15:00+00",
        "2024-01-03T14:15:00Z",
        "2024-01-03T15:15:00+01:00",
        "2024-01-03 14:15:00.000000+00",
    ],
)
def test_timestamps_parse_to_timezone_aware_utc(value: str):
    assert parse_utc_timestamp(value) == datetime(2024, 1, 3, 14, 15, tzinfo=UTC)
    assert parse_utc_timestamp(value).utcoffset().total_seconds() == 0


@pytest.mark.parametrize("value", [None, "", "not a time", "2024-01-03 14:15:00", 5])
def test_a_missing_naive_or_unparseable_time_is_none_never_a_guess(value):
    assert parse_utc_timestamp(value) is None


def test_fixtures_are_labelled_synthetic():
    fixtures = Path(__file__).resolve().parents[1] / "fixtures"
    for name in ("polymarket_markets.json", "kalshi_markets.json"):
        label = json.loads((fixtures / name).read_text())["_label"]
        assert label.startswith("SYNTHETIC —")
