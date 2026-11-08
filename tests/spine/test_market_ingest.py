"""Tests that drive both ingest commands end to end with fakes."""

import json
from pathlib import Path

import pytest
from fakes import ExplodingTransport, FakeClock, FakeTransport, ok, status
from market_fixtures import KALSHI_FIXTURE, POLYMARKET_FIXTURE, kalshi_transport

from precedent.config import POLYMARKET_GAMMA_BASE_URL
from precedent.spine.event_key import EventKey
from precedent.spine.ingest import kalshi, polymarket
from precedent.spine.market_record import is_transient_outcome
from precedent.spine.market_store import MarketStore
from precedent.spine.typers.base import TypedText
from precedent.spine.typers.fake import FakeTyper

PM = POLYMARKET_FIXTURE
TRANSIENT_STOP_LINE = (
    "ingest stopped: {n} market(s) could not be typed (transient: budget or transport); "
    "nothing past this page was fetched; re-run to continue"
)
KEY = EventKey("central_bank", "central_bank", ("bond",), "US", "high", True, -1)
FED_QUESTION = "Will the Fed cut rates in January 2024?"
TYPER = FakeTyper(
    {
        FED_QUESTION: TypedText(
            status="typed", key=KEY, reason="r", evidence_span="Fed", typer_version="fake-v1"
        )
    }
)


@pytest.fixture(autouse=True)
def data_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path))
    return tmp_path


def last_json_line(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out.strip().splitlines()[-1])


def run_polymarket(argv: list[str], transport: FakeTransport) -> int:
    return polymarket.run(argv, transport=transport, clock=FakeClock(), typer=TYPER)


def test_polymarket_ingest_fetches_types_stores_and_reports(data_root, capsys):
    transport = FakeTransport([ok({"markets": PM["response"]})])
    assert run_polymarket([], transport) == 0
    stored = MarketStore(data_root / "markets").read("polymarket", "fake-v1")
    assert [(r.market_id, r.typed.status) for r in stored] == [
        ("polymarket:0xaaa", "typed"),
        ("polymarket:0xbbb", "untyped"),
    ]
    line = last_json_line(capsys)
    assert line["report"]["kept"] == 2 and line["report"]["untyped"] == 1
    assert line["report"]["requests"] == 1 and line["report"]["seen"] == 5
    assert line["append"] == {"added": 2, "identical_skipped": 0}
    assert json.loads((data_root / "state" / "polymarket.json").read_text())["exhausted"] is True
    assert transport.requests[0].url.startswith(POLYMARKET_GAMMA_BASE_URL)


def test_category_filter_skips_null_and_unlisted_categories(data_root, capsys):
    transport = FakeTransport([ok({"markets": PM["response"]})])
    assert run_polymarket(["--category", "politics", "--category", "macro"], transport) == 0
    stored = MarketStore(data_root / "markets").read("polymarket", "fake-v1")
    assert [r.market_id for r in stored] == ["polymarket:0xaaa"]  # 0xbbb has an empty category
    assert last_json_line(capsys)["report"]["skipped_category"] == 1


def test_max_requests_defaults_to_fifty_and_can_be_lowered():
    def pages() -> FakeTransport:
        return FakeTransport(
            [
                ok({"markets": [{**PM["response"][0], "conditionId": f"0x{n}"}], "next_cursor": f"c{n}"})
                for n in range(60)
            ]
        )

    default = pages()
    assert run_polymarket([], default) == 0
    assert len(default.requests) == 50
    lowered = pages()
    run_polymarket(["--max-requests", "2"], lowered)
    assert len(lowered.requests) == 2


def test_a_failure_after_some_pages_keeps_them_prints_the_report_and_exits_one(data_root, capsys):
    transport = FakeTransport([ok({"markets": PM["response"][:2], "next_cursor": "c1"}), status(404)])
    assert run_polymarket([], transport) == 1
    assert len(MarketStore(data_root / "markets").read("polymarket", "fake-v1")) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.out.strip().splitlines()[-1])["append"]["added"] == 2
    assert "404" in captured.err
    assert json.loads((data_root / "state" / "polymarket.json").read_text())["cursor"] == "c1"


def test_a_usage_error_exits_two():
    assert run_polymarket(["--max-requests", "0"], FakeTransport([])) == 2
    assert polymarket.run(["--nope"]) == 2


def test_a_config_error_from_the_typer_exits_two(monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert polymarket.run([], transport=FakeTransport([]), clock=FakeClock()) == 2


def test_max_calls_and_dry_run_reach_the_typer_factory(monkeypatch):
    seen: dict = {}

    def fake_factory(max_calls=None, dry_run=False):
        seen.update(max_calls=max_calls, dry_run=dry_run)
        return TYPER

    monkeypatch.setattr("precedent.spine.ingest.market_run.typer_from_env", fake_factory)
    polymarket.run(
        ["--max-calls", "7", "--dry-run"],
        transport=FakeTransport([ok({"markets": []})]),
        clock=FakeClock(),
    )
    assert seen == {"max_calls": 7, "dry_run": True}


def test_kalshi_ingest_runs_end_to_end(data_root, capsys):
    code = kalshi.run(
        ["--category", "economics"],
        transport=kalshi_transport(KALSHI_FIXTURE),
        clock=FakeClock(),
        typer=TYPER,
    )
    assert code == 0
    stored = MarketStore(data_root / "markets").read("kalshi", "fake-v1")
    assert [r.market_id for r in stored] == ["kalshi:FED-26OCT-T4.25", "kalshi:FED-25MAR-T4.50"]
    line = last_json_line(capsys)
    assert line["report"]["kept"] == 2 and line["report"]["skipped_unresolved"] == 1
    assert json.loads((data_root / "state" / "kalshi.json").read_text())["phase"] == "done"


def test_a_budget_stop_is_stored_and_reported_in_the_summary_line(data_root, capsys):
    # cutoff, live page, event, series: the live page completes, the historical page cannot start
    code = kalshi.run(
        ["--max-requests", "4"],
        transport=kalshi_transport(KALSHI_FIXTURE),
        clock=FakeClock(),
        typer=TYPER,
    )
    assert code == 0
    line = last_json_line(capsys)
    assert line["report"]["stopped_by_budget"] is True
    assert (line["report"]["seen"], line["report"]["kept"]) == (2, 1)
    assert line["append"]["added"] == 1
    assert len(MarketStore(data_root / "markets").read("kalshi", "fake-v1")) == 1


def _tree_snapshot(root: Path) -> list[str]:
    return sorted(str(p.relative_to(root)) for p in root.rglob("*"))


@pytest.mark.parametrize("venue", ["polymarket", "kalshi"])
def test_dry_run_sends_no_request_and_writes_nothing(venue, data_root, capsys):
    module = {"polymarket": polymarket, "kalshi": kalshi}[venue]
    code = module.run(["--dry-run"], transport=ExplodingTransport(), clock=FakeClock(), typer=TYPER)
    assert code == 0
    assert _tree_snapshot(data_root) == []
    line = last_json_line(capsys)
    assert line["dry_run"] is True and "no request was sent" in line["planned"]
    assert line["report"]["requests"] == 0 and line["append"] == {"added": 0, "identical_skipped": 0}


@pytest.mark.parametrize("venue", ["polymarket", "kalshi"])
def test_dry_run_with_the_default_typer_also_touches_nothing(venue, data_root, capsys):
    module = {"polymarket": polymarket, "kalshi": kalshi}[venue]
    assert module.run(["--dry-run"], transport=ExplodingTransport(), clock=FakeClock()) == 0
    assert _tree_snapshot(data_root) == []


class _TransientTyper:
    """Answers the Fed question; everything else is a network blip."""

    version = "fake-v1"

    def type_text(self, text: str) -> TypedText:
        if text == FED_QUESTION:
            return TYPER.type_text(text)
        return TypedText(
            status="refused_invalid", key=None, reason="http 503", evidence_span=None,
            typer_version="fake-v1",
        )


def test_polymarket_transient_outcomes_are_not_stored_and_the_page_is_fetched_again(
    data_root, capsys
):
    first = FakeTransport([ok({"markets": PM["response"]})])
    code = polymarket.run([], transport=first, clock=FakeClock(), typer=_TransientTyper())
    assert code == 1  # Ruling P0-1: a transient stop is a partial failure
    captured = capsys.readouterr()
    assert TRANSIENT_STOP_LINE.format(n=1) in captured.err
    stored = MarketStore(data_root / "markets").read("polymarket", "fake-v1")
    assert [r.market_id for r in stored] == ["polymarket:0xaaa"]
    report = json.loads(captured.out.strip().splitlines()[-1])["report"]
    assert (report["kept"], report["not_stored_transient"], report["refused_invalid"]) == (1, 1, 1)
    assert report["stopped_by_transient"] is True
    assert not (data_root / "state" / "polymarket.json").exists()  # the cursor did not advance
    second = FakeTransport([ok({"markets": PM["response"]})])
    assert run_polymarket([], second) == 0
    again = MarketStore(data_root / "markets").read("polymarket", "fake-v1")
    assert [(r.market_id, r.typed.status) for r in again] == [
        ("polymarket:0xaaa", "typed"),
        ("polymarket:0xbbb", "untyped"),
    ]


def test_kalshi_transient_outcomes_are_not_stored(data_root, capsys):
    code = kalshi.run(
        [], transport=kalshi_transport(KALSHI_FIXTURE), clock=FakeClock(), typer=_TransientTyper()
    )
    assert code == 1  # Ruling P0-1
    captured = capsys.readouterr()
    report = json.loads(captured.out.strip().splitlines()[-1])["report"]
    assert TRANSIENT_STOP_LINE.format(n=report["not_stored_transient"]) in captured.err
    assert report["not_stored_transient"] >= 1 and report["stopped_by_transient"] is True
    stored = MarketStore(data_root / "markets").read("kalshi", "fake-v1")
    assert all(r.typed.reason != "http 503" for r in stored)
    assert len(stored) == report["kept"]


class _CountingTyper:
    """Types like TYPER and counts every call."""

    version = "fake-v1"

    def __init__(self) -> None:
        self.calls = 0

    def type_text(self, text: str) -> TypedText:
        self.calls += 1
        return TYPER.type_text(text)


def test_already_stored_markets_are_skipped_before_typing(data_root, capsys):
    first = _CountingTyper()
    code = polymarket.run(
        [], transport=FakeTransport([ok({"markets": PM["response"]})]), clock=FakeClock(),
        typer=first,
    )
    assert code == 0 and first.calls == 2
    store = MarketStore(data_root / "markets")
    stored_before = len(store.read("polymarket", "fake-v1"))
    capsys.readouterr()
    (data_root / "state" / "polymarket.json").unlink()  # crawl again from the first page
    second = _CountingTyper()
    code = polymarket.run(
        [], transport=FakeTransport([ok({"markets": PM["response"]})]), clock=FakeClock(),
        typer=second,
    )
    assert code == 0
    assert second.calls == 0
    report = last_json_line(capsys)["report"]
    assert report["already_stored"] == stored_before == 2
    assert report["kept"] == 0
    assert len(store.read("polymarket", "fake-v1")) == stored_before


class _BudgetTyper:
    """Spine `max_calls` semantics: after `budget` answers every later call is refused."""

    version = "fake-v1"

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._used = 0

    def type_text(self, text: str) -> TypedText:
        if self._used >= self._budget:
            return TypedText(
                status="refused_invalid", key=None, reason="max_calls reached",
                evidence_span=None, typer_version="fake-v1",
            )
        self._used += 1
        return TYPER.type_text(text)


def test_max_calls_below_page_size_progresses_across_runs(data_root, capsys):
    store = MarketStore(data_root / "markets")
    code = polymarket.run(
        [], transport=FakeTransport([ok({"markets": PM["response"]})]), clock=FakeClock(),
        typer=_BudgetTyper(1),
    )
    captured = capsys.readouterr()
    assert code == 1
    assert TRANSIENT_STOP_LINE.format(n=1) in captured.err
    assert len(store.read("polymarket", "fake-v1")) == 1
    assert not (data_root / "state" / "polymarket.json").exists()
    code = polymarket.run(
        [], transport=FakeTransport([ok({"markets": PM["response"]})]), clock=FakeClock(),
        typer=_BudgetTyper(1),
    )
    assert code == 0
    report = last_json_line(capsys)["report"]
    assert report["already_stored"] == 1 and report["kept"] == 1
    assert [r.market_id for r in store.read("polymarket", "fake-v1")] == [
        "polymarket:0xaaa",
        "polymarket:0xbbb",
    ]


class _ForbiddenTyper:
    """Answers the Fed question; everything else gets a permanent `http 403`."""

    version = "fake-v1"

    def type_text(self, text: str) -> TypedText:
        if text == FED_QUESTION:
            return TYPER.type_text(text)
        return TypedText(
            status="refused_invalid", key=None, reason="http 403", evidence_span=None,
            typer_version="fake-v1",
        )


def test_a_permanent_http_refusal_is_stored_and_does_not_stop_the_run(data_root, capsys):
    code = polymarket.run(
        [], transport=FakeTransport([ok({"markets": PM["response"]})]), clock=FakeClock(),
        typer=_ForbiddenTyper(),
    )
    assert code == 0
    report = last_json_line(capsys)["report"]
    assert report["stopped_by_transient"] is False and report["not_stored_transient"] == 0
    stored = MarketStore(data_root / "markets").read("polymarket", "fake-v1")
    assert [(r.market_id, r.typed.status, r.typed.reason) for r in stored] == [
        ("polymarket:0xaaa", "typed", "r"),
        ("polymarket:0xbbb", "refused_invalid", "http 403"),
    ]
    assert json.loads((data_root / "state" / "polymarket.json").read_text())["exhausted"] is True


def _typed_with(status: str, reason: str) -> TypedText:
    return TypedText(
        status=status, key=None, reason=reason, evidence_span=None, typer_version="fake-v1"
    )


@pytest.mark.parametrize(
    ("reason", "transient"),
    [
        ("http 0", True),
        ("http 408", True),
        ("http 429", True),
        ("http 503", True),
        ("http 400", False),
        ("http 401", False),
        ("http 403", False),
        ("http 404", False),
        ("http 422", False),
        ("http", True),
        ("http abc", True),
        ("max_calls reached", True),
        ("schema: reply is not an object", False),
    ],
)
def test_is_transient_outcome_by_refusal_reason(reason, transient):
    assert is_transient_outcome(_typed_with("refused_invalid", reason)) is transient


def test_is_transient_outcome_for_dry_run_and_ordinary_answers():
    assert is_transient_outcome(_typed_with("untyped", "dry-run")) is True
    assert is_transient_outcome(_typed_with("untyped", "not in mapping")) is False
