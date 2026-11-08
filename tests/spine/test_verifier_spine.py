"""Held-back adversarial tests written by the Task 21 verifier (SYNTHETIC data only).

None of these were written by the implementers of Tasks 1-16. A test marked
`xfail(strict=True, reason="FINDING V-n: ...")` pins a defect found by the verifier: it stays in
place so the suite is green and the finding cannot be fixed without removing the marker.
"""

import json
import socket
import time
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from fakes import FakeClock, FakeTransport, make_event_record, ok, ok_text, status
from fixtures.gdelt_rows import frame_of, gdelt_row

from precedent import cli
from precedent.spine.event_key import EventKey
from precedent.spine.event_store import EventStore
from precedent.spine.gdelt_events import (
    GdeltParams,
    dedupe_gdelt_records,
    events_from_gdelt_rows,
    load_cameo_map,
)
from precedent.spine.http import HttpError, HttpRequest, RateLimiter, send_with_retries
from precedent.spine.kalshi_client import KalshiClient
from precedent.spine.market_record import MarketIngestReport
from precedent.spine.pit_alignment import assign_effective_session, build_session_cutoffs
from precedent.spine.polymarket_client import polymarket_market_from
from precedent.spine.price_import import PriceImportError, import_prices_csv
from precedent.spine.typers.rules import RulesTyper
from precedent.spine.typers.typing_response import to_typed_text

CAMEO = load_cameo_map()
PARAMS = GdeltParams()
INGESTED = datetime(2025, 1, 1, tzinfo=UTC)


def gdelt_sessions(*rows: list[str]) -> list[date | None]:
    records, _ = events_from_gdelt_rows(frame_of(*rows), CAMEO, PARAMS, ingested_utc=INGESTED)
    return [r.effective_session for r in records]


# --- 1. GDELT rows on weekends, holidays and the half-day boundary -------------------------------


def test_v_gdelt_weekend_and_holiday_rows_roll_to_the_next_open_session():
    # One row per case, each with a distinct id; the sessions differ so none is a duplicate.
    cases = {
        "20240106100000": date(2024, 1, 8),  # Saturday 10:00 UTC -> Monday
        "20240115150000": date(2024, 1, 16),  # MLK Day (closed) -> Tuesday
        "20240704150000": date(2024, 7, 5),  # Independence Day (closed) -> Friday
        "20241128150000": date(2024, 11, 29),  # Thanksgiving (closed) -> the half-day Friday
    }
    rows = [
        gdelt_row(GLOBALEVENTID=str(2000 + i), DATEADDED=stamp)
        for i, stamp in enumerate(cases)
    ]
    assert gdelt_sessions(*rows) == list(cases.values())


def test_v_gdelt_half_day_boundary_is_inclusive_to_the_second():
    # 17:30:00 UTC + 30 min latency == 18:00:00 UTC == 13:00 New York on 2024-11-29 exactly.
    on_the_cutoff = gdelt_row(GLOBALEVENTID="3001", DATEADDED="20241129173000")
    one_second_late = gdelt_row(GLOBALEVENTID="3002", DATEADDED="20241129173001")
    assert gdelt_sessions(on_the_cutoff, one_second_late) == [date(2024, 11, 29), date(2024, 12, 2)]


# --- 2. Alignment: available_utc equal to a cutoff lands on that session -------------------------


def test_v_available_exactly_at_a_cutoff_lands_on_that_session_not_the_next():
    sessions = pd.DatetimeIndex(["2024-03-04", "2024-03-05", "2024-03-06"])
    cutoffs = build_session_cutoffs(sessions, pd.Timestamp("2024-03-01 16:00").time())
    exact = cutoffs.iloc[1]
    available = pd.Series(
        [exact - pd.Timedelta(nanoseconds=1), exact, exact + pd.Timedelta(nanoseconds=1)]
    )
    assigned = assign_effective_session(available, cutoffs)
    assert list(assigned) == [sessions[1], sessions[1], sessions[2]]


# --- 3. Polymarket: reversed outcome order and exactness of the prices ---------------------------


def raw_polymarket(**overrides) -> dict:
    raw = {
        "conditionId": "0xverifier",
        "question": "Will the verifier find a defect?",
        "closed": True,
        "umaResolutionStatus": "resolved",
        "outcomes": '["No", "Yes"]',
        "outcomePrices": '["1", "0"]',
        "createdAt": "2024-01-01T00:00:00Z",
        "closedTime": "2024-02-01 12:00:00+00",
        "endDate": "2024-02-01T00:00:00Z",
        "slug": "verifier",
    }
    raw.update(overrides)
    return raw


@pytest.mark.parametrize(
    ("outcomes", "prices", "winner"),
    [
        ('["No", "Yes"]', '["1", "0"]', "no"),  # reversed order, first outcome wins
        ('["No", "Yes"]', '["0", "1"]', "yes"),
        ('["Yes", "No"]', '["1.0", "0.0"]', "yes"),
        ('["Yes", "No"]', '["0.000", "1.000"]', "no"),
        (["No", "Yes"], [1, 0], "no"),  # already-decoded numeric lists
    ],
)
def test_v_polymarket_winner_follows_the_label_whatever_the_order(outcomes, prices, winner):
    report = MarketIngestReport()
    market = polymarket_market_from(
        raw_polymarket(outcomes=outcomes, outcomePrices=prices), report
    )
    assert market is not None and market.resolution == winner
    assert report.ambiguous == 0


@pytest.mark.parametrize("prices", ['["0.9999999", "0.0000001"]', '["1", "0.0001"]', '["NaN", "1"]'])
def test_v_polymarket_near_certain_prices_are_ambiguous_never_a_winner(prices):
    report = MarketIngestReport()
    assert polymarket_market_from(raw_polymarket(outcomePrices=prices), report) is None
    assert report.ambiguous == 1 and report.kept == 0


# --- 4. Typer evidence span: whitespace tolerated, case refused ----------------------------------

VALID_REPLY = {
    "event_type": "central_bank",
    "actor_kind": "central_bank",
    "impacted_kinds": ["country"],
    "country_group": "US",
    "severity_band": "high",
    "is_scheduled": False,
    "direction_sign": 1,
    "evidence_span": "cuts rates",
}


def test_v_evidence_span_may_differ_by_whitespace_but_not_by_case_or_a_dropped_word():
    text = "Fed unexpectedly cuts \t\n rates   in emergency move"
    spaced = to_typed_text(json.dumps({**VALID_REPLY, "evidence_span": "cuts  rates"}), text, "v")
    assert spaced.status == "typed"
    padded = {**VALID_REPLY, "evidence_span": "  unexpectedly   cuts rates "}
    assert to_typed_text(json.dumps(padded), text, "v").status == "typed"
    for bad in ("Cuts rates", "CUTS RATES", "cuts the rates", "cutsrates"):
        refused = to_typed_text(json.dumps({**VALID_REPLY, "evidence_span": bad}), text, "v")
        assert refused.status == "refused_invalid" and refused.key is None, bad


def test_v_a_model_reply_with_an_extra_key_or_a_refusal_string_is_refused_not_guessed():
    text = "Fed unexpectedly cuts rates"
    extra = json.dumps({**VALID_REPLY, "confidence": "high"})
    assert to_typed_text(extra, text, "v").status == "refused_invalid"
    for reply in ("I cannot help with that.", "", "null", "{}", '{"event_type": null}'):
        result = to_typed_text(reply, text, "v")
        assert result.status == "refused_invalid" and result.key is None


# --- 5. Kalshi: a market settled exactly at the historical cutoff is taken once ------------------

KALSHI = "https://api.elections.kalshi.com/trade-api/v2"


def kalshi_market(ticker: str, settled: str, result: str = "yes") -> dict:
    return {
        "ticker": ticker,
        "event_ticker": "EV-VERIFY",
        "title": f"Will {ticker} happen?",
        "result": result,
        "settlement_ts": settled,
        "created_time": "2025-01-01T00:00:00Z",
        "close_time": "2025-01-02T00:00:00Z",
    }


def test_v_kalshi_market_settled_exactly_at_the_cutoff_is_taken_once(tmp_path: Path):
    cutoff = "2026-09-08T00:00:00Z"
    at_cutoff = kalshi_market("AT-CUTOFF", cutoff)
    transport = FakeTransport(
        {
            f"{KALSHI}/historical/cutoff": ok({"market_settled_ts": cutoff}),
            f"{KALSHI}/markets": ok(
                {
                    "markets": [
                        at_cutoff,
                        kalshi_market("BEFORE", "2026-09-07T23:59:59Z"),
                        kalshi_market("AFTER", "2026-09-08T00:00:01Z", "no"),
                    ],
                    "cursor": "",
                }
            ),
            # the historical endpoint also serves the boundary market and the early one
            f"{KALSHI}/historical/markets": ok(
                {"markets": [at_cutoff, kalshi_market("BEFORE", "2026-09-07T23:59:59Z")], "cursor": ""}
            ),
            f"{KALSHI}/events/EV-VERIFY": status(404),
        }
    )
    report = MarketIngestReport()
    markets = KalshiClient(
        transport, FakeClock(), tmp_path / "kalshi.json", max_requests=20
    ).fetch_settled(report)
    ids = sorted(m.market_id for m in markets)
    assert ids == ["kalshi:AFTER", "kalshi:AT-CUTOFF", "kalshi:BEFORE"]
    assert len(ids) == len(set(ids))
    assert all(m.category is None for m in markets)  # a 404 event means unknown, not an error


# --- 6. Price CSV: a UTF-8 BOM, CRLF and re-import ------------------------------------------------


def test_v_price_csv_with_a_bom_and_crlf_imports_cleanly(tmp_path: Path):
    csv_path = tmp_path / "bom.csv"
    csv_path.write_bytes(
        b"\xef\xbb\xbfdate,symbol,open,close\r\n2024-01-02,aaa,10,11\r\n2024-01-03,AAA,11,12\r\n"
    )
    report = import_prices_csv(csv_path, tmp_path / "prices")
    assert (report.rows, report.symbols, report.first_date) == (2, 1, "2024-01-02")
    frame = pd.read_parquet(tmp_path / "prices" / "AAA.parquet")
    assert list(frame["close"]) == [11.0, 12.0]


@pytest.mark.parametrize(
    "price", ["nan", "inf", "-1", "0", "1e999", "", "10,5"], ids=lambda p: repr(p)
)
def test_v_price_csv_refuses_non_finite_non_positive_or_misquoted_prices(tmp_path: Path, price):
    csv_path = tmp_path / "bad.csv"
    csv_path.write_text(f"date,symbol,open,close\n2024-01-02,AAA,{price},11\n", encoding="utf-8")
    with pytest.raises(PriceImportError):
        import_prices_csv(csv_path, tmp_path / "prices")
    assert not (tmp_path / "prices" / "AAA.parquet").exists()  # nothing written on refusal


def test_v_price_reimport_of_a_symbol_replaces_it_as_the_plan_states(tmp_path: Path):
    prices = tmp_path / "prices"
    first = tmp_path / "a.csv"
    first.write_text("date,symbol,open,close\n2024-01-02,AAA,10,11\n", encoding="utf-8")
    second = tmp_path / "b.csv"
    second.write_text("date,symbol,open,close\n2024-02-01,AAA,20,21\n", encoding="utf-8")
    import_prices_csv(first, prices)
    import_prices_csv(second, prices)
    frame = pd.read_parquet(prices / "AAA.parquet")
    assert list(frame["close"]) == [21.0]
    manifest = json.loads((prices / "MANIFEST.json").read_text())
    assert manifest["symbols"]["AAA"]["first_date"] == "2024-02-01"


# --- 7. EventStore: one article under two parser versions ----------------------------------------


def test_v_the_same_article_under_two_parser_versions_keeps_both(tmp_path: Path):
    store = EventStore(tmp_path)
    one = make_event_record(article_id="gdelt:77", parser_version="rules-gdelt-v1")
    two = make_event_record(article_id="gdelt:77", parser_version="rules-gdelt-v2")
    assert store.append([one, two]).added == 2
    assert [r.parser_version for r in store.read("rules-gdelt-v1")] == ["rules-gdelt-v1"]
    assert [r.parser_version for r in store.read("rules-gdelt-v2")] == ["rules-gdelt-v2"]
    again = store.append([one, two])
    assert (again.added, again.identical_skipped) == (0, 2)


def test_v_two_parser_versions_that_share_a_file_slug_are_refused(tmp_path: Path):
    store = EventStore(tmp_path)
    left = make_event_record(parser_version="a/b")
    right = make_event_record(parser_version="a_b", article_id="gdelt:2")
    with pytest.raises(ValueError, match="slug"):
        store.append([left, right])
    assert list(tmp_path.glob("events_*")) == []  # all or nothing


# --- 8. Rules typer: degenerate and hostile input ------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["", "   \n\t ", "x" * 10_000, " " * 10_000, "a " * 5_000, "\x00" * 100, "‮" + "ï" * 5_000],
    ids=["empty", "blank", "10kB-x", "10kB-space", "10kB-words", "nul", "bidi"],
)
def test_v_rules_typer_returns_untyped_on_degenerate_text_without_raising(text):
    started = time.perf_counter()
    result = RulesTyper().type_text(text)
    assert time.perf_counter() - started < 5.0  # no catastrophic backtracking
    assert result.status == "untyped" and result.key is None and result.evidence_span is None


# --- 9. HTTP seam: rate limiter and retries ------------------------------------------------------


class BoundedClock(FakeClock):
    """A FakeClock that fails the test instead of looping forever."""

    def sleep(self, seconds: float) -> None:
        assert seconds > 0, "a zero or negative sleep would spin"
        super().sleep(seconds)
        assert self.slept_seconds < 1_000, "rate limiter did not terminate"


def test_v_rate_limiter_with_a_fractional_window_terminates_and_spaces_requests():
    clock = BoundedClock()
    limiter = RateLimiter(1, 1 / 3, clock)
    stamps = []
    for _ in range(30):
        limiter.acquire()
        stamps.append(clock.monotonic())
    gaps = [b - a for a, b in zip(stamps, stamps[1:], strict=False)]
    assert min(gaps) >= (1 / 3) - 1e-9
    assert clock.slept_seconds == pytest.approx(29 / 3, abs=1e-6)


def test_v_rate_limiter_refuses_a_zero_budget_or_window():
    for max_requests, window in ((0, 1.0), (1, 0.0), (1, -1.0)):
        with pytest.raises(ValueError):
            RateLimiter(max_requests, window, FakeClock())


def get_request() -> HttpRequest:
    return HttpRequest("GET", "https://example.test/x", headers={})


def test_v_send_with_retries_never_sleeps_on_success_or_a_permanent_error():
    for success in (ok({"a": 1}), ok_text("x")):
        clock = FakeClock()
        send_with_retries(FakeTransport([success]), clock, get_request())
        assert clock.slept_seconds == 0.0
    for permanent in (400, 401, 403, 404, 410):
        clock = FakeClock()
        with pytest.raises(HttpError) as raised:
            send_with_retries(FakeTransport([status(permanent)]), clock, get_request())
        assert raised.value.status == permanent and clock.slept_seconds == 0.0


def test_v_send_with_retries_is_bounded_and_sleeps_once_per_retry():
    clock = FakeClock()
    transport = FakeTransport([status(503)] * 4)
    with pytest.raises(HttpError):
        send_with_retries(transport, clock, get_request(), max_retries=3)
    assert len(transport.requests) == 4  # one try plus three retries, no more
    recovered = FakeClock()
    transport = FakeTransport([status(429), status(500), ok({"fine": True})])
    assert send_with_retries(transport, recovered, get_request()).json() == {"fine": True}
    assert recovered.slept_seconds > 0 and len(transport.requests) == 3


# --- 10. CLI routing and the typer dry run --------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["typers", "score"], ("typers", ["score"])),
        (["prices", "import", "x.csv"], ("prices", ["import", "x.csv"])),
        (["mcp"], ("mcp", [])),
        (["ask"], ("ask", [])),
        (["ask", "ask"], ("ask", ["ask"])),
        (["ingest"], ("ingest", [])),
        (["ingest", "gdelt", "--max-files", "4"], ("ingest gdelt", ["--max-files", "4"])),
        (["ingest", "nowhere", "x"], ("ingest", ["nowhere", "x"])),
        (["Fed", "cuts", "rates"], ("ask", ["Fed", "cuts", "rates"])),
        (["Typers rule the world"], ("ask", ["Typers rule the world"])),  # case differs: text
        ([], ("ask", [])),
    ],
)
def test_v_the_ask_sugar_only_fires_when_the_first_word_is_not_a_subcommand(argv, expected):
    assert cli.resolve_command(argv) == expected


def test_v_event_key_with_the_wrong_number_of_fields_is_a_value_error():
    for text in (
        "a/b/c/US/high/scheduled",  # 6 fields
        "a/b/c/US/high/scheduled/1/extra",  # 8 fields
        "a/b/c/US/high/scheduled/1/",  # trailing slash
        "",
        "/////",
    ):
        with pytest.raises(ValueError):
            EventKey.from_string(text)
    ok_key = EventKey.from_string("central_bank/central_bank/none/none/low/scheduled/0")
    assert ok_key.impacted_kinds == ()


def test_v_a_model_typer_dry_run_through_the_cli_opens_no_socket(tmp_path, monkeypatch):
    from precedent.spine.typers import cli as typers_cli

    def refuse(*_args, **_kwargs):
        raise AssertionError("a socket was opened during --dry-run")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-test-value")
    out = tmp_path / "score.md"
    code = typers_cli.run(
        ["score", "--typer", "anthropic", "--max-calls", "3", "--dry-run", "--out", str(out)]
    )
    assert code == 0
    assert out.read_text(encoding="utf-8").startswith("NO LIVE RUN —")


def test_v_a_model_typer_without_max_calls_is_refused_before_any_call(monkeypatch):
    from precedent.spine.typers import cli as typers_cli

    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-test-value")
    assert typers_cli.run(["score", "--typer", "anthropic"]) == 2


# --- 11. GDELT dedup: the in-run rule and the cross-run helper must agree ------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING V-2: dedupe_gdelt_records keys on linked_countries[0], which is Actor2's country "
        "when Actor1 has no country; events_from_gdelt_rows keys on Actor1CountryCode. The same "
        "two rows are merged in one conversion but kept apart across runs."
    ),
)
def test_v_gdelt_dedup_across_runs_matches_dedup_inside_one_run():
    # Actor1 has no country; the two rows differ only in Actor2's country and event id.
    def row(event_id: str, actor2_country: str) -> list[str]:
        return gdelt_row(
            GLOBALEVENTID=event_id,
            Actor1CountryCode="",
            Actor1Name="A MINISTER",
            Actor1Type1Code="GOV",
            Actor2CountryCode=actor2_country,
            Actor2Name="SOMEONE",
        )

    one_run, _ = events_from_gdelt_rows(
        frame_of(row("5001", "RUS"), row("5002", "CHN")), CAMEO, PARAMS, ingested_utc=INGESTED
    )
    separate_runs = []
    for event_id, country in (("5001", "RUS"), ("5002", "CHN")):
        records, _ = events_from_gdelt_rows(
            frame_of(row(event_id, country)), CAMEO, PARAMS, ingested_utc=INGESTED
        )
        separate_runs += records
    assert len(dedupe_gdelt_records(separate_runs)) == len(one_run)


# --- 12. Further probes pinned as findings --------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING V-3: check_known_sessions refuses holidays but not weekends, so a Saturday "
        "session (for example from a price CSV with a weekend row) gets a cutoff and can receive "
        "events that belong to Monday."
    ),
)
def test_v_a_weekend_session_is_refused_like_a_holiday():
    from precedent.spine.session_calendar import calendar_cutoffs, load_nyse_calendar

    with pytest.raises(ValueError):
        calendar_cutoffs(pd.DatetimeIndex(["2024-01-05", "2024-01-06"]), load_nyse_calendar())


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING V-4: edgar_8k_events._default_cutoffs does not cap its 15-day lookahead at the "
        "calendar's last year (gdelt_events does), so an 8-K accepted in the last two weeks of "
        "2027 raises instead of aligning or staying unaligned."
    ),
)
def test_v_an_8k_accepted_in_the_last_days_of_the_calendar_still_aligns():
    from precedent.spine.edgar_8k_events import events_from_submissions

    submissions = {
        "name": "Acme",
        "addresses": {"business": {"stateOrCountry": "NY"}},
        "filings": {
            "recent": {
                "accessionNumber": ["0001-27-000001"],
                "acceptanceDateTime": ["2027-12-20T21:05:00.000Z"],
                "form": ["8-K"],
                "items": ["2.02"],
            }
        },
    }
    records, _ = events_from_submissions(
        submissions, "ACME", ingested_utc=datetime(2027, 12, 21, tzinfo=UTC)
    )
    assert len(records) == 1


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING V-5: EventKey.from_string validates the field count, group, band, flag and sign "
        "but not event_type, actor_kind or impacted kinds, so a typo in a gold case or a stored "
        "key parses as a valid key."
    ),
)
def test_v_an_event_key_with_an_unknown_event_type_is_refused():
    with pytest.raises(ValueError):
        EventKey.from_string("garbage/garbage/foo+bar/US/low/scheduled/0")


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING V-6: `precedent --help` is treated as text for `ask` and exits 2 with "
        "'not built yet: ask'; there is no top-level usage."
    ),
)
def test_v_top_level_help_is_not_routed_to_ask(capsys):
    cli.main(["--help"])
    assert "not built yet" not in capsys.readouterr().err


@pytest.mark.xfail(
    strict=True,
    reason=(
        "FINDING V-7: price import accepts any ISO-8601 date form Python parses (20240102, "
        "2024-W01-1) although the contract and its own error message say YYYY-MM-DD."
    ),
)
def test_v_price_csv_date_must_be_yyyy_mm_dd(tmp_path: Path):
    csv_path = tmp_path / "compact.csv"
    csv_path.write_text("date,symbol,open,close\n20240102,AAA,10,11\n", encoding="utf-8")
    with pytest.raises(PriceImportError):
        import_prices_csv(csv_path, tmp_path / "prices")
