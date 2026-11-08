# Precedent developer manual

**Written:** 2026-11-07, from the code in this checkout and the documents under `docs/`.
**For:** the owner, as a developer new to this codebase and to event-study ideas.
**Status of what it describes:** the spine (spine plan Tasks 1 to 16) is built and tested on synthetic data only. Nothing has run end to end on live data or with a live model. The Precedent product track (libraries, card, `ask`, snapshot, MCP server) and the Dream, Ripple and Mechanism tracks are **planned**, not built. Every section below says which it is.

Where this manual gives a market fact, it comes from the product thesis (`docs/2026-11-07-product-thesis.md`) and is attributed to it. It contains no benchmark or result that the repository has not produced.

---

## 1. What Precedent is

Precedent answers one question: **"The last N times something like this happened, what happened next?"** You give it an event, as a headline or as a typed key. It finds similar events that were known strictly earlier, looks up what followed each one, and prints a card with the count, the spread and the sources. When it has too few independent precedents, it refuses instead of guessing.

It reports three kinds of "what happened next" (amendment section 4):

| Outcome type | What the number is | Data |
|---|---|---|
| `event_sequence` | Did a typed follow-on event (for example sanctions) happen within *h* sessions? A rate. | GDELT 2.0, free |
| `market_resolution` | How did prediction-market questions open at the time resolve? A YES rate. | Polymarket, Kalshi, free |
| `asset` | The abnormal return of an ETF or stock over 1, 5 or 20 sessions. A median and a band. | Prices the user brings |

The card (amendment section 3; the layout is fixed, the values below are the amendment's illustration, not a result):

```text
PRECEDENT  ·  <event in plain words, as typed>
typed as       <event_type> / <actor_kind> / <country_group> / severity <band> / <scheduled|unscheduled>
precedents     n=<n>  independent=<n_eff>  earliest <date>  latest <date>

<outcome type>                      horizon   rate|median   80% band        n   vs ordinary
<outcome key, e.g. trade_or_sanctions by G7>   20        14/23 (61%)   [43%, 77%]     23  outside
<outcome key, e.g. SPY abnormal return>        5         -0.8%         [-3.1%, +1.4%] 19  inside

nearest precedents
<date>  <one line>  similarity <s>  <source id>
library covers  <source>: <first date> to <last date>, <N> events; ...
These are past outcomes of similar events, not a forecast.
```

(The illustrative band `[43%, 77%]` is not what the planned Wilson code gives for 14 of 23; the plan's test pins about `[48%, 73%]`. Trust the plan's tests, not the illustration.)

The refusal, which the thesis picks as the README's first screenshot:

```text
REFUSED        <n> usable precedents, need <min_neff>. No band is drawn below that.
nearest precedents
<date>  <one line>  ...
library covers  ...
These are past outcomes of similar events, not a forecast.
```

**Who it is for** (thesis sections 1 and 4): forecasters and prediction-market users who want base rates, readers of a daily brief, and developers who call it from an AI assistant through the MCP server.

**What it never does.** It never forecasts, never gives advice, never places orders (design 1.3, A1). The one rule in `CLAUDE.md`: the language model reads and emits structure; it never predicts. Numbers come from price and resolution history, computed by deterministic code.

---

## 2. What is new about it, concretely

Each property below is a design choice with a reason. "What most tools do" is the alternative the design rejects.

**The time guard, per window.** A precedent may serve a query only if its outcome window had *closed* by the query's `available_utc` (design 8.3). A 1-session response becomes usable long before a 20-session one. *Why:* a precedent whose outcome was still unfolding is future information. *Usual alternative:* filter precedents by event date only, which lets a still-open 20-day window leak. *Where:* `response_complete_session` in the planned row schema; `find_precedents` in Precedent Task 7 (planned). The spine already provides the clocks it needs.

**Honest refusal below an effective sample size.** Twelve analogs from one crisis week are not twelve independent observations. The planned `effective_sample_size` takes the smaller of the Kish size and the count of non-overlapping windows; below `min_neff` (8, proposed) the card prints `REFUSED` and no band. *Usual alternative:* show a confident number from whatever matched. *Where:* Precedent Task 8 (planned).

**"Library covers" against selection bias.** Only reported events enter any library. The card must say which sources, dates and counts it drew on (thesis section 7). *Usual alternative:* a bare percentage with no denominator of what was searched.

**Three outcome types on one engine.** All three libraries produce rows of one shape, so retrieval, the time guard, aggregation and refusal are written and tested once (amendment section 4).

**Rules first; the model is optional and must earn its place.** Every source is typed by committed rules (CAMEO tables for GDELT, item tables for 8-K, a keyword table for free text). A model typer may replace the rules only if it beats them on the held-out gold split (design 8.4, `stages/03_parse/CONTEXT.md`). *Usual alternative:* an LLM in the loop from day one with no baseline to compare against. *Where:* `src/precedent/spine/typers/`.

**Code decides; the model proposes.** Model output is validated against a closed schema (`extra="forbid"`), and its quoted `evidence_span` must appear in the input text, or the answer is `refused_invalid`. Time, instrument identity and every number come from code. *Where:* `spine/typers/typing_response.py`, `spine/event_parser.py`.

**Local-first with free data.** EDGAR, GDELT, Polymarket and Kalshi are free; prices are the user's own CSV; nothing in the repository displays vendor prices. Unit tests never touch the network: every client takes an injected `Transport` and `Clock` (`spine/http.py`).

**Evidence banners and an independent verifier.** Every result carries one label (`SYNTHETIC`, `LIVE-DATA`, `LIVE-MODEL`, `NO LIVE RUN`) in `docs/STATUS.md`, and every run-produced file under `docs/examples/` starts with a banner; a test enforces it. Each track ends with a verifier that did not write the code.

---

## 3. The map

Four tracks share one spine (amendment A1, thesis section 3):

| Track | Role | Package (planned) | Status |
|---|---|---|---|
| Spine | Ingest, type, align, price, leak-test | `precedent.spine` | Built (Tasks 1 to 16) |
| Precedent | The product: libraries, engine, card, MCP | `precedent.libraries`, `precedent.engine`, `precedent.mcp_server` | Planned |
| Dream | Overnight contradiction diff; content engine for the brief | `precedent.dream` | Planned |
| Ripple | Shocks along a dated company-link graph; gated on a price licence | `precedent.ripple` | Planned |
| Mechanism | Small causal graphs and what-if rollouts; research and content only | `precedent.mechanism` | Planned |

Data flow from sources to the card:

```text
 SOURCES                    SPINE (built)                                   PRODUCT (planned)
 GDELT 2.0 15-min files ──► gdelt_fetch ─► gdelt_events (CAMEO rules) ─┐
 SEC EDGAR submissions ───► edgar_client ─► edgar_8k_events (item rules)├─► EventStore (JSONL,
                                                                        │   per parser_version)
                              pit_alignment + session_calendar ─────────┘        │
                              (available_utc, effective_session)                 │
 Polymarket Gamma ────────► polymarket_client ─┐                                 ├─► libraries:
 Kalshi trade API ────────► kalshi_client ─────┴► typers ─► MarketStore ─────────┤   event_sequence
                                                 (rules | anthropic | ollama)    │   market_resolution
 your prices CSV ─────────► price_import ─► ParquetPriceProvider                 │   asset
                              └► abnormal_returns (market model, CAR) ───────────┘        │
                                                                                          ▼
 headline ─► typer ─► EventKey ───────────────────────────► retrieval (time guard) ─► aggregate ─► card
                                                                                   (text, JSON, PNG, MCP)
 research side: news_factors, macro_factors ─► qlib_dataset ─► ablation, leak_checks ─► backtest
                (the last four need the research extra)
```

---

## 4. The spine, part by part

All paths are under `src/precedent/spine/` unless stated.

### 4.1 Event schema and closed enums — `event_schema.py`

`NewsEvent` is everything a parser may say about an article: `event_type` (21 values, from `earnings` to `no_market_event`), `actor` (`name`, `kind`, `country_iso3`), `action`, up to eight `impacted` entries each with `direction_step` in {-2..2}, `severity` 1..10, `macro_direction_step`, `confidence`, `horizon`, `novelty` (new, update, recap, opinion), `is_scheduled` and `evidence_span`.

**Why closed enums:** constrained decoders disagree about numeric bounds but all enforce enums (design 2, 3.3). Five direction steps are as much resolution as the signal has. `ticker_hint` is only a hint: tickers are resolved by code, never trusted from a model. The schema has no outcome field at all, so a model cannot be asked to remember the future. `references/event_schema_v1.json` is the exported JSON Schema.

### 4.2 EventRecord and the four clocks — `event_record.py`, `pit_alignment.py`

`EventRecord` wraps a `NewsEvent` in a code-written envelope: `article_id`, `body_sha256`, `source`, four time fields, `effective_session`, `parser_version`, `resolved_impacts`, `linked_countries`.

| Clock | Meaning | Used to align? |
|---|---|---|
| event time | when it happened (GDELT `SQLDATE`) | never |
| `published_utc` | the publisher's claim; may be missing or wrong | lower bound only |
| `first_seen_utc` | earliest moment we could have had it: GDELT `DATEADDED`, EDGAR `acceptanceDateTime`, or our fetch time live | yes; required |
| `ingested_utc` | when our row was written | audit only |

The rule is `available_utc = max(published_utc, first_seen_utc) + latency` (`compute_available_utc`). Taking the max means a publisher clock can only make an item later, never earlier. `ingested_utc` is excluded because a 2019 article backfilled today has a 2026 ingest time. Latency covers fetch, parse and decision time: 30 minutes for GDELT (`references/precedent_params.json`), 5 minutes for EDGAR (`DEFAULT_LATENCY` in `edgar_8k_events.py`), both proposed. A negative latency, a naive timestamp or a missing `first_seen_utc` raises.

**Look-ahead bias** is using information in a decision that was not available at that moment. **Point-in-time** data records what was knowable when. These two functions are where the spine enforces it.

### 4.3 Sessions, cutoffs and the NYSE calendar — `session_calendar.py`, `pit_alignment.py`

A **session** is one NYSE trading day. Each session has one decision **cutoff**: 16:00 New York, or the early close on a half-day. An event belongs to the first session whose cutoff is at or after its `available_utc` (`assign_effective_session`, a `searchsorted`). Weekend and holiday news rolls forward to the next session.

Worked example, from `tests/spine/test_pit_alignment.py` and the spine plan: an EDGAR item first seen Friday 2024-01-05 at 15:59 New York (20:59 UTC). With zero latency it lands on Friday. With the 5-minute latency, `available_utc` is 21:04 UTC, past Friday's 21:00 UTC cutoff, so `effective_session` is Monday 2024-01-08. One second after 16:00 also rolls to Monday.

Early close: on 2024-11-29 the cutoff is 13:00 New York (18:00 UTC), so news at 14:00 rolls to Monday 2024-12-02 instead of landing on the half-day (`test_news_after_a_half_day_close_rolls_to_monday`).

`references/nyse_holidays.json` covers 2010 to 2027. `calendar_cutoffs` refuses a session outside those years or on a holiday, so an unknown day never silently gets a 16:00 cutoff. Cutoffs are built per date in New York time and converted to UTC, so daylight-saving changes are handled. `resolve_sessions` prefers the price provider's sessions when there are any, and checks them against the calendar.

**Gotcha:** the file itself says the 2010 and 2011 holidays are rule-derived and unverified against an NYSE source, and that the July 3 and December 24 early closes in 2012 and 2013 are secondary-sourced only. Everything else, including the 2012, 2018 and 2025 special closures, is verified (checked 2026-11-08; see the file's `sources`).

### 4.4 The event key — `event_key.py`

`EventKey` is seven discrete fields: `event_type`, `actor_kind`, `impacted_kinds` (sorted tuple), `country_group` (`US`, `DM`, `EM`, `none`, from `references/country_groups.json`), `severity_band` (1-3 low, 4-6 mid, 7-10 high), `is_scheduled`, `direction_sign` (-1, 0, 1).

String form, which the stores and libraries index: `central_bank/central_bank/currency+index/US/high/unscheduled/-1`. An empty kinds tuple is written `none`. `EventKey.from_string` validates every part and raises on anything unknown.

`build_event_key(record)` uses only the parsed event, never time fields. Direction is the sign of `macro_direction_step`, or, when that is zero, the sign of the summed impact steps. **Why discrete:** precedents are found by exact key match first (baseline B1); similarity by embedding is a later, separately measured rung.

### 4.5 The event store — `event_store.py`, `jsonl_io.py`

One JSON Lines file per parser version: `$PRECEDENT_DATA_DIR/events/events_<slug>.jsonl`. Records are keyed by (`article_id`, `parser_version`).

- **Append-only.** An identical record is skipped; the same key with different content raises `EventConflictError` (the `ingested_utc` field is ignored in that comparison). An append is all or nothing.
- **Vintages.** Re-parsing with a new rule table or model writes a new `parser_version` file beside the old one. A backtest pins one version. Two versions whose slugs collide are refused.
- **Atomic writes.** `atomic_write_lines` writes a temporary sibling and renames it, so a crash never leaves half a file.
- `to_frame` returns one row per resolved impact plus a `FrameReport` counting unaligned records and unresolved impacts. Nothing is dropped silently.

### 4.6 GDELT 2.0 — `gdelt_fetch.py`, `gdelt_events.py`, `ingest/gdelt.py`

**What it is.** The GDELT Project machine-codes world news into events: who did what to whom, with a **CAMEO** code (a standard taxonomy of political actions: root codes 01 to 20, such as 14 PROTEST or 19 FIGHT). GDELT 2.0 publishes a file every 15 minutes from 2015-02-19. It holds no article text.

**Clocks.** `DATEADDED` (column 60, UTC) is when GDELT first saw the event; it becomes `first_seen_utc`. `SQLDATE` (column 2) is the event's own date and is never read for time. A row whose `SQLDATE` is days before its `DATEADDED` is aligned by `DATEADDED`.

**Fetcher.** `GdeltFetcher` reads the master file list, picks files stamped within `--since..--until`, downloads at most `--max-files` per run (default 96, one day), checks each file's MD5 from the list, writes it atomically under `raw/gdelt/`, and records progress in `gdelt_fetch_state.json`, so an interrupted run resumes. The MD5 guards against corruption, not tampering (the list is served over plain HTTP).

**Typing** (`events_from_gdelt_rows`, parser version `rules-gdelt-v1:<12 hex>`, a hash of `cameo_to_event_type.json` and the `gdelt` section of `precedent_params.json`, so editing either starts a new vintage). In order:

1. Drop rows with `NumSources < 3` (proposed, decision B4).
2. Map the CAMEO code through `references/cameo_to_event_type.json`. Roots 14 → `election_or_political`; 15, 18, 19, 20 → `armed_conflict`; 13 → `other`; subcode 163 (impose embargo, boycott or sanctions) with a government Actor1 → `trade_or_sanctions`; everything else `untyped` and dropped. Every row is marked `"proposed": true`; `stages/02_triage/CONTEXT.md` lists where the mapping is coarser than the code (for example protests are not elections).
3. Actor kind from `Actor1Type1Code` (GOV, MIL → government; IGO → international_org; BUS → company; else other). Severity from `|GoldsteinScale|` thirds, bumped one band at `NumSources >= 10`, mapped to 2, 5 or 8. Direction from the sign of `GoldsteinScale`. `is_scheduled` is always false.
4. Align, then dedup: one record per (event key, Actor1 country, effective session), earliest `DATEADDED` winning, ties to the smaller `GLOBALEVENTID`. `dedupe_gdelt_records` applies the same rule across runs.

`GdeltIngestReport` counts every row: `rows_read = kept + dropped_untyped + dropped_min_sources + dropped_duplicate`. GDELT's terms require a citation and link on any use or redistribution (`references/README.md`).

### 4.7 EDGAR 8-K — `edgar_client.py`, `edgar_8k_events.py`, `ingest/edgar.py`

An **8-K** is the form a US public company files to report a material event; each filing lists numbered items (2.02 results, 5.02 officer change, 1.05 cybersecurity incident, and so on). EDGAR's submissions JSON gives each filing's `acceptanceDateTime`, the moment the SEC accepted it, read as UTC (RECALLED until the first live fetch confirms the `Z` suffix; see the stage 01 contract). `filingDate` is never used: filings accepted after 17:30 Eastern get the next business day's date.

`events_from_submissions` (parser version `rules-8k-v1:<12 hex>`, a hash of `item_rules_8k.json`) makes one record per 8-K item that has a rule in `references/item_rules_8k.json` (event type, severity, scheduled flag). Unknown items, other forms and filings without items are counted, not dropped silently. `8-K/A` becomes `novelty = update`. A filer whose business address is a US state code (`references/us_state_codes.json`) links country `USA`. Every symbol on the filer's CIK (share classes) becomes one resolved impact of the same record; `ingest edgar` groups the `cik,symbol` lines by CIK.

**The no-contact gate.** The SEC requires a declared User-Agent with a contact. No contact is written into the repository; `EdgarClient.from_env` reads `PRECEDENT_SEC_USER_AGENT` and raises `ConfigError` when it is unset, so `precedent ingest edgar` exits 2 before any request. HTTP 403 and 429 stop the run. The client rate-limits itself and fetches at most `--max-requests` CIKs (default 50). Label: `NO LIVE RUN`; the item list and state codes are recalled, not confirmed against the SEC (`docs/STATUS.md`). Only the `filings.recent` page is read (about the latest 1,000 filings).

`universe.py` holds the dated index membership (`members_asof`) and `resolve_instrument`, which returns `None` for an unmapped CIK instead of guessing.

### 4.8 Prediction markets — `market_record.py`, `market_store.py`, `polymarket_client.py`, `kalshi_client.py`, `ingest/market_run.py`

`ResolvedMarket` holds `market_id`, `venue`, `question`, `created_utc`, `close_utc`, `resolved_utc`, `resolution` (yes or no), `category`, `source_ref`. `MarketRecord` adds `typed`, the typer's answer for the question. Timestamps without an explicit offset are rejected (`parse_utc_timestamp`).

**Polymarket.** The Gamma API has no winner field. `polymarket_market_from` derives it: the market must be closed, resolved (`umaResolutionStatus == "resolved"` when the key exists), have exactly the outcomes Yes and No, and have `outcomePrices` of exactly one 1 and one 0. Anything else is counted as `ambiguous` and skipped, never assigned a winner. The client pages `/markets/keyset` newest first by cursor, resumable from `state/polymarket.json`.

**Kalshi.** Settled markets newer than the `/historical/cutoff` timestamp come from `/markets?status=settled`; older ones only from `/historical/markets`. The client runs a live phase then a historical phase, and joins each market's category through its event's series.

**Typing the question.** `type_markets` sends every question through a typer. Untyped markets are kept and counted; a typer out of call budget yields `untyped` with reason `max_calls`. `MarketStore` writes `markets/markets_<venue>.jsonl`, keyed by (market id, typer version). Re-typing under the same version with a different answer raises `MarketConflictError` by design. The API shapes were read by unauthenticated probes on 2026-11-07 (`stages/01_ingest/CONTEXT.md`); no ingest result is recorded in `docs/STATUS.md`.

### 4.9 Typers — `typers/`

A **typer** turns free text (a market question or a typed-in headline) into an `EventKey`. The protocol (`typers/base.py`):

```python
class Typer(Protocol):
    version: str
    def type_text(self, text: str) -> TypedText: ...
```

`TypedText.status` is `typed` (with a key and an evidence span), `untyped`, or `refused_invalid` (neither). The validator enforces that combination.

- `RulesTyper` (`typers/rules.py`): keyword and pattern table `references/typing_rules.json`. Each event type scores by include matches, minus excludes; a tie is `untyped` ("ambiguous"). Country, severity and schedule come from their own phrase lists. Its version is `rules-typing-v1:<first 12 hex of the table's SHA-256>`, so editing the table changes the version. No network. The default.
- `AnthropicTyper`, `OllamaTyper` (`typers/anthropic_typer.py`, `typers/ollama_typer.py`, shared flow in `typers/typing_response.py`): send the text with the typing prompt (`references/typing_prompt_v1.txt`) over `httpx`, validate the reply strictly, reject an evidence span not in the text, and return `refused_invalid` on any failure. They honour `--dry-run` (no socket; returns `untyped`) and a hard `max_calls` that counts retries. Missing keys or URLs fail before any network call. **Not yet called against a live API:** the request and reply shapes are unconfirmed until an owner-gated run.
- `typers/select.py`: `typer_by_name`, and `typer_from_env` reading `PRECEDENT_TYPER`.
- `typers/fake.py`: a mapping-backed typer for tests.

**The gold set and the score.** `references/gold/typing_cases.json` holds 60 synthetic cases written for this repository: 40 `dev`, 20 `heldout`. `precedent typers score` scores per field. Here, *precision* is correct answers over cases the typer typed; *recall* is correct answers over cases that should be typed. From `docs/examples/typing_score_rules.md` (`SYNTHETIC`):

| split | typed / should be typed | `event_type` P / R | `country_group` P / R | `direction_sign` P / R |
|---|---|---|---|---|
| dev (40) | 33 / 33 | 1.000 / 1.000 | 1.000 / 1.000 | 1.000 / 1.000 |
| heldout (20) | 10 / 15 | 1.000 / 0.667 | 0.700 / 0.467 | 0.300 / 0.200 |

**Why held-out is worse, honestly:** the rules were written while reading the dev cases, so the dev score measures how well the table fits text it was built from. The held-out split was not looked at, so it is the only fair estimate. On it, the rules typed 10 of the 15 typeable cases and missed 5 phrasings the table has no pattern for; where it did type, the event type was always right, but direction was right in only 3 of 10. Both splits are small (one held-out case moves recall by about 7 points) and synthetic. Status precision of 1.000 means the rules never typed a case that should have stayed untyped. The bar for any model typer is to beat these held-out numbers.

### 4.10 Prices — `price_import.py`, `price_provider.py`, `prices_cli.py`, `qlib_provider.py`

`precedent prices import <csv>` takes columns `date,symbol,open,close`, validates the whole file (positive finite prices, no duplicate symbol-day, ISO dates), then writes one Parquet file per symbol and a `MANIFEST.json` with the source SHA-256 under `$PRECEDENT_DATA_DIR/prices/`, atomically.

`ParquetPriceProvider` implements the `PriceProvider` protocol (`sessions`, `prices`). **Return convention:** a return is indexed by the session on which it became known: `close_to_close[t] = close(t)/close(t-1) - 1`. `load_returns` never bridges a missing price (`pct_change(fill_method=None)`). Forward-looking values exist only in the Qlib label expression. `qlib_provider.py` (research extra) is the Qlib bridge: `init_qlib` always passes `region="us"` and an SQLite experiment store, and `QlibPriceProvider` reads Qlib data through the same protocol.

### 4.11 Abnormal returns — `abnormal_returns.py`

An **abnormal return** is the part of an asset's return the market does not explain. The market model fits `asset = alpha + beta * market` by least squares on the **estimation window**, sessions -250 to -21 relative to the event, ending well before it so the event cannot leak into its own baseline. The **CAR** (cumulative abnormal return) sums `asset - (alpha + beta * market)` over sessions +1 to +horizon, so the event window starts after the decision cutoff.

`cumulative_abnormal_return` returns **NaN, not zero**, when the estimation window has fewer than 120 usable rows, the market has no variance, or the event window runs past the data. Zero would be a claim that nothing happened. `estimate_betas` gives the trailing beta per session and asset; pass that frame (not its last row) to the macro factors.

### 4.12 News and macro factors — `news_factors.py`, `macro_factors.py` (Qlib-facing)

`build_news_factor_panel` turns events into a sparse (session, instrument) panel: same-day columns (`news_cnt_1d`, `news_sev_max_1d`, `news_signed_1d`, `news_novel_share_1d`) and decayed ones (`news_signed_decay`, `news_attention_decay`) with a 3-session half-life and a 20-session age limit. Age is counted in session positions, not calendar days. An event near the end of the calendar is truncated, never extrapolated.

`build_macro_factor_panel` sums decayed macro directions (`macro_direction_decay`) and scales them by each instrument's beta (`macro_x_beta`). A Series of betas is one static beta per instrument; a frame indexed by session supplies the beta at each session (NaN beta gives NaN, never a value filled from another session), so the product stays point-in-time.

**Truncation invariance** is the leak test every factor builder ships with: build from all events, then from only events up to day t; rows up to t must be identical. If they differ, a factor depends on later data. Both builders pass it through `assert_truncation_invariant` in `tests/fakes.py`.

### 4.13 The Qlib dataset and the ablation ladder — `qlib_dataset.py`, `ablation.py`, `purged_splits.py`, `rank_ic.py` (the first two need the research extra)

`make_alpha_handler` joins factor panels onto Qlib's Alpha158 price features with a `NestedDataLoader`, the **price loader first and `join="left"`**. The price index wins: a factor row for an instrument with no price row that day (a halt, not yet listed) is dropped and counted as `events_without_price_row`. `FillnaByPrefix` zero-fills **only** factor columns (`news_`, `prec_`, `ripple_`, `macro_`); price features keep their NaNs, because a blanket fill would turn missing prices into zeros. The label is `Ref($open, -2)/Ref($open, -1) - 1` (convention C1, below).

**Purged walk-forward** (`purged_walk_forward_segments`): training, validation and test windows step forward in time, separated by a gap of `label_horizon + embargo` sessions, because the last training rows' labels look into the next window. Qlib's own `RollingGen` only gaps before the test segment. `ablation.py` derives the horizon from the label expression and checks every gap.

**Rank IC** (`rank_ic.py`): each day, the Spearman correlation between predicted scores and realized labels across instruments (days with fewer than three names are dropped); `ic_summary` gives mean, ICIR (mean over std), hit rate.

`run_ablation` runs every rung on every fold, writes `ablation.csv` and a `run_manifest.json` with panel hashes, rung-file hashes and the git commit, and counts every trial. Rungs are YAML files in `references/rungs/`; today there are `m0_price.yaml` (no factors) and `m1_news.yaml` (`news_` factors). Tracks add rungs by adding files.

**The positive control.** `tests/spine/test_leak_checks.py::test_positive_control_detects_a_planted_lead` builds a synthetic market where the label is driven by the factor dated one session later. The aligned panel must show rank IC near zero; the panel shifted one session earlier must show rank IC above 0.2. A leak detector that cannot find a planted leak proves nothing.

### 4.14 Leak checks — `leak_checks.py` (research side; the positive-control test needs the research extra)

- `shift_panel(panel, sessions, k)` moves rows k sessions along the calendar (negative is earlier); rows that fall off are dropped.
- `cutoff_audit(events, cutoffs)` returns every event that breaks `previous cutoff < available_utc <= cutoff(effective_session)`. An empty frame means alignment is exact.
- `lag_curve` reports rank IC with the panel delayed 1, 2 and 5 sessions. A smooth fade is diffusion; a cliff at lag 1 suggests the signal lived in timestamp error.

### 4.15 Backtest wiring — `backtest_config.py`, `long_short.py` (research side; neither imports Qlib)

**Convention C1:** a signal dated t is decided at the close of t, entered at the open of t+1 and exited at the open of t+2, which is exactly what the label measures; `deal_price` is `open`. `build_port_analysis_config` builds Qlib's long-only `TopkDropoutStrategy` config (top 50, drop 5, 5 basis points per side, impact 0.1, minimum cost 1.0, benchmark SPY; all proposed, recorded in `references/cost_assumptions.json`).

Qlib cannot short, so `long_short_returns` does the long-short accounting outside it: equal-weighted top and bottom quantiles, half the capital per leg, cost charged on turnover, a daily borrow fee on the short leg. A selected name with no known return raises instead of counting as zero.

### 4.16 The CLI and config — `src/precedent/cli.py`, `src/precedent/config.py`

`cli.py` is a dispatch table (`TRACK_COMMANDS`) from a command to `module:function`, imported lazily. A first word that is not a command means `ask`. If the target module does not exist yet, the command prints `not built yet: <command>` and exits 2; if a research package is missing, it prints the install hint. Tracks never edit this file.

| Command | Target | Built? |
|---|---|---|
| `ingest gdelt` / `polymarket` / `kalshi` / `edgar` | `precedent.spine.ingest.*:run` | yes |
| `prices` | `precedent.spine.prices_cli:run` | yes |
| `typers` | `precedent.spine.typers.cli:run` | yes |
| `ask`, `fetch`, `build`, `evaluate`, `mcp` | `precedent.engine.*`, `precedent.libraries.build`, `precedent.mcp_server` | no |
| `dream`, `ripple`, `mechanism` | the track packages | no |

`config.py` is the only module allowed to name a model id, a base URL or a data-dir default (a test greps for it).

| Environment variable | Used by | Default |
|---|---|---|
| `PRECEDENT_DATA_DIR` | every store and the fetcher | `~/.cache/precedent` |
| `PRECEDENT_TYPER` | `typer_from_env` | `rules` |
| `PRECEDENT_SEC_USER_AGENT` | EDGAR client | none; required for EDGAR |
| `ANTHROPIC_API_KEY` | `AnthropicTyper` | none |
| `PRECEDENT_OLLAMA_BASE_URL`, `PRECEDENT_OLLAMA_MODEL` | `OllamaTyper` | none |
| `PRECEDENT_MODEL_BASE_URL` | the llama.cpp parser runtime | reserved; its module (spine Task 18) is not built |

---

## 5. The product track that comes next (all PLANNED)

Source: `docs/superpowers/plans/2026-11-06-precedent.md`. Nothing in this section exists in `src/` yet.

**Story clusters (Task 2).** The same story is reported many times. Events with the same key and primary actor within 3 sessions are chained into one cluster; the first report is the event. Without this, one story supplies thirty of its own precedents.

**One row schema, three libraries (Tasks 3 to 6).** Every row has `event_id`, `outcome_type`, `outcome_key`, `horizon`, `value`, `is_binary`, `effective_session`, `response_complete_session`, `source_ref`. Asset rows are CARs at 1, 5 and 20 sessions. Market rows qualify when the market was open at the precedent's `available_utc` (created before, resolved after) and its question typed; `value` is 1.0 for YES. Sequence rows ask whether a first-report event of type T linked to the same country followed within 5, 20 or 60 sessions. A row whose window has not closed is absent and counted, never zero.

**Retrieval (Task 7).** `find_precedents` keeps events available at or before the query, first reports only, never the query's own story, scores exact-key similarity, keeps the top 30, and returns only rows whose `response_complete_session` cutoff is at or before the query time.

**Aggregation (Task 8).** Continuous outcomes: similarity-weighted quantiles, the 10th to 90th percentile band, and a shrunk centre `median * n_eff / (n_eff + 10)`. Binary outcomes: a weighted rate with a **Wilson interval** at 80% (z = 1.2816), which, unlike the plain `p ± z·se` formula, stays inside [0, 1] and behaves at small n. **Effective sample size** is the smaller of Kish's `(Σw)² / Σw²` (how many equal-weight observations the weights are worth) and the count of non-overlapping windows. Example from the plan's tests: twelve events starting on consecutive sessions with a 5-session horizon collapse to n_eff = 3 (starts 0, 5, 10) and are refused. `vs ordinary` compares with a **placebo**: the same measure on random non-event windows (assets) or the unconditional rate (binary).

**Baseline ladder (Task 9).** B0 is the unconditional distribution, B1 exact-key rules typing; later B2 and B3 add model typing and embeddings. Events are scored in time order from earlier events only, with pinball loss and band coverage (continuous) or Brier score (binary).

**Bundle and `precedent build` (Task 10); the card (Task 11); `precedent ask` (Task 12).** The card renders as text, JSON with a `schema_version`, and PNG with a bundled monospace font; the disclaimer line and the `library covers` line are mandatory and tested. `ask` takes text or key flags (`--event-type` and so on), `--asof`, `--horizon`, `--outcome-type`, `--json`, `--card`, and `--dry-run` (rules typer, the committed sample, no socket). An untyped headline returns the refusal with a hint listing the key flags.

**Snapshot and `precedent fetch` (Task 13).** Downloads a release zip and `MANIFEST.json`, verifies every file's SHA-256 before reading anything, rejects path-traversal entries, and only then moves the library into place. `SNAPSHOT_RELEASES_URL` in `config.py` is `None` until the owner creates the GitHub repository.

**MCP server (Task 14).** `precedent-mcp` over stdio with the `mcp` SDK 2.x (`MCPServer`, not the removed `FastMCP`), two read-only tools: `precedents(...)` returns the card's JSON, `card(...)` the text.

**Committed sample (Task 15, owner gate).** About two months of real GDELT events and resolved markets under `data/sample/`, under 2 MB, cited, with no prices and no question text, so the offline demo runs and mostly refuses.

**Dream, Ripple, Mechanism (planned, separate plans).** Dream builds a bi-temporal claim layer and a nightly diff of tensions and silent shifts for the brief. Ripple propagates shocks along dated company links and must beat a shuffled-graph placebo. Mechanism fits small template causal graphs and shows a fan chart beside Precedent's band. All three reuse the spine's clocks and time guard.

---

## 6. How the workspace works

The repository is an **ICM** workspace (Interpretable Context Methodology): the filesystem is the orchestration layer and plain files are the state.

- `CLAUDE.md` (layer 0): what this is, the one rule, global constraints.
- `CONTEXT.md` (layer 1): routes a task type to a stage.
- `stages/NN_name/CONTEXT.md` (layer 2): one contract per stage: objective, inputs, process, outputs, verify, approval, failure behaviour, and a `## Loop` section or a one-line "single pass" reason. `tests/test_workspace_contract.py` checks every contract has its sections. Spine stages are 01 to 09; tracks will use 20s, 30s, 40s, 50s.
- `references/` (layer 3): stable rules that ship in the wheel. `stages/*/output/` and `$PRECEDENT_DATA_DIR` (layer 4): per-run artifacts, never committed.
- **Evidence labels** in `docs/STATUS.md` and **banners** on `docs/examples/` files keep "tested on synthetic data" from being read as "works on real data".
- **Owner gates:** no push, merge, model or dataset download, paid call or first live run without your yes. The repository has no remote.
- **An oracle** is the external check a loop iterates against (tests, a score script), never the worker's own opinion. Only the parser prompt tuning and model fitting stages are loops (`stages/03_parse`, `stages/07_train`); everything else is single pass.
- **Independent verifier:** spine Task 21 runs after Task 16 and writes to `docs/verification/`, which does not exist yet.
- **CI** (`.github/workflows/ci.yml`, actions pinned by SHA): `uv sync --frozen --all-extras`, `uv run ruff check .`, `uv run pytest -q`, `uv run python scripts/ci_ask_lifecycle.py`, then a check that the tree is clean. No network to any source or model. The amendment also asks for a secret scan step in CI; it is not in this workflow (a tracked-file secret test exists in `tests/test_workspace_contract.py`).

**Commands that work today:**

```bash
uv sync --frozen --all-extras            # install everything, from the lock file
uv run pytest -q                         # the test suite (synthetic data, no network)
uv run ruff check .                      # lint
uv run python scripts/ci_ask_lifecycle.py  # checks `ask` still reports "not built yet"
uv run precedent ingest                  # prints the ingest usage line
uv run precedent typers score --typer rules           # rewrites docs/examples/typing_score_rules.md
uv run precedent typers score --typer rules --split heldout
uv run precedent prices import path/to/prices.csv     # date,symbol,open,close
```

`uv run precedent --help` currently prints `not built yet: ask` and exits 2, because a non-command word is routed to `ask`. Use `uv run precedent typers --help` or `uv run precedent prices --help` for real help text.

**Commands that touch the network (owner gate before the first run):**

```bash
uv run precedent ingest gdelt --since 2024-01-02 --until 2024-01-02 --max-files 4   # live GDELT download
PRECEDENT_SEC_USER_AGENT="<name> <email>" uv run precedent ingest edgar --ciks ciks.csv --max-requests 10
uv run precedent ingest polymarket --max-requests 5 --category politics
uv run precedent ingest kalshi --max-requests 5
uv run precedent typers score --typer anthropic --max-calls 60                       # paid API calls
```

A model typer always needs an explicit `--max-calls`; `--dry-run` opens no socket. A live model score appends a `LIVE-MODEL` row to `docs/STATUS.md` automatically.

---

## 7. Where the risks are

- **GDELT is noisy.** Its CAMEO coding over-triggers (thesis section 7). Event-sequence base rates inherit that noise until re-typing of linked articles is measured. The `NumSources >= 3` and severity thresholds are proposals (B4).
- **The CAMEO mapping is a proposal.** Every row of `cameo_to_event_type.json` is marked proposed, and the triage contract lists coarse mappings (protest typed as political, force posture as armed conflict).
- **Matching market questions is the hard part.** Question text is heterogeneous; the rules typer's held-out recall on the synthetic gold set is 0.667, and its direction precision there is 0.300.
- **Selection bias.** Only reported events enter a library; the `library covers` line states it but does not remove it.
- **The calendar for 2010 and 2011 is unverified** (rule-derived), and the July 3 and December 24 early closes in 2012 and 2013 are secondary-sourced only. GDELT 2.0 starts in 2015, so this mainly affects EDGAR and prices.
- **Model-API shapes are unconfirmed.** The Anthropic and Ollama request and reply shapes are tested against fakes only; `ANTHROPIC_VERSION` and the model id in `config.py` have not been exercised.
- **EDGAR item rules and state codes are recalled,** not read from the SEC (`docs/STATUS.md`). The CIK-to-ticker map is current-only, so delisted names may stay unresolved.
- **Data display rights.** Equity price licences such as Tiingo's low tiers are internal-use only (thesis section 2); the repository never ships or displays prices, and the asset layer stays local until a licence exists. Free price data is also survivorship-biased (design 4.3).
- **The model's own memory.** A model trained after an event can remember its outcome. Only post-cutoff results are clean (design 4.4); rules typing avoids the issue.
- **Canada.** The thesis advises an hour with a securities lawyer before charging Canadians for anything that names a security.

---

## 8. Glossary

| Term | Meaning |
|---|---|
| Session | One NYSE trading day. |
| Cutoff | The instant a session's decisions are made: 16:00 New York, or the early close. |
| Point-in-time | Data as it was knowable at a moment, not as later revised. |
| Look-ahead bias | Using information before it was available; makes backtests look better than reality. |
| `available_utc` | `max(published, first_seen) + latency`: when the system could first act on an item. |
| Effective session | The first session whose cutoff is at or after `available_utc`. |
| Abnormal return, CAR | Return minus the market model's expectation; CAR sums it over the event window. |
| Base rate | How often an outcome followed a class of events in the past. |
| Effective sample size (n_eff) | How many independent observations a weighted, overlapping sample is worth. |
| Kish size | `(Σw)² / Σw²`, the effective count of a weighted sample. |
| Wilson interval | A confidence interval for a proportion that stays in [0, 1] and works at small n. |
| Placebo | The same measurement on random non-event windows, for comparison. |
| Story cluster | Reports of one story grouped so it counts once. |
| Purged walk-forward | Time-ordered train/valid/test windows with a gap at least as long as the label horizon. |
| Rank IC | Daily cross-sectional Spearman correlation between predictions and outcomes. |
| CAMEO | The event taxonomy GDELT uses, root codes 01 to 20. |
| 8-K | The SEC form for material corporate events, with numbered items. |
| Typer | Code that turns text into an `EventKey`: rules, or a model behind the same protocol. |
| Vintage | One `parser_version`'s set of records; re-parsing creates a new vintage. |
| Oracle | The external check a loop is judged by. |
| Owner gate | A step that waits for the owner's explicit yes. |

## 9. Read next, in order

1. `CLAUDE.md`, `CONTEXT.md`
2. `docs/superpowers/specs/2026-11-07-precedent-product-amendment.md`
3. `docs/superpowers/specs/2026-11-06-predate-design.md`, sections 3, 4 and 8
4. `src/precedent/spine/event_record.py`, `pit_alignment.py`, `session_calendar.py`, `event_key.py`
5. `tests/spine/test_pit_alignment.py`, `tests/spine/test_session_calendar.py`
6. `src/precedent/spine/gdelt_events.py` and `stages/02_triage/CONTEXT.md`
7. `src/precedent/spine/typers/base.py`, `rules.py`, `typing_response.py`; `docs/examples/typing_score_rules.md`
8. `src/precedent/spine/abnormal_returns.py`, `qlib_dataset.py`, `leak_checks.py`, `tests/spine/test_leak_checks.py`
9. `docs/superpowers/plans/2026-11-06-precedent.md`, Tasks 7, 8, 11 and 12
10. `docs/STATUS.md` before quoting any result
