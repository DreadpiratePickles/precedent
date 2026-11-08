# Stage: 01_ingest

## Objective
Fetch raw items from each source and append them, content-hashed and stamped with `first_seen_utc`, to an append-only article store.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `src/precedent/config.py` (source base URLs, data directory) | 3 | Authoritative for every URL and default | Yes | Source constants |
| Source feeds: SEC EDGAR submissions, GDELT masterfile list, Polymarket, Kalshi | 4 | Authoritative for item content and the source's own availability stamp | Yes | Per source, through an injected `Transport` |
| `references/` event schema (`first_seen_utc` rule) | 3 | Authoritative for record fields | Yes | Design 3.4 and 4.1 |

## Process
1. Deterministic: read the source's cursor from the store, request only newer items through the shared HTTP seam (timeouts, bounded retries, rate limit).
2. Deterministic: compute `body_sha256` of the exact text kept; set `first_seen_utc` from the source's availability stamp (backfill) or the fetch time (live). A missing stamp is an error, not a default.
3. Deterministic: append records keyed by `article_id` and `body_sha256`; never rewrite or delete. Third-party article text stays in the local store and is never committed.
4. Deterministic: write a manifest of what was fetched, skipped and failed.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| Article store under the data directory (`PRECEDENT_DATA_DIR`) | Append-only records, `first_seen_utc` required, naive timestamps rejected | `02_triage` |
| `stages/01_ingest/output/manifest.json` | JSON: counts, cursors, per-item failures, input hashes | Human review, `02_triage` |

## Verify
- Unit tests with injected `Transport` and `Clock` only; no test touches the network.
- A record without `first_seen_utc`, or with a naive timestamp, is rejected.
- Re-running the same fetch appends nothing (idempotent by `article_id` and `body_sha256`).
- `uv run pytest -q` exits 0.

## Approval
Review the manifest counts and failures. A live fetch against a real source, and any fetch that downloads a dataset, is an owner gate: name the source, size and cost first.

## Failure Behavior
Retry only transient failures (429, 5xx) with exponential backoff and jitter, at most 3 times. A partial failure is recorded per item in the manifest; a failed read never becomes an empty store. After the retry budget, stop, keep what was appended, and escalate with the manifest.

Loop: single pass. The work is a deterministic transform decided by tests, so there is no oracle-driven iteration to run (design 10.2).

Market typing outcomes that say nothing about the question are not stored: a spent call budget (`max_calls`), a transport refusal (`http ...`) and anything produced under `--dry-run`. They are counted (`not_stored_transient`), the crawl stops without saving the cursor past that page so a later run fetches it again, and `ingest polymarket|kalshi --dry-run` sends no request and writes nothing.

### Source: S&P 500 membership
- Source: `fja05680/sp500`, file `S&P 500 Historical Components & Changes.csv` (read 2026-11-07 from the raw file; its README gives no column layout). Not committed; `tests/fixtures/membership_sample.csv` is a synthetic file with the same layout.
- Layout: header `date,tickers`; one row per change date (ISO `YYYY-MM-DD`); `tickers` is one quoted, comma-separated list holding every member after that date.
- Delisted symbols may carry a `-YYYYMM` suffix (for example `AAL-199702`). `load_membership` keeps symbols exactly as written; `members_asof` uses only rows dated on or before the asked date and raises for a date before the first row.
- Limit: the free CIK-to-ticker map is current-only, so a CIK of a company that has since changed ticker or delisted may be unresolved. `resolve_instrument` returns None rather than guess.
### Source: prices

Research for Task 11, Step 0. Read from `microsoft/qlib` tag `v0.9.7` (commit `da920b7f954f48ab1bb64117c976710de198373e`), shallow-cloned into the gitignored `vendor/qlib` (about 15 MB). Nothing from it is committed.

**Commands (Yahoo through Qlib's collector; not run, owner gate in Task 11 Step 6).** From `vendor/qlib`, with the collector's `requirements.txt` installed:
1. Collect: `python scripts/data_collector/yahoo/collector.py download_data --source_dir <src> --start 2000-01-01 --end <YYYY-MM-DD> --delay 1 --interval 1d --region US` (`end` is an open bound and may not lie in the future).
2. Normalize: `python scripts/data_collector/yahoo/collector.py normalize_data --source_dir <src> --normalize_dir <nor> --region US --interval 1d`.
3. Dump: `python scripts/dump_bin.py dump_all --data_path <nor> --qlib_dir data/qlib_us --freq day --exclude_fields date,symbol --file_suffix .csv`.

**Findings that bear on Step 6.**
- The US collector takes no symbol list. `YahooCollectorUS.get_instrument_list` builds the universe from `get_us_stock_symbols`, which reads three third-party endpoints (an Eastmoney API, the Nasdaq Trader FTP symbol directory, an NYSE API) plus `^GSPC`, `^NDX`, `^DJI`. The dated S&P 500 union and SPY therefore cannot be requested directly; the options are a subclass or patch of `get_instrument_list` in a throwaway copy, or `limit_nums` for a trial. Whether SPY is in the universe is unverified until a run.
- Normalization rescales `open`, `high`, `low`, `close` by the adjusted close and then so that the first valid close is 1. Levels from the dumped provider are therefore not prices; returns are unaffected, which is all the spine uses (return convention in design 8.5).
- `dump_all` accepts a CSV directory with `date` and `symbol` columns and one column per field, so a hand-built CSV directory can be dumped without the collector.

**Binary layout (confirmed by reading `dump_bin.py`; the test fixture `tests/fixtures/tiny_provider.py` writes exactly this and Qlib reads it back in `tests/spine/test_qlib_provider.py`).**
- `calendars/day.txt`: one `YYYY-MM-DD` per line, sorted.
- `instruments/all.txt`: `SYMBOL<TAB>start<TAB>end`, upper-case symbols, dates `YYYY-MM-DD`.
- `features/<symbol lower-case>/<field>.day.bin`: little-endian float32 (`"<f"`). The first value is the calendar index of the symbol's first row; the rest are one value per calendar day from that row to the symbol's last row, NaN for a day the symbol skipped. Qlib returns a NaN row for such a day and no rows after the symbol's end.

**Qlib init form used (`precedent.spine.qlib_provider.init_qlib`).** `qlib.init(provider_uri=str(provider_dir), region="us", exp_manager={"class": "MLflowExpManager", "module_path": "qlib.workflow.expm", "kwargs": {"uri": "sqlite:///<workdir>/mlruns.db", "default_exp_name": "precedent"}})`. This form works unchanged with pyqlib 0.9.7 and mlflow 3.18.0: an experiment run created `mlruns.db` under the work directory. Qlib spawns worker processes for `D.features`; a script that calls it outside pytest must sit under `if __name__ == "__main__":` on macOS.

**Provider check.** `python -m precedent.spine.qlib_provider --check <provider_dir>` prints sessions, instruments, last session, whether SPY is present and SPY's last session. It reads the calendar and instrument files directly and does not initialise Qlib.

**Caveat for tasks that record experiments.** With this SQLite store, a recorded run writes its artifacts to `./mlruns` relative to the working directory (observed in a probe run), not next to `mlruns.db`. Tasks that call `R.start` must run from, or set an artifact location under, the work directory. `init_qlib` alone writes nothing there.

### Source: GDELT 2.0
- Source: Events export files, one every 15 minutes since 2015-02-19, named `YYYYMMDDHHMMSS.export.CSV.zip` in the master list `GDELT_MASTERFILELIST_URL` (`config.py`; lines `<bytes> <md5> <url>`). Fetch and parse code: `spine/gdelt_fetch.py`, `spine/gdelt_events.py`; command `precedent ingest gdelt --since YYYY-MM-DD --until YYYY-MM-DD [--max-files N]` (default 96, one day). Raw files go to `<data dir>/raw/gdelt/`, progress to `<data dir>/gdelt_fetch_state.json` (written atomically after each file whose MD5 matched). Nothing under the data directory is committed.
- Format: tab-separated, no header, exactly 61 columns; `read_export_csv` refuses any other count and names the line.
- Column positions, 1-based, read on 2026-11-07 from the GDELT Event Codebook V2.0 (381,630 bytes, `data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf`): `GLOBALEVENTID` 1, `SQLDATE` 2, `Actor1Code` 6, `Actor1Name` 7, `Actor1CountryCode` 8, `Actor1Type1Code` 13, `Actor2Name` 17, `Actor2CountryCode` 18, `Actor2Type1Code` 23, `IsRootEvent` 26, `EventCode` 27, `EventBaseCode` 28, `EventRootCode` 29, `QuadClass` 30, `GoldsteinScale` 31, `NumMentions` 32, `NumSources` 33, `NumArticles` 34, `AvgTone` 35, `ActionGeo_CountryCode` 54, `DATEADDED` 60, `SOURCEURL` 61. The codebook lists fields in order and has no position column: columns 1 to 35 are the fields up to `AvgTone`, then three location blocks of eight (`Type`, `FullName`, `CountryCode`, `ADM1Code`, `ADM2Code`, `Lat`, `Long`, `FeatureID`) for Actor1 (36-43), Actor2 (44-51) and Action (52-59), then `DATEADDED` (the codebook calls it "the second from the last field") and `SOURCEURL`. These positions were derived from that order; no export file was fetched to confirm the width.
- Header file caution: `gdeltproject.org/data/lookups/CSV.header.dailyupdates.txt` (893 bytes, read 2026-11-07) is the GDELT 1.0 layout of 58 columns, with no `ADM2Code` fields (`ActionGeo_CountryCode` 52, `DATEADDED` 57, `SOURCEURL` 58). Columns 1 to 35 agree with the 2.0 layout; the positions after 35 do not. This code uses the 2.0 layout only.
- `ActionGeo_CountryCode` is FIPS 10-4, not ISO 3166, and is not used in v1. `NumMentions`, `NumSources` and `NumArticles` count only the first 15-minute window in which the event was seen.
- Time: `first_seen_utc` is `DATEADDED` (UTC); `SQLDATE` is never read for time. `published_utc` is null.
- Dedup across runs: the store keys records by `article_id` only; consumers reading several runs together call `dedupe_gdelt_records`.
- Parser version: `rules-gdelt-v1:<12 hex>` (`gdelt_parser_version()`) is a hash of `references/cameo_to_event_type.json` and the `gdelt` section of `references/precedent_params.json`; editing either starts a new vintage instead of mixing rules inside one. The 8-K version, `rules-8k-v1:<12 hex>` (`edgar_parser_version()`), hashes `references/item_rules_8k.json`. Rows dated before the calendar's first year stay unaligned.
- Integrity: the master list is served over plain HTTP, so the MD5 guards against corruption, not tampering.
- Live run: the first run is an owner gate (it downloads real files); see the stage Approval section.

### Source: SEC EDGAR
- Label: `NO LIVE RUN`. The owner gate was not passed: SEC requires a declared contact in the User-Agent and no contact is written into this repository. Nothing below was read from a live response by this task.
- Shape assumed, from the field names checked 2026-11-07 (amendment spec section 12, `SEC EDGAR` row): `GET {SEC_SUBMISSIONS_BASE_URL}/CIK##########.json` (CIK zero-padded to 10 digits). Top level: `cik`, `entityType`, `sic`, `name`, `tickers`, `exchanges`, `addresses.business.stateOrCountry`, `filings.recent`, `filings.files`.
- `filings.recent` is columnar (parallel arrays of equal length): `accessionNumber`, `filingDate`, `acceptanceDateTime` (ISO-8601 UTC, for example `2026-11-06T21:30:47.000Z`), `form`, `items` (comma-separated item codes such as `2.02,9.01`; empty for forms without items), `primaryDocument`. `filings.files` (older filings in further documents) is not read: only `recent` is.
- Alignment: `first_seen_utc` and `published_utc` are `acceptanceDateTime`, never `filingDate`. Only forms `8-K` and `8-K/A` produce records, one per item code that has a rule; other items, other forms and 8-K filings whose `items` is empty, separator-only or null are counted in the report (`skipped_items`, `skipped_forms`, `skipped_filings_without_items`), never dropped silently. HTTP 403 (a blocked or undeclared User-Agent) and 429 stop the ingest run after the first occurrence. `8-K/A` is `novelty = update`.
- `references/us_state_codes.json`: `RECALLED`. The two-letter codes for the 50 states, DC and PR, VI, GU, AS, MP. EDGAR writes other countries as different strings (for example `X0` for the UK), so any code not in the file is non-US and the record links no country.
- `references/item_rules_8k.json`: `RECALLED`. The item numbers and titles were not confirmed against the SEC item list (the sec.gov HTML page redirected to an investor.gov glossary page, and the Form 8-K PDF could not be read as text here). Every entry is a proposal for the owner; the file says `"proposed": true`. Unlisted items (for example 7.01, 8.01, 9.01) are skipped and counted.
- Fixtures `tests/fixtures/submissions_sample.json` and `submissions_foreign_sample.json` are synthetic, in the shape above.
- Limit: the first page (`recent`) holds about the latest 1,000 filings; a company with older 8-Ks needs the `files` documents, which this version does not fetch.
- Identity: `article_id` is `<accession>:<item>` and carries no instrument. `events_from_submissions(submissions, instruments, ...)` makes one record per (filing, item) with one resolved impact per instrument on the CIK (share classes such as GOOG and GOOGL); `ingest edgar` groups the `cik,symbol` lines by CIK, fetches each CIK once, and catches a store conflict (`EventConflictError`) or `ValueError` per CIK, so the JSON summary always prints and the exit code is 1.
- Calendar years: a filing accepted before 1 January of the calendar's first year gets no session (`effective_session` is null) and the store counts it as `unaligned_records`; the default cutoffs start no earlier than that date.
- Owner-gate check, `acceptanceDateTime` time zone (`RECALLED` until confirmed): the code reads the `Z` suffix as true UTC. At the first live fetch, confirm it against `filingDate` and the SEC's 17:30 Eastern same-day rule (a filing accepted after 17:30 ET carries the next business day's `filingDate`). Under true UTC, 17:30 ET is 21:30Z (EDT) or 22:30Z (EST), so a filing stamped later than that on a business day must show the next business day's `filingDate`. If instead filings stamped between 17:30Z and that threshold already show the next day's `filingDate`, the stamps are Eastern wall time with a false `Z`: every 8-K's `available_utc` would be 4-5 hours early (systematic look-ahead for after-close 8-Ks), and no 8-K result may be trusted until the parser is fixed. Evidence so far, not proof: the 2026-11-07 unauthenticated probe showed `2026-11-06T21:30:47.000Z` for a same-day filing, consistent with UTC.

### Source: Polymarket
- LIVE-DATA, read 2026-11-07, unauthenticated, `GET /markets?closed=true` and `/markets/keyset?closed=true` on the Gamma base URL. No response body is stored in the repository.
- The response is a bare JSON list. `outcomes` and `outcomePrices` are JSON-encoded strings (for example `'["Yes", "No"]'`, `'["0", "1"]'`). There is no winner field.
- Time formats differ by field: `createdAt` and `endDate` are ISO-8601 with `Z`; `closedTime` is `YYYY-MM-DD HH:MM:SS[.ffffff]+00` (space separator, `+00` offset, no colon). `endDate` is absent on about 10% of old markets.
- `umaResolutionStatus` (singular) exists only on newer markets (seen as `resolved`); old markets carry only `umaResolutionStatuses` (plural, JSON-encoded list, `[]` for old ones) and no singular field. Old markets can also show `outcomePrices` `["0", "0"]`.
- Paging limits: `limit` is capped at 100 (a request for 500 or 101 returns 100 rows); `offset` is refused somewhere between 1000 and 3000 with "offset too large, use /markets/keyset for deeper pagination". `/markets/keyset` returns `{"markets": [...], "next_cursor": ...}` and takes `after_cursor`. Default order is oldest first; `order=closedTime&ascending=false` is accepted.
- Client rule (controller rulings 26-28): page `/markets/keyset?closed=true&limit=100&order=closedTime&ascending=false[&after_cursor=]`, newest first. A probe on 2026-11-07 confirmed the keyset endpoint accepts those order parameters and that `next_cursor` continues newest-first. Exhausted when `next_cursor` is empty or the page is empty. A market with `closed` true and no `umaResolutionStatus` key counts as resolved when `outcomePrices` are exactly one 1 and one 0 (counted as `resolved_by_prices_only`); a present status other than `resolved` is unresolved whatever the prices.
- Resetting: the client pages newest first and a finished feed keeps a permanent `exhausted` flag, so markets that close later are seen only after the state file `state/polymarket.json` is reset.
- Re-typing a market already stored under the same typer version (for example with a larger `--max-calls`) raises `MarketConflictError` by design (the store is keyed by market and typer version): reset the store or bump the typer version.

### Source: Kalshi
- LIVE-DATA, read 2026-11-07, unauthenticated, on the Kalshi base URL. No response body is stored in the repository.
- `GET /historical/cutoff` returns `market_settled_ts` `2026-09-08T00:00:00Z` (also `trades_created_ts`, `orders_updated_ts`, `market_positions_last_updated_ts`).
- `GET /markets?status=settled` and `GET /historical/markets` return `{"markets": [...], "cursor": "<opaque>"}`. Market fields seen: `ticker`, `event_ticker`, `title`, `yes_sub_title`, `result`, `settlement_ts`, `close_time`, `created_time`, `status`; all timestamps are ISO-8601 with `Z` (microseconds optional).
- `result` takes `yes`, `no` and also `scalar`; only `yes` and `no` are kept. `title` carries the full question on ordinary markets ("Will over 8.5 goals be scored?"); `yes_sub_title` is the shorter label. On multi-leg markets (`KXMVE...`) both are the same comma-joined leg list.
- `GET /events/{event_ticker}` returns `{"event": {..., "series_ticker", "category"}, "markets": [...]}`: the series ticker is `event.series_ticker` (the event also carries a `category`). `GET /series/{series_ticker}` returns `{"series": {..., "category", "categories"}}`.
- Settled markets are dominated by sports and multi-leg markets, so a category filter discards most rows only after the two category lookups per new event.
- A finished feed (`phase` `done`) is not re-read; reset `state/kalshi.json` to see markets that settle later. Re-typing a stored market under the same typer version raises `MarketConflictError` by design (keying rule): reset the store or bump the typer version.
