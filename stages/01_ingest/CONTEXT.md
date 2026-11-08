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

### Source: S&P 500 membership
- Source: `fja05680/sp500`, file `S&P 500 Historical Components & Changes.csv` (read 2026-11-07 from the raw file; its README gives no column layout). Not committed; `tests/fixtures/membership_sample.csv` is a synthetic file with the same layout.
- Layout: header `date,tickers`; one row per change date (ISO `YYYY-MM-DD`); `tickers` is one quoted, comma-separated list holding every member after that date.
- Delisted symbols may carry a `-YYYYMM` suffix (for example `AAL-199702`). `load_membership` keeps symbols exactly as written; `members_asof` uses only rows dated on or before the asked date and raises for a date before the first row.
- Limit: the free CIK-to-ticker map is current-only, so a CIK of a company that has since changed ticker or delisted may be unresolved. `resolve_instrument` returns None rather than guess.

### Source: Polymarket
- LIVE-DATA, read 2026-11-07, unauthenticated, `GET /markets?closed=true` and `/markets/keyset?closed=true` on the Gamma base URL. No response body is stored in the repository.
- The response is a bare JSON list. `outcomes` and `outcomePrices` are JSON-encoded strings (for example `'["Yes", "No"]'`, `'["0", "1"]'`). There is no winner field.
- Time formats differ by field: `createdAt` and `endDate` are ISO-8601 with `Z`; `closedTime` is `YYYY-MM-DD HH:MM:SS[.ffffff]+00` (space separator, `+00` offset, no colon). `endDate` is absent on about 10% of old markets.
- `umaResolutionStatus` (singular) exists only on newer markets (seen as `resolved`); old markets carry only `umaResolutionStatuses` (plural, JSON-encoded list, `[]` for old ones) and no singular field. Old markets can also show `outcomePrices` `["0", "0"]`.
- Paging limits: `limit` is capped at 100 (a request for 500 or 101 returns 100 rows); `offset` is refused somewhere between 1000 and 3000 with "offset too large, use /markets/keyset for deeper pagination". `/markets/keyset` returns `{"markets": [...], "next_cursor": ...}` and takes `after_cursor`. Default order is oldest first; `order=closedTime&ascending=false` is accepted.
- Client rule (controller rulings 26-28): page `/markets/keyset?closed=true&limit=100&order=closedTime&ascending=false[&after_cursor=]`, newest first. A probe on 2026-11-07 confirmed the keyset endpoint accepts those order parameters and that `next_cursor` continues newest-first. Exhausted when `next_cursor` is empty or the page is empty. A market with `closed` true and no `umaResolutionStatus` key counts as resolved when `outcomePrices` are exactly one 1 and one 0 (counted as `resolved_by_prices_only`); a present status other than `resolved` is unresolved whatever the prices.

### Source: Kalshi
- LIVE-DATA, read 2026-11-07, unauthenticated, on the Kalshi base URL. No response body is stored in the repository.
- `GET /historical/cutoff` returns `market_settled_ts` `2026-09-08T00:00:00Z` (also `trades_created_ts`, `orders_updated_ts`, `market_positions_last_updated_ts`).
- `GET /markets?status=settled` and `GET /historical/markets` return `{"markets": [...], "cursor": "<opaque>"}`. Market fields seen: `ticker`, `event_ticker`, `title`, `yes_sub_title`, `result`, `settlement_ts`, `close_time`, `created_time`, `status`; all timestamps are ISO-8601 with `Z` (microseconds optional).
- `result` takes `yes`, `no` and also `scalar`; only `yes` and `no` are kept. `title` carries the full question on ordinary markets ("Will over 8.5 goals be scored?"); `yes_sub_title` is the shorter label. On multi-leg markets (`KXMVE...`) both are the same comma-joined leg list.
- `GET /events/{event_ticker}` returns `{"event": {..., "series_ticker", "category"}, "markets": [...]}`: the series ticker is `event.series_ticker` (the event also carries a `category`). `GET /series/{series_ticker}` returns `{"series": {..., "category", "categories"}}`.
- Settled markets are dominated by sports and multi-leg markets, so a category filter discards most rows only after the two category lookups per new event.
