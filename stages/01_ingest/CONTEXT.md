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

### Source: GDELT 2.0
- Source: Events export files, one every 15 minutes since 2015-02-19, named `YYYYMMDDHHMMSS.export.CSV.zip` in the master list `GDELT_MASTERFILELIST_URL` (`config.py`; lines `<bytes> <md5> <url>`). Fetch and parse code: `spine/gdelt_fetch.py`, `spine/gdelt_events.py`; command `precedent ingest gdelt --since YYYY-MM-DD --until YYYY-MM-DD [--max-files N]` (default 96, one day). Raw files go to `<data dir>/raw/gdelt/`, progress to `<data dir>/gdelt_fetch_state.json` (written atomically after each file whose MD5 matched). Nothing under the data directory is committed.
- Format: tab-separated, no header, exactly 61 columns; `read_export_csv` refuses any other count and names the line.
- Column positions, 1-based, read on 2026-11-07 from the GDELT Event Codebook V2.0 (381,630 bytes, `data.gdeltproject.org/documentation/GDELT-Event_Codebook-V2.0.pdf`): `GLOBALEVENTID` 1, `SQLDATE` 2, `Actor1Code` 6, `Actor1Name` 7, `Actor1CountryCode` 8, `Actor1Type1Code` 13, `Actor2Name` 17, `Actor2CountryCode` 18, `Actor2Type1Code` 23, `IsRootEvent` 26, `EventCode` 27, `EventBaseCode` 28, `EventRootCode` 29, `QuadClass` 30, `GoldsteinScale` 31, `NumMentions` 32, `NumSources` 33, `NumArticles` 34, `AvgTone` 35, `ActionGeo_CountryCode` 54, `DATEADDED` 60, `SOURCEURL` 61. The codebook lists fields in order and has no position column: columns 1 to 35 are the fields up to `AvgTone`, then three location blocks of eight (`Type`, `FullName`, `CountryCode`, `ADM1Code`, `ADM2Code`, `Lat`, `Long`, `FeatureID`) for Actor1 (36-43), Actor2 (44-51) and Action (52-59), then `DATEADDED` (the codebook calls it "the second from the last field") and `SOURCEURL`. These positions were derived from that order; no export file was fetched to confirm the width.
- Header file caution: `gdeltproject.org/data/lookups/CSV.header.dailyupdates.txt` (893 bytes, read 2026-11-07) is the GDELT 1.0 layout of 58 columns, with no `ADM2Code` fields (`ActionGeo_CountryCode` 52, `DATEADDED` 57, `SOURCEURL` 58). Columns 1 to 35 agree with the 2.0 layout; the positions after 35 do not. This code uses the 2.0 layout only.
- `ActionGeo_CountryCode` is FIPS 10-4, not ISO 3166, and is not used in v1. `NumMentions`, `NumSources` and `NumArticles` count only the first 15-minute window in which the event was seen.
- Time: `first_seen_utc` is `DATEADDED` (UTC); `SQLDATE` is never read for time. `published_utc` is null.
- Dedup across runs: the store keys records by `article_id` only; consumers reading several runs together call `dedupe_gdelt_records`.
- Integrity: the master list is served over plain HTTP, so the MD5 guards against corruption, not tampering.
- Live run: the first run is an owner gate (it downloads real files); see the stage Approval section.
