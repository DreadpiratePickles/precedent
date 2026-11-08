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

### Source: SEC EDGAR
- Label: `NO LIVE RUN`. The owner gate was not passed: SEC requires a declared contact in the User-Agent and no contact is written into this repository. Nothing below was read from a live response by this task.
- Shape assumed, from the field names checked 2026-11-07 (amendment spec section 12, `SEC EDGAR` row): `GET {SEC_SUBMISSIONS_BASE_URL}/CIK##########.json` (CIK zero-padded to 10 digits). Top level: `cik`, `entityType`, `sic`, `name`, `tickers`, `exchanges`, `addresses.business.stateOrCountry`, `filings.recent`, `filings.files`.
- `filings.recent` is columnar (parallel arrays of equal length): `accessionNumber`, `filingDate`, `acceptanceDateTime` (ISO-8601 UTC, for example `2026-11-06T21:30:47.000Z`), `form`, `items` (comma-separated item codes such as `2.02,9.01`; empty for forms without items), `primaryDocument`. `filings.files` (older filings in further documents) is not read: only `recent` is.
- Alignment: `first_seen_utc` and `published_utc` are `acceptanceDateTime`, never `filingDate`. Only forms `8-K` and `8-K/A` produce records, one per item code that has a rule; other items and forms are counted in the report, never dropped silently. `8-K/A` is `novelty = update`.
- `references/us_state_codes.json`: `RECALLED`. The two-letter codes for the 50 states, DC and PR, VI, GU, AS, MP. EDGAR writes other countries as different strings (for example `X0` for the UK), so any code not in the file is non-US and the record links no country.
- `references/item_rules_8k.json`: `RECALLED`. The item numbers and titles were not confirmed against the SEC item list (the sec.gov HTML page redirected to an investor.gov glossary page, and the Form 8-K PDF could not be read as text here). Every entry is a proposal for the owner; the file says `"proposed": true`. Unlisted items (for example 7.01, 8.01, 9.01) are skipped and counted.
- Fixtures `tests/fixtures/submissions_sample.json` and `submissions_foreign_sample.json` are synthetic, in the shape above.
- Limit: the first page (`recent`) holds about the latest 1,000 filings; a company with older 8-Ks needs the `files` documents, which this version does not fetch.
