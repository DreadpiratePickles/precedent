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
