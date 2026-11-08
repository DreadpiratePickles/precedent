# Stage: 21_precedent_evaluate

## Objective
Score the baseline ladder (B0, B1) against the precedent estimate for each outcome type, in time order, so the precedent card is shown only where it beats a baseline.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `library/rows_*.parquet`, `library/placebo_asset.parquet` | 4 | Authoritative for outcomes | Yes | One file per outcome type |
| `library/events.parquet`, `library/sessions.parquet` | 4 | Authoritative for event time and key | Yes | `first_seen_utc`, key |
| `library/MANIFEST.json` | 4 | Authoritative for file hashes | Yes | Verified before any file is read |
| `references/precedent_params.json` | 3 | Authoritative for `top_k`, `min_neff`, `shrink_k` | Yes | `engine` section |

## Process
1. Deterministic: verify the manifest hashes, then load the library.
2. Deterministic: walk events in time order; for each, take only rows whose outcome was known before the event (an expanding window).
3. Deterministic: B0 is the unconditional placebo distribution; B1 is an exact match on a rules-based event type (design 8.4); the precedent estimate uses the top `top_k` key matches.
4. Deterministic: score each rung per outcome type with pinball loss per quantile and band coverage, with Wilson intervals at `wilson_z`.
5. Deterministic: write one table of scores per outcome type and a short report.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/21_precedent_evaluate/output/scores.parquet` | Outcome type, method, metric, interval, n | `23`, human review |
| `stages/21_precedent_evaluate/output/report.md` | Plain text, with the evidence label | Human review |

## Verify
- A score for an event never uses a row that resolved after the event's `first_seen_utc`.
- Truncating the library at a date gives the same earlier scores.
- The report states `SYNTHETIC`, `LIVE-DATA`, `LIVE-MODEL` or `NO LIVE RUN`.
- `uv run pytest -q` exits 0.

## Approval
A human reads the report before any claim of skill is made. No external action occurs in this stage.

## Failure Behavior
A manifest mismatch stops the stage before any file is read. An outcome type with fewer than `min_neff` effective matches is reported as insufficient, never scored. No retries. Output is written atomically.

Loop: single pass until Task 18 adds its `## Loop`; one pass over a fixed library is decided by tests.
