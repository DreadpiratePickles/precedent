# Stage: 20_precedent_libraries

## Objective
Build the three precedent libraries (asset, market resolution, event sequence) and a placebo table from the event and market stores, using code only.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `events/` under the data dir | 4 | Authoritative for typed, aligned event records | Yes | All records, `first_seen_utc`, `effective_session` |
| `markets/` under the data dir | 4 | Authoritative for resolved market records | Yes | Question, typed key, resolution time |
| `prices/` under the data dir | 4 | Authoritative for outcomes; read through the spine price provider | No | Daily bars of the asset panel; without it the asset library is not built |
| `references/precedent_params.json` | 3 | Authoritative for horizons, panel, placebo samples and seed | Yes | `engine` section; every value is proposed |
| `references/country_groups.json` | 3 | Authoritative for the country group of a key | Yes | `groups` |

## Process
1. Deterministic: read events and keep those with a `first_seen_utc`, a session and a key. Reject naive timestamps.
2. Deterministic: read markets under one typer version; `PRECEDENT_TYPER` selects it. Count untyped markets in the report; never drop one silently.
3. Deterministic: for each event and each horizon in `asset_horizons`, compute the abnormal return against `market` from prices at or after the effective session.
4. Deterministic: build market-resolution rows (resolved after the event, same key) and event-sequence rows (the next events at `sequence_horizons`).
5. Deterministic: draw `placebo_samples` placebo dates with `placebo_seed`, and compute the same outcomes for them.
6. Deterministic: write each file atomically and a MANIFEST with the SHA-256 of every file.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `library/events.parquet`, `library/sessions.parquet` | One row per event; session calendar | `21`, `22`, `23`, `24` |
| `library/rows_asset.parquet`, `library/rows_market_resolution.parquet`, `library/rows_event_sequence.parquet` | One row per event and outcome | `21`, `23` |
| `library/placebo_asset.parquet`, `library/markets.parquet` | Placebo outcomes; typed markets | `21`, `23` |
| `library/build_report.json`, `library/MANIFEST.json` | Counts, skips, typer version; file hashes | `24`, human review |

## Verify
- Row schema checked on every file; a row whose outcome starts before its event's effective session fails the time guard.
- Counts: events in equal rows out plus skips; the report reconciles each number.
- Truncating the event store gives the same earlier rows.
- `uv run pytest -q` exits 0.

## Approval
No approval inside the stage and no external action. A human reviews `build_report.json` before any snapshot is made.

## Failure Behavior
An empty event store exits 2 with the cause on stderr. A row that breaks the time guard stops the stage with its event id. Untyped markets are counted in the report, never dropped silently. No retries. Output is written atomically.

Loop: single pass. A deterministic build decided by tests.
