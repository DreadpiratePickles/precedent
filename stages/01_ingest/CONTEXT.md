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
