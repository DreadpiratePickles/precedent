FAIL — Precedent track verified against origin/main a3eb6a4 (track commits fa134fa..ead812e): 12 checks run, 9 PASS, 3 FAIL (V-1 Important, V-2 Important, V-3 to V-5 Minor); 7 live or owner-gated steps BLOCKED / NO LIVE RUN. 2090 tests passed, 1 skipped, ruff clean. No time-guard leak found; the failures are a false-zero in the event-sequence library and misleading header counts.

Verifier: independent, fresh context. Inputs read: product amendment, design section 8, the Precedent plan (Global Constraints, Review Focus, Task 19), CLAUDE.md. Label: every PASS below is SYNTHETIC (synthetic fixtures and probes) or NO LIVE RUN; nothing here touched live data, a model or the network. ANTHROPIC_API_KEY was unset and PRECEDENT_TYPER left at `rules`. Probe scripts are in the verifier's scratchpad (`verify-prec/`), not in the repository.

## Step 2: commands and exit codes

| Command | Exit | Result |
|---|---|---|
| `uv sync --frozen --all-extras --offline` | 0 | Checked 207 packages, no download |
| `uv run ruff check .` | 0 | All checks passed |
| `uv run pytest -q` (run with `-v`) | 0 | 2090 passed, 1 skipped (`tests/spine/test_typer_select.py::test_a_model_typer_whose_module_is_not_built_is_a_config_error_naming_it`), 145 s. A first run overlapped with another verifier's pytest and stalled at 0.1% CPU; it was killed and the clean re-run passed. |
| `precedent ask --dry-run --event-type central_bank --actor-kind central_bank --severity-band high --unscheduled` | 0 | REFUSED card, `library covers  nothing (no library loaded)`, disclaimer. `data/sample/` does not exist (owner gate), so no library. |
| `precedent ask --dry-run "zzzz qqqq"` | 0 | `typed as       UNTYPED`, REFUSED card, hint on stderr |
| `precedent ask --data-dir /nonexistent --event-type ... --unscheduled` | 2 | `error: no library at /nonexistent/library ...` plus fetch/build/dry-run hint |
| `precedent-mcp` over stdio (client script, mcp 2.x `stdio_client`) | 0 | Tools listed: `precedents` [horizon, key, outcome_type, text], `card` [horizon, key, text]. Calls with no library or both text and key return tool errors (`isError`), not crashes. |
| `PRECEDENT_TYPER=anthropic` without key / `ollama` without base URL | 2 | `config error: ... required and is not set`, before any network call |
| `precedent ask --dry-run --card x.png ...` | 2 | Card printed, then `ConfigError` because `references/fonts/DejaVuSansMono.ttf` is not bundled (owner gate); no PNG written |

## Step 3: check table

| # | Requirement | Verdict | Label | Evidence |
|---|---|---|---|---|
| 1 | Review Focus 1: a query's own story is not its own precedent | PASS | SYNTHETIC | `retrieval.py:84-85`; test `test_a_story_is_not_its_own_precedent`; probe: with `story_cluster_id="S3"` events a, c returned, own-story d and update b excluded. Caveat V-5: CLI and MCP never set a story id. |
| 2 | Review Focus 2: unfinished window absent and counted, never zero (asset, event sequence) | FAIL | SYNTHETIC | Asset OK: `asset.py:80` skips and counts `skipped_incomplete_window`; test `test_unfinished_windows_are_absent_and_counted`. Event sequence FAILS at the feed edge: V-1. |
| 3 | Review Focus 3: market created after, resolved before, or created at the same instant is no precedent; untyped counted | PASS | SYNTHETIC | `market_resolution.py:115` (`created < t < resolved`, strict); test `test_only_markets_open_at_the_event_qualify` (includes `polymarket:same`, created == T_E, excluded); `markets_untyped` counted, asserted in the same test |
| 4 | Review Focus 4: few independent precedents refuses, no band, still prints `library covers` and disclaimer | PASS | SYNTHETIC | `card.py:207-224`; tests `test_refused_card_text_is_exact_and_draws_no_band`, `test_every_card_ends_with_covers_then_disclaimer`; CLI probe horizon 60 on a 20-event library: `REFUSED 7 usable precedents, need 8`, covers and disclaimer printed |
| 5 | Review Focus 5: snapshot with one tampered byte refused before any file is parsed, nothing left in data dir | PASS | SYNTHETIC | Probe (CRC-valid tampered `events.parquet`, `pq.read_table`, `pq.ParquetFile` and `pd.read_parquet` patched to raise): refused `events.parquet: sha256 differs from the manifest`, `dest` empty. Raw byte flip is caught earlier by zip CRC. Tests `test_a_tampered_byte_is_refused_and_nothing_is_installed`, `test_an_unlisted_entry_is_refused`. |
| 6 | No path builds a band without `find_precedents` | PASS with caveat | SYNTHETIC | Callers of `summarize_responses`: `ask.py:323` (rows from `find_precedents` at `ask.py:299`), `evaluate.py:184`, `precedent_factors.py:178`, `mechanism/evaluate.py:198`, `mechanism/cli.py:235` (via `build_guarded_panel`, `panel.py:128`). Caveat: the card's header `n`/`independent` and the REFUSED decision use `select_precedent_events` alone (`ask.py:238-242`); no band, but see V-2. |
| 7 | Time guard behaviour of `find_precedents`, including the top_k question | FAIL (design defect, no leak) | SYNTHETIC | See "top_k probe" below and V-2 |
| 8 | Card mandatory lines on every code path | PASS | SYNTHETIC | Headline, typed as, precedents, nearest, `library covers`, disclaimer printed on: typed key, UNTYPED (text and JSON), no-library dry run, refused, asof before library, outcome type with no rows (`card.py:197-225`, `_covers_line`). Paths that print no card exit 2 by design: missing library without `--dry-run`, altered library, typer `ConfigError`. |
| 9 | `fetch_snapshot` reads nothing before `verify_files` | PASS | SYNTHETIC | `snapshot.py:256-270`: only the two manifests are parsed; library files are written byte-for-byte (`_copy_entry`), then `verify_files(temp, manifest)` (`manifest.py` also rejects unlisted files and symlinks), then atomic swap. Probe above. The manifest and zip share one origin, so a replaced pair passes; that is the design's trust root. |
| 10 | No model provider or Qlib import in core modules | PASS | n/a | grep for `anthropic\|openai\|ollama\|qlib\|lightgbm\|scipy\|yaml\|networkx` imports in `engine/`, `libraries/`, `mcp_server.py`, `cli.py`, `config.py`: none. `httpx` appears only in `snapshot.py` (release download). `test_urls_model_ids_and_data_dir_defaults_live_only_in_config` passes. |
| 11 | Nothing under `data/` but `data/sample/` is tracked | PASS | n/a | `git ls-files data` is empty; `.gitignore` has `data/*` and `!data/sample/`. `data/sample/` itself does not exist yet (Task 15, owner gate). |
| 12 | Every `docs/examples/` file has its banner | PASS | n/a | `brief_2024-06-03.md` and `typing_score_rules.md` begin `SYNTHETIC —`; `README.md` is the documented exemption; `test_every_example_file_starts_with_an_evidence_banner` passes. No Precedent card example exists (NO LIVE RUN). |
| 13 | MCP: two read-only tools over stdio, engine only | PASS | SYNTHETIC | `mcp_server.py:99-100`, listing above, `test_server_lists_two_tools_over_stdio`. Minor deviation V-4. |

## top_k before the completion guard: exact behaviour

Yes. `find_precedents` calls `select_precedent_events(..., top_k)` first (`retrieval.py:123`), which keeps the `top_k` most recent matches by `effective_position` (`retrieval.py:94-99`); the completion guard runs afterwards on those rows only (`retrieval.py:135-139`). Finished older analogs are never promoted to fill the gap.

Probe (`topk_probe.py`): key K, 111 first-report events, one every 3 sessions (positions 0..330), event-sequence rows at horizon 20 (complete at `p+20`), query at 30 minutes after the session-330 cutoff, `top_k=30`.

- Selected events: positions 243..330 (30 events).
- Rows surviving the guard: 23 (positions 243..309). The 7 newest selected events are unfinished and absent; none leaked (0 rows with a completion cutoff after the query).
- Finished analogs actually available at that moment: 104 (positions 0..309). 81 finished analogs are discarded because they are older than the 30th most recent match.
- Pre-filtering to finished events before `top_k` yields 30 rows instead of 23.
- Card level (`_precedent_card`): `n=30 independent=5` so `REFUSED 5 usable precedents, need 8`. The same refusal happens even with guard-first selection (independent=5), because 30 events spanning 90 sessions hold only 5 non-overlapping 20-session windows. So for any frequent key at horizon 20, `top_k=30` recency truncation alone forces a refusal while about 15 independent finished windows exist in the library.
- Frequent-key crisis case with a thin band: the 23 surviving rows are the only evidence used; the row-level band uses only those.
- Nothing unfinished enters a band. The defect is loss of usable evidence and refusals, not leakage. Design 8.2 P4 says "keep the top K", but P4 follows the time guard in the design's component order; the code inverts it.

## Findings

**V-1 (Important). Event-sequence library writes false 0.0 follow-on values for events in the last 60 sessions before the end of the feed.**
`libraries/build.py:59,140` extends the session calendar `LOOKAHEAD_DAYS = 120` past the latest record; `event_sequence.py:73` skips a window only when `position + horizon >= len(sessions)`. With 84 calendar sessions beyond the last event, a window that lies wholly or partly after the feed ends is emitted as `value = 0.0` (no follow-on observed) and `response_complete_session` in the future. Concrete input (probe `lookahead_probe.py`): records dated 2021-01-04 and 2021-06-30, one first report linked to USA at 2021-06-30 gives rows `(5, 0.0, 2021-07-08)`, `(20, 0.0, 2021-07-29)`, `(60, 0.0, 2021-09-24)` with `skipped_incomplete_window=0`. Once the snapshot ages past those dates the guard serves these as completed outcomes. This violates Review Focus 2 (absent and counted, never zero) and the spec's feed-outage rule (design pitfall 20). Fix: skip windows that extend past the last observed feed session, not the padded calendar. Also applies when a market feed extends beyond the event feed. Tests use calendars without lookahead, so none catches it.

**V-2 (Important). Header `n` / `independent` and the REFUSED decision count unfinished precedents; the row table does not.**
`ask.py:238-242,255-277` computes `n_eff` from `select_precedent_events`, so events whose windows have not closed are counted. Concrete input (`header_probe.py`): 7 independent events at positions 100..136 plus 1 event in the query's own session, horizon 5, event sequence. Card prints `precedents n=8 independent=8` and is not refused, yet the only row prints `refused: independent=7, need 8`. A reader sees "8 independent" next to a row that says 7. Related: the top_k-before-guard behaviour above (`retrieval.py:123`) drops finished analogs. Fix: compute `n`, `n_eff` and the refusal from the post-guard rows (or apply the guard before `top_k`), per outcome type.

**V-3 (Minor). A card with no row at the chosen horizon prints an empty table silently.**
`ask.py:277`, `card.py:171-184`. Input: `precedent ask --data-dir <library> --event-type central_bank --actor-kind central_bank --severity-band high --unscheduled --country-group US --horizon 1`: the `event_sequence` header prints with zero rows, no refusal, no `NOT BUILT` or "no rows at this horizon" line. Same for `--horizon 60` on asset rows. Indistinguishable from a bug; should say why.

**V-4 (Minor). MCP untyped text without a library returns a tool error, not the refusal card.**
`mcp_server.py:49`, `ask.py:185` (`run_card` loads the bundle before typing). Input: tool `card` with `text="zzzz qqqq"` and no library: `isError`, "no library at ...". Amendment section 6 says an untypable text returns the refusal with `typed_as: UNTYPED` and the key flags. Holds only when a library exists (CLI `--dry-run` handles it separately).

**V-5 (Minor). `precedent ask` and the MCP tools never set `story_cluster_id`.**
`ask.py:560`, `mcp_server.py:44`. A "real event" asked by key as of now can match its own earlier reports (probe: with no story id, the own-story event `d` is returned as a precedent). Only the Python API (`PrecedentQuery.story_cluster_id`) is protected; documentation should say CLI and MCP queries are hypotheticals.

## BLOCKED / NO LIVE RUN (owner-gated, not exercised)

All are BLOCKED / NO LIVE RUN, stated plainly:

1. Font download (`references/fonts/DejaVuSansMono.ttf` absent; `--card` PNG exits 2).
2. Release asset upload and a real `precedent fetch` against GitHub Releases (only mock transports tested).
3. `data/sample/` real GDELT and market slice (absent; Task 15).
4. First real run, milestone P-M1 (Task 16): no live library, no real card.
5. M2 Qlib ablation (Task 17 research rung): not run live.
6. Embedding model download and the B3 rung (Task 18).
7. Typing budget and any model typer call (Anthropic, Ollama): not called; the skipped test is the model-typer module guard.
8. Asset library with real prices: no price data in the repo or run.

Methodology note: the wiki/handoff steps from the verifier's instructions (`agent_os/handoff.md`, wiki log entry) were not performed in this task; the supervisor owns them.
