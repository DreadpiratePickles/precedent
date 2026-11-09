# Precedent Implementation Plan

> Amended 2026-11-07 against the product amendment spec. Supersedes the 2026-11-06 version.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For any typed-in or real event, return the strictly-earlier precedents and what followed them in three outcome libraries (asset response, market resolution, event sequence), as one card (text, JSON, PNG, MCP) that shows every number with its `n`, refuses below `min_neff`, and states what the library covers.

**Architecture:** Two packages on top of the spine. `precedent.libraries` builds three libraries whose rows share one schema; `precedent.engine` retrieves behind one time guard, aggregates (quantiles for continuous outcomes, a rate with a Wilson interval for binary ones), and renders the card. Typing goes through the spine's `Typer` protocol (rules by default, no model in tests or CI). Data arrives either as a verified snapshot (`precedent fetch`) or by rebuilding from the stores (`precedent build`); a small real sample under `data/sample/` runs the offline demo.

**Tech Stack:** As the spine: Python 3.11, uv, pandas, numpy, pydantic 2, pyarrow, httpx, `mcp>=2,<3`, Pillow (core); `pyqlib==0.9.7` and `lightgbm` only in the `research` extra (Tasks 17 and 18). Later, one local embedding model (owner gate).

**Spec:** `docs/superpowers/specs/2026-11-07-precedent-product-amendment.md` (sections 3 to 6, 9, 11), on top of `docs/superpowers/specs/2026-11-06-predate-design.md` section 8. Product context: `docs/2026-11-07-product-thesis.md` sections 3, 4.1 and 6.

**Depends on:** spine plan Tasks 1 to 8, 11 and 12; Task 16 (P-M1) also on spine 5, 8, 9 and 17; Task 17 (factors) also on spine 13 to 15; Task 18 also on spine 7 and 17 to 20. Each task's Interfaces block names the spine items it consumes. Runs in its own worktree; it touches only `src/precedent/engine/`, `src/precedent/libraries/`, `src/precedent/mcp_server.py`, `tests/engine/`, `tests/libraries/`, `tests/test_mcp_server.py`, `stages/2[0-4]_*/`, `scripts/build_sample_library.py`, `scripts/run_precedent_m1.py`, `data/sample/`, `references/precedent_params.json` (its `engine` section), `references/fonts/`, `references/rungs/m2_precedent.yaml`, `docs/STATUS.md` rows, `docs/verification/precedent-*.md`, and in `src/precedent/config.py` only the two accessors `sample_dir()` and `release_base_url()` if the spine has not defined them.

**Parallel groups:** Tasks 1 to 3 serially; then {4, 5, 6}, {7, 8}, {11}; then 9 (after 7, 8), 10 (after 4 to 6); then 12 (after 7, 8, 10, 11), 13 (after 10); then 14 (after 12); 15 to 18 are owner-gated or research; 19 last.

## Global Constraints

- Everything in the spine plan's Global Constraints applies.
- Package `precedent`; this plan's code lives in `precedent.engine.*` and `precedent.libraries.*` (formerly `predate.precedent.*`). Environment variables are `PRECEDENT_*`; this plan reads `PRECEDENT_DATA_DIR` and `PRECEDENT_TYPER` only, always through `precedent.config` (`config.data_dir() -> Path`, default `~/.cache/precedent/`). No URL, model id or data-dir default appears outside `config.py`.
- Data layout under the data dir: `events/` (EventStore), `markets/` (MarketStore), `prices/` (spine Task 11), `library/` (this plan's bundle, Task 10). `data/sample/` mirrors it: `data/sample/README.md` plus `data/sample/library/`.
- CLI dispatch targets the spine's `cli.py` must import lazily: `ask` -> `precedent.engine.ask:run`, `fetch` -> `precedent.engine.snapshot:run`, `evaluate` -> `precedent.engine.evaluate:run`, `build` -> `precedent.libraries.build:run`, `mcp` -> `precedent.mcp_server:run` (these are exactly the targets in spine Task 1's table). Every `run(argv: list[str]) -> int`; the console script `precedent-mcp` still points at `precedent.mcp_server:main`. This plan never edits `cli.py`.
- Parameters change only through `references/precedent_params.json`, section `engine` (the spine owns section `gdelt`). Proposed values: `top_k = 30`, `min_neff = 8`, `shrink_k = 10`, `story_window_sessions = 3`, `nearest_shown = 5`, `card_rows_per_type = 8`, `asset_horizons = [1, 5, 20]`, `sequence_horizons = [5, 20, 60]`, `wilson_z = 1.2816`, `placebo_samples = 500`, `placebo_seed = 0`, `market = "SPY"`, `asset_panel` as design 8.5.
- **Time guard:** a library row serves a query only if the decision cutoff of its `response_complete_session` is at or before the query's `available_utc`; a precedent event serves only if its own `available_utc` is at or before it. Every path to a band goes through `find_precedents`.
- **Timing per library:** asset: effective position `p`, trade opens at `p + 1`, horizon `h` sums `p + 2 .. p + 1 + h`, complete at `sessions[p + 1 + h]`. Event sequence: follow-ons at positions `p + 1 .. p + h`, complete at `sessions[p + h]`. Market resolution: complete on the session on which `resolved_utc` falls (first session whose cutoff is at or after it); `horizon` is that position minus `p`.
- Rows of all three libraries have exactly the columns `event_id, outcome_type, outcome_key, horizon, value, is_binary, effective_session, response_complete_session, source_ref`; `outcome_type` in {`asset`, `market_resolution`, `event_sequence`}; `is_binary` is false for `asset` and true for the other two; binary `value` is 1.0 or 0.0.
- No number without its `n`. Below `min_neff` independent precedents the card refuses and draws no band. Every card ends with the `library covers` line and then `These are past outcomes of similar events, not a forecast.`
- The card is built by code from summaries, never by a model. `engine/`, `libraries/` and `mcp_server.py` import no model provider; they obtain a typer only through `precedent.spine.typers.select.typer_from_env()` or a passed `Typer`.
- Nothing in the repo displays or redistributes vendor prices; the sample ships no asset rows and no market question text (market ids, typed keys, created and resolved dates and resolutions only).
- Tests and CI never open a network socket and never read `data/sample/`; they use synthetic fixtures built in `conftest.py`.
- Run-produced files under `docs/examples/` begin with `SYNTHETIC —`, `LIVE —` or `NO LIVE RUN —`; `docs/STATUS.md` rows use `SYNTHETIC | LIVE-DATA | LIVE-MODEL | NO LIVE RUN`.
- Owner gates (named "owner gate" in the steps): any dataset, font or model download, any API spend, publishing a release asset, any push or merge.

## Review Focus

1. A query whose own story is already in the library: its earlier reports must not come back as precedents. (Test in Task 7.)
2. A precedent near the end of the data whose window has not closed: the row is absent and counted, never zero. (Tests in Tasks 4 and 6.)
3. A market created after the precedent event, or resolved before it, or created at the same instant: it is not a precedent outcome; untyped markets are counted. (Test in Task 5.)
4. Few independent precedents (many analogs from one crisis week): the card prints `REFUSED`, no band, and still prints the `library covers` and disclaimer lines. (Tests in Tasks 8 and 11.)
5. A downloaded snapshot with one tampered byte: refused before any file is parsed, and nothing is left in the data dir. (Test in Task 13.)

## File Structure

```text
src/precedent/engine/
  __init__.py
  keys.py             EventKey re-exported from the spine; key_from_flags, key_from_dict, match_fields, typed_as_line
  params.py           EngineParams from references/precedent_params.json (engine section)
  story_clusters.py   first report against later reports
  retrieval.py        PrecedentQuery, time guard, type filter, ranking
  aggregate.py        effective sample size, continuous and binary summaries, Wilson interval, refusal
  evaluate.py         baseline ladder scoring
  card.py             CardModel, render_text, render_json, render_png
  ask.py              run_card, run (precedent ask)
  snapshot.py         build_snapshot, fetch_snapshot, run (precedent fetch)
  precedent_factors.py              research extra (Task 17)
  canonical_text.py, embedder.py    model-typed rungs (Task 18)
src/precedent/libraries/
  __init__.py
  common.py           shared row schema, validate_library_rows
  events_table.py     events table from EventRecords, one_line
  asset.py            asset response library and asset placebo
  market_resolution.py  market resolution library and unconditional market rate
  event_sequence.py   event sequence library and unconditional sequence rate
  manifest.py         Manifest models, sha256 helpers, LibraryIntegrityError
  bundle.py           LibraryBundle, write_bundle, load_bundle
  build.py            build_libraries, run (precedent build)
  sample.py           build_sample
src/precedent/mcp_server.py         precedent-mcp
scripts/build_sample_library.py, scripts/run_precedent_m1.py
references/precedent_params.json (engine section), references/fonts/ (font + licence), references/rungs/m2_precedent.yaml
stages/20_precedent_libraries/, 21_precedent_evaluate/, 22_precedent_factors/, 23_precedent_card_mcp/, 24_library_snapshot/   CONTEXT.md and output/ each
data/sample/README.md, data/sample/library/
tests/engine/, tests/libraries/ (each with conftest.py of synthetic builders), tests/test_mcp_server.py
docs/verification/precedent-<date>.md
```

Bundle files (Task 10), all under `library/`: `events.parquet`, `sessions.parquet` (`session`, `cutoff_utc`), `rows_asset.parquet`, `rows_market_resolution.parquet`, `rows_event_sequence.parquet`, `placebo_asset.parquet` (`outcome_key, horizon, value, response_complete_session`), `markets.parquet` (`market_id, outcome_key, created_utc, resolved_utc, value, response_complete_session`), `build_report.json`, `MANIFEST.json`.

Synthetic test calendar (both conftests): `SESSIONS = pd.bdate_range("2020-01-02", periods=400)` (dates), `CUTOFFS` = each session at 16:00 America/New_York in UTC, indexed by session.

## Task Cycle

Every task: Step 1 write the listed tests; Step 2 run them and see the stated failure; Step 3 implement the listed signatures (a task writes Step 3 out only when it settles a choice the signatures and tests leave open); Step 4 run the task's tests, then `uv run ruff check . && uv run pytest -q`, all green; Step 5 add a `docs/STATUS.md` row (label `SYNTHETIC` unless stated) and commit the listed files with the given message (explicit `git add` of each file, never `git add .`).

---

### Task 1: Engine skeleton: keys, parameters, stage contracts

**Files:**
- Create: `src/precedent/engine/__init__.py`, `engine/keys.py`, `engine/params.py`, `src/precedent/libraries/__init__.py`, `stages/20_precedent_libraries/CONTEXT.md`, `stages/21_precedent_evaluate/CONTEXT.md`, `stages/22_precedent_factors/CONTEXT.md`, `stages/23_precedent_card_mcp/CONTEXT.md`, `stages/24_library_snapshot/CONTEXT.md`
- Modify: `references/precedent_params.json` (add the `engine` section; create the file with only that section if spine Task 5 has not yet)
- Test: `tests/engine/test_keys.py`, `tests/engine/test_params.py`, `tests/engine/test_stage_contracts.py`

**Interfaces:**
- Consumes: `precedent.spine.event_key.{EventKey, build_event_key, severity_band}` (spine Task 2); the `EventType` and `ActorKind` enums (spine Task 2).
- Produces (`engine/keys.py`): re-exports `EventKey`, `build_event_key`, `severity_band` (same objects, not copies); `key_from_flags(event_type: str, actor_kind: str, severity_band: str, is_scheduled: bool, country_group: str = "none", impacted_kinds: tuple[str, ...] = (), direction_sign: int = 0) -> EventKey` (raises `ValueError("unknown event_type: ...")` and the like for values outside the enums); `key_from_dict(d: dict) -> EventKey` (same fields, unknown field -> `ValueError`); `match_fields(key: EventKey) -> tuple` = `(event_type, actor_kind, country_group, severity_band, is_scheduled)`; `typed_as_line(key: EventKey) -> str`.
- Produces (`engine/params.py`): `EngineParams` (pydantic, frozen, `extra="forbid"`; fields as Global Constraints, horizons as `tuple[int, ...]`); `load_engine_params(path: Path = Path("references/precedent_params.json")) -> EngineParams` reading only the `engine` section.

- [ ] **Step 0:** Read the spine's merged `event_key.py`, `pit_alignment.py`, `event_store.py`, `market_record.py`, `typers/` and `price_provider.py`. If any name this plan consumes differs, use the spine's name and say so in the commit message of the task that consumes it.
- [ ] **Step 1: Write the failing tests**

```python
def test_engine_reexports_the_spine_key():
    from precedent.engine import keys
    from precedent.spine import event_key
    assert keys.EventKey is event_key.EventKey and keys.build_event_key is event_key.build_event_key

def test_key_from_flags_defaults_and_validation():
    k = key_from_flags("central_bank", "central_bank", "high", False, country_group="US")
    assert (k.impacted_kinds, k.direction_sign) == ((), 0)
    with pytest.raises(ValueError, match="unknown event_type"):
        key_from_flags("weather", "government", "high", False)

def test_key_from_dict_rejects_unknown_fields():
    good = {"event_type": "central_bank", "actor_kind": "central_bank", "severity_band": "high", "is_scheduled": False}
    assert key_from_dict(good) == key_from_flags("central_bank", "central_bank", "high", False)
    with pytest.raises(ValueError):
        key_from_dict({**good, "colour": "red"})

def test_typed_as_line():
    k = key_from_flags("central_bank", "central_bank", "high", False, country_group="US")
    assert typed_as_line(k) == "central_bank / central_bank / US / severity high / unscheduled"

def test_engine_params_hold_the_proposed_values():
    p = load_engine_params()
    assert (p.top_k, p.min_neff, p.shrink_k, p.story_window_sessions, p.nearest_shown, p.card_rows_per_type) == (30, 8.0, 10.0, 3, 5, 8)
    assert p.asset_horizons == (1, 5, 20) and p.sequence_horizons == (5, 20, 60)
    assert p.wilson_z == pytest.approx(1.2816) and p.market == "SPY"

def test_unknown_engine_param_is_rejected(tmp_path):
    (tmp_path / "p.json").write_text(json.dumps({"engine": {**ENGINE_JSON, "top_kk": 3}}))
    with pytest.raises(ValidationError):
        load_engine_params(tmp_path / "p.json")

STAGES = ("20_precedent_libraries", "21_precedent_evaluate", "22_precedent_factors", "23_precedent_card_mcp", "24_library_snapshot")

def test_precedent_stage_contracts_are_complete_and_single_pass():
    for stage in STAGES:
        text = Path(f"stages/{stage}/CONTEXT.md").read_text()
        assert [h for h in REQUIRED_HEADINGS if h not in text] == [], stage
        assert "single pass" in text.lower(), stage
```

`ENGINE_JSON` is the committed `engine` section read in the test module; `REQUIRED_HEADINGS` are the seven headings of the spine's contract test.

- [ ] **Step 2: Run** `uv run pytest tests/engine -q`. Expected: FAIL, `ModuleNotFoundError: No module named 'precedent.engine'`.
- [ ] **Step 3: Implement.** Contracts follow the rulebook's stage standard. 20 builds the three libraries from the stores; 21 scores the ladder; 22 builds `prec_` factors; 23 renders cards and serves MCP; 24 builds and verifies the snapshot (publishing it is the owner's step, in Approval). Each says `single pass` with one reason: 20, 22, 23, 24 are deterministic builds with no iteration; 21 is single pass until Task 18 adds its `## Loop`.
- [ ] **Step 4: Run** the same command, then the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): key helpers, engine params, stage contracts 20-24`.

### Task 2: Story clusters

**Files:**
- Create: `src/precedent/engine/story_clusters.py`; Test: `tests/engine/test_story_clusters.py`

**Interfaces:**
- Produces: `assign_story_clusters(events: pd.DataFrame, window_sessions: int = 3) -> pd.DataFrame`. Input columns `event_id, key, primary_actor, effective_position, available_utc`; adds `story_cluster_id: str` (the `event_id` of the cluster's first report) and `is_first_report: bool`. Same `key` and `primary_actor`, at most `window_sessions` positions apart, chained; order by `effective_position`, then `available_utc`, then `event_id`. Input row order is preserved in the output.

- [ ] **Step 1: Write the failing tests**

```python
def test_first_report_is_the_event():
    events = events_frame([("e1", "k1", "ACME", 10), ("e2", "k1", "ACME", 12), ("e3", "k1", "ACME", 20), ("e4", "k2", "ACME", 12)])
    out = assign_story_clusters(events, window_sessions=3)
    assert out["is_first_report"].tolist() == [True, False, True, True]
    assert out["story_cluster_id"].tolist() == ["e1", "e1", "e3", "e4"]

def test_clusters_chain_across_the_window():
    out = assign_story_clusters(events_frame([("a", "k1", "X", 0), ("b", "k1", "X", 3), ("c", "k1", "X", 6)]), window_sessions=3)
    assert out["story_cluster_id"].tolist() == ["a", "a", "a"]
```

`events_frame(rows)` (conftest) builds the input frame with `available_utc` = the row's session cutoff minus one hour.

- [ ] **Step 2: Run** `uv run pytest tests/engine/test_story_clusters.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): story clustering`.

### Task 3: Shared row schema and the events table

**Files:**
- Create: `src/precedent/libraries/common.py`, `libraries/events_table.py`; Test: `tests/libraries/test_common.py`, `tests/libraries/test_events_table.py`

**Interfaces:**
- Consumes: `EventRecord` (spine Task 2, including `linked_countries`), `build_event_key`, `assign_story_clusters` (Task 2).
- Produces (`common.py`): `OUTCOME_TYPES = ("event_sequence", "market_resolution", "asset")` (also the card's display order); `LIBRARY_COLUMNS` (the nine columns, in Global Constraints order); `LibrarySchemaError(ValueError)`; `validate_library_rows(rows: pd.DataFrame) -> pd.DataFrame` (returns the frame with columns in `LIBRARY_COLUMNS` order and dtypes fixed; raises on missing or extra columns, unknown `outcome_type`, `is_binary` not matching the type, a binary value outside {0.0, 1.0}, a NaN value, `response_complete_session < effective_session`); `empty_library_rows() -> pd.DataFrame`.
- Produces (`events_table.py`): `build_events_table(records: list[EventRecord], sessions: pd.DatetimeIndex, window_sessions: int = 3) -> tuple[pd.DataFrame, EventsTableReport]` with columns `event_id` (= `article_id`), `key` (`EventKey.as_string()`), `event_type`, `country_group`, `available_utc`, `effective_session`, `effective_position`, `primary_actor`, `linked_countries` (list of ISO3), `story_cluster_id`, `is_first_report`, `one_line`, `source`, `source_ref` (= `article_id`). `primary_actor` is the first resolved impact's instrument if any, else `linked_countries[0]` if any, else `event.actor.name`. `EventsTableReport(records: int, kept: int, unaligned: int)`; a record without `effective_session` is skipped and counted. `one_line(record: EventRecord) -> str` = `f"{event.actor.name}: {event.action}"` cut to 60 characters.

- [ ] **Step 1: Write the failing tests**

```python
def test_valid_rows_pass_and_are_reordered():
    out = validate_library_rows(rows_frame([ROW_SEQ, ROW_ASSET])[list(reversed(LIBRARY_COLUMNS))])
    assert list(out.columns) == list(LIBRARY_COLUMNS)

@pytest.mark.parametrize("bad", [
    {"outcome_type": "weather"}, {"outcome_type": "asset", "is_binary": True},
    {"value": 0.5}, {"value": float("nan")},
    {"response_complete_session": date(2020, 1, 1)},          # before effective_session 2020-01-10
])
def test_invalid_rows_raise(bad):
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(rows_frame([{**ROW_SEQ, **bad}]))

def test_extra_column_raises():
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(rows_frame([ROW_SEQ]).assign(extra=1))

def test_events_table_skips_and_counts_unaligned_records():
    table, report = build_events_table([gdelt_record("g1", position=10), gdelt_record("g2", position=None)], SESSIONS)
    assert table["event_id"].tolist() == ["g1"] and (report.kept, report.unaligned) == (1, 1)

def test_primary_actor_prefers_the_instrument_then_the_country():
    table, _ = build_events_table([edgar_record("x:2.02", instrument="ACME", linked=["USA"]), gdelt_record("g1", linked=["MLI", "FRA"])], SESSIONS)
    assert table["primary_actor"].tolist() == ["ACME", "MLI"]

def test_one_line_is_actor_and_action_cut_to_60():
    assert one_line(gdelt_record("g1", actor="MALI", action="x" * 80)) == ("MALI: " + "x" * 80)[:60]
```

`ROW_SEQ` is an `event_sequence` row (`horizon` 5, `value` 1.0, `is_binary` True, `effective_session` 2020-01-10, `response_complete_session` 2020-01-17); `ROW_ASSET` an `asset` row with `value` 0.012. `gdelt_record` and `edgar_record` (conftest) build valid `EventRecord`s aligned to `SESSIONS[position]`.

- [ ] **Step 2: Run** `uv run pytest tests/libraries -q`. Expected: FAIL, `ModuleNotFoundError: No module named 'precedent.libraries.common'`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(libraries): shared row schema and events table`.

### Task 4: Asset response library

**Files:**
- Create: `src/precedent/libraries/asset.py`; Test: `tests/libraries/test_asset.py`

**Interfaces:**
- Consumes: `cumulative_abnormal_return(asset_returns, market_returns, event_pos, horizon, est_window=(-250, -21), min_obs=120) -> float` (spine Task 12; called with `event_pos = p + 1`); open-to-open returns as a frame indexed by session, one column per symbol (spine Task 11 loader).
- Produces: `build_asset_library(events: pd.DataFrame, returns: pd.DataFrame, market: str, sessions: pd.DatetimeIndex, assets_for_event: Callable[[pd.Series], list[str]], horizons: tuple[int, ...] = (1, 5, 20)) -> tuple[pd.DataFrame, AssetLibraryReport]`. Only first reports are used. Rows pass `validate_library_rows`; `outcome_key` = symbol, `value` = CAR. `AssetLibraryReport(rows: int, skipped_incomplete_window: int, skipped_short_history: int)`.
- Produces: `sample_asset_placebo(asset_returns: pd.Series, market_returns: pd.Series, sessions: pd.DatetimeIndex, horizon: int, n_samples: int, rng: np.random.Generator, exclude_positions: set[int]) -> pd.DataFrame` with columns `value, response_complete_session` (CARs at random non-event positions, same timing rule).

- [ ] **Step 1: Write the failing tests**

```python
def test_response_window_starts_after_the_trade_open(asset_inputs):
    # one event effective at position 300; asset = market (seeded noise) plus +3% at position 302
    library, _ = build_asset_library(**asset_inputs)
    row = library[library.horizon == 1].iloc[0]
    assert row.value == pytest.approx(0.03, abs=1e-9)
    assert row.response_complete_session == asset_inputs["sessions"][302]
    assert (row.outcome_type, row.is_binary) == ("asset", False)

def test_unfinished_windows_are_absent_and_counted(asset_inputs_near_end):   # event 3 sessions before the last
    library, report = build_asset_library(**asset_inputs_near_end)
    assert set(library.horizon) == {1} and report.skipped_incomplete_window == 2

def test_short_history_assets_are_counted_not_zeroed(asset_inputs_new_fund):
    library, report = build_asset_library(**asset_inputs_new_fund)
    assert "NEWFUND" not in set(library.outcome_key) and report.skipped_short_history == 3

def test_placebo_avoids_event_positions_and_is_seeded(asset_inputs):
    a = sample_asset_placebo(*placebo_args(asset_inputs), horizon=5, n_samples=50, rng=np.random.default_rng(0), exclude_positions={300})
    b = sample_asset_placebo(*placebo_args(asset_inputs), horizon=5, n_samples=50, rng=np.random.default_rng(0), exclude_positions={300})
    pd.testing.assert_frame_equal(a, b)
    assert len(a) == 50
```

- [ ] **Step 2: Run** `uv run pytest tests/libraries/test_asset.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(libraries): asset response library`.

### Task 5: Market resolution library

**Files:**
- Create: `src/precedent/libraries/market_resolution.py`; Test: `tests/libraries/test_market_resolution.py`

**Interfaces:**
- Consumes: `MarketRecord` (spine Task 8: `market_id, venue, question, created_utc, close_utc, resolved_utc, resolution, category, typed: TypedText, source_ref`), `TypedText` (spine Task 6), `assign_effective_session` semantics (spine Task 3).
- Produces: `build_market_resolution_library(events: pd.DataFrame, markets: list[MarketRecord], sessions: pd.DatetimeIndex, session_cutoffs_utc: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, MarketLibraryReport]` returning the rows, the `markets.parquet` frame (all typed markets, for the placebo) and the report. For each first-report event at `t = available_utc`, a market qualifies when `created_utc < t < resolved_utc` and `typed.status == "typed"`. `outcome_key = typed.key.as_string()`, `value` 1.0 for `yes`, `source_ref = market_id`. One row per `(event_id, outcome_key)`: when several qualify, the one resolving first (ties by `market_id`) is kept and the rest counted. `MarketLibraryReport(markets_read: int, markets_untyped: int, markets_refused_invalid: int, rows: int, collapsed_duplicates: int, skipped_no_session: int)` (`skipped_no_session`: resolved after the last session).
- Produces: `unconditional_market_rate(markets: pd.DataFrame, outcome_key: str, cutoff_utc: pd.Timestamp, session_cutoffs_utc: pd.Series) -> float | None`: the YES rate of markets with that key whose resolution session's cutoff is at or before `cutoff_utc`; `None` when there are none.

- [ ] **Step 1: Write the failing tests**

```python
# conftest: event E effective at position 10, t = CUTOFFS[10] - 2h; key K1 typed markets unless stated
def test_only_markets_open_at_the_event_qualify():
    rows, _, report = build_market_resolution_library(EVENTS_E, [
        market("polymarket:yes", created=T_E - pd.Timedelta("1D"), resolved=CUTOFFS.iloc[20] - pd.Timedelta("1h"), resolution="yes"),
        market("polymarket:late", created=T_E + pd.Timedelta("1h"), resolved=CUTOFFS.iloc[25]),
        market("polymarket:early", created=T_E - pd.Timedelta("5D"), resolved=T_E - pd.Timedelta("1h")),
        market("polymarket:same", created=T_E, resolved=CUTOFFS.iloc[25]),
        market("kalshi:untyped", created=T_E - pd.Timedelta("1D"), resolved=CUTOFFS.iloc[25], typed=UNTYPED),
    ], SESSIONS, CUTOFFS)
    assert rows["source_ref"].tolist() == ["polymarket:yes"]
    row = rows.iloc[0]
    assert (row.value, row.is_binary, row.horizon) == (1.0, True, 10)
    assert row.response_complete_session == SESSIONS[20] and row.outcome_key == K1.as_string()
    assert report.markets_untyped == 1

def test_several_markets_with_one_key_collapse_to_the_first_resolving():
    rows, _, report = build_market_resolution_library(EVENTS_E, [MARKET_YES_AT_20, MARKET_NO_AT_30], SESSIONS, CUTOFFS)
    assert rows["source_ref"].tolist() == ["polymarket:yes"] and report.collapsed_duplicates == 1

def test_unconditional_rate_respects_the_cutoff():
    frame = markets_frame([("K1", "yes", 20), ("K1", "yes", 21), ("K1", "no", 22), ("K1", "yes", 23), ("K1", "yes", 90)])
    assert unconditional_market_rate(frame, K1.as_string(), CUTOFFS.iloc[23], CUTOFFS) == pytest.approx(0.75)
    assert unconditional_market_rate(frame, K1.as_string(), CUTOFFS.iloc[19], CUTOFFS) is None
```

- [ ] **Step 2: Run** `uv run pytest tests/libraries/test_market_resolution.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(libraries): market resolution library`.

### Task 6: Event sequence library

**Files:**
- Create: `src/precedent/libraries/event_sequence.py`; Test: `tests/libraries/test_event_sequence.py`

**Interfaces:**
- Consumes: the events table (Task 3); `references/country_groups.json` (spine Task 2) as `dict[str, str]` ISO3 -> group.
- Produces: `build_event_sequence_library(events: pd.DataFrame, sessions: pd.DatetimeIndex, event_types: tuple[str, ...], horizons: tuple[int, ...] = (5, 20, 60)) -> tuple[pd.DataFrame, SequenceLibraryReport]`. For each first report `e` at position `p` with actor country `c = e.linked_countries[0]`, each `T` in `event_types` and each `h`: `value` 1.0 if a first report of type `T`, from a different story, with `c in linked_countries`, has `effective_position` in `p + 1 .. p + h`, else 0.0. `outcome_key = f"{T}/linked_country"`, `source_ref` = the precedent's `source_ref`, `response_complete_session = sessions[p + h]`. `SequenceLibraryReport(rows: int, skipped_incomplete_window: int, skipped_no_country: int)`; `skipped_incomplete_window` counts each `(e, T, h)` with `p + h` past the last session.
- Produces: `unconditional_sequence_rate(events: pd.DataFrame, event_type: str, country_group: str, horizon: int, end_position: int, country_groups: dict[str, str]) -> float | None`: over every country `c` in the group that appears in any `linked_countries`, and every start `s` in `0 .. end_position - horizon`, the share of `(c, s)` with a first report of `event_type` linked to `c` at a position in `s + 1 .. s + horizon`; `None` when the group has no country.

- [ ] **Step 1: Write the failing tests**

```python
def test_follow_on_events_are_counted_by_horizon():
    events = table([
        ("p", "armed_conflict", ["MLI", "FRA"], 10),
        ("f1", "trade_or_sanctions", ["FRA", "MLI"], 13),
        ("f2", "trade_or_sanctions", ["USA"], 12),           # not linked to MLI
        ("f3", "trade_or_sanctions", ["MLI"], 10),           # same session, not after
        ("f4", "election_or_political", ["MLI"], 40),
    ])
    rows, _ = build_event_sequence_library(events, SESSIONS, ("trade_or_sanctions", "election_or_political"))
    got = rows[rows.event_id == "p"].set_index(["outcome_key", "horizon"])["value"].to_dict()
    assert got == {("trade_or_sanctions/linked_country", 5): 1.0, ("trade_or_sanctions/linked_country", 20): 1.0,
                   ("trade_or_sanctions/linked_country", 60): 1.0, ("election_or_political/linked_country", 5): 0.0,
                   ("election_or_political/linked_country", 20): 0.0, ("election_or_political/linked_country", 60): 1.0}
    assert rows[(rows.event_id == "p") & (rows.horizon == 20)]["response_complete_session"].iloc[0] == SESSIONS[30]

def test_windows_past_the_data_are_absent_and_counted():
    events = table([("p", "armed_conflict", ["MLI"], len(SESSIONS) - 30)])
    rows, report = build_event_sequence_library(events, SESSIONS, ("trade_or_sanctions", "election_or_political"))
    assert set(rows.horizon) == {5, 20} and report.skipped_incomplete_window == 2

def test_later_reports_of_a_story_are_not_follow_ons():
    rows, _ = build_event_sequence_library(table([("p", "armed_conflict", ["MLI"], 10), ("u", "trade_or_sanctions", ["MLI"], 12, False)]),
                                           SESSIONS, ("trade_or_sanctions",))
    assert rows[rows.event_id == "p"]["value"].max() == 0.0

def test_unconditional_rate_hand_value():
    events = table([("x", "trade_or_sanctions", ["MLI"], 5)])
    assert unconditional_sequence_rate(events, "trade_or_sanctions", "EM", 2, 9, {"MLI": "EM"}) == pytest.approx(0.25)   # s in 0..7; hits at s = 3, 4
    assert unconditional_sequence_rate(events, "trade_or_sanctions", "DM", 2, 9, {"MLI": "EM"}) is None
```

`table(rows)` (conftest) builds an events table; a fifth tuple element sets `is_first_report` (default True, own story).

- [ ] **Step 2: Run** `uv run pytest tests/libraries/test_event_sequence.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(libraries): event sequence library`.

### Task 7: Retrieval with the time guard

**Files:**
- Create: `src/precedent/engine/retrieval.py`; Test: `tests/engine/test_retrieval.py`

**Interfaces:**
- Produces: `PrecedentQuery` (frozen dataclass): `key: EventKey`, `available_utc: pd.Timestamp`, `story_cluster_id: str | None = None`, `canonical_text: str | None = None`.
- Produces: `exact_key_similarity(query: PrecedentQuery, events: pd.DataFrame) -> pd.Series`: 1.0 where `match_fields` are equal and, when `query.key.direction_sign != 0`, the direction sign is equal too; else 0.0.
- Produces: `select_precedent_events(query: PrecedentQuery, events: pd.DataFrame, top_k: int = 30, similarity: Callable[[PrecedentQuery, pd.DataFrame], pd.Series] = exact_key_similarity) -> pd.DataFrame` (the events table plus `similarity`). The unguarded pool of look-alikes, not what a card shows. Fixed order: event `available_utc <= query.available_utc`, first reports only, drop `query.story_cluster_id`, similarity, drop 0.0, top K by similarity then latest `effective_position` then `event_id`.
- Produces: `find_precedents(query: PrecedentQuery, events: pd.DataFrame, rows: pd.DataFrame, session_cutoffs_utc: pd.Series, horizon: int, outcome_type: str, top_k: int = 30, similarity=exact_key_similarity) -> pd.DataFrame` with `LIBRARY_COLUMNS` plus `similarity`: REVISED by ruling P-42 (the original text selected `top_k` events first and then guarded their rows, which left frequent keys with empty or refused bands). Guard first, then cut: candidates are look-alike events with a completed row of `outcome_type` (and `horizon`, except for `market_resolution`, whose rows carry their own horizons) by `query.available_utc` (`response_complete_session` cutoff at or before it); slots are counted per instrument for asset rows; selection takes events by similarity, then latest `effective_position`, skipping an event whose outcome window overlaps one already accepted for that instrument (starts at least one horizon apart), until `top_k`.

- [ ] **Step 1: Write the failing tests**

```python
def test_an_unfinished_analog_is_never_returned():
    # analog A completes on session 110, analog B on session 130; the query is available just after session 120's cutoff
    found = find_precedents(QUERY_AFTER_120, EVENTS_AB, ROWS_AB, CUTOFFS, horizon=5, outcome_type="event_sequence")
    assert found["event_id"].unique().tolist() == ["A"]

def test_time_guard_holds_for_random_libraries():
    for seed in range(50):
        query, events, rows = random_query_and_library(seed)
        for otype in OUTCOME_TYPES:
            found = find_precedents(query, events, rows, CUTOFFS, horizon=5, outcome_type=otype)
            assert (CUTOFFS.loc[found["response_complete_session"]].to_numpy() <= query.available_utc).all()

def test_a_story_is_not_its_own_precedent():
    found = find_precedents(QUERY_IN_STORY_S1, EVENTS_WITH_S1, ROWS_WITH_S1, CUTOFFS, horizon=5, outcome_type="event_sequence")
    assert "S1" not in set(found.merge(EVENTS_WITH_S1, on="event_id")["story_cluster_id"])

def test_exact_similarity_uses_direction_only_when_the_query_sets_it():
    events = events_with_keys([KEY_DOWN, KEY_UP, KEY_OTHER_COUNTRY])
    assert exact_key_similarity(query_for(KEY_ANY_DIRECTION), events).tolist() == [1.0, 1.0, 0.0]
    assert exact_key_similarity(query_for(KEY_DOWN), events).tolist() == [1.0, 0.0, 0.0]

def test_top_k_keeps_the_latest_matches():
    picked = select_precedent_events(QUERY_LATE, forty_matching_events(), top_k=30)
    assert len(picked) == 30 and picked["effective_position"].min() == 10      # positions 0..39; the 30 latest

def test_market_rows_keep_their_own_horizons():
    found = find_precedents(QUERY_LATE, EVENTS_M, ROWS_M_HORIZONS_3_AND_40, CUTOFFS, horizon=5, outcome_type="market_resolution")
    assert sorted(found["horizon"]) == [3, 40]
```

- [ ] **Step 2: Run** `uv run pytest tests/engine/test_retrieval.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): retrieval behind a time guard`.

### Task 8: Aggregation: continuous, binary, refusal

**Files:**
- Create: `src/precedent/engine/aggregate.py`; Test: `tests/engine/test_aggregate.py`

**Interfaces:**
- Produces: `effective_sample_size(weights: np.ndarray, window_start_positions: np.ndarray, horizon: int) -> float`: the smaller of the Kish size `(sum w)^2 / sum w^2` and the count of non-overlapping windows (greedy in position order, starts at least `horizon` apart); 0.0 for empty input.
- Produces: `wilson_interval(k: float, n: float, z: float = 1.2816) -> tuple[float, float]`, clipped to [0, 1]; `ValueError` for `n <= 0`.
- Produces: `PrecedentSummary` (frozen dataclass): `mode: Literal["continuous", "binary"]`, `n: int`, `n_eff: float`, `refused: bool`, `quantiles: dict[float, float] | None` (0.1, 0.25, 0.5, 0.75, 0.9, weighted), `median: float | None`, `centre_shrunk: float | None` (`median * n_eff / (n_eff + shrink_k)`), `hit_rate: float | None` (share of values > 0), `k: int | None` (unweighted count of 1.0), `rate: float | None` (weighted mean), `interval: tuple[float, float] | None` (`wilson_interval(rate * n_eff, n_eff, z)`), `placebo_rate: float | None`, `inside_ordinary_range: bool | None`.
- Produces: `summarize_responses(values: np.ndarray, weights: np.ndarray, window_start_positions: np.ndarray, horizon: int, placebo: np.ndarray | float | None, *, is_binary: bool = False, shrink_k: float = 10.0, min_neff: float = 8.0, z: float = 1.2816) -> PrecedentSummary`. `refused = n_eff < min_neff`; a refused summary keeps `n`, `n_eff`, `k` and leaves every band field `None`. Continuous: `inside_ordinary_range` when the median lies within the placebo's 10th to 90th percentile. Binary: `placebo` is the unconditional rate; inside when it lies within `interval`. `placebo=None` gives `inside_ordinary_range=None`.

- [ ] **Step 1: Write the failing tests**

```python
def test_separate_windows_keep_full_sample_size_and_shrink_as_specified():
    s = summarize_responses(np.full(12, 0.02), np.ones(12), np.arange(12) * 10, horizon=5, placebo=PLACEBO)
    assert s.n_eff == pytest.approx(12.0) and s.refused is False
    assert s.centre_shrunk == pytest.approx(0.02 * 12 / 22)

def test_overlapping_windows_collapse_sample_size_and_refuse():
    s = summarize_responses(np.full(12, 0.02), np.ones(12), np.arange(12), horizon=5, placebo=PLACEBO)
    assert s.n_eff == pytest.approx(3.0) and s.refused is True and s.quantiles is None   # starts 0, 5, 10

def test_a_median_inside_the_placebo_range_is_flagged():
    s = summarize_responses(np.full(20, 0.01), np.ones(20), np.arange(20) * 10, 5, np.linspace(-0.05, 0.05, 101))
    assert s.inside_ordinary_range is True

def test_wilson_hand_values():
    assert wilson_interval(14, 23) == pytest.approx((0.475243, 0.727658), abs=1e-6)
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.141078), abs=1e-6)
    assert wilson_interval(10, 10) == pytest.approx((0.858922, 1.0), abs=1e-6)

def test_binary_summary_rate_interval_and_placebo():
    values = np.array([1.0] * 9 + [0.0] * 3)
    s = summarize_responses(values, np.ones(12), np.arange(12) * 100, horizon=20, placebo=0.2, is_binary=True)
    assert (s.mode, s.k, s.n, s.rate) == ("binary", 9, 12, pytest.approx(0.75))
    assert s.interval == pytest.approx((0.566669, 0.873133), abs=1e-6)
    assert s.inside_ordinary_range is False and s.quantiles is None
    assert summarize_responses(values, np.ones(12), np.arange(12) * 100, 20, 0.7, is_binary=True).inside_ordinary_range is True

def test_binary_refusal_keeps_counts_and_drops_the_interval():
    s = summarize_responses(np.ones(5), np.ones(5), np.arange(5) * 100, 20, 0.5, is_binary=True)
    assert s.refused is True and (s.k, s.n) == (5, 5) and s.interval is None
```

- [ ] **Step 2: Run** `uv run pytest tests/engine/test_aggregate.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): continuous and binary aggregation, Wilson interval, refusal`.

### Task 9: Baseline ladder scoring

**Files:**
- Create: `src/precedent/engine/evaluate.py`; Test: `tests/engine/test_evaluate.py`

**Interfaces:**
- Consumes: `select_precedent_events`, `find_precedents` (Task 7), `summarize_responses` (Task 8).
- Produces: `pinball_loss(realized: np.ndarray, predicted_quantile: np.ndarray, q: float) -> float`; `brier_score(realized: np.ndarray, predicted_rate: np.ndarray) -> float`.
- Produces: `score_precedent_ladder(events: pd.DataFrame, rows: pd.DataFrame, session_cutoffs_utc: pd.Series, outcome_type: str, rungs: tuple[str, ...] = ("B0", "B1"), horizon: int = 5, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)) -> pd.DataFrame` with columns `rung, outcome_type, event_type, quantile, pinball_loss, coverage_80, brier, n_scored, n_refused` (continuous types fill `quantile, pinball_loss, coverage_80`; binary types fill `brier`, the others NaN). Events are scored in time order, each from earlier events only, through `find_precedents`. B0 = all earlier rows of the outcome key regardless of type (the unconditional distribution or rate); B1 = `exact_key_similarity`.
- Produces: `run(argv: list[str]) -> int` for `precedent evaluate [--data-dir PATH] [--horizon 5]`: loads the bundle under the data dir, scores B0 and B1 per outcome type with `score_precedent_ladder` and prints the table; 0 on success, 2 when no library is present (stderr names `precedent build`).

- [ ] **Step 1: Write the failing tests**

```python
def test_loss_hand_values():
    assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)
    assert pinball_loss(np.array([0.0]), np.array([1.0]), 0.9) == pytest.approx(0.1)
    assert brier_score(np.array([1.0, 0.0]), np.array([0.8, 0.4])) == pytest.approx(0.10)

def test_typed_rung_beats_untyped_when_types_differ(separable_asset_events):    # type A followed by +2%, type B by -2%
    s = score_precedent_ladder(**separable_asset_events, outcome_type="asset").query("quantile == 0.5").groupby("rung")["pinball_loss"].mean()
    assert s["B1"] < s["B0"]

def test_typed_rung_beats_untyped_on_binary_outcomes(separable_sequence_events):  # type A always followed, type B never
    s = score_precedent_ladder(**separable_sequence_events, outcome_type="event_sequence").groupby("rung")["brier"].mean()
    assert s["B1"] < s["B0"]

def test_coverage_is_near_nominal_on_stationary_data(stationary_asset_events):
    c = score_precedent_ladder(**stationary_asset_events, outcome_type="asset").query("rung == 'B1'")["coverage_80"].mean()
    assert 0.7 <= c <= 0.9

def test_evaluate_command_exits_2_without_a_library(tmp_path, capsys):
    assert run(["--data-dir", str(tmp_path)]) == 2 and "precedent build" in capsys.readouterr().err
```

- [ ] **Step 2: Run** `uv run pytest tests/engine/test_evaluate.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): time-ordered ladder scoring for all outcome types`.

### Task 10: Library bundle, manifest and `precedent build`

**Files:**
- Create: `src/precedent/libraries/manifest.py`, `libraries/bundle.py`, `libraries/build.py`; Test: `tests/libraries/test_manifest.py`, `tests/libraries/test_bundle.py`, `tests/libraries/test_build.py`

**Interfaces:**
- Consumes: `EventStore(root).read(parser_version)` (spine Task 4), `MarketStore(root).read(...)` (spine Task 8), `build_session_cutoffs` (spine Task 3), the spine Task 11 price loader (absent prices are normal), Tasks 3 to 6.
- Produces (`manifest.py`): `ManifestFile(path: str, sha256: str, rows: int | None)`; `ManifestSource(name: str, first_date: date, last_date: date, rows: int, citation: str)`; `Manifest(files: list[ManifestFile], sources: list[ManifestSource], built_from_commit: str (40 lowercase hex, or `unknown` outside a git checkout), precedent_version: str, created_utc: datetime (tz-aware))`, all pydantic `extra="forbid"`; `sha256_file(path: Path) -> str`; `verify_files(root: Path, manifest: Manifest) -> None` raising `LibraryIntegrityError(path, reason)` on a missing file, a sha mismatch or a file present under `root` but not listed (other than `MANIFEST.json`).
- Produces (`bundle.py`): `LibraryBundle` (frozen dataclass): `events, rows, placebo_asset, markets: pd.DataFrame`, `sessions: pd.DatetimeIndex`, `session_cutoffs_utc: pd.Series`, `manifest: Manifest`; `write_bundle(dest: Path, events, rows, placebo_asset, markets, sessions, session_cutoffs_utc, build_report: dict, sources: list[ManifestSource], commit: str) -> Manifest` (rows split by type into the three parquet files, each through `validate_library_rows`; write to a sibling temp dir, then rename); `load_bundle(library_dir: Path) -> LibraryBundle` (raises `LibraryMissingError` when `MANIFEST.json` is absent; calls `verify_files` before reading any parquet).
- Produces (`build.py`): `BuildReport` (pydantic): `events: EventsTableReport`, `asset: AssetLibraryReport | None`, `asset_not_built_reason: str | None`, `market_resolution: MarketLibraryReport`, `event_sequence: SequenceLibraryReport`; `build_libraries(data_dir: Path, params: EngineParams, commit: str, parser_versions: tuple[str, ...] = ("rules-gdelt-v1", "rules-8k-v1")) -> BuildReport` writing `data_dir / "library"`; `run(argv: list[str]) -> int` for `precedent build [--data-dir PATH]`: prints the report's counts (kept, dropped, untyped) and returns 0; 2 when the event store is empty. With no prices, the asset library is not built and `asset_not_built_reason == "no prices"`. Event types for the sequence library are those present among the first reports, minus `no_market_event`. `commit` comes from `git rev-parse HEAD` in `run` (argument array, no shell), or `unknown` when git or the checkout is absent.

- [ ] **Step 1: Write the failing tests**

```python
def test_bundle_round_trips_and_lists_every_file(tmp_path, bundle_parts):
    manifest = write_bundle(tmp_path / "library", **bundle_parts)
    loaded = load_bundle(tmp_path / "library")
    pd.testing.assert_frame_equal(loaded.rows.sort_values(SORT).reset_index(drop=True), validate_library_rows(bundle_parts["rows"]).sort_values(SORT).reset_index(drop=True))
    assert sorted(f.path for f in manifest.files) == sorted(BUNDLE_FILES)

def test_a_tampered_file_is_refused_before_it_is_parsed(tmp_path, bundle_parts):
    write_bundle(tmp_path / "library", **bundle_parts)
    (tmp_path / "library" / "rows_event_sequence.parquet").write_bytes(b"not parquet")
    with pytest.raises(LibraryIntegrityError, match="rows_event_sequence.parquet"):
        load_bundle(tmp_path / "library")

def test_an_unlisted_file_is_refused(tmp_path, bundle_parts):
    write_bundle(tmp_path / "library", **bundle_parts)
    (tmp_path / "library" / "extra.parquet").write_bytes(b"x")
    with pytest.raises(LibraryIntegrityError, match="extra.parquet"):
        load_bundle(tmp_path / "library")

def test_missing_library_is_its_own_error(tmp_path):
    with pytest.raises(LibraryMissingError):
        load_bundle(tmp_path / "nothing")

def test_manifest_rejects_a_short_commit():
    with pytest.raises(ValidationError):
        Manifest(**{**MANIFEST_DICT, "built_from_commit": "abc123"})

def test_build_without_prices_builds_two_libraries_and_counts_untyped(tmp_path, populated_stores):
    report = build_libraries(populated_stores, PARAMS, commit="0" * 40)
    assert report.asset is None and report.asset_not_built_reason == "no prices"
    assert report.market_resolution.markets_untyped == 2 and report.event_sequence.rows > 0
    bundle = load_bundle(populated_stores / "library")
    assert set(bundle.rows["outcome_type"]) == {"market_resolution", "event_sequence"}
    assert json.loads((populated_stores / "library" / "build_report.json").read_text())["market_resolution"]["markets_untyped"] == 2
```

`bundle_parts` (`tests/libraries/conftest.py`) holds synthetic events across `SESSIONS` with rows of all three types; `bundle_root` is a data dir with those parts written by `write_bundle` (reused in Task 15). `populated_stores` writes 30 synthetic GDELT-style records and 6 markets (2 untyped) into an EventStore and MarketStore under `tmp_path` and returns that data dir.

- [ ] **Step 2: Run** `uv run pytest tests/libraries/test_manifest.py tests/libraries/test_bundle.py tests/libraries/test_build.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** them and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(libraries): verified library bundle and precedent build`.

### Task 11: The card: model, text, JSON, PNG

**Files:**
- Create: `src/precedent/engine/card.py`; Test: `tests/engine/test_card.py`
- Create (owner gate, Step 6): `references/fonts/DejaVuSansMono.ttf`, `references/fonts/LICENSE`

**Interfaces:**
- Produces: `DISCLAIMER = "These are past outcomes of similar events, not a forecast."`; `KEY_FLAGS_HINT = "could not type this text; ask by key instead: --event-type <type> --actor-kind <kind> --severity-band low|mid|high --scheduled|--unscheduled [--country-group US|DM|EM|none] [--direction-sign -1|0|1]"`; `SOURCE_UNITS = {"polymarket": "markets", "kalshi": "markets"}` (any other source: `events`).
- Produces (pydantic, frozen, `extra="forbid"`): `CardRow(outcome_type, outcome_key: str, horizon: int | None (None for market rows, printed "res"), is_binary: bool, n: int, n_eff: float, refused: bool, k: int | None, rate: float | None, interval: tuple[float, float] | None, median: float | None, centre_shrunk: float | None, band: tuple[float, float] | None (p10, p90), vs_ordinary: Literal["inside", "outside"] | None)`; `NearestPrecedent(date: date, one_line: str, similarity: float, source_ref: str)`; `SourceCoverage(name: str, first_date: date, last_date: date, rows: int)`; `CardModel(schema_version: Literal["1.0"] = "1.0", headline: str, typed_as: str ("UNTYPED" or `typed_as_line`), typed_reason: str | None, key: str | None, asof_utc: datetime, horizon: int, outcome_type: str, n: int, n_eff: float, earliest: date | None, latest: date | None, refused: bool, min_neff: float, rows: list[CardRow], not_built: dict[str, str] (outcome type -> reason), nearest: list[NearestPrecedent], covers: list[SourceCoverage], hint: str | None, disclaimer: Literal[DISCLAIMER] = DISCLAIMER)`.
- Produces: `format_outcome_row(row: CardRow, min_neff: float) -> str`; `render_text(card: CardModel) -> str`; `render_json(card: CardModel) -> str` (`model_dump_json(indent=2)`); `render_png(card: CardModel, path: Path, font_path: Path | None = DEFAULT_FONT_PATH) -> Path` (`DEFAULT_FONT_PATH = Path("references/fonts/DejaVuSansMono.ttf")`; `None` uses Pillow's built-in default font; black text on white, 16 px padding, one text line per card line; deterministic bytes).

Layout rules (amendment section 3; exact strings below are the oracle): header lines `PRECEDENT  ·  {headline}`, `typed as       {typed_as}`, `precedents     n={n}  independent={floor(n_eff)}  earliest {earliest}  latest {latest}` (`-` for a missing date), then a blank line. If `refused`: `REFUSED        {floor(n_eff)} usable precedents, need {min_neff:g}. No band is drawn below that.` then straight to `nearest precedents`. Otherwise, for each outcome type in `OUTCOME_TYPES` order that was asked for: `f"{otype:<36}NOT BUILT: {reason}"` when in `not_built`, else the header `f"{otype:<36}horizon   rate|median   80% band        n   vs ordinary"` and up to `card_rows_per_type` rows; then a blank line and `nearest precedents`. Row: `f"{key:<36}{h:>7}   {centre:<14}{band:<13}{n:>4}   {vs}"`, where binary `centre = f"{k}/{n} ({rate:.0%})"`, `band = f"[{lo:.0%}, {hi:.0%}]"`, continuous `centre = f"{median:+.1%}"`, `band = f"[{p10:+.1%}, {p90:+.1%}]"`; a refused row is `f"{key:<36}{h:>7}   refused: independent={floor(n_eff)}, need {min_neff:g}"`; overlong values push the line right, nothing is truncated. Nearest: `f"{date}  {one_line}  similarity {s:.2f}  {source_ref}"`, or `(none)`. Then `library covers  ` + `"; ".join(f"{name}: {first} to {last}, {rows} {unit}")`, or `library covers  nothing (no library loaded)`. Last line `DISCLAIMER`.

- [ ] **Step 1: Write the failing tests**

```python
EXPECTED = "\n".join([
    "PRECEDENT  ·  a central bank raises rates unexpectedly",
    "typed as       central_bank / central_bank / US / severity high / unscheduled",
    "precedents     n=12  independent=12  earliest 2016-12-14  latest 2023-07-26",
    "",
    "event_sequence                      horizon   rate|median   80% band        n   vs ordinary",
    "election_or_political/linked_country     20   9/12 (75%)    [57%, 87%]     12   outside",
    "market_resolution                   horizon   rate|median   80% band        n   vs ordinary",
    "central_bank/scheduled/US               res   refused: independent=5, need 8",
    "asset                               NOT BUILT: no prices",
    "",
    "nearest precedents",
    "2023-07-26  Federal Reserve raises policy rate  similarity 1.00  gdelt:1112223334",
    "library covers  gdelt2: 2015-02-19 to 2024-03-31, 41233 events; kalshi: 2021-07-01 to 2024-03-31, 812 markets",
    "These are past outcomes of similar events, not a forecast.",
])

REFUSED_EXPECTED = "\n".join([
    "PRECEDENT  ·  a cyber attack on a mid-size bank",
    "typed as       cyber_incident / company / US / severity high / unscheduled",
    "precedents     n=3  independent=3  earliest 2019-03-04  latest 2023-11-09",
    "",
    "REFUSED        3 usable precedents, need 8. No band is drawn below that.",
    "nearest precedents",
    "2023-11-09  8-K item 1.05 cybersecurity incident  similarity 1.00  0000950170-23-061234:1.05",
    "library covers  edgar_8k: 2015-01-02 to 2024-03-29, 9120 events",
    "These are past outcomes of similar events, not a forecast.",
])

def test_card_text_is_exact(golden_card):
    assert render_text(golden_card) == EXPECTED

def test_refused_card_text_is_exact_and_draws_no_band(refused_card):
    text = render_text(refused_card)
    assert text == REFUSED_EXPECTED and "80% band" not in text and "%" not in text.split("REFUSED")[1].split("library covers")[0]

def test_continuous_row_format():
    row = CardRow(outcome_type="asset", outcome_key="SPY", horizon=5, is_binary=False, n=19, n_eff=19.0, refused=False, k=None, rate=None,
                  interval=None, median=-0.008, centre_shrunk=-0.0052, band=(-0.031, 0.014), vs_ordinary="inside")
    assert format_outcome_row(row, 8.0) == "SPY                                       5   -0.8%         [-3.1%, +1.4%]  19   inside"

def test_every_card_ends_with_covers_then_disclaimer(golden_card, refused_card, untyped_card):
    for card in (golden_card, refused_card, untyped_card):
        lines = render_text(card).splitlines()
        assert lines[-2].startswith("library covers  ") and lines[-1] == DISCLAIMER

def test_untyped_card_says_so(untyped_card):
    assert "typed as       UNTYPED" in render_text(untyped_card) and untyped_card.hint == KEY_FLAGS_HINT

def test_disclaimer_cannot_be_changed(golden_card):
    with pytest.raises(ValidationError):
        CardModel(**{**golden_card.model_dump(), "disclaimer": "trust me"})

def test_json_round_trips_with_schema_version(golden_card):
    data = json.loads(render_json(golden_card))
    assert data["schema_version"] == "1.0" and data["disclaimer"] == DISCLAIMER
    assert CardModel.model_validate_json(render_json(golden_card)) == golden_card

def test_png_is_written_and_deterministic(tmp_path, golden_card):
    a = render_png(golden_card, tmp_path / "a.png", font_path=None).read_bytes()
    b = render_png(golden_card, tmp_path / "b.png", font_path=None).read_bytes()
    assert a[:8] == b"\x89PNG\r\n\x1a\n" and a == b
    assert len(Image.open(tmp_path / "a.png").getcolors(maxcolors=1_000_000)) > 1

def test_png_with_a_missing_font_is_a_config_error(tmp_path, golden_card):
    with pytest.raises(ConfigError, match="card font not bundled"):
        render_png(golden_card, tmp_path / "a.png", font_path=tmp_path / "missing.ttf")
```

`golden_card`, `refused_card` and `untyped_card` (conftest) build `CardModel`s with exactly the values printed above (the market row is a refused binary row with `n=5, n_eff=5.0, horizon=None`; the untyped card has `n=0, n_eff=0.0, refused=True, typed_as="UNTYPED"`, no nearest precedents).

- [ ] **Step 2: Run** `uv run pytest tests/engine/test_card.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): the card in text, JSON and PNG`.
- [ ] **Step 6: Owner gate.** Ask to download DejaVu Sans Mono (`DejaVuSansMono.ttf`, DejaVu Fonts licence, from the dejavu-fonts GitHub release; state the file size at the time of asking). On yes, commit the font and its licence text: `chore: bundle the card font`. Until then `render_png` with the default path raises `ConfigError("card font not bundled: <path>")` (the test above) and `precedent ask --card` exits 2 with that message.

### Task 12: `precedent ask` and `run_card`

**Files:**
- Create: `src/precedent/engine/ask.py`; Test: `tests/engine/test_ask.py`

**Interfaces:**
- Consumes: `Typer`, `TypedText` (spine Task 6), `typer_from_env()` and `RulesTyper` (spine Task 6), `FakeTyper(mapping)` (spine Task 6, tests), `ConfigError` (spine Task 1 `config.py`), `config.data_dir()`, `config.sample_dir() -> Path` (spine Task 1 `config.py`: the repository's `data/sample/` in a checkout, else `data_dir()`), Tasks 7, 8, 10, 11.
- Produces: `run_card(query: PrecedentQuery | str, horizon: int, outcome_type: str, data_dir: Path, typer: Typer) -> CardModel`. `outcome_type` in `("all",) + OUTCOME_TYPES`; `horizon` in `asset_horizons ∪ sequence_horizons`, else `ValueError`. A `str` is typed with `typer.type_text`; `untyped` or `refused_invalid` gives the untyped card (`typed_as="UNTYPED"`, `typed_reason` from the typer, `hint=KEY_FLAGS_HINT`, `refused=True`); a typed text becomes `PrecedentQuery(key, available_utc=now UTC)`. Loads `load_bundle(data_dir / "library")`; when absent raises `LibraryMissingError`. Event-level `n`, `n_eff` come from `select_precedent_events` and `effective_sample_size` at `horizon`; below `min_neff` the card is refused with `rows=[]`. Otherwise one `CardRow` per `(outcome_type, outcome_key)` from `find_precedents` and `summarize_responses` with the placebo per type (asset: `placebo_asset` rows under the time guard; market: `unconditional_market_rate`; sequence: `unconditional_sequence_rate` with `end_position` = last session whose cutoff is at or before `available_utc`). A type with no rows in the library goes to `not_built` (`asset`: `"no prices"`; others: `"no library rows"`). `covers` from `manifest.sources`; `nearest` = first `nearest_shown` selected events.
- Produces: `run(argv: list[str]) -> int` for `precedent ask [TEXT] [--event-type T --actor-kind A --severity-band B (--scheduled|--unscheduled) [--country-group G] [--direction-sign S]] [--asof DATE] [--horizon 5] [--outcome-type all] [--json] [--card PATH.png] [--data-dir PATH] [--dry-run]`. When `--card PATH.png` is under `docs/examples/`, `run` also writes `PATH.md` (same stem) beside it: line 1 the evidence banner (`SYNTHETIC —` when the library is `data/sample/` or a synthetic fixture, else `LIVE —`), line 2 the query and data dir (spine rule: a PNG carries its banner in a same-stem `.md` sidecar). `--asof` takes an ISO date (meaning 23:59:59 UTC that day) or datetime; default now. `--dry-run` uses `RulesTyper()` and `config.sample_dir()` (the checkout's `data/sample/`, or the library under `$PRECEDENT_DATA_DIR`; a pip user with no checkout runs `precedent fetch` first), opens no socket, and when the sample is absent prints the refused card with `library covers  nothing (no library loaded)` and returns 0. Exit codes: 0 for any card, including a refusal or `UNTYPED` (then `KEY_FLAGS_HINT` goes to stderr); 2 for could-not-run: bad flags, both or neither of text and key, `LibraryMissingError` outside dry-run (stderr names `precedent fetch`, `precedent build` and `--dry-run`), `LibraryIntegrityError`, `ConfigError` from a typer (stderr also prints `KEY_FLAGS_HINT`).

- [ ] **Step 1: Write the failing tests**

```python
KEY_FLAGS = ["--event-type", "central_bank", "--actor-kind", "central_bank", "--severity-band", "high", "--unscheduled", "--country-group", "US"]

def test_key_query_prints_card_and_disclaimer(capsys, library_root):
    assert run(KEY_FLAGS + ["--asof", "2024-06-03", "--data-dir", str(library_root)]) == 0
    out = capsys.readouterr().out
    assert out.rstrip().endswith(DISCLAIMER) and "independent=" in out and "event_sequence" in out

def test_too_few_precedents_is_a_refusal_not_a_band(capsys, sparse_library_root):
    assert run(["--event-type", "cyber_incident", "--actor-kind", "company", "--severity-band", "high", "--unscheduled", "--data-dir", str(sparse_library_root)]) == 0
    out = capsys.readouterr().out
    assert "REFUSED" in out and "80% band" not in out

def test_untyped_text_is_a_refusal_with_the_key_flags(capsys, library_root, monkeypatch):
    monkeypatch.setenv("PRECEDENT_TYPER", "rules")
    assert run(["zzzz qqqq", "--data-dir", str(library_root)]) == 0
    captured = capsys.readouterr()
    assert "typed as       UNTYPED" in captured.out and "--event-type" in captured.err

def test_typer_config_error_exits_2(capsys, library_root, monkeypatch):
    monkeypatch.setattr(ask, "typer_from_env", raising_config_error("ANTHROPIC_API_KEY is not set"))
    assert run(["a central bank raises rates unexpectedly", "--data-dir", str(library_root)]) == 2
    err = capsys.readouterr().err
    assert "ANTHROPIC_API_KEY" in err and "--event-type" in err

def test_missing_library_exits_2_and_says_what_to_do(capsys, tmp_path):
    assert run(KEY_FLAGS + ["--data-dir", str(tmp_path)]) == 2
    assert "precedent fetch" in capsys.readouterr().err

def test_dry_run_reads_the_sample_and_opens_no_socket(capsys, monkeypatch, library_root):
    monkeypatch.setattr(config, "sample_dir", lambda: library_root)          # a synthetic data dir, never data/sample
    monkeypatch.setattr(socket, "socket", forbid_socket)
    assert run(["a central bank raises rates unexpectedly", "--dry-run"]) == 0
    assert capsys.readouterr().out.rstrip().endswith(DISCLAIMER)

def test_dry_run_without_a_sample_still_renders_a_refusal(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "sample_dir", lambda: tmp_path)
    assert run(KEY_FLAGS + ["--dry-run"]) == 0
    assert "library covers  nothing (no library loaded)" in capsys.readouterr().out

def test_json_flag_emits_the_card_object(capsys, library_root):
    assert run(KEY_FLAGS + ["--json", "--data-dir", str(library_root)]) == 0
    assert json.loads(capsys.readouterr().out)["schema_version"] == "1.0"

def test_run_card_with_a_fake_typer_matches_the_key_query(library_root):
    typer = FakeTyper({"rates up": typed(CENTRAL_BANK_US_HIGH_KEY)})
    by_text = run_card("rates up", 5, "all", library_root, typer)
    by_key = run_card(PrecedentQuery(CENTRAL_BANK_US_HIGH_KEY, by_text.asof_utc), 5, "all", library_root, typer)
    assert (by_text.n, by_text.rows) == (by_key.n, by_key.rows) and by_text.headline == "rates up"

def test_query_before_any_completion_has_no_precedents(library_root):
    card = run_card(PrecedentQuery(CENTRAL_BANK_US_HIGH_KEY, pd.Timestamp("2000-01-01", tz="UTC")), 5, "all", library_root, FakeTyper({}))
    assert card.n == 0 and card.refused is True

def test_asset_without_prices_is_not_built(library_root):
    assert run_card(PrecedentQuery(CENTRAL_BANK_US_HIGH_KEY, NOW), 5, "all", library_root, FakeTyper({})).not_built == {"asset": "no prices"}
```

`library_root` (`tests/engine/conftest.py`) is a data dir whose `library/` is written by `write_bundle` from synthetic events (20 independent `central_bank / US / high` events in 2020 and 2021, with sequence and market rows and no asset rows); `sparse_library_root` holds 3 `cyber_incident / company / US / high` events. Tasks 13 and 14 reuse both.

- [ ] **Step 2: Run** `uv run pytest tests/engine/test_ask.py -q`. Expected: FAIL, `ModuleNotFoundError: No module named 'precedent.engine.ask'`.
- [ ] **Step 4: Run** it, the suite, and `uv run precedent ask --dry-run --event-type central_bank --actor-kind central_bank --severity-band high --unscheduled`. Expected: PASS; the command prints a card (a refusal while the sample is absent) and exits 0.
- [ ] **Step 5: Commit** `feat(engine): precedent ask and run_card`.

### Task 13: Snapshot and `precedent fetch`

**Files:**
- Create: `src/precedent/engine/snapshot.py`; Test: `tests/engine/test_snapshot.py`

**Interfaces:**
- Consumes: `Manifest`, `verify_files`, `LibraryIntegrityError`, `load_bundle` (Task 10); `config.release_base_url() -> str | None` (owner sets it at publish time; `None` until then).
- Produces: `snapshot_name(tag: str) -> str` = `f"precedent-library-{tag}.zip"`; `build_snapshot(library_dir: Path, out_dir: Path, tag: str) -> Path` (stdlib `zipfile`, entries sorted, fixed timestamp 1980-01-01, `MANIFEST.json` inside and copied beside the zip); `fetch_snapshot(tag: str, dest: Path, transport: httpx.BaseTransport, base_url: str) -> Manifest`: GETs `{base_url}/{tag}/MANIFEST.json` and `{base_url}/{tag}/{snapshot_name(tag)}`, extracts into a temp dir beside `dest`, rejects absolute or `..` entry names, requires the inner `MANIFEST.json` to equal the downloaded one, runs `verify_files`, and only then renames the temp dir to `dest / "library"` (an existing one is replaced only after verification). Errors: `SnapshotFetchError` (HTTP status other than 200, timeout), `LibraryIntegrityError`. `run(argv: list[str]) -> int` for `precedent fetch [--version TAG] [--data-dir PATH]`: 0 on success (prints the manifest's sources); 2 on any error or when `release_base_url()` is `None` (`no release repository configured`).

- [ ] **Step 1: Write the failing tests**

```python
def test_fetch_installs_a_verified_snapshot(tmp_path, snapshot_server):
    manifest = fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)
    assert load_bundle(tmp_path / "library").manifest == manifest

def test_a_tampered_byte_is_refused_and_nothing_is_installed(tmp_path, snapshot_server):
    snapshot_server.flip_byte("rows_event_sequence.parquet")
    with pytest.raises(LibraryIntegrityError, match="rows_event_sequence.parquet"):
        fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)
    assert sorted(p.name for p in tmp_path.iterdir()) == []

def test_a_path_escaping_entry_is_refused(tmp_path, snapshot_server):
    snapshot_server.add_entry("../evil.txt", b"x")
    with pytest.raises(LibraryIntegrityError):
        fetch_snapshot("v0.1.0", tmp_path / "d", snapshot_server.transport, BASE)
    assert not (tmp_path / "evil.txt").exists()

def test_inner_and_outer_manifests_must_agree(tmp_path, snapshot_server):
    snapshot_server.replace_outer_manifest(rows_changed=True)
    with pytest.raises(LibraryIntegrityError, match="MANIFEST"):
        fetch_snapshot("v0.1.0", tmp_path, snapshot_server.transport, BASE)

def test_missing_release_is_a_fetch_error(tmp_path, snapshot_server):
    with pytest.raises(SnapshotFetchError):
        fetch_snapshot("v9.9.9", tmp_path, snapshot_server.transport, BASE)

def test_build_snapshot_is_byte_identical_across_runs(tmp_path, library_root):
    a = build_snapshot(library_root / "library", tmp_path / "a", "v0.1.0").read_bytes()
    b = build_snapshot(library_root / "library", tmp_path / "b", "v0.1.0").read_bytes()
    assert a == b

def test_fetch_without_a_configured_release_exits_2(capsys, monkeypatch):
    monkeypatch.setattr(config, "release_base_url", lambda: None)
    assert run(["--version", "v0.1.0"]) == 2 and "no release repository configured" in capsys.readouterr().err
```

`snapshot_server` (conftest) builds a snapshot from `library_root` with `build_snapshot` and serves it through `httpx.MockTransport`; its helpers rewrite the served bytes.

- [ ] **Step 2: Run** `uv run pytest tests/engine/test_snapshot.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): verified snapshot build and precedent fetch`. Publishing a release asset is the owner's step (stage 24 Approval).

### Task 14: MCP server

**Files:**
- Create: `src/precedent/mcp_server.py`; Test: `tests/test_mcp_server.py`

**Interfaces:**
- Consumes: `run_card`, `render_text`, `KEY_FLAGS_HINT`, `key_from_dict`, `typer_from_env`, `config.data_dir()`.
- Produces: `server = MCPServer("precedent")` (`from mcp.server.mcpserver import MCPServer`); plain module functions `precedents(text: str | None = None, key: dict | None = None, horizon: int = 5, outcome_type: str = "all") -> dict` (the card's JSON as a dict) and `card(text: str | None = None, key: dict | None = None, horizon: int = 5) -> str` (the text card; an untyped card appends `"\n" + KEY_FLAGS_HINT`), registered with `server.tool()(precedents)` and `server.tool()(card)` so the module names stay callable; both raise `ValueError` unless exactly one of `text` and `key` is given. `main() -> None` calls `server.run()` (stdio default); `run(argv: list[str]) -> int` (the target of the spine's `mcp` command) calls `main()` and returns 0; the module ends with `if __name__ == "__main__": main()`. Console script `precedent-mcp` (spine Task 1). Read-only: no tool writes anything.

- [ ] **Step 0:** `uv run python -c "from mcp.server.mcpserver import MCPServer"` must succeed under the locked `mcp` version. If it does not, stop and report; do not substitute `FastMCP`.
- [ ] **Step 1: Write the failing tests**

```python
def test_precedents_returns_card_json(monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    data = mcp_server.precedents(key=CENTRAL_BANK_US_HIGH_DICT, horizon=5)
    assert data["schema_version"] == "1.0" and data["disclaimer"] == DISCLAIMER

def test_untyped_text_returns_the_refusal_and_the_key_flags(monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    monkeypatch.setenv("PRECEDENT_TYPER", "rules")
    text = mcp_server.card(text="zzzz qqqq")
    assert "typed as       UNTYPED" in text and "--event-type" in text

@pytest.mark.parametrize("kwargs", [{}, {"text": "x", "key": CENTRAL_BANK_US_HIGH_DICT}, {"key": {**CENTRAL_BANK_US_HIGH_DICT, "colour": "red"}}])
def test_bad_arguments_raise(kwargs, monkeypatch, library_root):
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(library_root))
    with pytest.raises(ValueError):
        mcp_server.precedents(**kwargs)

def test_server_lists_two_tools_over_stdio(tmp_path):
    proc = subprocess.Popen([sys.executable, "-m", "precedent.mcp_server"], stdin=PIPE, stdout=PIPE, text=True,
                            env={**os.environ, "PRECEDENT_DATA_DIR": str(tmp_path)})
    send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}}})
    read_response(proc, 1, timeout_s=20)
    send(proc, {"jsonrpc": "2.0", "method": "notifications/initialized"})
    send(proc, {"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    names = sorted(t["name"] for t in read_response(proc, 2, timeout_s=20)["result"]["tools"])
    proc.terminate()
    assert names == ["card", "precedents"]

def test_core_modules_import_no_provider_and_no_research_extra():
    for path in [*Path("src/precedent/engine").glob("*.py"), *Path("src/precedent/libraries").glob("*.py"), Path("src/precedent/mcp_server.py")]:
        if path.name in {"precedent_factors.py", "embedder.py"}:
            continue
        text = path.read_text()
        for banned in ("anthropic_typer", "ollama_typer", "api.anthropic.com", "import qlib", "import lightgbm"):
            assert banned not in text, (path, banned)
```

`send` writes one JSON line and flushes; `read_response` reads lines until the given `id` arrives or the timeout passes (then fails).

- [ ] **Step 2: Run** `uv run pytest tests/test_mcp_server.py -q`. Expected: FAIL, `ModuleNotFoundError: No module named 'precedent.mcp_server'`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat: precedent-mcp server with precedents and card tools`.

### Task 15: Committed sample library (owner gate on the download)

**Files:**
- Create: `src/precedent/libraries/sample.py`, `scripts/build_sample_library.py`; Test: `tests/libraries/test_sample.py`
- Create after the owner gate: `data/sample/README.md`, `data/sample/library/*`; Modify: `docs/STATUS.md`

**Interfaces:**
- Consumes: `load_bundle`, `write_bundle` (Task 10).
- Produces: `SampleTooLargeError(ValueError)`; `build_sample(library_dir: Path, out_dir: Path, since: date, until: date, evidence_label: Literal["LIVE", "SYNTHETIC"], citation: str, max_bytes: int = 2_000_000) -> Manifest`: keeps events with `effective_session` in `[since, until]` and their market and sequence rows, drops all asset rows and `placebo_asset` (no prices ship), trims `markets` and `sessions` to the window (`markets` keeps only `market_id, outcome_key, created_utc, resolved_utc, value, response_complete_session`; no question text, venue title or description is ever copied), writes `out_dir / "library"` with `write_bundle`, then `out_dir / "README.md"` whose line 1 is `f"{evidence_label} — GDELT 2.0 events and resolved Polymarket/Kalshi market outcomes (ids, typed keys and dates; no question text), {since} to {until}"` followed by the citation and the manifest's source lines. Over `max_bytes` (sum of library files) raises before anything is left in `out_dir`. `scripts/build_sample_library.py --data-dir PATH --since DATE --until DATE --out data/sample` calls it with `evidence_label="LIVE"`.

- [ ] **Step 1: Write the failing tests**

```python
def test_sample_keeps_the_window_and_ships_no_prices(tmp_path, bundle_root):
    build_sample(bundle_root / "library", tmp_path / "s", date(2021, 1, 1), date(2021, 2, 28), "SYNTHETIC", CITATION)
    bundle = load_bundle(tmp_path / "s" / "library")
    assert bundle.events["effective_session"].between(date(2021, 1, 1), date(2021, 2, 28)).all()
    assert bundle.rows.query("outcome_type == 'asset'").empty and bundle.placebo_asset.empty

def test_sample_ships_no_question_text(tmp_path, bundle_root):
    build_sample(bundle_root / "library", tmp_path / "s", date(2021, 1, 1), date(2021, 2, 28), "SYNTHETIC", CITATION)
    markets = load_bundle(tmp_path / "s" / "library").markets
    assert list(markets.columns) == ["market_id", "outcome_key", "created_utc", "resolved_utc", "value", "response_complete_session"]

def test_readme_banner_and_citation(tmp_path, bundle_root):
    build_sample(bundle_root / "library", tmp_path / "s", date(2021, 1, 1), date(2021, 2, 28), "SYNTHETIC", CITATION)
    lines = (tmp_path / "s" / "README.md").read_text().splitlines()
    assert lines[0].startswith("SYNTHETIC — GDELT 2.0 events") and CITATION in lines

def test_oversize_sample_is_refused_and_leaves_nothing(tmp_path, bundle_root):
    with pytest.raises(SampleTooLargeError):
        build_sample(bundle_root / "library", tmp_path / "s", date(2020, 1, 1), date(2022, 12, 31), "SYNTHETIC", CITATION, max_bytes=1000)
    assert not (tmp_path / "s" / "library").exists()
```

- [ ] **Step 2: Run** `uv run pytest tests/libraries/test_sample.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS. **Commit** `feat(libraries): sample library builder`.
- [ ] **Step 5: Owner gate.** Propose the two-month window (default 2024-07-01 to 2024-08-31) and state the download: the GDELT master file list, then every export file in the window, total bytes summed from the list's `<bytes>` column (printed by `precedent ingest gdelt --since --until` before it downloads), plus bounded Polymarket and Kalshi reads (`--max-requests`). Quote the GDELT citation requirement from gdeltproject.org/about.html. Wait for the owner's yes.
- [ ] **Step 6:** On yes: `uv run precedent ingest gdelt --since 2024-07-01 --until 2024-08-31`, `uv run precedent ingest polymarket --max-requests 500`, `uv run precedent ingest kalshi --max-requests 500`, `uv run precedent build`, `uv run python scripts/build_sample_library.py --since 2024-07-01 --until 2024-08-31 --out data/sample`. Expected: exit 0 each; `du -sk data/sample/library` under 2000; README line 1 starts `LIVE —`. Run `uv run precedent ask --dry-run "..."` on three headlines and record which refuse.
- [ ] **Step 7:** Owner reviews size, citation and contents (amendment B5; the sample stores market ids, typed keys, created and resolved dates and resolutions but no question text, so the owner confirms the venues' terms allow redistributing those facts, and the GDELT citation is printed). On yes, commit `data/sample/` and a `LIVE-DATA` status row: `data: two-month GDELT and market sample library`.

### Task 16: First real run (milestone P-M1, owner gate)

**Files:**
- Create: `scripts/run_precedent_m1.py`; output under `stages/21_precedent_evaluate/output/<run_id>/` (not committed); Modify: `docs/STATUS.md`

**Interfaces:**
- Consumes: spine live paths (Tasks 5, 8, 9, 17 and, if the owner imports prices, 11), Tasks 9 and 10.

- [ ] **Step 1: Owner gate.** Name the downloads (GDELT range and bytes, market request budgets, EDGAR CIK list) and wait for the yes.
- [ ] **Step 2:** Build the libraries with `precedent build` from `rules-gdelt-v1` and `rules-8k-v1` events and the typed market store; build the asset library only if prices were imported (`precedent prices import <csv>`).
- [ ] **Step 3:** `uv run python scripts/run_precedent_m1.py --run-id <id>` scores B0 and B1 per outcome type at its horizons, holding out the most recent year. Expected: exit 0 and `report.md` in the output dir.
- [ ] **Step 4:** `report.md` holds pinball loss and coverage (asset), Brier (binary types) per rung and event type, refusal counts, every `BuildReport` count including untyped markets, and the ten event types where B1 improves most and least on B0. Label `LIVE-DATA`, plus `EXPLORATORY (survivorship-biased prices)` for asset rows.
- [ ] **Step 5:** Commit the script and the status row: `feat(engine): P-M1 run script and result line`. Stop for the owner's review. Success criterion P2's coverage band (70% to 90%) is checked here for B1 asset rows.

### Task 17: Precedent factors for Qlib (`research` extra)

**Files:**
- Create: `src/precedent/engine/precedent_factors.py`, `references/rungs/m2_precedent.yaml`; Test: `tests/engine/test_precedent_factors.py`

**Interfaces:**
- Consumes: `make_alpha_handler` and `run_ablation` (spine Tasks 13 and 14), leak checks (spine Task 15). Qlib is imported inside functions only.
- Produces: `build_precedent_factor_panel(events: pd.DataFrame, rows: pd.DataFrame, session_cutoffs_utc: pd.Series, sessions: pd.DatetimeIndex, horizon: int = 5, max_age_sessions: int = 5) -> pd.DataFrame` indexed (datetime, instrument) with columns `prec_car5_med, prec_car5_iqr, prec_neff, prec_hit_rate, prec_placebo_z`, from `asset` rows only. A row exists while an event for that instrument is at most `max_age_sessions` old. Refused summaries give `prec_neff` and zeros elsewhere.

- [ ] **Step 1: Write the failing tests**

```python
def test_precedent_panel_passes_truncation_invariance(panel_inputs):
    for cut in CUT_SESSIONS:
        full = build_precedent_factor_panel(**panel_inputs).loc[:cut]
        truncated = build_precedent_factor_panel(**truncate_inputs(panel_inputs, cut)).loc[:cut]
        pd.testing.assert_frame_equal(full, truncated)

def test_every_column_carries_the_prefix(panel_inputs):
    assert all(c.startswith("prec_") for c in build_precedent_factor_panel(**panel_inputs).columns)
```

`truncate_inputs` removes events after the cut and rows whose `response_complete_session` is after the cut.

- [ ] **Step 2: Run** `uv run pytest tests/engine/test_precedent_factors.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** it and the suite. Expected: PASS.
- [ ] **Step 5: Commit** `feat(engine): precedent factors and M2 rung`.
- [ ] **Step 6 (owner gate, needs imported prices):** run the ablation with rung M2 on real data; append the result to the P-M1 report; status row `LIVE-DATA`.

### Task 18: Model-typed and embedded rungs (owner gate on spend and download)

**Files:**
- Create: `src/precedent/engine/canonical_text.py`, `src/precedent/engine/embedder.py`; Modify: `engine/evaluate.py` (rungs B2 and B3), `stages/21_precedent_evaluate/CONTEXT.md` (`## Loop`)
- Test: `tests/engine/test_canonical_text.py`, `tests/engine/test_embedder.py`

**Interfaces:**
- Consumes: model typers (spine Task 7) for market questions; model-typed `NewsEvent`s (spine Tasks 17 to 20).
- Produces: `canonical_text(record: EventRecord) -> str`, one line built only from enum values and role tokens, no entity name. `TextEmbedder` (Protocol): `embed(texts: list[str]) -> np.ndarray`, `release_date: date` (recorded in `precedent_version`). `embedding_similarity(embedder: TextEmbedder, weight: float) -> Callable[[PrecedentQuery, pd.DataFrame], pd.Series]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_canonical_text_contains_no_entity_names():
    text = canonical_text(record_about("Acme Corp", impacted=["Globex Inc"]))
    assert "Acme" not in text and "Globex" not in text and "[COMPANY]" in text

def test_similarity_uses_the_embedder_through_the_guarded_path(fake_embedder):
    found = find_precedents(QUERY_WITH_TEXT, EVENTS_WITH_TEXT, ROWS_WITH_TEXT, CUTOFFS, horizon=5, outcome_type="event_sequence",
                            similarity=embedding_similarity(fake_embedder, weight=0.5))
    assert (CUTOFFS.loc[found["response_complete_session"]].to_numpy() <= QUERY_WITH_TEXT.available_utc).all()
```

- [ ] **Step 2: Run** `uv run pytest tests/engine/test_canonical_text.py tests/engine/test_embedder.py -q`. Expected: FAIL, `ModuleNotFoundError`.
- [ ] **Step 4: Run** them and the suite. Expected: PASS. Commit `feat(engine): canonical text and embedding similarity seam`.
- [ ] **Step 5: Owner gate.** Name the embedding model, its licence, size and release date (candidates checked on 2026-11-06: `Qwen/Qwen3-Embedding-0.6B`, Apache-2.0; `BAAI/bge-m3`, MIT), and the API budget for typing market questions with the Anthropic typer (`max_calls`). All embedding candidates were released after January 2024, so earlier events may be in their training data; B2, which uses no embedding, is the control.
- [ ] **Step 6:** Score B2 and B3 against B1 on the held-out year. Tuning the similarity weight is a ratchet: oracle `score_precedent_ladder` on development events, held-out check the most recent year, budget 10 iterations (proposed), no-progress stop after 3, mutable file `references/precedent_params.json` (`engine` section) only. Write that `## Loop` contract before running it.
- [ ] **Step 7:** Report against success criterion P2: if model-typed precedents do not beat rule-typed ones on pinball loss and Brier, say so and keep B1 and `RulesTyper` as the defaults. Commit `feat(engine): model-typed and embedded rungs` with a `LIVE-MODEL` status row.

### Task 19: Independent verification

**Files:**
- Create: `docs/verification/precedent-<date>.md`

- [ ] **Step 1:** Dispatch a verifier that did not write this track (fresh context, given only the amendment spec, design section 8, this plan and the branch diff; told to assume the work is broken).
- [ ] **Step 2:** The verifier runs, and records with exit codes: `uv sync --frozen --all-extras`, `uv run ruff check .`, `uv run pytest -q`, `uv run precedent ask --dry-run --event-type central_bank --actor-kind central_bank --severity-band high --unscheduled`, `uv run precedent ask --dry-run "zzzz qqqq"` (expects `UNTYPED`, exit 0), `uv run precedent ask --data-dir /nonexistent --event-type central_bank --actor-kind central_bank --severity-band high --unscheduled` (expects exit 2), and the MCP stdio tool listing.
- [ ] **Step 3:** It checks by reading code and running probes: each Review Focus item; that no path builds a band without `find_precedents`; the card's mandatory lines on every code path (including errors that still print a card); `fetch_snapshot` reads nothing before `verify_files`; no model provider or Qlib import in core modules; nothing under `data/` but `data/sample/` is tracked; every `docs/examples/` file has its banner.
- [ ] **Step 4:** It writes `docs/verification/precedent-<date>.md`: `PASS`, `FAIL` or `BLOCKED` per check, with commands and results. Commit `docs: independent verification of the Precedent track`.
- [ ] **Step 5:** Fix each finding test-first in `fix: independent-review findings` commits, re-run the failed checks, and append the re-run results to the report. Merging is the owner's step.
