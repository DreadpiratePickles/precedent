# Precedent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For any real or typed-in event, return the most similar strictly-earlier events and the distribution of what assets did afterwards, with honest bands, a refusal when precedents are too few, and factors Qlib can test.

**Architecture:** A package `predate.precedent` on top of the spine. A typed event key and a response library (abnormal returns per event, asset and horizon) are built by code. Retrieval applies a time guard before anything else. The first milestone uses rule-typed events only, so it needs no language model; model-typed events and embeddings are added last and must beat the rules.

**Tech Stack:** As the spine. Later, one local embedding model (owner-approved download).

**Spec:** `docs/superpowers/specs/2026-11-06-predate-design.md`, section 8.

**Depends on:** spine plan Tasks 1 to 11 (`EventStore`, `pit_alignment`, `price_provider`, `abnormal_returns`, `make_alpha_handler`, `run_ablation`). Runs in its own worktree alongside the Ripple plan; it touches only `src/predate/precedent/`, `tests/precedent/`, `stages/2*_precedent_*/` and `references/rungs/m2_precedent.yaml`.

## Global Constraints

- Everything in the spine plan's Global Constraints applies.
- **Time guard:** an analog serves a query only if the decision cutoff of its `response_complete_session` is at or before the query's `available_utc`. Every retrieval path goes through `find_precedents`; nothing reads the response library directly to build a band.
- **Return timing:** responses use open-to-open returns indexed by the session on which they became known. For an event with effective session at position `p`, the trade opens at `p + 1`, the response of horizon `h` sums positions `p + 2` to `p + 1 + h`, and `response_complete_session = sessions[p + 1 + h]`.
- Horizons are 1, 5 and 20 sessions. Published evidence suggests hard news (earnings, guidance) drifts for weeks while soft news reverses, so results are always reported per event type.
- Proposed values, changeable only through `references/precedent_params.json`: `top_k = 30`, `min_neff = 8`, `shrink_k = 10`, story window 3 sessions.
- No number is ever shown without its sample size. Below `min_neff` the tool refuses to draw a band.
- Output always carries the line `These are past outcomes of similar events, not a forecast.`

## Review Focus

1. A query whose own story is already in the library: its earlier reports must not come back as precedents. (Test in Task 4.)
2. An event near the end of the data, whose 20-session window has not closed: the row is absent, not zero. (Test in Task 2.)
3. An asset with too little history for the market model (a fund launched recently): the response is NaN and counted, and the asset is left out of that event's band. (Test in Task 2.)
4. Many analogs from one crisis week: effective sample size collapses and the tool refuses. (Test in Task 5.)
5. A typed-in what-if when no model runtime is configured: exit code 2 with a message naming the key flags that work without a model. (Test in Task 8.)

## File Structure

```text
src/predate/precedent/
  event_key.py        typed key from an event record
  responses.py        response library
  story_clusters.py   first report against later reports
  retrieval.py        time guard, type filter, ranking
  aggregate.py        quantiles, effective sample size, placebo, shrinkage, refusal
  evaluate.py         baseline ladder scoring
  precedent_factors.py  factors for Qlib
  whatif.py           the command
  canonical_text.py, embedder.py   model-typed rungs (Task 9)
references/precedent_params.json, references/country_groups.json, references/rungs/m2_precedent.yaml
stages/20_precedent_responses/, 21_precedent_evaluate/, 22_precedent_factors/   CONTEXT.md and output/ each
tests/precedent/
```

## Task Cycle

As in the spine plan: write the listed tests, see them fail for the stated reason, implement the listed signatures, run the task's tests then `uv run pytest -q`, commit the listed files and add a line to `docs/STATUS.md`.

---

### Task 1: Event key

**Files:**
- Create: `src/predate/precedent/event_key.py`, `references/country_groups.json`, `references/precedent_params.json`, `stages/20_precedent_responses/CONTEXT.md`, `stages/21_precedent_evaluate/CONTEXT.md`, `stages/22_precedent_factors/CONTEXT.md`
- Test: `tests/precedent/test_event_key.py`

**Interfaces:**
- Consumes: `EventRecord` (spine Task 2).
- Produces: `EventKey` (frozen dataclass): `event_type: str`, `actor_kind: str`, `impacted_kinds: tuple[str, ...]` (sorted, unique), `country_group: str` (`US`, `DM`, `EM` or `none`), `severity_band: str`, `is_scheduled: bool`, `direction_sign: int`. `severity_band(severity: int) -> str` gives `low` for 1-3, `mid` for 4-6, `high` for 7-10. `build_event_key(record: EventRecord) -> EventKey`.

- [ ] **Step 1: Write the failing tests**

```python
def test_severity_bands():
    assert [severity_band(s) for s in (1, 3, 4, 6, 7, 10)] == ["low", "low", "mid", "mid", "high", "high"]

def test_key_ignores_names():
    assert build_event_key(record_about("Acme Corp")) == build_event_key(record_about("Globex Inc"))

def test_unknown_country_maps_to_none():
    assert build_event_key(record_with_country(None)).country_group == "none"
```

- [ ] **Step 2: Run.** Expected: FAIL, `No module named 'predate.precedent'`.
- [ ] **Step 3: Implement.** Stage contracts follow the spine's contract standard; all three say `Loop: single pass`.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(precedent): typed event key and stage contracts`.

### Task 2: Response library

**Files:**
- Create: `src/predate/precedent/responses.py`, Test: `tests/precedent/test_responses.py`

**Interfaces:**
- Consumes: `cumulative_abnormal_return` (spine Task 8), `load_returns(..., kind="open_to_open")` (spine Task 7).
- Produces: `build_response_library(events: pd.DataFrame, returns: pd.DataFrame, market: str, sessions: pd.DatetimeIndex, assets_for_event: Callable[[pd.Series], list[str]], horizons: tuple[int, ...] = (1, 5, 20)) -> tuple[pd.DataFrame, ResponseReport]`. The frame has columns `event_id, asset, horizon, car, effective_session, response_complete_session`. `ResponseReport` has `rows, skipped_incomplete_window, skipped_short_history`.

- [ ] **Step 1: Write the failing tests**

```python
def test_response_window_starts_after_the_trade_open(library_inputs):
    # one event effective at position 300; +3% planted in the open-to-open return indexed at position 302
    library, _ = build_response_library(**library_inputs)
    row = library[library.horizon == 1].iloc[0]
    assert row.car == pytest.approx(0.03, abs=1e-9)
    assert row.response_complete_session == library_inputs["sessions"][302]

def test_unfinished_windows_are_absent_and_counted(library_inputs_near_end):     # event 3 sessions before the last one
    library, report = build_response_library(**library_inputs_near_end)
    assert set(library.horizon) == {1}
    assert report.skipped_incomplete_window == 2

def test_short_history_assets_are_counted_not_zeroed(library_inputs_new_fund):
    library, report = build_response_library(**library_inputs_new_fund)
    assert "NEWFUND" not in set(library.asset) and report.skipped_short_history == 3
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(precedent): response library with completion sessions`.

### Task 3: Story clusters

**Files:**
- Create: `src/predate/precedent/story_clusters.py`, Test: `tests/precedent/test_story_clusters.py`

**Interfaces:**
- Produces: `assign_story_clusters(events: pd.DataFrame, sessions: pd.DatetimeIndex, window_sessions: int = 3) -> pd.DataFrame` adding `story_cluster_id: str` and `is_first_report: bool`. Two events share a cluster when they have the same `EventKey` and primary actor and are at most `window_sessions` apart, chained.

- [ ] **Step 1: Write the failing test**

```python
def test_first_report_is_the_event():
    events = events_frame([("k1", "ACME", 10), ("k1", "ACME", 12), ("k1", "ACME", 20), ("k2", "ACME", 12)])   # key, actor, session position
    out = assign_story_clusters(events, SESSIONS, window_sessions=3)
    assert out["is_first_report"].tolist() == [True, False, True, True]
    assert out["story_cluster_id"].nunique() == 3
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(precedent): story clustering`.

### Task 4: Retrieval with the time guard

**Files:**
- Create: `src/predate/precedent/retrieval.py`, Test: `tests/precedent/test_retrieval.py`

**Interfaces:**
- Produces: `PrecedentQuery` (frozen dataclass): `key: EventKey`, `available_utc: pd.Timestamp`, `story_cluster_id: str | None`, `canonical_text: str | None`.
- Produces: `find_precedents(query: PrecedentQuery, library: pd.DataFrame, session_cutoffs_utc: pd.Series, horizon: int, top_k: int = 30, similarity: Callable[[PrecedentQuery, pd.DataFrame], pd.Series] = exact_key_similarity) -> pd.DataFrame` with columns `event_id, asset, car, similarity, effective_session, response_complete_session`. Order of operations is fixed: time guard, first reports only, drop the query's own story, similarity, top K.
- Produces: `exact_key_similarity` (1.0 on equal keys, else 0.0; rows at 0.0 are dropped).

- [ ] **Step 1: Write the failing tests**

```python
def test_an_unfinished_analog_is_never_returned():
    # analog A completes on session 110, analog B on session 130; the query is available just after session 120's cutoff
    found = find_precedents(QUERY_AFTER_120, LIBRARY, CUTOFFS, horizon=5)
    assert found["event_id"].unique().tolist() == ["A"]

def test_time_guard_holds_for_random_libraries():
    for seed in range(50):
        query, library = random_query_and_library(seed)
        found = find_precedents(query, library, CUTOFFS, horizon=5)
        assert (CUTOFFS.loc[found["response_complete_session"]].to_numpy() <= query.available_utc).all()

def test_a_story_is_not_its_own_precedent():
    found = find_precedents(QUERY_IN_STORY_S1, LIBRARY_WITH_STORY_S1, CUTOFFS, horizon=1)
    assert "S1" not in set(found.merge(LIBRARY_WITH_STORY_S1, on="event_id")["story_cluster_id"])
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(precedent): retrieval behind a time guard`.

### Task 5: Aggregation, placebo and refusal

**Files:**
- Create: `src/predate/precedent/aggregate.py`, Test: `tests/precedent/test_aggregate.py`

**Interfaces:**
- Produces: `summarize_responses(cars: np.ndarray, weights: np.ndarray, window_start_positions: np.ndarray, horizon: int, placebo_cars: np.ndarray, shrink_k: float = 10.0, min_neff: float = 8.0) -> PrecedentSummary`. Fields: `n: int`, `n_eff: float`, `quantiles: dict[float, float]` for 0.1, 0.25, 0.5, 0.75, 0.9 (weighted), `centre_shrunk: float`, `hit_rate: float`, `inside_ordinary_range: bool`, `refused: bool`.
- `n_eff` is the smaller of the Kish size `(sum w)^2 / sum w^2` and the number of non-overlapping windows (greedy, windows at least `horizon` positions apart). `centre_shrunk = median * n_eff / (n_eff + shrink_k)`. `inside_ordinary_range` is true when the median lies between the placebo's 10th and 90th percentiles. `refused = n_eff < min_neff`.
- Produces: `sample_placebo_cars(asset_returns, market_returns, horizon, n_samples, rng, exclude_positions) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
def test_separate_windows_keep_full_sample_size_and_shrink_as_specified():
    summary = summarize_responses(np.full(12, 0.02), np.ones(12), np.arange(12) * 10, horizon=5, placebo_cars=PLACEBO)
    assert summary.n_eff == pytest.approx(12.0) and summary.refused is False
    assert summary.centre_shrunk == pytest.approx(0.02 * 12 / 22)

def test_overlapping_windows_collapse_sample_size_and_refuse():
    summary = summarize_responses(np.full(12, 0.02), np.ones(12), np.arange(12), horizon=5, placebo_cars=PLACEBO)
    assert summary.n_eff == pytest.approx(3.0) and summary.refused is True       # only starts 0, 5 and 10 are independent

def test_a_median_inside_the_placebo_range_is_flagged():
    placebo = np.linspace(-0.05, 0.05, 101)
    assert summarize_responses(np.full(20, 0.01), np.ones(20), np.arange(20) * 10, 5, placebo).inside_ordinary_range is True
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(precedent): aggregation, placebo comparison, refusal`.

### Task 6: Baseline ladder scoring

**Files:**
- Create: `src/predate/precedent/evaluate.py`, Test: `tests/precedent/test_evaluate.py`

**Interfaces:**
- Produces: `pinball_loss(realized: np.ndarray, predicted_quantile: np.ndarray, q: float) -> float`.
- Produces: `score_precedent_ladder(events, library, session_cutoffs_utc, rungs: tuple[str, ...] = ("B0", "B1"), horizon: int = 5, quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)) -> pd.DataFrame` with columns `rung, event_type, quantile, pinball_loss, coverage_80, n_scored, n_refused`. Events are scored in time order, each from earlier events only, through `find_precedents`.

- [ ] **Step 1: Write the failing tests**

```python
def test_pinball_loss_hand_values():
    assert pinball_loss(np.array([1.0]), np.array([0.0]), 0.9) == pytest.approx(0.9)
    assert pinball_loss(np.array([0.0]), np.array([1.0]), 0.9) == pytest.approx(0.1)

def test_typed_rung_beats_untyped_when_types_differ(separable_events):   # type A events are followed by +2%, type B by -2%
    scores = score_precedent_ladder(**separable_events).query("quantile == 0.5").groupby("rung")["pinball_loss"].mean()
    assert scores["B1"] < scores["B0"]

def test_coverage_is_near_nominal_on_stationary_data(stationary_events):
    coverage = score_precedent_ladder(**stationary_events).query("rung == 'B1'")["coverage_80"].mean()
    assert 0.7 <= coverage <= 0.9
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(precedent): time-ordered ladder scoring`.

### Task 7: First real run (milestone P-M1, owner gate)

**Files:**
- Create: `scripts/run_precedent_m1.py`; output under `stages/21_precedent_evaluate/output/<run_id>/` (not committed); Modify: `docs/STATUS.md`

- [ ] **Step 1:** Build the response library from the `rules-8k-v1` events (spine Task 12) for each subject stock, its sector fund and SPY, and from `rules-gdelt-v1` events (spine Task 16, if built) for the asset panel in spec 8.5.
- [ ] **Step 2:** Score B0 and B1 at horizons 1, 5 and 20, holding out the most recent year of events.
- [ ] **Step 3:** Write `report.md`: pinball loss and coverage per rung and event type, refusal counts, the `ResponseReport` counts, and the ten event types where B1 improves most and least on B0. Label `LIVE-DATA`, `EXPLORATORY (survivorship-biased prices)`.
- [ ] **Step 4:** Commit the script and the status line. Stop for the owner's review. Success criterion P2's coverage band (70% to 90%) is checked here for B1.

### Task 8: The what-if command

**Files:**
- Create: `src/predate/precedent/whatif.py`, Test: `tests/precedent/test_whatif.py`

**Interfaces:**
- Produces: `run(argv: list[str]) -> int`, reached through `predate whatif` (spine Task 1).
  - By key, no model needed: `predate whatif --event-type central_bank --actor-kind central_bank --severity-band high --unscheduled [--country-group US] [--asof 2024-06-03] [--horizon 5] [--json]`.
  - By text: `predate whatif "<text>"`. Needs the model runtime (spine Task 13); without it, return 2 and print the key flags.
- Output layout as in spec 8.5.

- [ ] **Step 1: Write the failing tests**

```python
def test_key_query_prints_band_and_disclaimer(capsys, populated_stores):
    assert run(["--event-type", "earnings", "--actor-kind", "company", "--severity-band", "mid", "--asof", "2024-06-03"]) == 0
    out = capsys.readouterr().out
    assert "These are past outcomes of similar events, not a forecast." in out and "independent" in out

def test_too_few_precedents_is_a_refusal_not_a_band(capsys, sparse_stores):
    assert run(["--event-type", "cyber_incident", "--actor-kind", "company", "--severity-band", "high"]) == 0
    out = capsys.readouterr().out
    assert "fewer than 8 usable precedents" in out and "p10" not in out

def test_text_query_without_a_runtime_exits_2(capsys, populated_stores, monkeypatch):
    monkeypatch.delenv("PREDATE_MODEL_BASE_URL", raising=False)
    assert run(["a central bank raises rates unexpectedly"]) == 2
    assert "--event-type" in capsys.readouterr().err
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(precedent): what-if command`.

### Task 9: Precedent factors for Qlib

**Files:**
- Create: `src/predate/precedent/precedent_factors.py`, `references/rungs/m2_precedent.yaml`, Test: `tests/precedent/test_precedent_factors.py`

**Interfaces:**
- Consumes: `make_alpha_handler` and `run_ablation` (spine Tasks 9 and 10).
- Produces: `build_precedent_factor_panel(events, library, session_cutoffs_utc, sessions, horizon: int = 5, max_age_sessions: int = 5) -> pd.DataFrame` indexed (datetime, instrument) with columns `prec_car5_med, prec_car5_iqr, prec_neff, prec_hit_rate, prec_placebo_z`. A row exists while an event for that instrument is at most `max_age_sessions` old. Refused summaries give `prec_neff` and zeros elsewhere.

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

`truncate_inputs` removes events after the cut and library rows whose `response_complete_session` is after the cut.

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5:** Run the ablation with rung M2 on real data; append the result to the P-M1 report. Commit `feat(precedent): precedent factors and M2 rung`.

### Task 10: Model-typed and embedded rungs (after spine Task 15; owner gate on the download)

**Files:**
- Create: `src/predate/precedent/canonical_text.py`, `src/predate/precedent/embedder.py`; Modify: `evaluate.py` (rungs B2 and B3), `stages/21_precedent_evaluate/CONTEXT.md` (`## Loop`)
- Test: `tests/precedent/test_canonical_text.py`, `test_embedder.py`

**Interfaces:**
- Produces: `canonical_text(record: EventRecord) -> str`, a single line built only from enum values and role tokens; it contains no entity name from the record.
- Produces: `TextEmbedder` (Protocol): `embed(texts: list[str]) -> np.ndarray`, with a `release_date: date` attribute recorded in `precedent_version`. `embedding_similarity(embedder, weight: float) -> Callable`.

- [ ] **Step 1: Write the failing tests**

```python
def test_canonical_text_contains_no_entity_names():
    record = record_about("Acme Corp", impacted=["Globex Inc"])
    text = canonical_text(record)
    assert "Acme" not in text and "Globex" not in text and "[COMPANY]" in text

def test_similarity_uses_the_embedder_through_the_guarded_path(fake_embedder):
    found = find_precedents(QUERY_WITH_TEXT, LIBRARY_WITH_TEXT, CUTOFFS, horizon=5, similarity=embedding_similarity(fake_embedder, weight=0.5))
    assert (CUTOFFS.loc[found["response_complete_session"]].to_numpy() <= QUERY_WITH_TEXT.available_utc).all()
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Owner gate.** Name the embedding model, its licence, size and release date (candidates checked on 2026-11-06: `Qwen/Qwen3-Embedding-0.6B`, Apache-2.0; `BAAI/bge-m3`, MIT). All candidates were released after January 2024, so any earlier event may be in their training data; B2, which uses no embedding, is the control.
- [ ] **Step 6:** Score B2 and B3 against B1 on the held-out year. Tuning the similarity weight is a ratchet: oracle `score_precedent_ladder` on development events, held-back check the most recent year, budget 10 iterations (proposed), mutable file `references/precedent_params.json` only. Write that `## Loop` contract before running it.
- [ ] **Step 7:** Report against success criterion P2: if model-typed precedents do not beat rule-typed ones on pinball loss, say so and keep B1 as the default. Commit `feat(precedent): model-typed and embedded rungs`.
