# Mechanism Implementation Plan

> Amended 2026-11-07 against the product amendment spec. Supersedes the 2026-11-07 version.
>
> Research and content only; feeds the brief's one-mechanism slot; not a product surface (thesis section 3).

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** For any real or typed-in event, build a small structural causal model whose *structure* comes from a document (template rules first, a language model later) and whose *numbers* come from price history, run Monte Carlo rollouts and `do()` interventions over it, and show the resulting fan chart beside Precedent's empirical band so the two can disagree in public.

**Architecture:** A package `precedent.mechanism` on top of the spine and Precedent. A `CausalGraph` is a typed DAG over a closed node vocabulary (event variables, intermediate variables, the asset panel) with signed, lag-banded edges. Code fits one linear structural equation per node from the Precedent response library under the same time guard, estimates the residual covariance, and rolls the system forward. Interventions set a node and cut its incoming edges. The first milestone uses hand-written template graphs per event type, so it needs no language model; model-proposed graphs are added last and must beat the templates on held-out events or they are reported as not having earned their cost.

**Tech Stack:** As the spine, plus `networkx` for DAG checks (research extra, `uv sync --extra research`). The cockpit is one static HTML file with the fitted coefficients embedded as JSON; rollouts for slider moves run in the browser, so no server is needed.

**Spec:** `docs/superpowers/specs/2026-11-06-predate-design.md`, sections 1.2, 3.3, 4 and 8. This plan is the spec for the Mechanism track; there is no separate design section yet.

**Depends on:** spine plan Tasks 1 to 4 and 9 to 15 (old Tasks 1 to 11); spine Task 2 (`EventKey`) and spine Task 6 (`typer_from_env`); Precedent Tasks 2 to 8 (story clusters, shared row schema, the response libraries, `find_precedents`, `summarize_responses`). Task 8 also needs spine Task 18. Task 9 imports Qlib and so also needs the `research` extra (`uv sync --all-extras`). Runs in its own worktree; it touches only `src/precedent/mechanism/`, `tests/mechanism/`, `stages/4*_mechanism_*/`, `references/mechanism/` and `references/rungs/m4_mechanism_*.yaml`.

**No `cli.py` edit needed:** spine Task 1's `cli.py` already lists `ask`, `ripple`, `mechanism` and `dream` as lazily imported subcommands (`mechanism -> precedent.mechanism.cli:run`; a track that is absent prints `not built yet` and returns 2), so this track never edits `src/precedent/cli.py`. This track creates `src/precedent/mechanism/cli.py` (Task 7) with `run(argv: list[str]) -> int`.

## What the research check changed

Checked on 2026-11-07. Each item shapes a constraint or a test below.

- **No end-to-end system exists.** The pieces are separate: PyWhy-LLM (py-why/pywhyllm, "experimental") has a model propose DAG edges and critique graphs; FinCARE (arXiv 2510.20221) adds LLM-proposed edges to PC/GES/NOTEARS discovery, fits linear structural equations and propagates interventions, but from 10-K text and on one synthetic dataset; SOCIA (arXiv 2505.12006) has agents write and repair simulator code, with code not released. Nothing was found that validates such a model on market data. The design below is FinCARE's shape with Precedent as the validator.
- **In FinCARE the only ablation that mattered was the model's edge proposals.** So the structure is where the model may add value, and the structure is the only thing it is allowed to emit.
- **Model causal graphs are mostly recall.** Discovery F1 tracks how often a relation appears in pretraining (0.88 vs 0.2 across frequency bins, arXiv 2407.19638). Negative context cut affirmation of correct relations to 0.20 for an 8B model. Hence: every model-proposed edge needs an evidence span in the document, and the node vocabulary is closed.
- **Models infer X causes Y from mention order** (92.6% vs 1.9% when the order is flipped, arXiv 2406.12158; 70B did not fix it). Hence the order-flip test in Task 8: a passage and its sentence-reversed twin must yield the same edge directions, and the flip rate is reported.
- **Verbalized confidence is worse than chance for small models** (AUROC 0.358; over 80% of false edges carried at least 80% confidence; agreement across samples was better calibrated, arXiv 2608.23660). Hence the schema has no confidence field for edges; edge support is the agreement count over `k` independent samples.
- **Hard schema decoding raises validity and lowers accuracy for sub-3B models** (validity 61.5% to 100%, accuracy 19.7% to 11.0%, arXiv 2605.26128). Hence Task 8 compares constrained and prompt-only decoding on the gold graphs instead of assuming the constrained one is better.
- **Frontier models lose 25 to 40 points moving from interventional to counterfactual questions** (arXiv 2510.01539). Hence the model is never asked a counterfactual question. It emits structure; the counterfactual is computed.

## Global Constraints

- Everything in the spine plan's Global Constraints applies. Only `parser_graph.py` may call a language model directly; `cli.py` reaches one only through the spine's `typer_from_env()` seam when the owner opts in.
- **The model emits structure, never numbers.** `CausalGraph` has no coefficients, no probabilities, no magnitudes. Every number in a fan chart is fitted by code from price history or sampled from fitted residuals.
- **Closed node vocabulary.** A node is an event variable (`event:<event_type>` or `event:<event_type>:<field>` for `severity_band`, `direction_sign`, `is_scheduled`), an intermediate variable from `references/mechanism/intermediates.json` (proposed first list: `rates_expectation`, `risk_appetite`, `oil_supply`, `usd_demand`, `credit_spread`, `sector_demand:<sector>`), or an asset from the Precedent asset panel (spec 8.5). A graph naming any other node is rejected.
- **Edges are enums.** `sign` in {+1, -1}, `lag_band` in {same_session, days, weeks}. No weights. `source_doc_id` and `evidence_span` are required on every edge whose `graph_version` does not start with `template-`.
- **Fitting is time-guarded.** The equations for a query available at `available_utc` are fitted only from response-library rows whose `response_complete_session` cutoff is at or before `available_utc`. The fit goes through `fit_sem`, which takes the guarded rows from `find_precedents`; nothing reads the library directly.
- **Rollouts are deterministic given a seed.** `seed` is part of every result manifest. Default 10,000 paths (proposed, in `references/mechanism/params.json`).
- **An intervention is `do()`, not conditioning.** `intervene(node, value)` sets the node and removes its incoming edges before simulation. The cockpit never exposes "observe".
- **Every fan chart is printed beside Precedent's empirical band for the same query, asset and horizon**, with both sample sizes. A mechanism result without its Precedent band is not reported. Below Precedent's `min_neff` the mechanism also refuses: there is nothing to fit.
- Output always carries the line `A fitted model of past co-movement under an assumed structure, not a forecast.`

## Review Focus

1. A graph with a cycle, or an edge from an asset into an event variable: rejected at validation, never silently pruned. (Test in Task 1.)
2. A query whose own story is in the library: the fit must not use it. (Test in Task 3, through `find_precedents`.)
3. Fewer guarded rows than parameters for some node: that node's equation falls back to intercept-only and is flagged `underdetermined`; the chart shows the flag. (Test in Task 3.)
4. An intervention on a node with no descendants among the assets: the fan chart equals the no-intervention chart, and the output says so. (Test in Task 4.)
5. A model-proposed edge whose evidence span is not in the passage, or whose node is outside the vocabulary: dropped and counted; never repaired by the code. (Test in Task 8.)

## File Structure

```text
src/precedent/mechanism/
  graph_schema.py      CausalGraph, Node, Edge, validation, JSON schema
  graph_store.py       append-only store keyed by (graph_id, graph_version)
  templates.py         rule-built graphs per event type (Task 2)
  fit_sem.py           guarded parameter fitting (Task 3)
  simulate.py          rollouts, interventions, quantiles (Task 4)
  evaluate.py          ladder scoring against Precedent (Task 5)
  cli.py               the `precedent mechanism` command: arguments, typing the text, text/JSON output (Task 7)
  cockpit.py           the static HTML page (Task 7)
  parser_graph.py      model-proposed graphs, consensus, order-flip test (Task 8)
  mechanism_factors.py factors for Qlib (Task 9, optional)
references/mechanism/intermediates.json, params.json, templates/<event_type>.json
references/mechanism/graph_schema_v1.json, references/mechanism/graph_prompt_v1.txt
references/rungs/m4_mechanism_template.yaml, m4_mechanism_model.yaml, m4_mechanism_placebo.yaml
stages/40_mechanism_graphs/, 41_mechanism_fit/, 42_mechanism_evaluate/, 43_mechanism_cockpit/   CONTEXT.md and output/ each
tests/mechanism/
```

## Task Cycle

As in the spine plan: write the listed tests, see them fail for the stated reason, implement the listed signatures, run the task's tests then `uv run pytest -q`, commit the listed files and add a line to `docs/STATUS.md`.

---

### Task 1: Causal graph schema and store

**Files:**
- Create: `src/precedent/mechanism/graph_schema.py`, `src/precedent/mechanism/graph_store.py`, `references/mechanism/intermediates.json`, `references/mechanism/params.json`, `references/mechanism/graph_schema_v1.json`, `stages/40_mechanism_graphs/CONTEXT.md`, `41_mechanism_fit/CONTEXT.md`, `42_mechanism_evaluate/CONTEXT.md`, `43_mechanism_cockpit/CONTEXT.md`
- Test: `tests/mechanism/test_graph_schema.py`, `tests/mechanism/test_graph_store.py`

**Interfaces:**
- Produces: `NodeKind` (str Enum): `event`, `intermediate`, `asset`. `Node` (pydantic, frozen): `id: str`, `kind: NodeKind`. `node_vocabulary(intermediates_path: Path, asset_panel: list[str], event_types: list[str]) -> frozenset[str]`.
- Produces: `Edge` (pydantic, `extra="forbid"`): `source: str`, `target: str`, `sign: Literal[-1, 1]`, `lag_band: Literal["same_session", "days", "weeks"]`, `source_doc_id: str | None`, `evidence_span: str | None`, `support: int = 1`.
- Produces: `CausalGraph` (pydantic): `graph_id: str`, `graph_version: str`, `event_key: EventKey`, `nodes: list[Node]`, `edges: list[Edge]`. Validation: every edge endpoint is in `nodes`; every node id is in the vocabulary; the graph is a DAG; no edge has an `asset` source and a non-asset target; every edge in a graph whose `graph_version` does not start with `template-` has `source_doc_id` and a non-empty `evidence_span`. `graph_json_schema() -> dict`.
- Produces: `GraphStore(root: Path)` with `.append(graphs) -> AppendReport`, `.read(graph_version: str) -> list[CausalGraph]`, `.for_key(event_key: EventKey, graph_version: str) -> CausalGraph | None`. One JSON Lines file per `graph_version`, written to a temporary file and renamed; a key that exists with different content raises `GraphConflictError`.

- [ ] **Step 1: Write the failing tests**

```python
def test_cycle_is_rejected():
    with pytest.raises(ValidationError):
        graph(edges=[("event:central_bank", "rates_expectation"), ("rates_expectation", "risk_appetite"), ("risk_appetite", "rates_expectation")])

def test_asset_cannot_cause_an_event_variable():
    with pytest.raises(ValidationError):
        graph(edges=[("SPY", "event:central_bank")])

def test_unknown_node_is_rejected_not_added():
    with pytest.raises(ValidationError):
        graph(edges=[("event:central_bank", "my_new_variable"), ("my_new_variable", "SPY")])

def test_model_graph_needs_evidence_on_every_edge():
    with pytest.raises(ValidationError):
        graph(edges=[("event:central_bank", "rates_expectation", None)], version="llm-graph-v1")   # no evidence span
    graph(edges=[("event:central_bank", "rates_expectation", None)], version="template-v1")           # templates are exempt

def test_reference_schema_file_matches_the_code():
    assert json.loads(Path("references/mechanism/graph_schema_v1.json").read_text()) == graph_json_schema()

def test_store_is_idempotent_and_refuses_conflicts(tmp_path):
    store = GraphStore(tmp_path)
    assert store.append([G]).added == 1 and store.append([G]).identical_skipped == 1
    with pytest.raises(GraphConflictError):
        store.append([G.model_copy(update={"edges": G.edges[:-1]})])
```

- [ ] **Step 2: Run.** Expected: FAIL, `No module named 'precedent.mechanism'`.
- [ ] **Step 3: Implement.** Stage contracts follow the spine's standard; `40_mechanism_graphs` carries a `## Loop` section for the graph prompt (filled in Task 8); the others say `Loop: single pass`. `intermediates.json` holds the proposed list above with a one-line definition each.
- [ ] **Step 4: Run.** Expected: PASS.
- [ ] **Step 5: Commit** `feat(mechanism): causal graph schema, vocabulary and store`.

### Task 2: Template graphs (no model)

**Files:**
- Create: `src/precedent/mechanism/templates.py`, `references/mechanism/templates/central_bank.json`, `macro_data_release.json`, `trade_or_sanctions.json`, `commodity_supply.json`, `earnings.json`, `guidance.json`, `armed_conflict.json`, `natural_disaster.json` (the eight event types with the most 8-K and GDELT events, proposed; add others as data shows volume)
- Test: `tests/mechanism/test_templates.py`

**Interfaces:**
- Produces: `load_templates(templates_dir: Path, vocabulary: frozenset[str]) -> dict[str, CausalGraph]` keyed by `event_type`, each with `graph_version="template-v1"`. `template_for(key: EventKey, templates) -> CausalGraph | None`: the template for the key's `event_type`, with the `direction_sign` event node set from the key.
- Each template is a hand-written DAG: the event node, at most four intermediates, and the asset panel. For example `central_bank`: `event:central_bank:direction_sign -> rates_expectation (+)`, `rates_expectation -> TLT (-)`, `rates_expectation -> risk_appetite (-)`, `risk_appetite -> SPY (+)`, `risk_appetite -> HYG (+)`, `rates_expectation -> UUP (+)`. Templates are written from textbook relations and say so in a `rationale` field that is documentation, not data.

- [ ] **Step 1: Write the failing tests**

```python
def test_every_template_validates_against_the_vocabulary():
    templates = load_templates(TEMPLATES_DIR, VOCAB)
    assert set(templates) >= {"central_bank", "earnings"} and all(g.graph_version == "template-v1" for g in templates.values())

def test_template_reaches_at_least_one_asset_from_the_event_node():
    for graph in load_templates(TEMPLATES_DIR, VOCAB).values():
        assert descendants(graph, graph.event_node_id) & set(ASSET_PANEL)

def test_unknown_event_type_has_no_template():
    assert template_for(key_with(event_type="other"), load_templates(TEMPLATES_DIR, VOCAB)) is None
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(mechanism): template graphs per event type`.

### Task 3: Guarded parameter fitting

**Files:**
- Create: `src/precedent/mechanism/fit_sem.py`, Test: `tests/mechanism/test_fit_sem.py`

**Interfaces:**
- Consumes: `find_precedents` (Precedent Task 7), the response libraries (Precedent Tasks 3 to 6), `load_returns(..., kind="open_to_open")` (spine Task 11).
- Produces: `GuardedPanel`: the per-precedent observations the fit may see. `build_guarded_panel(query: PrecedentQuery, library, session_cutoffs_utc, returns, market, horizon, top_k) -> GuardedPanel`. It calls `find_precedents` once per asset and assembles, per precedent event, the event variables (from the `EventKey`: `direction_sign`, `severity_band` as ordinal 0,1,2, `is_scheduled`), and each asset's abnormal return over the response window. Intermediates are latent; see below.
- Produces: `fit_sem(graph: CausalGraph, panel: GuardedPanel, ridge: float = 1.0) -> FittedSEM`. For each non-event node in topological order: an asset node's equation regresses its abnormal return on its parents; an intermediate node has no observed series, so it is identified as the ridge-regressed linear combination of *its* parents that best explains *its* children jointly (one latent factor, sign-anchored by the template's edge signs); the fit of each child then uses that latent score. `FittedSEM` has `coefficients: dict[str, dict[str, float]]`, `intercepts`, `residual_cov: np.ndarray` over asset nodes, `n_obs: int`, `underdetermined: set[str]`, `graph_id`, `graph_version`, `fitted_asof_utc`.
- An edge whose fitted coefficient has the opposite sign from the graph's `sign` is kept, but listed in `FittedSEM.sign_disagreements`; the chart shows them.
- A node with fewer than `parents + 2` observations is fitted as intercept-only and added to `underdetermined`.

- [ ] **Step 1: Write the failing tests**

```python
def test_fit_uses_only_time_guarded_rows(guard_fixture):
    # analog B's window completes after the query is available; its +50% planted return must not move any coefficient
    with_b, without_b = fit_sem(GRAPH, build_guarded_panel(**guard_fixture.with_b)), fit_sem(GRAPH, build_guarded_panel(**guard_fixture.without_b))
    assert with_b.coefficients == without_b.coefficients

def test_recovers_planted_linear_structure(planted_sem):
    # event -> X (+0.02 per sign unit) -> SPY (coef 1.0), noise sd 0.001, 200 events
    fitted = fit_sem(planted_sem.graph, planted_sem.panel, ridge=1e-6)
    assert fitted.coefficients["SPY"]["rates_expectation"] * fitted.coefficients["rates_expectation"]["event:central_bank:direction_sign"] == pytest.approx(0.02, abs=2e-3)

def test_too_few_rows_is_intercept_only_and_flagged(tiny_panel):     # three precedents, a node with three parents
    fitted = fit_sem(GRAPH, tiny_panel)
    assert "HYG" in fitted.underdetermined and fitted.coefficients["HYG"] == {}

def test_sign_disagreement_is_recorded_not_corrected(contrary_panel):  # data says rates up -> TLT up
    fitted = fit_sem(GRAPH, contrary_panel)
    assert ("rates_expectation", "TLT") in fitted.sign_disagreements and fitted.coefficients["TLT"]["rates_expectation"] > 0
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(mechanism): time-guarded structural equation fitting`.

### Task 4: Rollouts and interventions

**Files:**
- Create: `src/precedent/mechanism/simulate.py`, Test: `tests/mechanism/test_simulate.py`

**Interfaces:**
- Produces: `Intervention` (frozen dataclass): `node: str`, `value: float`.
- Produces: `simulate(fitted: FittedSEM, graph: CausalGraph, event_values: dict[str, float], interventions: tuple[Intervention, ...] = (), n_paths: int = 10_000, seed: int = 0) -> Rollout`. Topological evaluation; residuals drawn jointly from `residual_cov` (Cholesky; falls back to the diagonal with a flag when not positive definite). An intervention sets the node to its value on every path and removes its incoming edges first. `Rollout` has `paths: np.ndarray` (n_paths x assets), `assets: list[str]`, `seed`, `interventions`, `unreachable_interventions: list[str]` (intervened nodes with no asset descendant).
- Produces: `fan(rollout: Rollout, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9)) -> pd.DataFrame` indexed by asset.
- Produces: `counterfactual_pair(fitted, graph, event_values, intervention, n_paths, seed) -> tuple[Rollout, Rollout]`: factual and intervened rollouts drawn with the *same* residuals, so their difference isolates the intervention (abduction-action-prediction over the fitted noise).

- [ ] **Step 1: Write the failing tests**

```python
def test_same_seed_same_paths(fitted_graph):
    a, b = simulate(*fitted_graph, EVENT, seed=3), simulate(*fitted_graph, EVENT, seed=3)
    np.testing.assert_array_equal(a.paths, b.paths)

def test_do_cuts_incoming_edges(fitted_graph):
    # with rates_expectation forced to 0, the event's sign must not reach TLT
    rollout = simulate(*fitted_graph, EVENT, interventions=(Intervention("rates_expectation", 0.0),), seed=0)
    assert abs(fan(rollout).loc["TLT", 0.5]) < 1e-3

def test_counterfactual_pair_shares_noise(fitted_graph):
    factual, cf = counterfactual_pair(*fitted_graph, EVENT, Intervention("rates_expectation", 1.0), n_paths=1000, seed=1)
    diff = cf.paths - factual.paths
    assert np.allclose(diff.std(axis=0), 0.0, atol=1e-9)     # the difference is deterministic across paths

def test_intervention_with_no_asset_descendant_is_reported(fitted_graph_with_leaf):
    rollout = simulate(*fitted_graph_with_leaf, EVENT, interventions=(Intervention("dead_end", 1.0),))
    assert rollout.unreachable_interventions == ["dead_end"]
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(mechanism): Monte Carlo rollouts and do-interventions`.

### Task 5: Ladder scoring against Precedent

**Files:**
- Create: `src/precedent/mechanism/evaluate.py`, `references/rungs/m4_mechanism_template.yaml`, `references/rungs/m4_mechanism_placebo.yaml`, Test: `tests/mechanism/test_placebo.py`

**Interfaces:**
- Consumes: `pinball_loss`, `score_precedent_ladder` (Precedent Task 9).
- Produces: `shuffle_edges_preserving_degree(graph: CausalGraph, seed: int) -> CausalGraph`: the placebo graph. Same nodes, same in- and out-degree per node, re-wired at random subject to the DAG and kind rules; `graph_version="placebo-<seed>"`.
- Produces: `score_mechanism_ladder(events, library, session_cutoffs_utc, returns, market, graphs_by_rung: dict[str, Callable[[EventKey], CausalGraph | None]], horizon: int = 5, quantiles=(0.1, 0.5, 0.9), n_paths: int = 2_000, seed: int = 0) -> pd.DataFrame` with columns `rung, event_type, quantile, pinball_loss, coverage_80, n_scored, n_refused, n_underdetermined`. Rungs: `P-B1` (Precedent's empirical band, from `score_precedent_ladder`), `M-T` (template graph), `M-P` (placebo of the template). Events are scored in time order, each fitted from earlier events only.

- [ ] **Step 1: Write the failing tests**

```python
def test_placebo_keeps_degrees_and_stays_a_valid_dag():
    placebo = shuffle_edges_preserving_degree(TEMPLATE, seed=5)
    assert degrees(placebo) == degrees(TEMPLATE) and placebo.graph_version == "placebo-5"
    assert edge_set(placebo) != edge_set(TEMPLATE)

def test_true_structure_beats_placebo_on_planted_mechanism(planted_world):
    # SPY responds to the event only through rates_expectation; the placebo wires the event elsewhere
    table = score_mechanism_ladder(**planted_world, graphs_by_rung={"M-T": planted_world.template, "M-P": planted_world.placebo})
    med = table.query("quantile == 0.5").groupby("rung")["pinball_loss"].mean()
    assert med["M-T"] < med["M-P"]

def test_mechanism_and_precedent_are_scored_on_identical_events(planted_world):
    table = score_mechanism_ladder(**planted_world, graphs_by_rung={"M-T": planted_world.template})
    assert table.groupby("rung")["n_scored"].sum()["M-T"] == table.groupby("rung")["n_scored"].sum()["P-B1"]
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(mechanism): ladder scoring against Precedent with a placebo graph`.

### Task 6: First real run (milestone M-M1, owner gate)

**Files:**
- Create: `scripts/run_mechanism_m1.py`; output under `stages/42_mechanism_evaluate/output/<run_id>/` (not committed); Modify: `docs/STATUS.md`

- [ ] **Step 1:** Using the response libraries from Precedent Task 16 (first real run), score `P-B1`, `M-T` and `M-P` at horizons 1, 5 and 20 on the same held-out year, per event type.
- [ ] **Step 2:** Write `report.md`: pinball loss and coverage per rung and event type; `n_underdetermined`; the sign-disagreement table (template edges the data contradicts, with counts); the five event types where `M-T` most helps and most hurts against `P-B1`. Label `LIVE-DATA`, `EXPLORATORY (survivorship-biased prices)`.
- [ ] **Step 3:** State the outcome in one of three sentences: templates beat Precedent on pinball loss; templates match it; templates lose to it. If they lose, Task 8 still runs (the model may propose better structure than the templates), but Task 9 does not.
- [ ] **Step 4:** Commit the script and the status line. Stop for the owner's review.

### Task 7: The cockpit

**Files:**
- Create: `src/precedent/mechanism/cli.py`, `src/precedent/mechanism/cockpit.py`, `src/precedent/mechanism/cockpit_template.html`, Test: `tests/mechanism/test_cockpit.py` (the tests below import `run` from `precedent.mechanism.cli`)

**Interfaces:**
- Produces: `cli.run(argv: list[str]) -> int` in `src/precedent/mechanism/cli.py`, the target of the spine's `mechanism` command (`precedent.mechanism.cli:run`); `cockpit.py` renders the static page for it.
  - By key, no model: `precedent mechanism --event-type central_bank --actor-kind central_bank --severity-band high --unscheduled --direction -1 [--asof 2024-06-03] [--horizon 5] [--do rates_expectation=0] [--html PATH] [--json]`.
  - By text: `precedent mechanism "<text>"`. The event key comes from `precedent.spine.typers.select.typer_from_env()` (spine Task 6; the rules typer by default, a model typer only if the owner sets `PRECEDENT_TYPER`). When the typer returns `untyped` (or `refused_invalid`), exit 2 and print the key flags (`--event-type ... --direction`) the user can pass instead. After Task 8, a model-proposed graph replaces the template graph only when one exists for the typed event type; typing the text never needs a model.
- Text output, after Precedent's own block for the same query:

```text
mechanism      template-v1 for central_bank, fitted on <n_obs> precedents as of <date>; <k> equations underdetermined
asset   window   mechanism p10/median/p90      precedent p10/median/p90   agree?
TLT     5        ...                           ...                        yes / no (median outside the other's band)
...
edges the data disagrees with: rates_expectation -> TLT (template -, fitted +, n=<n>)
A fitted model of past co-movement under an assumed structure, not a forecast.
```

- `--html` writes one self-contained page: the DAG drawn as SVG, one slider per intermediate and event node (range from the fitted panel's 5th to 95th percentile), the fan chart per asset, and Precedent's band drawn as a fixed grey band behind it. The page embeds `FittedSEM` as JSON and re-runs `simulate` in JavaScript with a seeded PRNG on each slider move; it never calls a server. The disclaimer line is in the page footer and in the `<title>`.
- `--json` writes `FittedSEM`, the fan, the Precedent summary, and a manifest (`graph_version`, `fitted_asof_utc`, `seed`, `n_paths`, code commit).

- [ ] **Step 1: Write the failing tests**

```python
def test_key_query_prints_both_bands_and_the_disclaimer(capsys, populated_stores):
    assert run(["--event-type", "central_bank", "--actor-kind", "central_bank", "--severity-band", "high", "--unscheduled", "--direction", "-1", "--asof", "2024-06-03"]) == 0
    out = capsys.readouterr().out
    assert "mechanism p10/median/p90" in out and "precedent p10/median/p90" in out
    assert "A fitted model of past co-movement under an assumed structure, not a forecast." in out

def test_do_flag_is_applied_and_named(capsys, populated_stores):
    assert run(KEY_FLAGS + ["--do", "rates_expectation=0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["interventions"] == [{"node": "rates_expectation", "value": 0.0}]

def test_html_is_self_contained(tmp_path, populated_stores):
    out = tmp_path / "cockpit.html"
    assert run(KEY_FLAGS + ["--html", str(out)]) == 0
    html = out.read_text()
    assert "<script src=" not in html and "fetch(" not in html and "not a forecast" in html

def test_js_simulation_matches_python_on_the_same_seed(tmp_path, populated_stores, node_runtime):
    # run the embedded simulate() with node on the embedded JSON; medians must match Python's within tolerance
    run(KEY_FLAGS + ["--html", str(tmp_path / "c.html"), "--json"])
    assert js_median(tmp_path / "c.html", "SPY", seed=0) == pytest.approx(py_median("SPY", seed=0), abs=5e-4)

def test_untyped_text_query_exits_2_and_prints_the_key_flags(capsys, populated_stores, monkeypatch):
    monkeypatch.delenv("PRECEDENT_TYPER", raising=False)   # the rules typer
    assert run(["the weather in Lisbon was pleasant"]) == 2
    assert "--event-type" in capsys.readouterr().err
```

The fourth test is skipped when `node` is not on the path and the skip is recorded in `docs/STATUS.md`.

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Commit** `feat(mechanism): cockpit command and static what-if page`.

### Task 8: Model-proposed graphs (after spine Task 18; owner gate on the gold set)

**Files:**
- Create: `src/precedent/mechanism/parser_graph.py`, `references/mechanism/graph_prompt_v1.txt`, `references/rungs/m4_mechanism_model.yaml`; Modify: `stages/40_mechanism_graphs/CONTEXT.md` (`## Loop`), `evaluate.py` (rungs `M-L` model graph, `M-LP` its placebo)
- Test: `tests/mechanism/test_parser_graph.py`

**Interfaces:**
- Consumes: `EventExtractor` (spine Task 2), `OpenAICompatExtractor` (spine Task 18).
- Produces: `ProposedEdge` (pydantic, closed enums): `source: str`, `target: str`, `sign`, `lag_band`, `evidence_span: str`. `ProposedGraph` wraps `edges: list[ProposedEdge]` (at most 12).
- Produces: `parse_graph(passage: str, event_key: EventKey, source_doc_id: str, extractor: EventExtractor, graph_version: str, vocabulary: frozenset[str], k_samples: int = 5, min_support: int = 3, seed: int = 0) -> tuple[CausalGraph | None, GraphParseReport]`. The model is sampled `k_samples` times at distinct seeds; an edge is kept when it appears in at least `min_support` samples with the same sign, its span is found in the passage, and both nodes are in the vocabulary. `support` on the kept edge is its count. The event node and the asset nodes are always present; a result with no path from the event node to any asset returns `None`. `GraphParseReport` has `proposed, evidence_missing, unknown_node, below_support, cycle_dropped, kept, order_flip_rate`.
- Produces: `order_flip_rate(passage, ...) -> float`: `parse_graph` is also run on the passage with its sentences reversed; the rate is the share of kept edges whose direction differs between the two runs. The kept graph is the one from the original passage; the rate is reported beside it.

- [ ] **Step 1: Write the failing tests**

```python
def test_edge_without_evidence_in_the_passage_is_dropped():
    extractor = FakeExtractor(json_graph([("event:trade_or_sanctions", "oil_supply", -1, "days", "this sentence is not in the passage")]) * 5)
    graph, report = parse_graph(PASSAGE, KEY, "doc-1", extractor, "llm-graph-v1", VOCAB)
    assert graph is None and report.evidence_missing == 5

def test_node_outside_vocabulary_is_dropped_not_added():
    extractor = FakeExtractor(json_graph([("event:trade_or_sanctions", "shipping_insurance_premium", 1, "days", SPAN_IN_PASSAGE)]) * 5)
    graph, report = parse_graph(PASSAGE, KEY, "doc-1", extractor, "llm-graph-v1", VOCAB)
    assert report.unknown_node == 5 and (graph is None or "shipping_insurance_premium" not in {n.id for n in graph.nodes})

def test_support_counts_agreement_across_samples():
    samples = [json_graph([("event:trade_or_sanctions", "oil_supply", -1, "days", SPAN_A), ("oil_supply", "USO", 1, "days", SPAN_B)])] * 4 \
            + [json_graph([("event:trade_or_sanctions", "oil_supply", 1, "days", SPAN_A)])]
    graph, report = parse_graph(PASSAGE, KEY, "doc-1", FakeExtractor(samples), "llm-graph-v1", VOCAB, k_samples=5, min_support=3)
    oil = next(e for e in graph.edges if e.target == "oil_supply")
    assert oil.sign == -1 and oil.support == 4 and report.below_support == 1

def test_order_flip_is_measured_not_hidden():
    extractor = OrderSensitiveFakeExtractor()      # emits A->B on the original, B->A on the reversed passage
    _, report = parse_graph(PASSAGE, KEY, "doc-1", extractor, "llm-graph-v1", VOCAB)
    assert report.order_flip_rate == pytest.approx(1.0)

def test_request_has_temperature_zero_per_sample_and_distinct_seeds(recording_extractor):
    parse_graph(PASSAGE, KEY, "doc-1", recording_extractor, "llm-graph-v1", VOCAB, k_samples=3)
    assert len({r.seed for r in recording_extractor.requests}) == 3 and all(r.temperature == 0 for r in recording_extractor.requests)
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5: Owner gate.** Label a gold sample with the owner: 60 passages (news or filing excerpts describing an event) with the edges a careful reader would draw, split 40 development and 20 held out. Write the `## Loop` contract: ratchet; oracle edge precision and recall on development; held-back check on the 20, run once; mutable file `references/mechanism/graph_prompt_v*.txt` only; budget 10 iterations (proposed).
- [ ] **Step 6:** Run the gold scoring twice, with `response_format` schema-constrained and prompt-only (validation in code either way), and record both precision figures and the `order_flip_rate`. Choose the decoding mode with the higher held-out precision and record why.
- [ ] **Step 7:** Build model graphs for the held-out year's events; score rungs `M-L` and `M-LP` beside `M-T`, `M-P` and `P-B1`. Append to the M-M1 report. If `M-L` does not beat `M-T` on pinball loss, say so and keep templates as the cockpit default. Commit `feat(mechanism): evidence-backed model-proposed graphs`.

### Task 9: Mechanism factors for Qlib (only if Task 6 or 8 beat Precedent; `research` extra)

**Files:**
- Create: `src/precedent/mechanism/mechanism_factors.py`, `references/rungs/m4_mechanism_factors.yaml`, Test: `tests/mechanism/test_mechanism_factors.py`

**Interfaces:**
- Consumes: `make_alpha_handler`, `run_ablation` (spine Tasks 13 and 14). These import Qlib and LightGBM, so this task's code and tests need the `research` extra (`pyqlib==0.9.7`, `lightgbm`); the test module is skipped, and the skip recorded in `docs/STATUS.md`, when the extra is absent.
- Produces: `build_mechanism_factor_panel(events, library, session_cutoffs_utc, sessions, returns, market, graphs: Callable[[EventKey], CausalGraph | None], horizon: int = 5, max_age_sessions: int = 5, n_paths: int = 2_000, seed: int = 0) -> pd.DataFrame` indexed (datetime, instrument) with columns `mech_med, mech_iqr, mech_vs_prec` (mechanism median minus Precedent median), `mech_underdetermined` (0/1), `mech_sign_disagreements`.

- [ ] **Step 1: Write the failing tests**

```python
def test_mechanism_panel_passes_truncation_invariance(panel_inputs):
    for cut in CUT_SESSIONS:
        full = build_mechanism_factor_panel(**panel_inputs).loc[:cut]
        truncated = build_mechanism_factor_panel(**truncate_inputs(panel_inputs, cut)).loc[:cut]
        pd.testing.assert_frame_equal(full, truncated)

def test_every_column_carries_the_prefix(panel_inputs):
    assert all(c.startswith("mech_") for c in build_mechanism_factor_panel(**panel_inputs).columns)
```

- [ ] **Steps 2-4:** as the Task Cycle.
- [ ] **Step 5:** Run the ablation with rung M4 beside M0 to M3 on real data; append to the report. Commit `feat(mechanism): mechanism factors and M4 rung`.

### Task 10: Independent verifier

**Files:**
- Create: `docs/verification/mechanism-<date>.md`; Modify: `docs/STATUS.md`

The verifier is a separate agent from the one that built Tasks 1 to 9. It receives this plan, `docs/superpowers/specs/2026-11-07-precedent-product-amendment.md`, the diff of the track against its baseline commit and the test output, assumes the work is broken, and does not read the builder's summary.

- [ ] **Step 1:** Run and record each command with its exit code: `uv sync --frozen --all-extras`, `uv run ruff check .`, `uv run pytest -q tests/mechanism`, `uv run pytest -q`, `precedent mechanism --event-type central_bank --actor-kind central_bank --severity-band high --unscheduled --direction -1 --html <tmp>` on the sample library, then `test -z "$(git status --porcelain)"`.
- [ ] **Step 2:** Map each Review Focus item (1 to 5) and each Global Constraint to the test that proves it. A constraint with no test is a `FAIL`. Check by grep that no coefficient or probability field exists on `CausalGraph` or `Edge`, that only `parser_graph.py` calls a model directly (`cli.py` only through `typer_from_env()`), that no file outside the allowed paths changed (including `cli.py`), and that no model id, base URL or data-dir default appears outside `src/precedent/config.py`.
- [ ] **Step 3:** Confirm every fan chart in the text, JSON and HTML output is printed beside Precedent's band with both sample sizes and carries the disclaimer line, and that an input below `min_neff` is refused.
- [ ] **Step 4:** Write `docs/verification/mechanism-<date>.md`: the commands and exit codes, a `PASS`, `FAIL` or `BLOCKED` per requirement, and the evidence class of each result (`SYNTHETIC`, `LIVE-DATA`, `LIVE-MODEL`, `NO LIVE RUN`). Steps that need an owner gate (Tasks 6, 8, 9 live runs) and were not run are `NO LIVE RUN`, not `PASS`. Add the matching rows to `docs/STATUS.md`.
- [ ] **Step 5:** Commit `docs(mechanism): independent verification report`. Each finding is fixed in a separate commit `fix: independent-review findings`, then Steps 1 to 4 are re-run and the report updated.

## Open decisions for the owner

| # | Decision | Default if you say nothing |
|---|---|---|
| M1 | Confirm the closed intermediate vocabulary (six entries proposed). | As listed in Task 1. |
| M2 | Which eight event types get templates first. | The eight named in Task 2. |
| M3 | Whether intermediates are latent (fitted as one factor) or dropped so the model's structure reduces to event-to-asset edges. | Latent, with the drop as rung `M-T0` if time allows. |
| M4 | Gold-set labelling time: 60 passages. | 40 development, 20 held out. |
| M5 | Whether Task 9 runs if templates lose to Precedent but model graphs win. | Yes, on the model graphs. |
