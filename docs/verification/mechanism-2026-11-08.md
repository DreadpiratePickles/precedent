FAIL

# Mechanism independent verification, 2026-11-08

Verifier: Task 10, independent of Tasks 1-7 (Tasks 6, 8 and 9 are owner-gated and were not built). Branch `claude/track-mechanism-66ef31`, HEAD `9e84536` (before this commit). Python 3.11, uv 0.12.6, node v26.8.1. Inputs read: the plan, the product amendment, design sections 1.2, 3.3, 4 and 8, `CLAUDE.md`, the track's diff (`git diff 2ead3a5 HEAD`) and the controller's rulings file. No builder report, brief or review diff was read. Source was not modified; probes live outside the repository.
Reason for FAIL: three requirements fail (V-1 and V-2: the "beside Precedent's band, both sample sizes, refuse below `min_neff`" constraint is broken in the by-key command and in the HTML page; V-3: the plan's placebo comparison cannot discriminate structure). No Critical finding. Everything the track could run without an owner gate is green; everything that needs one is `NO LIVE RUN`.

## Commands run

| # | Command | Exit | Key output line |
|---|---|---|---|
| 1 | `uv sync --frozen --all-extras` | 0 | `Checked 207 packages` |
| 2 | `uv run ruff check .` | 0 | `All checks passed!` |
| 3 | `uv run pytest -q tests/mechanism -rs` | 0 | `198 passed in 26.32s`; no skip, so the node tests ran |
| 4 | `uv run pytest -q -rs` | 0 | `1146 passed, 1 skipped in 68.79s`; the skip is `tests/spine/test_typer_select.py:30` (ollama module exists) |
| 5 | `git status --porcelain` after 3-4 | 0 | empty |
| 6 | `precedent mechanism --event-type central_bank --actor-kind central_bank --severity-band high --unscheduled --direction -1 --html <scratch>/c.html --data-dir <scratch>/data` (empty temporary data dir; there is no `data/sample/` library) | 2 | stdout empty; stderr `not built yet: mechanism needs precedent.libraries.bundle from the Precedent track`; no HTML written. BLOCKED, see V-R24 |
| 7 | `UV_PROJECT_ENVIRONMENT=<scratch>/venv-core uv sync --frozen --extra dev` (core-only environment outside the repo) | 0 | synced |
| 8 | core-only `uv run --no-sync pytest -q tests/mechanism` | 0 | `198 passed in 39.20s` |
| 9 | core-only `uv run --no-sync pytest -q` | 0 | `1096 passed, 6 skipped` (research tests skip) |
| 10 | `git status --porcelain` after 7-9 | 0 | empty |
| 11 | greps of Step 2 (below) | see below | all clean |
| 12 | five verifier probes (scratch directory, never committed): `probe_cli.py`, `probe_story.py`, `probe_js.py`, `probe_placebo.py`, `probe_out.py` | 0 | results quoted in the findings |
| 13 | `test -z "$(git status --porcelain)"` after the report commit | recorded in the reply | |

Step 2 greps. `CausalGraph` fields are `graph_id, graph_version, event_key, nodes, edges`; `Edge` fields are `source, target, sign, lag_band, source_doc_id, evidence_span, support`; no coefficient, weight, probability, confidence or magnitude field exists (`grep -n "coefficient\|probability\|weight\|confidence\|magnitude" graph_schema.py`: no match; `support` is the agreement count the plan specifies). `parser_graph.py` does not exist (Task 8 gated); in `src/precedent/mechanism/` the only model seam is `cli.py:41,151` (`typer_from_env().type_text`), and no module imports `httpx`, `requests`, `urllib`, `socket`, `anthropic` or `openai`. `git diff 2ead3a5 HEAD --stat -- src/precedent/cli.py src/precedent/config.py` is empty. `grep -rnE "claude-|gpt-|https?://|\.cache/precedent|localhost:"` over `src/precedent/mechanism`, `references/mechanism`, `stages/4*_mechanism_*` matches only the SVG namespace in `cockpit_template.html`, and `tests/test_workspace_contract.py::test_urls_model_ids_and_data_dir_defaults_live_only_in_config` (which scans `*.py`) passes. Files changed outside the plan's allowed paths since `2ead3a5`: `.gitignore` and `.gitleaks.toml` (commits 2e4734f and b0cd7f0, which came from `origin/main` through the earlier merge, not from this track), `docs/STATUS.md` (permitted by the Task Cycle) and one line of the plan itself (V-12).

## Review Focus items

1. Cycle or asset-to-event edge rejected, never pruned. PASS, SYNTHETIC. `test_graph_schema.py::test_cycle_is_rejected` (:62), `test_self_loop_is_a_cycle` (:73), `test_asset_cannot_cause_an_event_variable` (:80), `test_asset_cannot_cause_an_intermediate` (:85). The validator only raises (`graph_schema.py` `_validate_graph`); nothing prunes.
2. A query's own story is not in the fit. PASS, SYNTHETIC, with V-5. The fit goes through one `find_precedents` call (`panel.py:128`). `test_panel.py::test_fit_uses_only_time_guarded_rows` (:90) plants a +50% analog whose window finishes after the query, shows the coefficients, intercepts and residual covariance are identical with and without it, and has a control (the same analog moves the fit once its window has finished). Verifier probe `probe_story.py`: with `story_cluster_id` set to a finished earlier event's story, that event is absent from the panel (present without it). No Mechanism test pins the story case (V-5).
3. Too few rows gives intercept-only, flagged `underdetermined`. PASS, SYNTHETIC, with V-6. `test_fit_sem.py::test_too_few_rows_is_intercept_only_and_flagged` (:183), `test_exactly_parents_plus_two_rows_is_fitted_not_flagged` (:306), `test_latent_with_too_few_rows_is_underdetermined_with_zero_score` (:195). The text header and the page subtitle show the count of flagged equations; neither names the node (V-6).
4. An intervention with no asset descendant leaves the fan unchanged, and the output says so. PASS, SYNTHETIC. `test_simulate.py::test_intervention_with_no_asset_descendant_is_reported` (:238), `test_unreachable_intervention_leaves_the_fan_unchanged` (:243); text line `test_cockpit.py::test_text_unreachable_intervention_line_and_disclaimer_last` (:216); the page prints the same sentence (`cockpit_template.html` `render()` notes). Ruling T4-1 (an intervened asset counts as reaching itself) is sound: the literal rule would report an effective do() on an asset as unreachable.
5. Model-proposed edge with a span not in the passage or a node outside the vocabulary is dropped and counted. BLOCKED, NO LIVE RUN: `parser_graph.py` is Task 8, gated on the owner and spine Task 18. The schema-level guards exist and pass (`test_unknown_node_is_rejected_not_added` :94, `test_model_graph_needs_evidence_on_every_edge` :150).

## Requirements

Evidence classes: SYNTHETIC (the synthetic world of `tests/mechanism/mechanism_world.py`, built through the real `build_asset_library`, `find_precedents` and `summarize_responses`, or hand-built fixtures), NO LIVE RUN (needs an owner gate or an unmerged Precedent module). There is no LIVE-DATA and no LIVE-MODEL result anywhere in this track.

| ID | Requirement | Verdict | Class | Evidence and rulings |
|---|---|---|---|---|
| V-R1 | Plan GC: the model emits structure, never numbers | PASS | SYNTHETIC | `test_no_weight_probability_or_confidence_field_exists` (`test_graph_schema.py:192`); the grep above |
| V-R2 | Plan GC: closed node vocabulary; an unknown node is rejected | PASS | SYNTHETIC | `test_unknown_node_is_rejected_not_added` (:94), `test_vocabulary_contents` (:198), `test_node_kind_must_match_its_id` (:106) |
| V-R3 | Plan GC: edges are enums; evidence on every non-template edge | PASS | SYNTHETIC | `test_edge_rejects_unknown_fields_and_enum_values` (:182), `test_model_graph_needs_evidence_on_every_edge` (:150); deviations V-8 |
| V-R4 | Plan GC: fitting is time-guarded, through `find_precedents`, nothing reads the library directly for the fit | PASS | SYNTHETIC | `test_fit_uses_only_time_guarded_rows` (`test_panel.py:90`), `test_panel_index_is_exactly_what_find_precedents_returned` (:120); `evaluate.py` reads `rows` only to obtain realised values to score, never to fit |
| V-R5 | Plan GC: rollouts deterministic given a seed; seed in the manifest; 10,000 default paths in `params.json` | PASS | SYNTHETIC | `test_same_seed_same_paths` (`test_simulate.py:147`), `test_params_file` (`test_graph_schema.py:226`); JSON carries `seed`, `n_paths`, `code_commit`, `fitted.graph_version`, `fitted.fitted_asof_utc` (probe `probe_out.py`) |
| V-R6 | Plan GC: an intervention is `do()`, not conditioning; the cockpit never exposes "observe" | PASS | SYNTHETIC | `test_do_cuts_incoming_edges` (:169), `test_do_value_is_broadcast_and_downstream_uses_it` (:181); page: intermediates have a `do()` checkbox, event sliders set the exogenous value, no observe control (code read); `--do` on an event node is exit 2 (T7-4, sound: an event variable is exogenous) |
| V-R7 | Plan GC: every fan chart is beside Precedent's band for the same query, asset and horizon, with both sample sizes; below `min_neff` the mechanism refuses | FAIL | SYNTHETIC | Text and JSON carry both sizes (`n=<fit>/<band>`; `fitted.n_obs`, `precedent[].n`, `n_eff`) and the fit refuses below `min_neff` (`test_command_refuses_when_the_fit_has_too_few_effective_precedents`, exit 2, empty stdout). Broken: V-1 (mechanism printed beside a refused Precedent band) and V-2 (the page shows no Precedent sample size) |
| V-R8 | Plan GC: output carries the disclaimer line | PASS | SYNTHETIC | text last line (`test_key_query_prints_both_bands_and_the_disclaimer`), page `<title>` and footer (`test_html_is_self_contained_and_carries_the_disclaimer`), JSON `disclaimer` field (Literal) |
| V-R9 | Plan GC: only `parser_graph.py` calls a model directly; `cli.py` only through `typer_from_env()` | FAIL | SYNTHETIC | True by grep today, but no test pins it (the plan: a constraint with no test is a FAIL); V-4. `test_typed_text_query_uses_the_typers_key` (`test_cockpit.py:495`) covers only the positive path |
| V-R10 | Stage contracts 40-43 follow the contract standard; stage 40 carries a complete `## Loop`; others say single pass | PASS | SYNTHETIC | `tests/test_workspace_contract.py` passes; stage 40 Loop names kind, oracle, held-out check, unit, mutable and protected files, isolation, runner, state, context, stop numbers, on-stop; ruling P9 sound; V-13 |
| V-R11 | Task 1: graph schema and store | PASS | SYNTHETIC | `test_graph_schema.py` (24 tests), `test_graph_store.py` (6: idempotent, conflict all-or-nothing, slug collision); the reference schema file equals `graph_json_schema()` (:245). Ruling P1 sound |
| V-R12 | Task 2: eight template graphs, no model | PASS | SYNTHETIC | `test_templates.py` (9). Rulings T2-1 to T2-5 accepted. T2-2 is grounded: `references/typing_prompt_v1.txt` rule 8 defines `direction_sign` as 1 favourable and -1 unfavourable, so the plan's `central_bank` `event -> rates_expectation (+)` contradicts the spine and the template uses (-). T2-3 (drop the macro rates edge) is sound |
| V-R13 | Task 3: guarded panel and fit (3a, 3b) | PASS | SYNTHETIC | `test_fit_sem.py` (24), `test_panel.py` (27); recovers a planted structure at the default ridge; sign disagreements recorded not corrected (:204). Rulings P4, P5, T3-1 to T3-4, T3B-1 to T3B-4 accepted; T3-3/T3-4 changed one assertion to a stricter one (centring), not a weakening |
| V-R14 | Task 4: rollouts, do(), counterfactual pairs, shared PRNG | PASS | SYNTHETIC | `test_simulate.py` (25); `test_counterfactual_pair_shares_noise` (:281). Ruling P6 sound; see V-R19 |
| V-R15 | Task 5a: degree-preserving placebo graph, rung files | PASS | SYNTHETIC | `test_placebo.py` (13): degrees kept, still a DAG, event and asset kind rules, deterministic by seed |
| V-R16 | Task 5b: ladder scores P-B1, M-T, M-P on identical units in time order | PASS | SYNTHETIC | `test_ladder.py::test_mechanism_and_precedent_are_scored_on_identical_events` (:166), `test_precedent_unit_helper_agrees_with_the_engine_ladder` (:178), `test_events_are_visited_in_time_order` (:201). Rulings T5B-1 to T5B-4 accepted |
| V-R17 | Task 5: the true structure beats the placebo on the planted mechanism (plan test `test_true_structure_beats_placebo_on_planted_mechanism`) | FAIL | SYNTHETIC | Ruling T5B-6/T5B-5 replaces the test with a characterisation that M-P equals M-T. I accept the diagnosis and reject the outcome: see V-3 |
| V-R18 | Task 7: by-key command on the Precedent engine, text/JSON/HTML | PASS | SYNTHETIC | `test_cockpit_cli.py` (17) and `test_cockpit.py` (37). `test_key_query_prints_both_bands_and_the_disclaimer`, `test_do_flag_is_applied_and_named`, `test_html_is_self_contained`, `test_untyped_text_query_exits_2_and_prints_the_key_flags` (`test_cockpit.py:488`). Ruling T7B-1 (a synthetic world with a loader seam replaces the plan's missing `populated_stores`) is sound: everything after the loader is real engine code. Only the on-disk loader is unproven (V-R24) |
| V-R19 | Task 7: the page's JavaScript simulate equals Python's on the same seed (plan tolerance 5e-4) | PASS | SYNTHETIC | node v26.8.1 ran the test (no skip). Probe `probe_js.py`: five seeds (0, 7, 123456789, 2**40+3, -5) times three intervention sets (none; do on a latent; do on an asset and a latent), max absolute difference 3.5e-18 |
| V-R20 | Toolchain: Python 3.11, uv, ruff, no `addopts`; core-only install passes, research tests skip | PASS | SYNTHETIC | commands 1-4, 7-9. The mechanism package imports nothing from the research extra |
| V-R21 | CLAUDE.md: model ids, base URLs and data-dir defaults only in `config.py` | PASS | SYNTHETIC | `test_urls_model_ids_and_data_dir_defaults_live_only_in_config`; grep above |
| V-R22 | CLAUDE.md: time is tz-aware UTC, naive timestamps rejected | PASS | SYNTHETIC | `test_panel_rejects_naive_asof`, `test_panel_rejects_non_utc_asof` (`test_fit_sem.py:390,398`), `test_mechanism_query_converts_to_utc_and_rejects_naive_time`, `test_result_rejects_unknown_fields_and_naive_timestamps` (`test_cockpit.py:243`). `CausalGraph` records carry no timestamp (V-11) |
| V-R23 | CLAUDE.md: no data or weights committed; tests offline, write only under `tmp_path`; unique test basenames; clean tree after the suite | PASS | SYNTHETIC | `git ls-files` shows no price, article or model file under the track; no network or SDK import in `tests/mechanism`; `find tests -name '*.py'` duplicates only `conftest.py`; command 5 and 10 empty |
| V-R24 | Task 10 Step 1: `precedent mechanism ... --html <tmp>` on the sample library | BLOCKED | NO LIVE RUN | There is no `data/sample/` library (owner gate) and `precedent.libraries.bundle` (Precedent Task 10) is not merged, so `_load_inputs` (`cli.py:190`) has run only against a fake bundle module and synthetic parquet files. Command 6 returns exit 2 with the `not built yet` message and writes nothing, as the code intends |
| V-R25 | Task 6: first real run, milestone M-M1 | BLOCKED | NO LIVE RUN | Owner gate; `scripts/run_mechanism_m1.py` does not exist; no result on real data |
| V-R26 | Task 8: model-proposed graphs | BLOCKED | NO LIVE RUN | Owner gate and spine Task 18; `parser_graph.py`, `graph_prompt_v1.txt` and `m4_mechanism_model.yaml` do not exist |
| V-R27 | Task 9: mechanism factors for Qlib, and CLAUDE.md "every factor builder ships a truncation-invariance test" | BLOCKED | NO LIVE RUN | Conditional on Task 6 or 8 beating Precedent; not built. No factor builder exists in the track, so the truncation-invariance rule has nothing to apply to yet |
| V-R28 | CLAUDE.md: zero spend, nothing pushed or merged, no download; every result in STATUS with a label | PASS | SYNTHETIC | `git branch -r --contains HEAD` empty; STATUS rows for Tasks 1, 2, 3a, 3b, 4, 5a, 5b, 7a, 7b carry a label; no file under `docs/examples/` was added by the track, so no banner is owed |
| V-R29 | Step 2: no file outside the allowed paths changed; `cli.py` untouched | PASS | SYNTHETIC | see Step 2 greps; V-12 for the plan line |

Counts: see the last line of this file.

## Findings

| ID | Severity | File:line | What and why |
|---|---|---|---|
| V-1 | Important | `src/precedent/mechanism/cli.py:313-319`, `cockpit.py:142`, `tests/mechanism/test_cockpit_cli.py:244` | When Precedent's exact-key band is below `min_neff` (or has no precedent) but the direction-free fit passes, the command exits 0 and prints the mechanism fan beside `refused: n_eff=...` (text) or the same string (page). Probe `probe_cli.py` on the synthetic world, direction +1, `--asof` at sessions 396 to 462: exit 0 and all 22 Precedent cells `refused` while 8 to 13 precedents are in the fit. The plan: "A mechanism result without its Precedent band is not reported. Below Precedent's `min_neff` the mechanism also refuses". The test `test_an_exact_key_with_no_precedent_refuses_every_band_but_not_the_command` pins the opposite. Why it matters: the product's rule is that the two models are shown together; a fan with no Precedent band beside it is exactly the unaccompanied model output the constraint forbids. Fix: refuse (exit 2, the same line) when the exact-key band is refused, or print only Precedent's refusal; owner may rule otherwise, in which case the plan sentence changes |
| V-2 | Important | `src/precedent/mechanism/cockpit_template.html:297,377` | The page shows neither Precedent's sample size nor its numbers: the grey band is drawn behind the fan and the row verdict is `yes`, `no (...)` or `refused: n_eff=`; `grep -n "b\.n\b"` on the page script finds nothing, and the subtitle gives only the fit's `n_obs`. Plan Task 7 and Step 3: both sample sizes beside every fan, in text, JSON and HTML. Text and JSON have them (`n=29/12`; `precedent[].n`). Fix: print Precedent's `n` and `n_eff` and its p10/median/p90 in each fan row |
| V-3 | Important | `src/precedent/mechanism/evaluate.py:101-145`, `references/rungs/m4_mechanism_placebo.yaml:3` | The M-P rung cannot discriminate structure. Probe `probe_placebo.py`: for the eight templates, 60 seeds each, a degree-preserving placebo reaches exactly the same assets from the event node as the template in all 360 cases of the six templates that can be rewired, and `PlaceboError` is raised for `earnings` and `macro_data_release` (120 of 120). With one exogenous column and free per-asset slopes the placebo therefore forecasts like the template (`test_on_the_central_bank_template_the_placebo_forecasts_like_the_template`, `test_ladder.py:142`). Ruling T5B-5 diagnoses this correctly and discloses it; I accept the diagnosis and the honesty, but the plan's acceptance ("true structure beats placebo") is unmet, and the rung file's own comment ("must lose to M-T or the structure earned nothing") can never come out as "lost", so a Task 6 reader would conclude the wrong thing. Needs the owner decision the controller already raised (redefine M-P to be able to disconnect assets, add exogenous columns, or constrain signs) before Task 6 |
| V-4 | Minor | tests: none | Nothing pins "only `parser_graph.py` calls a model; `cli.py` only through `typer_from_env()`". Holds by grep today. Add a test that scans `src/precedent/mechanism/*.py` for imports of `httpx`, `requests`, `urllib`, `socket` and SDK names, allowing only `parser_graph.py` |
| V-5 | Minor | `tests/mechanism/` (no file) | Review Focus 2's "own story" case has no Mechanism test: every event in the synthetic world has its own `story_cluster_id`, and the one test that passes a story id (`test_panel.py:310`) only checks that `mechanism_query` carries it. My probe shows `build_guarded_panel` excludes the story through `find_precedents`, and `evaluate._forecast` passes `event["story_cluster_id"]`; add a world with two events in one story and assert the panel and ladder exclude it. The by-key and by-text CLI paths pass no story id at all (by design: a typed query has none) |
| V-6 | Minor | `cockpit.py:176-178`, `cockpit_template.html:297`, `cockpit.py:71` | Review Focus 3 asks that "the chart shows the flag": text and page show only a count of underdetermined equations, not which node. `diagonal_fallback` (the covariance fell back to its diagonal) is in the JSON but in neither text nor page. Per-asset marginal fans are unaffected by the fallback, so this is a disclosure gap, not a wrong number |
| V-7 | Minor | `cockpit_template.html:205,222` | Plan Task 7: sliders run "from the fitted panel's 5th to 95th percentile". The page uses fixed ranges (event enums; intermediates -3 to 3, widened to include an initial `--do`); the payload carries no panel percentiles. A latent score's scale is not -3..3 in general |
| V-8 | Minor | `graph_schema.py:20,100,104,176` | Three looseness points in the schema: (a) graphs whose `graph_version` starts `placebo-` are also exempt from the evidence rule, which the plan exempts only for `template-` (needed for M-LP, but the version string is caller-chosen, so any graph named `placebo-x` skips evidence; pinned by `test_placebo_graph_is_exempt_from_the_evidence_rule`); (b) `Edge.sign` accepts `True` and `1.0` and coerces them to `1` (probe; `"1"`, `2`, `0` are rejected), against "reject unknown enum values"; (c) `Edge.support` has no lower bound (0 and negative accepted) although Task 8 will read it as an agreement count |
| V-9 | Minor | `panel.py:134-141`, `cli.py:222-258` | The SPY row of the by-key table: the library has no market row, so the panel's market column is the raw cumulative return (ruling T3B-3) and the "Precedent band" for SPY is computed by this module with `summarize_responses` on raw returns. Every other row is an abnormal return against the market model. The two are not marked apart in text or page, and Precedent itself would show no SPY band. Label the row or leave SPY out of the Precedent column |
| V-10 | Minor | `cli.py:190-215`, `cli.py:403-410` | `_load_inputs` has never run on a real bundle (BLOCKED, V-R24), and ruling T7B-8 (map `LibraryIntegrityError` to exit 2) is still open: once the bundle exists, a corrupt library gives a traceback. Also `evaluate.py:274` refuses a ladder event on `panel.n_obs < min_neff` (raw count) while `cli.py:309-314` refuses on `effective_sample_size` (effective count); a ladder score and the command can disagree on whether a query is thin |
| V-11 | Minor | `graph_schema.py` (CausalGraph), `graph_store.py` | CLAUDE.md: "`first_seen_utc` is required on every record". `CausalGraph` and `GraphStore` records carry no creation time, and no code appends template graphs to the store or writes stage 40's listed outputs (`graphs_<version>.jsonl`). Harmless for templates; for Task 8 a stored model graph needs a creation time before it can be time-guarded against an event |
| V-12 | Minor | `docs/superpowers/plans/2026-11-07-mechanism.md:231`, `docs/STATUS.md` | The track edited the plan (a test file rename) and its own STATUS rows for Task 3a ("the time-guard test runs only once `build_guarded_panel` exists") and Task 7a ("by-key path reports `not built yet` until the Precedent engine lands") are superseded by the 3b and 7b rows but still read as current. History is append-only, so add a superseded note rather than edit |
| V-13 | Minor | `stages/40_mechanism_graphs/CONTEXT.md` (Loop) | "Oracle protection: ... read-only to the worker" does not say how (a read-only path, a hook, a separate checkout); the rulebook says instructions alone do not protect an oracle. Fill in before Task 8 runs |
| V-14 | Minor | `cli.py:151-156`, `cockpit.py:155` | The plan's text format prints the mechanism table "after Precedent's own block for the same query"; the command prints no Precedent block (its card), only the table column. Acceptable if intended; say so in stage 43 |

## Adversarial probes (scratch, outside the repository)

| Probe | Target | Outcome |
|---|---|---|
| `probe_cli.py` | by-key command across 26 as-of positions on the synthetic world | positions 330-390 exit 2 with the refusal line (n_eff 2.0-7.0 below 8.0); 396-462 exit 0 with every Precedent cell refused (V-1); 468+ exit 0 with 4 real bands |
| `probe_story.py` | `story_cluster_id` through `build_guarded_panel` | the named story's event is excluded (V-5 is about the missing test) |
| `probe_js.py` | page JS vs Python, 5 seeds including 2**40+3 and -5, three intervention sets | max difference 3.5e-18 |
| `probe_placebo.py` | placebo reach over 8 templates x 60 seeds | same reach 360/360 where defined; `PlaceboError` for earnings and macro_data_release (V-3) |
| `probe_out.py` | text/JSON/HTML shape; `--do` on an unknown node; `--n-paths 0`; untyped text; typed text on a key the world lacks | text has both bands and the disclaimer last; JSON keys `asof_utc, code_commit, diagonal_fallback, disclaimer, event_values, fitted, graph, interventions, n_paths, precedent, rollout_fan, seed, unreachable_interventions`; page has no `<script src=`, `fetch(`, `XMLHttpRequest` or `import(`, and the disclaimer in `<title>` and footer; bad `--do` and `--n-paths 0` exit 2; untyped text exit 2 with the key flags; a typed key with no precedents exit 2 with `n_eff=0.0` |
| `Edge(sign=...)` | enum strictness | `True` and `1.0` accepted (V-8b) |

## Not verified

- Live data and live model: none run. Every number in this track comes from the synthetic world or hand-built fixtures; the plan's thresholds, templates and the `n_paths` default are proposals.
- The on-disk loader `_load_inputs` against a real `precedent.libraries.bundle` and a real price directory.
- The HTML page in a browser: the page's script was run under node (simulation) and read, but its DOM code (sliders, SVG, the fan rows) was not rendered; the in-app browser showed it as a static snapshot with no script execution.
- Tasks 6, 8, 9 and their stages' held-out checks; the stage 40 loop contract is a proposal that has never run.
- Precedent's own engine code (retrieval, aggregation, asset library) beyond what the Mechanism tests exercise.

Counts: PASS 22, FAIL 3, BLOCKED 4 (29 requirement rows; Review Focus: PASS 4, BLOCKED 1).


## Addendum 2026-11-08 — fix-wave ruling I-2 (implementation record)

Source: "Opus final review, Mechanism hand-over note I-2; implemented by Kimi". This addendum
records how finding V-1 (above) was resolved; it is not an owner ruling. The owner-rulings
file stays owner-only.

V-1 as first worded in the Opus review would over-refuse. Ruling I-2 (hand-over note,
`predate-git/.superpowers/handoff-2026-11-08/notes/mechanism-handover-2026-11-08.md`): refuse
per asset row instead. Implemented on `kimi/mechanism-fix-wave`:

- A row survives only if its Precedent band is not refused (exact-key precedent exists,
  `n_eff >= min_neff`) and the asset has observations. `min_neff` is the engine's, from
  `references/precedent_params.json` via `load_engine_params`.
- A non-surviving row shows no fan. It prints the reason — "no data" (zero observations), or
  "band refused: n_eff=X < min_neff=Y" — in the text output and on the page, and the JSON
  lists the assets: `no_data_assets` and `refused_assets`.
- Exit 2 only when no row survives, with two distinct messages: every asset with data has a
  refused band (`MechanismRefused`, band=True), or every asset lacks rows (`NoDataAssets`).
  In particular, with no exact-key precedent at all every row is refused, so the command
  still exits 2 — consistent with this ruling.
- Suppressed assets are still simulated (simulate-then-drop), so the counter-based PRNG
  draws stay identical for the surviving rows; the page's JavaScript twin suppresses the same
  rows and a node test pins the parity.
- The market row's band and fan are its own cumulative return, labelled `(raw)`; the JSON
  band rows carry `basis` (`abnormal_return` by default, `raw_return` for the market).

A mixed panel (one thin asset among good ones) succeeds: good rows show fans, the thin row
prints the refusal with no fan and no fabricated "+0.0%".


## Addendum 2026-11-08 — scoped re-review of the fix wave and re-run of the FAIL and BLOCKED rows (at 28d78a6)

Run by an independent Sonnet reviewer and verifier for the supervising session; it wrote none of the fixes. Probe scripts were kept outside the repository and are not committed.


### Review and verification: mechanism fix wave, 44c3eba..28d78a6

Checkout: scratchpad/wt-mech, detached at 28d78a6. I wrote none of this code. Reverts were done by script (review-work-mech2/exp.py), each restored with `git checkout -- <file>`. `git status --porcelain` is empty at the end. No network, no model call, no data download, no commit.

### Part 1. Per-item table

"Revert" = I removed the fix hunk in wt-mech and ran the named test (no `-x`); "fails" means the named test went red for the right reason. H = template `src/precedent/mechanism/cockpit_template.html`.

| Item | Requirement (quoted) | Code | Pinning test | Fails when reverted? | Verdict |
|---|---|---|---|---|---|
| V-10 pin | ladder "refuses on n_eff (effective_sample_size) like the CLI, not raw n_obs" | evaluate.py:277-284 (`n_eff < min_neff`) | test_ladder.py::test_the_ladder_guard_refuses_on_the_effective_size_not_the_raw_count | Yes. `if panel.n_obs < min_neff` makes it FAIL (1 failed, 16 passed). The test asserts `n_obs >= min_neff` and `n_eff < min_neff` first, so it is not vacuous | fixed |
| M-5 pin | "Use `event_values_of` everywhere"; the earlier review found the test vacuous | cli.py:195-207 `_event_values` -> panel.event_values_of | test_cockpit_cli.py::test_event_values_follow_the_panel_encoding_for_every_key_field (graph with severity_band and is_scheduled event nodes, all 3x2 key values) | Yes. A private map with high=5.0 FAILS. The earlier vacuity is closed | fixed |
| raw label (page) | "(raw)" label unpinned | H `mechanismFanRows` (` (raw)` on name and Precedent cell); cockpit.py `render_html` RAW_NOTE | test_cockpit.py::test_the_market_row_is_labelled_raw_on_the_page; test_cockpit_page.py::test_page_market_row_stays_labelled_raw_beside_its_band (runs the page under node) | Yes. Removing both ` (raw)` literals fails both tests; blanking RAW_NOTE fails the first | fixed. Note: the first test checks only that the substring ` (raw)` exists in the template file; the node test is the real pin |
| min_neff default | "no hard-coded min_neff"; must come from engine params | cockpit.py:90 `min_neff: float`, :123 required keyword; cli.py:277,431 `params.min_neff`; evaluate.py:205,400 | structural: every construction in tests passes `load_engine_params().min_neff` | Omitting it is a `TypeError` (field and keyword have no default). `grep -n "8\.0" src/precedent/mechanism/*.py`: none. Only literal left is the test fixture value 3.2 / 8.0 text in a hand-built row | fixed |
| docstring | stale pointer to owner-rulings file | test_cockpit_cli.py:275-277 now names the addendum of docs/verification/mechanism-2026-11-08.md | n/a (docstring) | n/a | fixed |
| V4 | "a test pinning that only config-named typers/extractors reach a model from mechanism (grep test)" | cli.py:41 is the only `precedent.spine.typers` import | test_model_seam.py (3 tests, incl. a self-test of the scanner) | Yes. `import requests` in panel.py fails test_no_network...; `from precedent.spine.typers.anthropic import AnthropicTyper` fails test_the_only_model_seam | fixed; scanner has gaps, see N-1 |
| V5 | "a mechanism test of own-story exclusion" | panel.py:80-84 `story_cluster_id` into PrecedentQuery | test_panel.py::test_query_story_excludes_that_story_from_the_fit_panel (also asserts the event IS in the panel without the story id) | Yes (`story_cluster_id=None` makes it FAIL) | fixed |
| V12 | stale STATUS rows 3a and 7a refreshed; plan line 231 edit noted | docs/STATUS.md rows 3a, 7a | none (docs) | n/a | fixed. The 7a row still says "At the time ... `not built yet`; superseded by 7b", accurate |
| V13 | "stage 40 Loop names the oracle protection mechanism ... still marked proposed" | stages/40_mechanism_graphs/CONTEXT.md Oracle protection | none (docs); workspace-contract test still green | n/a | fixed. The claim "neither the gold set, the pin file nor the test exists" is true: `references/gold` holds only `typing_cases.json`, no hash-pin file |
| I-1 wording | "M-P equals M-T by construction (a theorem)" fixed in the rung comment, STATUS 5b row and stage 42 | m4_mechanism_placebo.yaml:3-6; STATUS 5b; stage 42 Process step 2 note | test_ladder.py::test_on_the_central_bank_template_the_placebo_forecasts_like_the_template (characterisation, unchanged) | n/a (wording) | fixed. I reproduced the premise: all 8 templates have latents with exactly one parent (lat.py); 40 seeds x 6 templates give the same reached asset set and a real rewiring (reach.py); earnings and macro_data_release raise PlaceboError (no placebo at all) |
| M-8 | "stage 41 CONTEXT:12 still names params.json as authoritative for the asset panel" | stages/41_mechanism_fit/CONTEXT.md:12-13 now names `references/precedent_params.json` via `load_engine_params` and limits params.json to `n_paths` | existing test_mechanism_asset_panel_cannot_drift_from_the_engine_panel | n/a (docs) | fixed. params.json still has an `asset_panel` key (stated as a placeholder) |
| M-10 | "ledger figure is stale ... M-T 0.00260 vs P-B1 0.00271, coverage_80 0.74 vs 0.64" | STATUS 5b row: M-T 0.00260, P-B1 0.00271, coverage 0.73 / 0.64 | none | n/a | REPRODUCED (m10.py, `ladder` helper over `make_world()`, h=5, 500 paths): pinball {M-T 0.002599, M-P 0.002599, P-B1 0.002712}; coverage_80 {0.727, 0.727, 0.636}. The row's 0.73 is right; the handover's 0.74 was a rounding of an older run |
| V2 | "the HTML page shows Precedent's p10/median/p90 and both sample sizes per asset, like the text" | H `mechanismFanRows`, `drawFans`; cockpit_template grid columns | test_cockpit_page.py::test_page_rows_show_precedent_p10_median_p90_and_both_sample_sizes, ::test_page_suppressed_rows_keep_their_reason_and_no_fan, ::test_page_market_row_stays_labelled_raw_beside_its_band | Yes. Precedent cell -> "n/a": FAIL; sizes -> "": FAIL; removing the refused reason: FAIL | fixed. I-2 per-row refusal still correct on the page: see below |
| V6 | "text names the underdetermined equations and prints a line when the covariance fell back to the diagonal" (text and page) | cockpit.py `disclosure_lines`, DIAGONAL_LINE; H `mechanismDisclosures` | test_cockpit_page.py::test_text_names_the_underdetermined_equations, ::test_text_prints_one_line_when_the_covariance_fell_back_to_the_diagonal, ::test_page_mirrors_both_disclosure_lines | Yes (text `return []` fails the text test; page `return []` fails the parity test) | fixed |
| V7 | "sliders run from the fitted panel's 5th to 95th percentile" widened to the query value | fit_sem.py:314-318 `node_ranges` (np.nanpercentile 5/95 of the fitted series, assets excluded); cockpit.py `slider_ranges=dict(fitted.node_ranges)`; H `sliderBounds` | test_cockpit_page.py::test_fit_records_each_non_asset_nodes_panel_percentiles, ::test_payload_embeds_the_ranges_and_the_page_has_no_fixed_range, ::test_slider_range_is_the_panel_percentile_widened_to_include_the_query_value, ::test_a_node_without_a_panel_range_is_pinned_to_its_value_not_a_default_range | Yes, four separate reverts (page fixed [-3,3]; fit node_ranges {}; payload {}; no widening) each fail a different test | fixed. End to end on the CLI (sliders.py, `--do rates_expectation=5.0`): `slider_ranges == fitted.node_ranges`; page controls: rates min -0.690 max 5 (widened), risk -1.310..0.690, event -1..1 step 1; no `EVENT_RANGES` or `Math.min(-3` left. A latent range is its centred fitted score, which the new test recomputes independently |
| M-1 | "embed the factor in the payload" | simulate.py `Rollout.factor` from `_draw_residuals`; cockpit.py `residual_factor=rollout.factor.tolist()`; H `residualDraws(factor, ...)` | test_payload_factor_is_the_one_simulate_used (bit equality with numpy, PD and non-PD); test_js_fans_match_python_on_seeded_near_boundary_matrices_and_the_page_never_factorises (120 matrices) | Yes. Scaling the embedded factor by 1.01 fails 3 tests incl. both pins; re-adding a `choleskyFactor` to the template fails the "cholesky not in html" check | fixed. The page has no factorisation (grep: no "cholesky" in the template; `residualDraws` only multiplies). The test's matrices matter: the removed JS factoriser disagrees with numpy's PD decision on 3 of those 120 (oldchol.py), so the parity test would fail against the old code |
| Ruling MF-1 | fixture gains `node_ranges` with the old fixed values, no assertion changed | tests/mechanism/test_cockpit.py `NODE_RANGES`, `make_fitted(... node_ranges=dict(NODE_RANGES))` | test_js_do_intervention_and_controls_follow_python | Removing the fixture line makes exactly that one test fail, because controls are now pinned without ranges | JUDGED SOUND. The fixture supplies data the new field requires; the diff has no changed assertion (see section "tests in the range"). Side effect worth knowing: that test now checks payload-to-page plumbing only; the real range behaviour is tested in test_cockpit_page.py |

Targeted checks requested:
- Per-row refusal (I-2) after V2: cli sweep (sweep2.py, 12 as-of positions, both directions) on the synthetic world. Thin fit exits 2 with "the fit has"; thin exact-key band exits 2 with "Precedent's exact-key band"; both with empty stdout. Where it exits 0 (positions 468, 500, 700 at direction +1) the JSON suppressed set, the page rows (run under node) and the text rows agree, every graph asset is covered once, and `rollout_fan` keys equal assets minus suppressed. Page row for a survivor: `['HYG','','+0.5% / +1.1% / +1.7%','+0.7% / +1.0% / +2.0%','yes','n=29/21 n_eff=21.0']`; suppressed rows show `no data`. The mixed refused/surviving case is only reachable by hand-built fixtures (test_page_suppressed_rows_keep_their_reason_and_no_fan, HYG refused, TLT and SPY survive).
- PRNG draw parity (simulate-then-drop): `build_result` simulates every graph asset and only filters `rollout_fan` afterwards (cockpit.py:140-150); the page drops after `simulate` (H:192-193). 12 runs through the real CLI (jsparity.py: seeds 0, 7, 123456789, 2**40+3 x three do-sets, 2000 paths, all with diagonal_fallback True): max |JS - Python| = 3.5e-18.
- Embedded factor used by JS, no factorisation in the template: confirmed by reading `residualDraws` (throws if the payload factor is not nAssets x nAssets) and the "cholesky" grep test.
- Sliders: confirmed above. No hard-coded min_neff: confirmed.
- Docs numbers I could not reproduce: none that are checkable offline. The 5b-row figures reproduce (above). "Seeded 120-matrix near-boundary test" is 120 (test_cockpit_page.py). The handover's "8 to 43 of 200 matrices" disagreement count was not reproduced (my generator is the test's, 3 of 120); STATUS does not repeat that number.

### Tests in the range (git diff 44c3eba 28d78a6 -- tests)

- Removed tests: none (`git diff` has no removed `def test_`).
- Renamed tests: none.
- Loosened or changed assertions: none. Only two removed lines in test_cockpit.py: `min_neff=8.0` (now `load_engine_params().min_neff`, a stricter provenance, not a loosening) and a one-line `precedent=[], seed=1, ...` split to add `min_neff`.
- Fixture changes: `make_fitted` gains `node_ranges` (MF-1); `hand_result`, `simulated_result`, `no_data_fixture`, `build_result` calls gain the now-required `min_neff` argument.
- New test files/tests: test_cockpit_page.py (18 tests), test_model_seam.py (3), plus 1 each in test_cockpit.py, test_cockpit_cli.py, test_ladder.py, test_panel.py. Count 215 -> 234 in tests/mechanism; no skip, no xfail added.

### New defects and observations

| ID | File:line | Concrete failing input | Severity |
|---|---|---|---|
| N-1 | tests/mechanism/test_model_seam.py:`seam_imports`, `forbidden_imports` | Add to any mechanism module `from precedent.spine import typers`, or `import precedent.spine.typers.anthropic as T`, or `importlib.import_module("httpx")`: all three leave test_model_seam.py green (probed, panel.py reverted each time). Only `import httpx as _h` and `from precedent.spine.typers import anthropic` are caught. The V-4 guarantee is a static name scan, not a proof | Minor |
| N-2 | src/precedent/mechanism/cockpit_template.html `sliderBounds` (step = span/60 to 1 s.f., min not on the step grid) | CLI `--do rates_expectation=5.0` gives min -0.6897, max 5, step 0.09: 5 is off the browser's step grid (min + k*step). A real range input snaps its value to the grid, and `read()` (on the do() checkbox `change`) takes `slider.value`, so ticking do() on an untouched slider can silently set the value to the snapped number, not the query's. Same for a free latent at 0 on a grid anchored at -1.3103. NOT verified in a browser (only a node DOM stub). Fix: round min down to the step grid or use `step="any"` | Minor, unverified |
| N-3 | stages/43_mechanism_cockpit/CONTEXT.md step 3 vs cockpit.py `_row` | Step 3 says each row shows "fit n, band n, and Precedent's `n_eff`". The text row prints `n=29/21` only (no n_eff, sweep2.py text rows); only the page prints n_eff for surviving rows. STATUS B1 says "the page shows", so the status row is right and the contract wording is broader than the text output | Minor |
| N-4 | cli.py `_load_inputs` docstring (:224-229), docs/STATUS.md rows for 7b `_load_inputs` and Task 10 Step 1, test_cockpit_cli.py:405-408 | These say `precedent.libraries.bundle` "is not merged into this track". It is: `git merge-base --is-ancestor 032f110 a3eb6a4` is true (bundle, manifest, sample builder all exist in the checkout). The real loader now runs (see V-R24 below). Stale claim, not a behaviour bug; predates this range but the range re-touched these rows | Minor |
| N-5 | cockpit.py `residual_factor: list = []`, `slider_ranges: dict = {}` defaults | A MechanismResult built without them (or an old JSON) validates; the page then throws "payload has no residual factor of size N" for any non-empty asset set, and sliders pin to their value. Failing loudly is intended for the factor; defaults on a required-in-practice field is looseness | Minor |
| N-6 | H `mechanismFanRows` | If an asset is in `refused_assets` but has no band row, the page prints the generic reason "no data" (fan absent from the table) where the text prints a refusal. Not reachable from the CLI (refused assets always have a refused band row); only a hand-built payload triggers it | Minor, unreachable |
| N-7 | test_ladder.py V-10 pin docstring | The test says a natural scored unit never reaches the effective-size guard (P-B1's own rule scores it first), so in the real ladder the guard is close to dead code and is exercised only through the patched-panel seam. Honest, but the V-10 behaviour has no natural-path test | Minor |

No Critical or Important defect found. I could not construct an input where the command refuses while a row would have survived, or prints a fan beside a refused band.

### Part 2. Requirement rows re-run

Same evidence classes as the original report. SYNTHETIC = synthetic world through real engine code; NO LIVE RUN = needs an owner gate or live library.

| ID | Requirement | Verdict | Label | Evidence |
|---|---|---|---|---|
| V-R7 | Every fan chart beside Precedent's band for the same query, asset and horizon, with both sample sizes; below `min_neff` the mechanism refuses (as amended by I-2: per asset row; exit 2 only when no row survives) | PASS (was FAIL) | SYNTHETIC | Text, JSON and page all carry both sizes (page also n_eff and Precedent's p10/median/p90): sweep2.py, node-run page rows equal text rows. Refusals: fit thin exit 2, exact-key band thin exit 2, empty stdout (12 positions). Pins and revert results above (V2, I-2 rows). Caveat: "refuses" now means per row, by ruling I-2; the text omits n_eff on surviving rows (N-3) |
| V-R9 | Only `parser_graph.py` calls a model directly; `cli.py` only through `typer_from_env()` | PASS (was FAIL) | SYNTHETIC | test_model_seam.py pins it and fails on both a network import and a second typers import. Caveats: `parser_graph.py` does not exist yet (Task 8), so "only parser_graph" is vacuous today; scanner gaps N-1 |
| V-R17 | True structure beats the placebo on the planted mechanism | FAIL (stays) | SYNTHETIC | No test named `test_true_structure_beats_placebo_on_planted_mechanism` exists (`grep -rn true_structure_beats tests`: only the unrelated "...leaves_the_event_out"). M-P equals M-T by construction: M-P pinball 0.002599 = M-T 0.002599, coverage 0.727 = 0.727 (m10.py); all 8 templates' latents have one parent; 240 placebo draws reach the same asset set, and 2 of 8 templates have no placebo (reach.py). Wording is honest and not hidden: rung file comment, STATUS 5b row ("cannot rank anything ... pooling plus a slope, not structure ... open OWNER decision (M6)"), stage 42 step 2 note, and the unchanged characterisation test. No document I found still claims M-T beats M-P (grep for "beats the placebo", "must lose to M-T", "forecasts exactly like": only the verifier's own report rows). Owner decision M6 open |
| V-R24 | `precedent mechanism ... --html` on the sample library | BLOCKED | NO LIVE RUN | There is still no `data/sample/` or any real library (`ls data`: absent) and the sample is an owner gate. The recorded reason "bundle not merged" is stale (N-4). New partial evidence, SYNTHETIC: with the REAL `write_bundle` and `load_bundle`, the synthetic world and synthetic SPY prices in a scratch dir (realbundle.py), the unmodified `run()` exits 0, writes a 34,685-byte page and prints 4 bands plus 18 "no data" rows; flipping one byte of events.parquet gives exit 2, empty stdout, "events.parquet: sha256 differs from the manifest", no traceback. An empty data dir gives exit 2 "no library ... MANIFEST.json is absent". Still no real library, so the row stays BLOCKED |
| V-R25 | Task 6 first real run (M-M1) | BLOCKED | NO LIVE RUN | Owner gate; `scripts/run_mechanism_m1.py` absent (`ls scripts`) |
| V-R26 | Task 8 model-proposed graphs | BLOCKED | NO LIVE RUN | `parser_graph.py`, `graph_prompt_v1.txt`, `m4_mechanism_model.yaml` absent (`ls src/precedent/mechanism`, `references/mechanism`); owner gate |
| V-R27 | Task 9 mechanism factors for Qlib | BLOCKED | NO LIVE RUN | No factor builder in the track; conditional on Task 6 or 8 |
| V-R5 | Deterministic rollouts given a seed; seed, n_paths, code_commit in manifest | PASS (spot) | SYNTHETIC | `uv run pytest -q tests/mechanism` includes test_same_seed_same_paths and test_params_file; the JSON from the CLI carries `seed`, `n_paths`, `code_commit`; 12 seeded runs reproduce in JS to 3.5e-18 |
| V-R8 | Disclaimer line in text, page title and footer, JSON field | PASS (spot) | SYNTHETIC | test_key_query_prints_both_bands_and_the_disclaimer, test_html_is_self_contained_and_carries_the_disclaimer green; the new disclosure lines are inserted before the disclaimer, and test_text_prints_one_line_when_the_covariance_fell_back asserts the disclaimer is still last |
| V-R18 | Task 7 by-key command, text/JSON/HTML | PASS (spot) | SYNTHETIC | test_cockpit_cli.py and test_cockpit.py green (suite below); sweep2.py runs text, JSON and HTML on the same query |
| V-R19 | Page JS simulate equals Python on the same seed (tolerance 5e-4) | PASS (spot) | SYNTHETIC | node v26.8.1, no skip. My own 12 runs through the CLI (4 seeds x 3 do-sets), max diff 3.5e-18; plus the new 120-matrix near-boundary test |

Counts for the 11 rows re-run: PASS 6, FAIL 1, BLOCKED 4. (The original report's own counts, PASS 22 / FAIL 3 / BLOCKED 4, are not updated by this review; docs/verification/mechanism-2026-11-08.md is unchanged in this range.)

### Commands and exit codes (wt-mech)

| Command | Exit | Result |
|---|---|---|
| `uv run ruff check .` | 0 | All checks passed! |
| `uv run pytest -q` | 0 | 2126 passed, 1 skipped in 141.62s (the one skip is outside mechanism) |
| `uv run pytest -q tests/mechanism -rs` | 0 | 234 passed in 17.75s; no skip listed, so no node test skipped |
| `git status --porcelain` | 0 | empty (checked after every revert batch and at the end) |
| revert experiments (exp.py, 18 variants, all restored) | | all red for the intended reason, table above |
| probes: sweep2.py, jsparity.py, sliders.py, realbundle.py, m10.py, reach.py, lat.py, oldchol.py | 0 | results quoted above |

Scratch files: <scratch>/review-work-mech2/ (exp.py and the probes; realbundle_data/ holds the synthetic bundle).

### Could not verify

- The page in a real browser: only the script under node with a DOM stub. N-2 (range input step snapping) therefore stays unverified, and SVG layout of the new six-column row grid was not seen.
- Any real library, real prices or model output: none exist offline (V-R24 to V-R27).
- The handover's "8 to 43 of 200" and "diagonal fallback almost always on at n<=22": not reproduced; I only reproduced a PD-decision mismatch (3 of 120) between numpy and the removed JS factoriser, and that the fallback is on in every one of my 12 CLI runs on the synthetic world.
- Mixed refused/surviving rows from a natural world: the synthetic world cannot produce them; covered by hand-built fixtures only.
