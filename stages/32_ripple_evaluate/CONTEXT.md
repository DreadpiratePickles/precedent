# Stage: 32_ripple_evaluate

## Objective
Measure whether each Ripple graph's factors add rank information beyond a price-only baseline, and report every graph beside its degree-preserving placebo, at 1, 5, 20 and 60 sessions.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| Parquet prices: `--provider-dir` (default `<data_dir>/prices`) | 4 | Authoritative for sessions and close-to-close returns | Yes | One `<SYMBOL>.parquet` per instrument, including `--market` (default `SPY`) |
| Qlib provider: `--qlib-dir` (default `<data_dir>/qlib_us`) | 4 | Authoritative for Alpha158 features and the forward label; must be built from the same prices | Yes | `calendars/day.txt` and the symbol bins |
| Event store: `--events-dir` (default `<data_dir>/events`), file for `--parser-version` (default the `rules-8k` version) | 4 | Authoritative for events and `effective_session` | Yes | Resolved impacts with an effective session inside `--start` to `--end` |
| Sector map: `--sector-map`, JSON `{instrument: sector}` | 4 | Authoritative for G1 membership; static, so membership looks ahead | Yes when `g1` is selected | All entries naming a priced instrument |
| `references/ripple_params.json` | 3 | Authoritative for alpha, hops, half-life, age cap, G2 window and peers, and the placebo noise floor | Yes | `ripple` section |
| `references/rungs/m0_price.yaml`, `m1_news.yaml`, `m3_ripple_*.yaml` | 3 | Authoritative for which columns each rung adds | Yes | One file per rung |

## Process
1. Deterministic: load sessions, close-to-close returns and cutoffs (`calendar_cutoffs`); build price shocks from the events (`price_shocks`) and the abnormal-return frame (`abnormal_return_frame`); a shock that cannot be built is counted, not filled.
2. Deterministic: build G1 (sector membership, `--graphs g1`) and G2 (residual-correlation peers, rebuilt every `--rebuild-every` sessions from the `--corr-window` sessions ending then) as dated links in `<out_dir>/links/<graph>/`; read them only through `graph_asof`.
3. Deterministic: build each graph's placebo (`<graph>_placebo`) by degree-preserving shuffle with `--seed`, once per snapshot.
4. Deterministic: for each graph and horizon, build the `ripple_` panel, then fit the baseline rung (price only, M0) and the graph rung on purged walk-forward folds (`--train-len`, `--valid-len`, `--test-len`, `--step`; purge gap from the label look-ahead) with the spine ablation; the label for horizon `h` is `Ref($open, -(h+1))/Ref($open, -1) - 1`.
5. Deterministic: write the ladder, the per-fold rank ICs and the report. A graph beats its placebo at a horizon only when the margin exceeds the larger of `placebo_margin_floor` and `placebo_margin_se_multiple` standard errors of the per-fold difference (the floor alone with one fold); otherwise the report says "within noise".
6. Only with `--publish`: write the ledger files `shocks.parquet`, `abnormal_returns.parquet` and `MANIFEST.json` under `<data_dir>/ripple/`.

`uv run python scripts/run_ripple_m1.py --dry-run ...` checks that every input exists, prints the plan and writes nothing.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `<out_dir>/report.md` (default `stages/32_ripple_evaluate/output/<run_id>/`) | Markdown, evidence line first (`LIVE —` or `SYNTHETIC —`), `EXPLORATORY (survivorship-biased prices)`, the research-check note, the tables with 20 first, each graph beside its placebo with margin and tolerance, counts | `33_ripple_report`, the owner |
| `<out_dir>/ladder.csv` | `graph, horizon, rank_ic_mean, icir, incremental_rank_ic_vs_m1, baseline`; one row per graph and horizon | `33_ripple_report` |
| `<out_dir>/ladder_folds.csv` | `graph, horizon, fold, rank_ic_mean`; the per-fold rank ICs behind the noise tolerance | The report writer |
| `<out_dir>/<graph>_h<horizon>/ablation.csv` and `run_manifest.json` | Spine ablation result and manifest (rungs, folds, panel hashes, trial count) | The owner |
| `<out_dir>/links/<graph>/links.jsonl` | Dated links (`Link` schema) | Audit |
| `<out_dir>/run_manifest.json` | JSON: run id, evidence label, parser version, seed, horizons, counts, link-store and event-file hashes, code commit | The owner |
| `<data_dir>/ripple/{shocks.parquet, abnormal_returns.parquet, MANIFEST.json}` | Parquet and JSON; written only with `--publish`; the manifest carries `evidence_label` (`LIVE-DATA` or `SYNTHETIC`) and file hashes | `31_ripple_factors`, `33_ripple_report` |

Nothing is written under `<out_dir>` until the inputs check passes, and `<out_dir>` must not already hold files.

## Verify
- A graph is never reported without its placebo: the report writer raises if any graph lacks a `<graph>_placebo` row.
- The planted-spillover test shows the true graph beats its shuffle on generated data (true about 0.47, placebo about 0.00 rank IC); the verdict tests show a +0.0003 margin reads "within noise" and the "No graph beats its placebo at any horizon." headline.
- The correlation snapshot at a session is unchanged by later returns (tested); links are read only through `graph_asof`.
- `uv run ruff check . && uv run pytest -q` exits 0.

## Approval
The real run on live prices and 8-K events is an owner gate: the owner supplies the price source and the sector map, and runs `scripts/run_ripple_m1.py` after a `--dry-run`. The run is compute-only; it displays and redistributes no vendor price (outputs are derived statistics), and the report is labelled `LIVE-DATA`, `EXPLORATORY (survivorship-biased prices)`. `--publish` writes the persistent ledger and needs the owner's yes. The owner reads the report beside its placebos before any further rung; if neither control graph beats its placebo at any horizon, the owner decides whether Tasks 7 and 8 are still worth building.

## Failure Behavior
Exit 0 on success, 2 on a usage error (a bad or missing flag), 1 on a data error with the message on stderr: a missing or empty price, event or Qlib input, an unknown market, a missing sector map, no shock that could be built, a graph that produced no ripple rows, or a non-empty output directory. A missing input is never replaced with zeros. No retries. Report, ladder and manifests are written atomically; a failed run leaves its partial `<out_dir>` for inspection and is never resumed in place.

Loop: single pass. The evaluation runs once per graph and the owner reviews it; repeating it would fit to the result.
