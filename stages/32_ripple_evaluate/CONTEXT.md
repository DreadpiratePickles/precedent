# Stage: 32_ripple_evaluate

## Objective
Measure whether each Ripple factor adds rank information beyond the rungs below it, and report every graph rung beside its shuffled placebo.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `<data_dir>/ripple/ripple_factors.parquet` and `<data_dir>/ripple/shocks.parquet` | 4 | Authoritative for Ripple factor values | Yes | All columns |
| Factor panels of the lower rungs (news and Precedent factors) | 4 | Authoritative for their own factors | Yes | Same sessions and instruments |
| `references/rungs/m3_ripple_*.yaml` | 3 | Authoritative for which columns each rung adds | Yes | One file per rung |

## Process
1. Deterministic: for each rung, build purged folds and fit the rung with the spine ablation machinery.
2. Deterministic: build each graph rung's placebo by shuffling the links among receivers with the same seed, and score it identically.
3. Deterministic: compute rank IC per rung and per placebo; write the ablation table.
4. Deterministic: write the run manifest (input hashes, rung files, seed, code version) and a report that puts each rung beside its placebo.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/32_ripple_evaluate/output/<run_id>/ablation.csv` | One row per rung and placebo: rank IC and fold spread | `33_ripple_report`, the owner |
| `stages/32_ripple_evaluate/output/<run_id>/run_manifest.json` | JSON: input hashes, rung files, seed, code version | The owner |
| `stages/32_ripple_evaluate/output/<run_id>/report.md` | Markdown, evidence banner first | The owner |

## Verify
- Leak checks from the spine pass on the Ripple panels.
- A rung is never reported without its placebo.
- `uv run pytest -q` exits 0.

## Approval
No external action occurs in this stage; it is compute-only and no vendor price is displayed. The owner reads the report beside its placebo before any further rung.

## Failure Behavior
An invalid input row stops the stage with its identifier and cause; a missing input is never replaced with zeros. No retries. Outputs are written atomically.

Loop: single pass. The evaluation runs once per graph and the owner reviews it; repeating it would fit to the result.
