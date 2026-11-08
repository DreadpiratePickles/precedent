# Stage: 42_mechanism_evaluate

## Objective
Score the mechanism's forecasts against the ablation ladder and against Precedent's analog history, on the same sessions, so the result says what the mechanism adds; no model is called in this phase and nothing is published.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/41_mechanism_fit/output/fits_<graph_version>.jsonl` | 4 | Authoritative for fitted graphs | Yes | All |
| Precedent engine outputs | 4 | Authoritative for the analog baseline | Yes | Same event keys |
| `references/rungs/` ladder definitions | 3 | Authoritative for the rungs compared | Yes | All |
| Test-period boundary from the spine | 3 | Authoritative for what is held out | Yes | Design 10 |

## Process
1. Deterministic: run the mechanism and each ladder rung over the same purged folds.
2. Deterministic: compare against Precedent per event key and in total, with the leak checks the spine uses.
3. Deterministic: write the comparison table with its label; the held-out period is read once.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/42_mechanism_evaluate/output/comparison.parquet` | One row per (event key, rung) with metric and sample size | `43_mechanism_cockpit` |

## Verify
- Leak tests pass; the comparison reproduces from the stored fits.
- `uv run pytest -q` exits 0.

## Approval
The owner reads the comparison before any claim is made from it. Results are labelled in `docs/STATUS.md`. No external action occurs in this stage.

## Failure Behavior
A missing or misaligned input stops the stage with its identifier; no score is computed from partial data. No retries. Output is written atomically.

Loop: single pass. A scored comparison read once by a human; nothing is tuned against it.
