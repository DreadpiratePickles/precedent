# Stage: 08_evaluate

## Objective
Score the trained model and each ablation rung with purged walk-forward rank IC under identical folds and costs, and run the leak tests.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/07_train/output/results.tsv` and the chosen model | 4 | Authoritative for the trial count and the model | Yes | Best-so-far row |
| Dataset handler from `06_dataset` | 4 | Authoritative for features and labels | Yes | Valid and test segments |
| Ablation ladder definition (M0 to M4) | 3 | Authoritative for rungs | Yes | Design 7.4 |

## Process
1. Deterministic: for each rung compute mean rank IC, ICIR, and the top-minus-bottom spread under identical folds.
2. Deterministic: split every metric at the base model's cutoff; a rung that does not improve on the one below it after the cutoff is reported as not helping.
3. Deterministic: run the leak tests of design 4.5 that apply here: positive control (features shifted one session earlier on purpose; rank IC must jump) and the lag curve (features shifted 1, 2 and 5 sessions later).
4. The final test period is read once, here.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/08_evaluate/output/evaluation.json` | JSON: per-rung metrics, split at cutoff, trial count | `09_backtest`, human review |
| `stages/08_evaluate/output/leak_tests.json` | JSON: test name, result | Human review |

## Verify
- Metric functions are tested against hand-computed fixtures.
- The positive control detects a planted leak; a suite that cannot see one proves nothing.
- `uv run pytest -q` exits 0.

## Approval
A human reads the evaluation, including rungs that did not help, before any claim is made. Every claim is labelled with one of `SYNTHETIC`, `LIVE-DATA`, `LIVE-MODEL`, `NO LIVE RUN` in `docs/STATUS.md`.

## Failure Behavior
A failed rung is reported as failed, never dropped. Stop on a leak-test failure and escalate with the failing evidence; do not tune to pass it. No retries.

Loop: single pass. Looping on the evaluation is how backtests get overfit (design 10.2).
