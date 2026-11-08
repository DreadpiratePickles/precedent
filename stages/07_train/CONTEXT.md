# Stage: 07_train

## Objective
Fit a LightGBM model on the dataset, first on a rank-normalised label, and select hyperparameters by mean validation rank IC under purged folds.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| Dataset handler from `06_dataset` | 4 | Authoritative for features and labels | Yes | Train and valid segments |
| Purged fold generator and fold definitions | 3 | Authoritative for gaps (`label_horizon` + embargo) | Yes | Design 7.1 |
| Model and trial configuration | 3 | Authoritative for the search space and trial budget | Yes | Design 6.4, 7.2 |
| Results table of earlier trials | 4 | Authoritative for the trial count | Yes | All rows |

## Process
1. Deterministic: build purged walk-forward folds; fit one model per fold on the `CSRankNorm` label.
2. Deterministic: score each fold by mean rank IC and ICIR on the validation segment.
3. Deterministic: log every configuration ever scored on validation, so the trial count is known.
4. The final test period is not read in this stage.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/07_train/output/results.tsv` | One row per trial: configuration, fold scores, mean rank IC, ICIR, seed | Human review, `08_evaluate` |
| `stages/07_train/output/model/` | Model artefacts of the best-so-far configuration (not committed) | `08_evaluate` |

## Verify
- Fold tests: no label window crosses a segment boundary.
- The same seed and configuration reproduce the same score.
- Tests of this stage are research-module tests and start with `pytest.importorskip("lightgbm")`.
- `uv run pytest -q` exits 0.

## Approval
A human reviews the results table and the trial count before any claim about the best configuration. Renting compute is an owner gate.

## Failure Behavior
A failed trial is logged with its error line and counted as a trial; it is never dropped from the table. Stop at the trial budget or on repeated identical failure and keep the best-so-far result.

## Loop
Kind: ratchet
Oracle: mean validation rank IC across purged folds, computed by the fold scorer; a trial is kept only when it beats the best-so-far beyond the noise tolerance recorded in the results table. Green is not acceptance.
Oracle protection: the fold generator, scorer, labels and test-period boundary are read-only to the worker; a change stops the loop and goes to a human.
Held-out check: the final test period, touched once, at acceptance in `08_evaluate`.
Unit per iteration: one hyperparameter configuration (one logged trial).
Revises against: the results table, especially the fold scores of the best-so-far and the last failed trial.
Mutable / protected: the model configuration file / the fold generator, scorer, dataset, labels, existing tests (new tests may be added).
Isolation: a dedicated branch or worktree cut from the recorded baseline commit; nothing inside the loop publishes, deploys or spends.
Worker / judge: a script proposes configurations; no model judge.
Runner: a script counts trials, enforces the budget and the stops, and keeps or reverts the configuration.
State: `stages/07_train/output/results.tsv` and git commits; read first, updated last.
Context: one session per run, since each trial is a script call and the state is the table.
Stop: 30 logged trials; or no improvement in mean validation rank IC after 6 trials; or the same first error line 3 times in a row; or blocked (a valid exit).
On stop: the best-so-far configuration is kept, the trial count is recorded in the results table and `docs/STATUS.md`, and the owner is told.
Single-pass baseline: the default LightGBM configuration scored once under the same folds, recorded before the first trial.
