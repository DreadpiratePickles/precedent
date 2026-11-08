# CONTEXT.md

Layer 1: where do I go? Match the task, open that stage's `CONTEXT.md`, and load only the inputs it declares.

## Spine stages

| Task type | Stage |
|---|---|
| Fetch raw items from a source into the article store | `stages/01_ingest/` |
| Deduplicate, pre-match entities, apply the daily cap | `stages/02_triage/` |
| Parse an article into a validated event (the only model-calling stage) | `stages/03_parse/` |
| Event timing, `effective_session`, name to instrument matching | `stages/04_align/` |
| News, Precedent or Ripple factors | `stages/05_factors/` |
| Qlib handler, price features, labels | `stages/06_dataset/` |
| Fit and tune the model | `stages/07_train/` |
| Rank IC, ablation ladder, leak tests | `stages/08_evaluate/` |
| Costed backtest | `stages/09_backtest/` |

## Track stage ranges

| Track | Stage numbers |
|---|---|
| Precedent (analog engine) | 20 to 24 |
| Ripple (second-order effects) | 30 to 33 |
| Mechanism | 40 to 43 |
| Dream | 50 to 53 |

A track's stage folders are created by that track's plan; a track never edits `src/precedent/cli.py`.

## Shared resources

- Stable rules: `references/`
- Design and plans: `docs/superpowers/`
- Result labels: `docs/STATUS.md`
