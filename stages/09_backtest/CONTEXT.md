# Stage: 09_backtest

## Objective
Simulate a costed long-only strategy in Qlib from the model's scores and account for long-short results outside Qlib.

## Inputs
| Path or source | Layer | Authority | Required | Relevant section |
|---|---:|---|---:|---|
| `stages/08_evaluate/output/evaluation.json` and the chosen model | 4 | Authoritative for which model and rung to backtest | Yes | Chosen rung |
| `references/` cost assumptions and strategy settings | 3 | Authoritative for costs and strategy | Yes | Design 7.3 |
| Price provider data (local, never committed) | 4 | Authoritative for prices | Yes | Backtest period |

## Process
1. Deterministic: run `TopkDropoutStrategy` (`topk=50`, `n_drop=5`), long-only, with `deal_price` of `open`, benchmark SPY, and the recorded open and close costs.
2. Deterministic: compute long-short accounting outside Qlib, because Qlib cannot short.
3. Deterministic: report excess return over SPY net of costs, turnover and maximum drawdown, split at the base model's cutoff.

## Outputs
| Path | Schema or format | Consumer |
|---|---|---|
| `stages/09_backtest/output/backtest.json` | JSON: metrics, costs used, period, input hashes | Human review |

## Verify
- Accounting functions are tested against hand-computed fixtures.
- The report states the cost assumptions and that the figures are simulated, not a forecast.
- `uv run pytest -q` exits 0.

## Approval
A human reviews the backtest before it is cited. Nothing here trades, publishes or displays vendor prices.

## Failure Behavior
A missing benchmark or price gap stops the stage with a typed error; it never becomes a zero return. No retries.

Loop: single pass. Looping on a backtest is how it gets overfit (design 10.2).

Note (Task 15): Qlib's last trading step reads the following session, so a backtest must end at least one session before the provider's last session, and the final signal day is never traded.
