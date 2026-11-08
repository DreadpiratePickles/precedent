"""Research: the costed long-only backtest config for Qlib's `PortAnaRecord` (design 7.3).

Pure dict construction; Qlib is not imported here. The signal is the `"<PRED>"` placeholder that
`PortAnaRecord` fills from the saved predictions. Convention C1: a signal dated t is decided at
the close of t and traded at the open of t+1 (`deal_price` is `open`). The numbers are proposals;
`references/cost_assumptions.json` records each with its basis.
"""

from __future__ import annotations


def build_port_analysis_config(
    start: str,
    end: str,
    topk: int = 50,
    n_drop: int = 5,
    cost_per_side: float = 0.0005,
    impact_cost: float = 0.1,
    min_cost: float = 1.0,
    benchmark: str = "SPY",
    account: float = 1_000_000,
) -> dict:
    """Return the `config` argument of `PortAnaRecord`: strategy, executor and backtest blocks.

    Qlib's last step reads the following session, so a backtest must end at least one session
    before the provider's last, and the final signal day is never traded.
    """
    return {
        "strategy": {
            "class": "TopkDropoutStrategy",
            "module_path": "qlib.contrib.strategy",
            "kwargs": {"signal": "<PRED>", "topk": topk, "n_drop": n_drop},
        },
        "executor": {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {"time_per_step": "day", "generate_portfolio_metrics": True},
        },
        "backtest": {
            "start_time": start,
            "end_time": end,
            "account": account,
            "benchmark": benchmark,
            "exchange_kwargs": {
                "freq": "day",
                "limit_threshold": None,
                "deal_price": "open",
                "open_cost": cost_per_side,
                "close_cost": cost_per_side,
                "min_cost": min_cost,
                "impact_cost": impact_cost,
            },
        },
    }
