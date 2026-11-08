"""The costed Qlib backtest config, and the C1 timing it must produce (research)."""

import copy
import inspect
import json
import warnings
from pathlib import Path
from unittest import mock

import pytest

pytest.importorskip("qlib")

import pandas as pd  # noqa: E402
from qlib.backtest import backtest  # noqa: E402
from qlib.backtest.decision import Order  # noqa: E402
from qlib.backtest.exchange import Exchange  # noqa: E402
from qlib.data import D  # noqa: E402

from precedent.spine.backtest_config import build_port_analysis_config  # noqa: E402

# Third-party warnings from inside Qlib's simulator (a nanmean over an empty slice when the tiny
# provider has no quote for a step; pandas/numpy deprecations); our own warnings stay visible.
pytestmark = pytest.mark.filterwarnings(
    "ignore::DeprecationWarning",
    "ignore::FutureWarning",
    "ignore:Mean of empty slice:RuntimeWarning",
)

COST_FILE = Path(__file__).resolve().parents[2] / "references" / "cost_assumptions.json"

# The open of AAA on each session of the tiny provider (tests/spine/conftest.py).
OPEN_ON = {"2024-01-02": 100.0, "2024-01-03": 101.0, "2024-01-04": 102.0, "2024-01-05": 103.0}


@pytest.fixture(autouse=True)
def _work_in_tmp(tmp_path, monkeypatch):
    """Qlib and mlflow write relative to the working directory; keep the tree clean."""
    monkeypatch.chdir(tmp_path)


def run_tiny_backtest(signal_top_on: str, config: dict) -> pd.DataFrame:
    """Backtest `config` on the tiny provider with AAA ranked above BBB on `signal_top_on`.

    Returns one row per executed order: instrument, direction, session, price.
    """
    config = copy.deepcopy(config)
    # Qlib's last trading step reads the session after it, so a backtest must end before the
    # provider's final session; clamp the requested end to the second-to-last one.
    sessions = D.calendar(start_time=config["backtest"]["start_time"], freq="day")
    config["backtest"]["end_time"] = min(
        pd.Timestamp(config["backtest"]["end_time"]), pd.Timestamp(sessions[-2])
    )
    day = pd.Timestamp(signal_top_on)
    config["strategy"]["kwargs"]["signal"] = pd.Series(
        [1.0, 0.0],
        index=pd.MultiIndex.from_tuples([(day, "AAA"), (day, "BBB")], names=["datetime", "instrument"]),
    )
    executed: list[dict] = []
    real_deal_order = Exchange.deal_order

    def recording_deal_order(self, order, *args, **kwargs):
        trade_value, trade_cost, trade_price = real_deal_order(self, order, *args, **kwargs)
        if order.deal_amount > 0:
            executed.append(
                {
                    "instrument": order.stock_id,
                    "direction": "buy" if order.direction == Order.BUY else "sell",
                    "session": pd.Timestamp(order.start_time).normalize(),
                    "price": trade_price,
                }
            )
        return trade_value, trade_cost, trade_price

    with mock.patch.object(Exchange, "deal_order", recording_deal_order):
        backtest(
            executor=config["executor"],
            strategy=config["strategy"],
            **config["backtest"],
        )
    return pd.DataFrame(executed, columns=["instrument", "direction", "session", "price"])


def test_config_has_the_shape_port_ana_record_expects():
    config = build_port_analysis_config("2024-01-02", "2024-01-10")
    assert config["strategy"]["class"] == "TopkDropoutStrategy"
    assert config["strategy"]["kwargs"] == {"signal": "<PRED>", "topk": 50, "n_drop": 5}
    assert config["executor"]["class"] == "SimulatorExecutor"
    assert config["executor"]["kwargs"]["generate_portfolio_metrics"] is True
    backtest_kwargs = config["backtest"]
    assert (backtest_kwargs["start_time"], backtest_kwargs["end_time"]) == ("2024-01-02", "2024-01-10")
    assert backtest_kwargs["benchmark"] == "SPY"
    assert backtest_kwargs["account"] == 1_000_000
    assert backtest_kwargs["exchange_kwargs"] == {
        "freq": "day",
        "limit_threshold": None,
        "deal_price": "open",
        "open_cost": 0.0005,
        "close_cost": 0.0005,
        "min_cost": 1.0,
        "impact_cost": 0.1,
    }


def test_arguments_override_the_defaults():
    config = build_port_analysis_config(
        "2024-01-02", "2024-01-10", topk=1, n_drop=1, cost_per_side=0.001, benchmark="AAA"
    )
    assert config["strategy"]["kwargs"]["topk"] == 1
    exchange = config["backtest"]["exchange_kwargs"]
    assert exchange["open_cost"] == exchange["close_cost"] == 0.001
    assert config["backtest"]["benchmark"] == "AAA"


def test_cost_assumptions_file_agrees_with_the_function_defaults():
    recorded = json.loads(COST_FILE.read_text(encoding="utf-8"))["settings"]
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(build_port_analysis_config).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    for name in ("topk", "n_drop", "impact_cost", "min_cost", "benchmark", "account"):
        assert recorded[name]["value"] == defaults[name], name
    assert recorded["open_cost"]["value"] == recorded["close_cost"]["value"] == defaults["cost_per_side"]
    built = build_port_analysis_config("2024-01-02", "2024-01-10")
    assert recorded["deal_price"]["value"] == built["backtest"]["exchange_kwargs"]["deal_price"]
    assert recorded["region"]["value"] == "us"
    for name, entry in recorded.items():
        assert entry["basis"].strip(), name
        assert entry["proposed"] is True, name


def test_strategy_acts_on_the_previous_session_signal(tiny_provider):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        trades = run_tiny_backtest(
            signal_top_on="2024-01-03",
            config=build_port_analysis_config(
                "2024-01-02", "2024-01-10", topk=1, n_drop=1, benchmark="AAA"
            ),
        )
    first_buy = trades[trades.direction == "buy"].iloc[0]
    assert str(first_buy.session.date()) == "2024-01-04" and first_buy.price == pytest.approx(
        OPEN_ON["2024-01-04"]
    )
