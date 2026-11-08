"""Long-short accounting outside Qlib: gross, cost, borrow and net, per day."""

import pandas as pd
import pytest

from precedent.spine.long_short import long_short_returns

DAYS = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"])
NAMES = ["A", "B", "C", "D"]


def _pred(rows: list[list[float]]) -> pd.Series:
    index = pd.MultiIndex.from_product([DAYS[: len(rows)], NAMES], names=["datetime", "instrument"])
    return pd.Series([score for row in rows for score in row], index=index)


# Day 1 and 2: A, B are the top half (long), C, D the bottom half (short).
# Day 3: B and D swap sides, so half of each leg is replaced.
PRED = _pred([[4, 3, 2, 1], [4, 3, 2, 1], [4, 1, 2, 3]])
RETURNS = pd.DataFrame(
    {"A": [0.02, 0.02, 0.02], "B": [0.02, 0.02, 0.02], "C": [-0.01] * 3, "D": [-0.01] * 3},
    index=DAYS,
)
COST_PER_SIDE = 0.001
ANNUAL_BORROW = 0.0252


def _run():
    return long_short_returns(
        PRED, RETURNS, quantile=0.5, cost_per_side=COST_PER_SIDE, annual_borrow_fee=ANNUAL_BORROW
    )


def test_long_short_charges_costs_and_borrow():
    out = _run()
    # first day: long leg +2%, short leg -1%, both legs fully turned over
    assert out["net"].iloc[0] == pytest.approx(0.015 - 0.001 - 0.00005)


def test_columns_and_net_identity():
    out = _run()
    assert list(out.columns) == ["gross", "cost", "borrow", "net"]
    assert list(out.index) == list(DAYS)
    assert out["gross"].iloc[0] == pytest.approx(0.015)
    assert out["borrow"].iloc[0] == pytest.approx(0.0252 / 252 / 2)
    assert list(out["net"]) == pytest.approx(list(out["gross"] - out["cost"] - out["borrow"]))


def test_cost_follows_turnover_of_each_leg():
    out = _run()
    assert out["cost"].iloc[1] == pytest.approx(0.0)  # same names as day 1
    # one of two names per leg replaced: turnover 0.5 on both legs
    assert out["cost"].iloc[2] == pytest.approx(COST_PER_SIDE * 0.5)


def test_short_leg_loss_is_a_loss_not_a_gain():
    gains = RETURNS.copy()
    gains[["C", "D"]] = 0.03  # shorted names rally
    out = long_short_returns(PRED, gains, 0.5, cost_per_side=0.0, annual_borrow_fee=0.0)
    assert out["gross"].iloc[0] == pytest.approx((0.02 - 0.03) / 2)


@pytest.mark.parametrize("quantile", [0.0, -0.1, 0.6, 1.0])
def test_quantile_must_leave_two_disjoint_legs(quantile):
    with pytest.raises(ValueError, match="quantile"):
        long_short_returns(PRED, RETURNS, quantile, cost_per_side=0.0, annual_borrow_fee=0.0)


def test_a_day_without_known_returns_is_an_error_not_a_zero():
    with pytest.raises(ValueError, match="2024-01-04"):
        long_short_returns(
            PRED, RETURNS.iloc[:2], 0.5, cost_per_side=0.0, annual_borrow_fee=0.0
        )
