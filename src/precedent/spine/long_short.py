"""Long-short accounting outside Qlib, which cannot short (design 7.3). Pandas only.

Capital is split half long, half short, each leg equal-weighted over its names. `pred[t]` is the
signal for the position held over `known_returns[t]`; the caller aligns the two (under C1 the
return row dated t is the open-to-open return of the session the signal trades).
"""

from __future__ import annotations

import math

import pandas as pd

TRADING_DAYS_PER_YEAR = 252
LEG_COLUMNS = ("gross", "cost", "borrow", "net")


def _leg_weights(scores: pd.Series, count: int) -> tuple[pd.Series, pd.Series]:
    """Equal weights (summing to 1) for the top and bottom `count` names, indexed by instrument.

    The short leg is drawn from the names left after the long leg, so tied scores cannot put one
    name on both sides (ties break by position in `scores`).
    """
    long_names = scores.nlargest(count).index
    short_names = scores.drop(long_names).nsmallest(count).index
    return (
        pd.Series(1.0 / count, index=long_names),
        pd.Series(1.0 / count, index=short_names),
    )


def _turnover(today: pd.Series, yesterday: pd.Series) -> float:
    """Fraction of the leg's weight that is new since the previous day (1.0 on the first day)."""
    before = yesterday.reindex(today.index, fill_value=0.0)
    return float((today - before).clip(lower=0.0).sum())


def long_short_returns(
    pred: pd.Series,
    known_returns: pd.DataFrame,
    quantile: float,
    cost_per_side: float,
    annual_borrow_fee: float,
) -> pd.DataFrame:
    """Daily `gross`, `cost`, `borrow` and `net` of a dollar-neutral long-short book.

    `pred` has a (datetime, instrument) MultiIndex; `known_returns` is dates by instruments.
    Each day goes long the top `quantile` of names by score and short the bottom `quantile`.
    `known_returns[t]` is the forward return of the session the day-t signal trades (the caller
    shifts). A selected name with a NaN or absent return, or a day with fewer than two scored
    names, raises `ValueError` rather than counting as zero.
    """
    if not 0.0 < quantile <= 0.5:
        raise ValueError(f"quantile must be in (0, 0.5] so the legs are disjoint, got {quantile}")
    scores_by_day = pred.dropna().groupby(level="datetime")
    missing = [day for day in scores_by_day.groups if day not in known_returns.index]
    if missing:
        days = ", ".join(f"{day:%Y-%m-%d}" for day in sorted(missing))
        raise ValueError(f"no known returns for {days}")

    borrow_per_day = annual_borrow_fee / TRADING_DAYS_PER_YEAR / 2
    previous_long = pd.Series(dtype="float64")
    previous_short = pd.Series(dtype="float64")
    rows = {}
    for day, scores in scores_by_day:
        scores = scores.droplevel("datetime")
        if len(scores) < 2:
            raise ValueError(f"{day:%Y-%m-%d}: need at least two scored names for two legs")
        count = max(1, math.floor(len(scores) * quantile))
        long_weights, short_weights = _leg_weights(scores, count)
        day_returns = known_returns.loc[day]
        selected = long_weights.index.append(short_weights.index)
        unknown = day_returns.reindex(selected)[lambda found: found.isna()].index
        if len(unknown) > 0:
            raise ValueError(
                f"no known return on {day:%Y-%m-%d} for {', '.join(map(str, unknown))}"
            )
        long_return = float((long_weights * day_returns.reindex(long_weights.index)).sum())
        short_return = float((short_weights * day_returns.reindex(short_weights.index)).sum())
        gross = (long_return - short_return) / 2
        turnover_long = _turnover(long_weights, previous_long)
        turnover_short = _turnover(short_weights, previous_short)
        cost = cost_per_side * (turnover_long + turnover_short) / 2
        rows[day] = {
            "gross": gross,
            "cost": cost,
            "borrow": borrow_per_day,
            "net": gross - cost - borrow_per_day,
        }
        previous_long, previous_short = long_weights, short_weights
    return pd.DataFrame.from_dict(rows, orient="index", columns=list(LEG_COLUMNS))
