"""Daily cross-sectional Spearman rank IC and its summary statistics."""
import numpy as np
import pandas as pd

MIN_NAMES_PER_DAY = 3


def daily_rank_ic(pred: pd.Series, label: pd.Series) -> pd.Series:
    """Return the per-datetime Spearman correlation; days with < 3 names or no variance
    are dropped."""
    for name, values in (("pred", pred), ("label", label)):
        if "datetime" not in values.index.names:
            raise ValueError(
                f"{name} must have an index level named 'datetime', got {list(values.index.names)}"
            )
    pairs = pd.concat({"pred": pred, "label": label}, axis=1, join="inner").dropna()
    by_day = pairs.groupby(level="datetime")
    pairs = pairs[by_day["pred"].transform("size") >= MIN_NAMES_PER_DAY]
    ranks = pairs.groupby(level="datetime").rank(method="average")
    centred = ranks - ranks.groupby(level="datetime").transform("mean")
    day = centred.index.get_level_values("datetime")
    cross = (centred["pred"] * centred["label"]).groupby(day).sum()
    pred_var = (centred["pred"] ** 2).groupby(day).sum()
    label_var = (centred["label"] ** 2).groupby(day).sum()
    denominator = np.sqrt(pred_var * label_var)
    ic = (cross / denominator.where(denominator > 0)).dropna()
    return ic.rename("rank_ic").rename_axis("datetime")


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Return mean, std (ddof=1), icir = mean / std, hit_rate = share of days with IC > 0,
    and n_days."""
    values = ic.dropna()
    if values.empty:
        raise ValueError("ic_summary received no days with a defined IC")
    mean, std = float(values.mean()), float(values.std(ddof=1))
    icir = mean / std if std > 0 else float("nan")
    return {
        "mean": mean,
        "std": std,
        "icir": icir,
        "hit_rate": float((values > 0).mean()),
        "n_days": int(len(values)),
    }
