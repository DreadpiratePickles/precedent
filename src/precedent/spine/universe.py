"""The dated index universe and identifier resolution.

Membership comes from a dated table (one row per change date, listing every member after that
date). A lookup uses only rows on or before the asked date, so a past date never sees a
constituent that joined later.
"""

from pathlib import Path

import pandas as pd

_COLUMNS = ("date", "tickers")


def load_membership(csv_path: Path) -> pd.DataFrame:
    """Read a membership table: columns `date` and `tickers` (comma-separated list per date).

    Returns a frame sorted by `date` (Timestamp) with `tickers` as a frozenset of symbols.
    Symbols are kept exactly as the source writes them (a delisted symbol may carry a
    `-YYYYMM` suffix).
    """
    raw = pd.read_csv(csv_path, dtype=str)
    if tuple(raw.columns) != _COLUMNS:
        raise ValueError(
            f"{csv_path}: expected columns date, tickers; found {', '.join(raw.columns)}"
        )
    if raw["date"].isna().any() or raw["tickers"].isna().any():
        raise ValueError(f"{csv_path}: a row has an empty date or ticker list")
    dates = pd.to_datetime(raw["date"], format="%Y-%m-%d", errors="raise")
    if dates.duplicated().any():
        raise ValueError(f"{csv_path}: duplicate membership date")
    members = raw["tickers"].map(
        lambda cell: frozenset(symbol.strip() for symbol in cell.split(",") if symbol.strip())
    )
    table = pd.DataFrame({"date": dates, "tickers": members})
    return table.sort_values("date").reset_index(drop=True)


def members_asof(membership: pd.DataFrame, on: pd.Timestamp) -> frozenset[str]:
    """Members in force on `on`: the latest row dated on or before it."""
    eligible = membership[membership["date"] <= on]
    if eligible.empty:
        first = membership["date"].min()
        raise ValueError(f"{on.date()} is before the first membership row ({first.date()})")
    return eligible.iloc[-1]["tickers"]


def resolve_instrument(
    cik: int, cik_to_symbol: dict[int, str], provider_symbols: set[str]
) -> str | None:
    """Symbol for a CIK, or None when the CIK is unmapped or the provider lacks the symbol."""
    symbol = cik_to_symbol.get(cik)
    if symbol is None or symbol not in provider_symbols:
        return None
    return symbol
