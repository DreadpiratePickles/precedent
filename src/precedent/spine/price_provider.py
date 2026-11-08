"""Price access behind a Protocol: sessions and open/close prices, plus the return convention.

Return convention: a return is indexed by the session on which it became known.
`close_to_close[t] = close(t)/close(t-1) - 1`, `open_to_open[t] = open(t)/open(t-1) - 1`.
Forward-looking values exist only in the Qlib label expression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

import pandas as pd

from precedent.config import ConfigError, data_dir

PriceField = Literal["open", "close"]
ReturnKind = Literal["open_to_open", "close_to_close"]
_RETURN_FIELD: dict[str, PriceField] = {"open_to_open": "open", "close_to_close": "close"}
_EARLIEST = "1900-01-01"
_IMPORT_HINT = "run `precedent prices import <csv>` first"


class PriceProvider(Protocol):
    def sessions(self, start: str, end: str) -> pd.DatetimeIndex: ...

    def prices(
        self, instruments: list[str], start: str, end: str, field: PriceField
    ) -> pd.DataFrame: ...


class ParquetPriceProvider:
    """Reads the `<SYMBOL>.parquet` files that `precedent prices import` writes.

    Files are read once per instance; build a new provider after a re-import.
    """

    def __init__(self, prices_dir: Path) -> None:
        self._prices_dir = prices_dir
        self._frames: dict[str, pd.DataFrame] = {}
        self._all_sessions: pd.DatetimeIndex | None = None

    def _symbol_files(self) -> dict[str, Path]:
        return {path.stem: path for path in sorted(self._prices_dir.glob("*.parquet"))}

    def _frame(self, symbol: str) -> pd.DataFrame:
        if symbol not in self._frames:
            path = self._symbol_files().get(symbol)
            if path is None:
                raise ConfigError(f"no prices for {symbol} in {self._prices_dir}; {_IMPORT_HINT}")
            self._frames[symbol] = pd.read_parquet(path).set_index("date").sort_index()
        return self._frames[symbol]

    def _sessions_everywhere(self) -> pd.DatetimeIndex:
        if self._all_sessions is None:
            days = [self._frame(symbol).index for symbol in self._symbol_files()]
            union = pd.DatetimeIndex([]).append(days).unique() if days else pd.DatetimeIndex([])
            self._all_sessions = union.sort_values()
        return self._all_sessions

    def sessions(self, start: str, end: str) -> pd.DatetimeIndex:
        everywhere = self._sessions_everywhere()
        return everywhere[(everywhere >= pd.Timestamp(start)) & (everywhere <= pd.Timestamp(end))]

    def prices(
        self, instruments: list[str], start: str, end: str, field: PriceField
    ) -> pd.DataFrame:
        window = self.sessions(start, end)
        columns = {symbol: self._frame(symbol)[field].reindex(window) for symbol in instruments}
        return pd.DataFrame(columns, index=window, columns=list(instruments))


def default_price_provider() -> PriceProvider:
    prices_dir = data_dir() / "prices"
    if not any(prices_dir.glob("*.parquet")):
        raise ConfigError(f"no prices under {prices_dir}; {_IMPORT_HINT}")
    return ParquetPriceProvider(prices_dir)


def load_sessions(start: str, end: str, provider: PriceProvider | None = None) -> pd.DatetimeIndex:
    return (provider or default_price_provider()).sessions(start, end)


def load_returns(
    instruments: list[str],
    start: str,
    end: str,
    kind: ReturnKind,
    provider: PriceProvider | None = None,
) -> pd.DataFrame:
    """Returns indexed by the session they became known; never bridged across a missing price."""
    if kind not in _RETURN_FIELD:
        raise ValueError(f"unknown return kind {kind!r}; use open_to_open or close_to_close")
    source = provider or default_price_provider()
    earlier = source.sessions(_EARLIEST, end)
    before_start = earlier[earlier < pd.Timestamp(start)]
    fetch_from = before_start[-1].strftime("%Y-%m-%d") if len(before_start) else start
    levels = source.prices(instruments, fetch_from, end, _RETURN_FIELD[kind])
    returns = levels.pct_change(fill_method=None)
    return returns.loc[pd.Timestamp(start) : pd.Timestamp(end)]
