"""Research: the Qlib bridge. `init_qlib`, a `PriceProvider` over Qlib data, and a provider check.

Run scripts that call `init_qlib` and then read data under `if __name__ == "__main__":`; Qlib
spawns worker processes and, under the macOS `spawn` start method, re-imports the main module.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import qlib
from qlib.data import D

from precedent.config import ConfigError
from precedent.spine.price_provider import PriceField

SPY_SYMBOL = "SPY"
FAILURE = 1


def init_qlib(provider_dir: Path, workdir: Path) -> None:
    """Point Qlib at `provider_dir` (region `us`) with an SQLite-backed experiment manager."""
    workdir.mkdir(parents=True, exist_ok=True)
    qlib.init(
        provider_uri=str(provider_dir),
        region="us",
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {
                "uri": f"sqlite:///{workdir / 'mlruns.db'}",
                "default_exp_name": "precedent",
            },
        },
    )


class QlibPriceProvider:
    """A `PriceProvider` reading the provider Qlib was initialised on (call `init_qlib` first)."""

    def sessions(self, start: str, end: str) -> pd.DatetimeIndex:
        calendar = D.calendar(start_time=start, end_time=end, freq="day")
        return pd.DatetimeIndex(calendar).sort_values()

    def prices(
        self, instruments: list[str], start: str, end: str, field: PriceField
    ) -> pd.DataFrame:
        wanted = [symbol.upper() for symbol in instruments]
        known = set(D.list_instruments(D.instruments("all"), as_list=True))
        missing = [symbol for symbol in wanted if symbol not in known]
        if missing:
            raise ConfigError(f"no prices for {', '.join(missing)} in the Qlib provider")
        window = self.sessions(start, end)
        frame = D.features(wanted, [f"${field}"], start_time=start, end_time=end, freq="day")
        wide = frame[f"${field}"].unstack(level="instrument").reindex(window)
        return wide.reindex(columns=wanted).set_axis(list(instruments), axis=1)


@dataclass(frozen=True)
class ProviderReport:
    sessions: int
    instruments: int
    last_session: date | None
    has_spy: bool
    spy_last_session: date | None = None


def _read_lines(path: Path, what: str) -> list[str]:
    if not path.is_file():
        raise ConfigError(f"Qlib provider: {what} not found at {path}")
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check_provider(provider_dir: Path) -> ProviderReport:
    """Read the provider's calendar and instruments directly; no Qlib initialisation needed."""
    calendar = _read_lines(provider_dir / "calendars" / "day.txt", "calendar")
    instrument_lines = _read_lines(provider_dir / "instruments" / "all.txt", "instrument list")
    spy_last: date | None = None
    for line in instrument_lines:
        symbol, _start, end = line.split("\t")
        if symbol.upper() == SPY_SYMBOL:
            spy_last = date.fromisoformat(end)
    return ProviderReport(
        sessions=len(calendar),
        instruments=len(instrument_lines),
        last_session=date.fromisoformat(calendar[-1]) if calendar else None,
        has_spy=spy_last is not None,
        spy_last_session=spy_last,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m precedent.spine.qlib_provider")
    parser.add_argument("--check", type=Path, required=True, metavar="DIR")
    args = parser.parse_args(argv)
    try:
        report = check_provider(args.check)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return FAILURE
    print(f"sessions: {report.sessions}")
    print(f"instruments: {report.instruments}")
    print(f"last_session: {report.last_session or 'none'}")
    print(f"SPY present: {'yes' if report.has_spy else 'no'}")
    print(f"SPY last session: {report.spy_last_session or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
