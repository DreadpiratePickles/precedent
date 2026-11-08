"""A generated Qlib provider for tests (research): calendar, instruments and float32 bins.

Layout, confirmed against Qlib v0.9.7 `scripts/dump_bin.py`:
- `calendars/day.txt`: one `YYYY-MM-DD` per line.
- `instruments/all.txt`: `SYMBOL<TAB>start<TAB>end`, upper-case symbols.
- `features/<symbol lower-case>/<field>.day.bin`: little-endian float32; the first value is the
  calendar index of the symbol's first row, then one value per calendar day up to its last row
  (NaN for a calendar day the symbol skipped).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def write_tiny_provider(root: Path, prices: dict[str, pd.DataFrame]) -> None:
    """Write a provider under `root` from frames indexed by date, one column per field."""
    calendar = sorted({pd.Timestamp(day) for frame in prices.values() for day in frame.index})
    (root / "calendars").mkdir(parents=True, exist_ok=True)
    (root / "instruments").mkdir(parents=True, exist_ok=True)
    (root / "calendars" / "day.txt").write_text(
        "".join(f"{day:%Y-%m-%d}\n" for day in calendar), encoding="utf-8"
    )
    instrument_lines = []
    for symbol, frame in prices.items():
        frame = frame.copy()
        frame.index = pd.DatetimeIndex(frame.index)
        first, last = frame.index.min(), frame.index.max()
        instrument_lines.append(f"{symbol.upper()}\t{first:%Y-%m-%d}\t{last:%Y-%m-%d}\n")
        span = [day for day in calendar if first <= day <= last]
        start_index = calendar.index(first)
        feature_dir = root / "features" / symbol.lower()
        feature_dir.mkdir(parents=True, exist_ok=True)
        for field in frame.columns:
            values = frame[field].reindex(span).to_numpy(dtype="float64")
            record = np.hstack([[start_index], values]).astype("<f4")
            record.tofile(feature_dir / f"{field.lower()}.day.bin")
    (root / "instruments" / "all.txt").write_text("".join(instrument_lines), encoding="utf-8")
