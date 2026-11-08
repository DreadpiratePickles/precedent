"""Research: join factor panels to Qlib price features. The price index wins; absent factors are 0.

Factor panels are sparse `(datetime, instrument)` frames, one per family and keyed by a column
prefix (`news_`, `prec_`, `ripple_`, `macro_`). They are left-joined onto the price loader's
index, so a factor row for an instrument or day without a price row is dropped (and counted in
the `JoinReport`), never added. Missing factor values become 0 at inference; price features are
left as they are.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import DataLoader, NestedDataLoader, QlibDataLoader, StaticDataLoader
from qlib.data.dataset.processor import Processor

# Decide at the close of t, enter at the open of t+1, exit at the open of t+2.
DEFAULT_LABEL_EXPR = "Ref($open, -2)/Ref($open, -1) - 1"
PANEL_INDEX_NAMES = ["datetime", "instrument"]


@dataclass(frozen=True)
class JoinReport:
    """What the join set aside: factor rows in the time window with no price row."""

    events_without_price_row: int


class FillnaByPrefix(Processor):
    """Fill NaN in ("feature", name) columns whose name starts with a prefix; leave the rest."""

    def __init__(self, prefixes: Sequence[str], fill_value: float = 0.0) -> None:
        self.prefixes = tuple(prefixes)
        self.fill_value = fill_value

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("FillnaByPrefix expects (group, name) MultiIndex columns")
        matching = [
            column
            for column in df.columns
            if column[0] == "feature" and str(column[1]).startswith(self.prefixes)
        ]
        if not matching:
            raise ValueError(
                f"no ('feature', name) column starts with prefix in {self.prefixes}; "
                f"columns are {list(df.columns)[:5]}..."
            )
        filled = df.copy()
        filled[matching] = df[matching].fillna(self.fill_value)
        return filled

    def readonly(self) -> bool:
        """The input frame is copied, never written."""
        return True


def _check_factor_panel(key: str, panel: pd.DataFrame) -> None:
    if not key:
        raise ValueError("a factor panel key (column prefix) must not be empty")
    if list(panel.index.names) != PANEL_INDEX_NAMES:
        raise ValueError(
            f"panel '{key}' index must be named {PANEL_INDEX_NAMES}, got {list(panel.index.names)}"
        )
    if panel.index.has_duplicates:
        raise ValueError(
            f"panel '{key}' has duplicate (datetime, instrument) rows; "
            "the join would repeat price rows"
        )
    unprefixed = [str(name) for name in panel.columns if not str(name).startswith(key)]
    if unprefixed:
        raise ValueError(
            f"panel '{key}' columns must start with '{key}' so they are zero-filled: {unprefixed}"
        )
    if panel.isna().any().any():
        raise ValueError(f"panel '{key}' contains NaN; absent factors must be an absent row")


def _count_rows_without_price_row(
    panels: Mapping[str, pd.DataFrame],
    price_index: pd.MultiIndex,
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
) -> int:
    start, end = pd.Timestamp(start_time), pd.Timestamp(end_time)
    absent = 0
    for panel in panels.values():
        dates = panel.index.get_level_values("datetime")
        in_window = panel.index[(dates >= start) & (dates <= end)]
        absent += int((~in_window.isin(price_index)).sum())
    return absent


def make_alpha_handler(
    price_loader: DataLoader,
    factor_panels: Mapping[str, pd.DataFrame],
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
    learn_processors: list | None = None,
) -> tuple[DataHandlerLP, JoinReport]:
    """Left-join factor panels onto the price loader's index; zero-fill the factors at inference.

    `factor_panels` maps a column prefix to a panel whose every column starts with that prefix.
    An empty mapping gives the price-only handler. The report counts factor rows inside
    `[start_time, end_time]` that have no price row and were therefore dropped.
    """
    for key, panel in factor_panels.items():
        _check_factor_panel(key, panel)
    factor_loaders = [
        StaticDataLoader(config={"feature": panel.sort_index()}) for panel in factor_panels.values()
    ]
    joined_loader = NestedDataLoader(
        dataloader_l=[price_loader, *factor_loaders],
        join="left",  # the first loader's index wins
    )
    if learn_processors is None:
        learn_processors = [
            {"class": "DropnaLabel"},
            {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}},
        ]
    infer_processors = [FillnaByPrefix(tuple(factor_panels), 0.0)] if factor_panels else []
    with warnings.catch_warnings():
        # None means every provider instrument, which is intended; Qlib warns about it anyway.
        warnings.filterwarnings("ignore", message="`instruments` is not set")
        handler = DataHandlerLP(
            instruments=None,
            start_time=start_time,
            end_time=end_time,
            data_loader=joined_loader,
            infer_processors=infer_processors,
            learn_processors=learn_processors,
        )
    price_index = handler.fetch(col_set="__raw", data_key=DataHandlerLP.DK_R).index
    report = JoinReport(
        events_without_price_row=_count_rows_without_price_row(
            factor_panels, price_index, start_time, end_time
        )
    )
    return handler, report


def make_alpha158_price_loader(label_expr: str = DEFAULT_LABEL_EXPR) -> QlibDataLoader:
    """Alpha158 price features plus one forward-return label named LABEL0.

    Loading needs a real provider with enough history; construction does not.
    """
    if not label_expr.strip():
        raise ValueError("label_expr must be a non-empty Qlib expression")
    return QlibDataLoader(
        config={"feature": Alpha158DL.get_feature_config(), "label": ([label_expr], ["LABEL0"])}
    )
