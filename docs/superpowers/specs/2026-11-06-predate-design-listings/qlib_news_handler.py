"""Join the news factor panel to price features inside Qlib: the price index wins, no news means zero events."""
from pathlib import Path

import pandas as pd
import qlib
from qlib.contrib.data.loader import Alpha158DL
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import DataLoader, NestedDataLoader, QlibDataLoader, StaticDataLoader
from qlib.data.dataset.processor import Processor

NEWS_PREFIX = "news_"
DEFAULT_LABEL_EXPR = "Ref($open, -2)/Ref($open, -1) - 1"  # decide at close of t, enter at open t+1, exit at open t+2


class FillnaByPrefix(Processor):
    """Fill NaN in ("feature", name) columns whose name starts with `prefix`; all other columns are untouched."""

    def __init__(self, prefix: str, fill_value: float = 0.0) -> None:
        self.prefix = prefix
        self.fill_value = fill_value

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df.columns, pd.MultiIndex):
            raise ValueError("FillnaByPrefix expects (group, name) MultiIndex columns")
        matching = [col for col in df.columns if col[0] == "feature" and str(col[1]).startswith(self.prefix)]
        if not matching:
            raise ValueError(f"no ('feature', name) column starts with prefix '{self.prefix}'; columns are {list(df.columns)[:5]}...")
        filled = df.copy()
        filled[matching] = df[matching].fillna(self.fill_value)
        return filled

    def readonly(self) -> bool:
        """The input frame is copied, never written."""
        return True


def init_qlib_for_static_data(workdir: Path) -> None:
    """Minimum qlib.init that works without provider data; the sqlite experiment store lives under workdir."""
    workdir.mkdir(parents=True, exist_ok=True)
    qlib.init(
        provider_uri=str(workdir / "no_provider_data"),
        region="us",
        exp_manager={
            "class": "MLflowExpManager",
            "module_path": "qlib.workflow.expm",
            "kwargs": {"uri": f"sqlite:///{workdir / 'mlflow.db'}", "default_exp_name": "Experiment"},
        },
    )


def _check_news_panel(news_panel: pd.DataFrame) -> None:
    if list(news_panel.index.names) != ["datetime", "instrument"]:
        raise ValueError(f"news_panel index must be named ['datetime', 'instrument'], got {list(news_panel.index.names)}")
    if news_panel.index.has_duplicates:
        raise ValueError("news_panel has duplicate (datetime, instrument) rows; a left join would repeat price rows")
    unprefixed = [str(name) for name in news_panel.columns if not str(name).startswith(NEWS_PREFIX)]
    if unprefixed:
        raise ValueError(f"news_panel columns must start with '{NEWS_PREFIX}' to be zero-filled at inference: {unprefixed}")
    if news_panel.isna().any().any():
        raise ValueError("news_panel contains NaN; absent news must be an absent row, not a NaN")


def make_news_alpha_handler(
    price_loader: DataLoader,
    news_panel: pd.DataFrame,
    start_time: str | pd.Timestamp,
    end_time: str | pd.Timestamp,
    learn_processors: list | None = None,
) -> DataHandlerLP:
    """Left-join the news panel onto the price loader's index; zero-fill news (only) at inference."""
    _check_news_panel(news_panel)
    news_loader = StaticDataLoader(config={"feature": news_panel.sort_index()})
    joined_loader = NestedDataLoader(dataloader_l=[price_loader, news_loader], join="left")  # first loader's index wins
    if learn_processors is None:
        learn_processors = [{"class": "DropnaLabel"}, {"class": "CSRankNorm", "kwargs": {"fields_group": "label"}}]
    return DataHandlerLP(
        instruments=None,  # StaticDataLoader accepts None or a list; QlibDataLoader reads None as every provider instrument
        start_time=start_time,
        end_time=end_time,
        data_loader=joined_loader,
        infer_processors=[FillnaByPrefix(NEWS_PREFIX, 0.0)],
        learn_processors=learn_processors,
    )


def make_alpha158_price_loader(label_expr: str = DEFAULT_LABEL_EXPR) -> QlibDataLoader:
    """Alpha158 price features plus one forward-return label named LABEL0; loading needs real provider data."""
    if not label_expr.strip():
        raise ValueError("label_expr must be a non-empty Qlib expression")
    return QlibDataLoader(config={"feature": Alpha158DL.get_feature_config(), "label": ([label_expr], ["LABEL0"])})
