"""Integration tests: news panel joined to price features inside Qlib, on synthetic data only."""
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from qlib.contrib.data.loader import Alpha158DL
from qlib.contrib.model.gbdt import LGBModel
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader, StaticDataLoader
from qlib.workflow import R

from news_factors import PANEL_COLUMNS, build_news_factor_panel
from purged_splits import purged_walk_forward_segments
from rank_ic import daily_rank_ic
from qlib_news_handler import (
    FillnaByPrefix,
    init_qlib_for_static_data,
    make_alpha158_price_loader,
    make_news_alpha_handler,
)

N_DAYS, N_INSTRUMENTS = 300, 20
SESSIONS = pd.bdate_range("2020-01-01", periods=N_DAYS, name="datetime")
INSTRUMENTS = [f"S{i:02d}" for i in range(N_INSTRUMENTS)]


def synthetic_price_data(rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.MultiIndex.from_product([SESSIONS, INSTRUMENTS], names=["datetime", "instrument"])
    features = pd.DataFrame(rng.normal(size=(len(index), 5)), index=index, columns=[f"f{i}" for i in range(5)])
    label = pd.DataFrame({"LABEL0": 0.5 * features["f0"].to_numpy() + rng.normal(size=len(index))}, index=index)
    label.iloc[-2 * N_INSTRUMENTS :, 0] = np.nan  # the last two sessions have no forward label
    return features, label


def synthetic_news_panel(rng: np.random.Generator, n_events: int = 280) -> pd.DataFrame:
    events = pd.DataFrame(
        {
            "effective_session": SESSIONS[rng.integers(0, N_DAYS, size=n_events)],
            "instrument": rng.choice(INSTRUMENTS, size=n_events),
            "severity": rng.integers(1, 11, size=n_events),
            "direction": rng.choice([-1.0, -0.5, 0.0, 0.5, 1.0], size=n_events),
            "confidence_weight": rng.choice([0.3, 0.6, 0.9], size=n_events),
            "novelty": rng.choice(["new", "update", "recap", "opinion"], size=n_events),
        }
    )
    return build_news_factor_panel(events, SESSIONS, half_life_sessions=2.0, max_age_sessions=5)


@pytest.fixture(scope="module")
def inputs() -> dict:
    rng = np.random.default_rng(0)
    features, label = synthetic_price_data(rng)
    features.iloc[N_INSTRUMENTS * 10 + 3, 2] = np.nan  # one deliberately missing price feature
    return {"features": features, "label": label, "news": synthetic_news_panel(rng)}


@pytest.fixture(scope="module")
def handler(inputs: dict) -> DataHandlerLP:
    price_loader = StaticDataLoader(config={"feature": inputs["features"], "label": inputs["label"]})
    return make_news_alpha_handler(price_loader, inputs["news"], SESSIONS[0], SESSIONS[-1])


@pytest.fixture(scope="module")
def qlib_workdir(tmp_path_factory: pytest.TempPathFactory):
    """Initialise Qlib with its sqlite experiment store in a temp dir, and keep the cwd there too."""
    workdir = tmp_path_factory.mktemp("qlib_workdir")
    previous_cwd = Path.cwd()
    os.chdir(workdir)
    try:
        init_qlib_for_static_data(workdir)
        yield workdir
    finally:
        if R.exp_manager.active_experiment is not None:  # LGBModel.fit starts one implicitly; end it while logging works
            R.end_exp("FINISHED")
        os.chdir(previous_cwd)


def test_news_panel_is_sparse_but_not_empty(inputs: dict) -> None:
    coverage = len(inputs["news"]) / len(inputs["features"])
    assert 0.10 < coverage < 0.40


def test_raw_row_count_equals_price_row_count_and_news_sits_in_the_feature_group(
    handler: DataHandlerLP, inputs: dict
) -> None:
    raw = handler.fetch(col_set="__raw", data_key=DataHandlerLP.DK_R)
    assert len(raw) == len(inputs["features"])
    for column in PANEL_COLUMNS:
        assert ("feature", column) in raw.columns
        assert ("label", column) not in raw.columns
    assert raw["label"].columns.tolist() == ["LABEL0"]


def test_price_index_wins_over_news_rows_for_unknown_instruments(inputs: dict) -> None:
    stray = pd.DataFrame(
        0.0,
        index=pd.MultiIndex.from_tuples([(SESSIONS[5], "ZZZ")], names=["datetime", "instrument"]),
        columns=PANEL_COLUMNS,
    )
    news = pd.concat([inputs["news"], stray]).sort_index()
    price_loader = StaticDataLoader(config={"feature": inputs["features"], "label": inputs["label"]})
    handler = make_news_alpha_handler(price_loader, news, SESSIONS[0], SESSIONS[-1])
    raw = handler.fetch(col_set="__raw", data_key=DataHandlerLP.DK_R)
    assert len(raw) == len(inputs["features"])
    assert "ZZZ" not in raw.index.get_level_values("instrument")


def test_inference_fills_news_with_zero_but_leaves_price_nan_alone(handler: DataHandlerLP, inputs: dict) -> None:
    raw = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_R)
    infer = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    assert raw[PANEL_COLUMNS].isna().any().all()  # absent news rows are NaN after the join
    assert not infer[PANEL_COLUMNS].isna().any().any()
    assert infer["f2"].isna().sum() == 1 and inputs["features"]["f2"].isna().sum() == 1
    assert infer.loc[raw["news_cnt_1d"].isna(), PANEL_COLUMNS].eq(0.0).all().all()


def test_default_learn_processors_drop_unlabelled_rows_and_rank_normalise(handler: DataHandlerLP, inputs: dict) -> None:
    learn = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_L)
    assert not learn.isna().any().any()
    assert len(learn) == (N_DAYS - 2) * N_INSTRUMENTS
    assert learn["LABEL0"].abs().max() <= 0.5 * 3.46 + 1e-9  # CSRankNorm maps pct-ranks to (rank - 0.5) * 3.46
    raw_label = inputs["label"]["LABEL0"].loc[learn.index]
    assert daily_rank_ic(learn["LABEL0"], raw_label).round(9).eq(1.0).all()  # order within each day is preserved


def test_news_panel_with_bad_structure_is_rejected(inputs: dict) -> None:
    loader = StaticDataLoader(config={"feature": inputs["features"], "label": inputs["label"]})
    news = inputs["news"]
    with pytest.raises(ValueError, match="duplicate"):
        make_news_alpha_handler(loader, pd.concat([news, news.iloc[:1]]), SESSIONS[0], SESSIONS[-1])
    with pytest.raises(ValueError, match="index"):
        make_news_alpha_handler(loader, news.rename_axis(["date", "instrument"]), SESSIONS[0], SESSIONS[-1])
    with pytest.raises(ValueError, match="news_"):
        make_news_alpha_handler(loader, news.rename(columns={"news_cnt_1d": "cnt"}), SESSIONS[0], SESSIONS[-1])
    holes = news.copy()
    holes.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        make_news_alpha_handler(loader, holes, SESSIONS[0], SESSIONS[-1])


def test_fillna_by_prefix_touches_only_matching_feature_columns() -> None:
    columns = pd.MultiIndex.from_tuples([("feature", "news_a"), ("feature", "px_a"), ("label", "news_label")])
    frame = pd.DataFrame(np.nan, index=range(2), columns=columns)
    out = FillnaByPrefix("news_", 0.0)(frame)
    assert out[("feature", "news_a")].eq(0.0).all()
    assert out[("feature", "px_a")].isna().all() and out[("label", "news_label")].isna().all()
    assert frame.isna().all().all(), "the input frame must not be modified"
    assert FillnaByPrefix("news_").is_for_infer()
    with pytest.raises(ValueError, match="prefix"):
        FillnaByPrefix("absent_")(frame)


def test_alpha158_price_loader_builds_without_provider_data() -> None:
    # NOT loadable here: QlibDataLoader.load() needs real provider data. Only construction is checked.
    loader = make_alpha158_price_loader()
    assert isinstance(loader, QlibDataLoader)
    assert loader.fields["label"] == (["Ref($open, -2)/Ref($open, -1) - 1"], ["LABEL0"])
    expected_exprs, expected_names = Alpha158DL.get_feature_config()
    assert loader.fields["feature"] == (expected_exprs, expected_names) and len(expected_names) == 158
    assert not any(re.search(r"Ref\(.*?,\s*-", expr) for expr in expected_exprs), "features must not look forward"
    assert make_alpha158_price_loader("Ref($close, -1)/$close - 1").fields["label"][0] == ["Ref($close, -1)/$close - 1"]
    with pytest.raises(ValueError, match="label_expr"):
        make_alpha158_price_loader("  ")


def test_dataset_and_lgbmodel_fit_and_predict_one_row_per_test_row(
    qlib_workdir: Path, handler: DataHandlerLP
) -> None:
    fold = purged_walk_forward_segments(
        SESSIONS, train_len=150, valid_len=40, test_len=40, step=40, label_horizon=2
    )[0]
    dataset = DatasetH(handler=handler, segments=fold)
    model = LGBModel(loss="mse", num_leaves=15, learning_rate=0.1, num_threads=2)
    model.fit(dataset, num_boost_round=20, early_stopping_rounds=50, verbose_eval=0)
    assert model.model.current_iteration() == 20
    prediction = model.predict(dataset, segment="test")
    test_rows = dataset.prepare("test", col_set="feature", data_key=DataHandlerLP.DK_I)
    assert list(prediction.index.names) == ["datetime", "instrument"]
    assert len(prediction) == len(test_rows) == 40 * N_INSTRUMENTS
    assert prediction.index.equals(test_rows.index)
    assert prediction.notna().all()
    assert (qlib_workdir / "mlflow.db").exists()
