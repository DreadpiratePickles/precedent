"""Research: factor panels joined to Qlib price features; the price index wins, absent news is zero."""

import pytest

pytest.importorskip("qlib")

import re  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from qlib.config import C  # noqa: E402
from qlib.contrib.data.loader import Alpha158DL  # noqa: E402
from qlib.data.dataset.handler import DataHandlerLP  # noqa: E402
from qlib.data.dataset.loader import QlibDataLoader  # noqa: E402

from precedent.spine.news_factors import PANEL_COLUMNS, build_news_factor_panel  # noqa: E402
from precedent.spine.qlib_dataset import (  # noqa: E402
    FillnaByPrefix,
    JoinReport,
    make_alpha158_price_loader,
    make_alpha_handler,
)

# Qlib 0.9.7 still passes `inplace=` to pandas, which pandas 3 deprecates. The warning comes from
# third-party code we do not control; it is filtered for this module only.
pytestmark = pytest.mark.filterwarnings("ignore:The inplace keyword in DataFrame:Warning")

START, END = "2024-01-02", "2024-01-05"
SESSIONS = pd.DatetimeIndex(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
PRICE_FEATURES = ["open_to_close", "close_over_open_2d"]


@pytest.fixture
def synthetic_price_loader(tiny_provider, tmp_path, monkeypatch):
    """A QlibDataLoader over the tiny provider (AAA, BBB with a missing day, SPY)."""
    monkeypatch.chdir(tmp_path)
    # Qlib loads features through a worker pool; on macOS each spawned worker re-imports mlflow
    # (about 30 s per load here). One kernel loads in-process, which is all four rows need.
    monkeypatch.setitem(C, "kernels", 1)
    return QlibDataLoader(
        config={
            "feature": (["$close/$open", "Ref($close, 1)/$open"], PRICE_FEATURES),
            "label": (["Ref($open, -2)/Ref($open, -1) - 1"], ["LABEL0"]),
        }
    )


def news_events(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "effective_session": pd.to_datetime([day for day, _ in rows]),
            "instrument": [name for _, name in rows],
            "severity": 6,
            "direction": 1.0,
            "confidence_weight": 0.5,
            "novelty": "new",
        }
    )


def news_panel(rows: list[tuple[str, str]]) -> pd.DataFrame:
    return build_news_factor_panel(news_events(rows), SESSIONS, max_age_sessions=1)


def news_panel_with_extra_row(instrument: str) -> pd.DataFrame:
    panel = news_panel([("2024-01-03", "AAA")])
    extra = pd.DataFrame(
        0.0,
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-04"), instrument)], names=["datetime", "instrument"]
        ),
        columns=PANEL_COLUMNS,
    )
    return pd.concat([panel, extra]).sort_index()


PANEL = news_panel([("2024-01-03", "AAA")])
EMPTY_NEWS_PANEL = build_news_factor_panel(news_events([]), SESSIONS)


def test_factor_rows_without_a_price_row_are_counted(synthetic_price_loader):
    panel = news_panel_with_extra_row(instrument="NOT_LISTED")
    handler, report = make_alpha_handler(synthetic_price_loader, {"news_": panel}, START, END)
    assert report == JoinReport(events_without_price_row=1)
    raw = handler.fetch(col_set="__raw", data_key=DataHandlerLP.DK_R)
    assert "NOT_LISTED" not in raw.index.get_level_values("instrument")


def test_factor_rows_with_a_price_row_are_not_counted(synthetic_price_loader):
    _, report = make_alpha_handler(synthetic_price_loader, {"news_": PANEL}, START, END)
    assert report.events_without_price_row == 0


def test_a_day_with_no_events_gives_zeros(synthetic_price_loader):
    handler, _ = make_alpha_handler(synthetic_price_loader, {"news_": EMPTY_NEWS_PANEL}, START, END)
    frame = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    assert sorted(frame.filter(like="news_").columns) == sorted(PANEL_COLUMNS)
    assert len(frame) > 0
    assert (frame.filter(like="news_") == 0).all().all()


def test_no_factor_panels_is_the_price_only_rung(synthetic_price_loader):
    handler, report = make_alpha_handler(synthetic_price_loader, {}, START, END)
    frame = handler.fetch(col_set="feature")
    assert sorted(frame.columns) == sorted(PRICE_FEATURES)
    assert not any(c.startswith("news_") for c in frame.columns)
    assert report.events_without_price_row == 0


def test_a_panel_with_a_wrongly_prefixed_column_is_rejected(synthetic_price_loader):
    bad = PANEL.rename(columns={"news_cnt_1d": "cnt"})
    with pytest.raises(ValueError, match="news_"):
        make_alpha_handler(synthetic_price_loader, {"news_": bad}, START, END)


def test_news_lands_in_the_feature_group_and_is_zero_filled_only_at_inference(
    synthetic_price_loader,
):
    handler, _ = make_alpha_handler(synthetic_price_loader, {"news_": PANEL}, START, END)
    raw = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_R)
    inferred = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    assert sorted(inferred.columns) == sorted(PRICE_FEATURES + PANEL_COLUMNS)
    assert raw["news_cnt_1d"].isna().any()
    assert not inferred[PANEL_COLUMNS].isna().any().any()
    hit = inferred.loc[(pd.Timestamp("2024-01-03"), "AAA")]
    assert hit["news_cnt_1d"] == 1
    assert hit["news_signed_1d"] == pytest.approx(0.6 * 0.5)
    assert inferred.loc[(pd.Timestamp("2024-01-02"), "AAA"), PANEL_COLUMNS].eq(0.0).all()
    assert inferred.loc[(pd.Timestamp("2024-01-03"), "SPY"), PANEL_COLUMNS].eq(0.0).all()


def test_price_features_are_left_untouched_including_their_nans(synthetic_price_loader):
    handler, _ = make_alpha_handler(synthetic_price_loader, {"news_": PANEL}, START, END)
    price_only, _ = make_alpha_handler(synthetic_price_loader, {}, START, END)
    inferred = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    expected = price_only.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    assert expected[PRICE_FEATURES].isna().any().any()  # BBB has no 2024-01-03 row
    pd.testing.assert_frame_equal(inferred[PRICE_FEATURES], expected[PRICE_FEATURES])


def test_the_price_index_wins_the_join(synthetic_price_loader):
    handler, _ = make_alpha_handler(
        synthetic_price_loader, {"news_": news_panel_with_extra_row("ZZZ")}, START, END
    )
    price_only, _ = make_alpha_handler(synthetic_price_loader, {}, START, END)
    raw = handler.fetch(col_set="__raw", data_key=DataHandlerLP.DK_R)
    price_raw = price_only.fetch(col_set="__raw", data_key=DataHandlerLP.DK_R)
    assert raw.index.equals(price_raw.index)


def test_several_panels_each_zero_filled_under_their_own_prefix(synthetic_price_loader):
    precedent = pd.DataFrame(
        {"prec_score": [0.7]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-04"), "SPY")], names=["datetime", "instrument"]
        ),
    )
    handler, report = make_alpha_handler(
        synthetic_price_loader, {"news_": PANEL, "prec_": precedent}, START, END
    )
    inferred = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_I)
    assert sorted(inferred.columns) == sorted(PRICE_FEATURES + PANEL_COLUMNS + ["prec_score"])
    assert inferred.loc[(pd.Timestamp("2024-01-04"), "SPY"), "prec_score"] == pytest.approx(0.7)
    assert not inferred[PANEL_COLUMNS + ["prec_score"]].isna().any().any()
    assert report.events_without_price_row == 0


def test_the_default_learn_processors_drop_unlabelled_rows_and_rank_normalise(
    synthetic_price_loader,
):
    handler, _ = make_alpha_handler(synthetic_price_loader, {"news_": PANEL}, START, END)
    raw_label = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)["LABEL0"]
    learn = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_L)["LABEL0"]
    assert raw_label.isna().any()
    assert len(learn) > 0
    assert not learn.isna().any()
    assert (
        learn.abs().max() <= 0.5 * 3.46 + 1e-9
    )  # CSRankNorm maps pct-ranks to (rank - 0.5) * 3.46
    # Order within each day is preserved.
    by_day = {"level": "datetime"}
    learn_rank = learn.groupby(**by_day).rank()
    raw_rank = raw_label.loc[learn.index].groupby(**by_day).rank()
    assert (learn_rank == raw_rank).all()


def test_learn_features_have_no_nan_in_the_factor_columns(synthetic_price_loader):
    handler, _ = make_alpha_handler(synthetic_price_loader, {"news_": PANEL}, START, END)
    learn = handler.fetch(col_set="feature", data_key=DataHandlerLP.DK_L)
    assert len(learn) > 0
    assert sorted(learn.filter(like="news_").columns) == sorted(PANEL_COLUMNS)
    assert not learn[PANEL_COLUMNS].isna().any().any()


def test_custom_learn_processors_replace_the_defaults(synthetic_price_loader):
    handler, _ = make_alpha_handler(
        synthetic_price_loader, {}, START, END, learn_processors=[{"class": "DropnaLabel"}]
    )
    raw_label = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_R)["LABEL0"]
    learn = handler.fetch(col_set="label", data_key=DataHandlerLP.DK_L)["LABEL0"]
    pd.testing.assert_series_equal(learn, raw_label.dropna())  # no rank normalisation


def test_a_panel_with_a_bad_structure_is_rejected(synthetic_price_loader):
    def build(panel):
        return make_alpha_handler(synthetic_price_loader, {"news_": panel}, START, END)

    with pytest.raises(ValueError, match="duplicate"):
        build(pd.concat([PANEL, PANEL.iloc[:1]]))
    with pytest.raises(ValueError, match="index"):
        build(PANEL.rename_axis(["date", "instrument"]))
    holes = PANEL.copy()
    holes.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        build(holes)


def test_a_panel_key_must_not_be_empty(synthetic_price_loader):
    with pytest.raises(ValueError, match="key"):
        make_alpha_handler(synthetic_price_loader, {"": PANEL}, START, END)


def test_fillna_by_prefix_touches_only_matching_feature_columns():
    columns = pd.MultiIndex.from_tuples(
        [("feature", "news_a"), ("feature", "prec_a"), ("feature", "px_a"), ("label", "news_label")]
    )
    frame = pd.DataFrame(np.nan, index=range(2), columns=columns)
    out = FillnaByPrefix(("news_", "prec_"), 0.0)(frame)
    assert out[("feature", "news_a")].eq(0.0).all()
    assert out[("feature", "prec_a")].eq(0.0).all()
    assert out[("feature", "px_a")].isna().all()
    assert out[("label", "news_label")].isna().all()
    assert frame.isna().all().all(), "the input frame must not be modified"
    assert FillnaByPrefix(("news_",)).is_for_infer()
    with pytest.raises(ValueError, match="prefix"):
        FillnaByPrefix(("absent_",))(frame)


def test_alpha158_price_loader_builds_without_provider_data():
    # Only construction is checked; loading Alpha158 needs a full provider.
    loader = make_alpha158_price_loader()
    assert isinstance(loader, QlibDataLoader)
    assert loader.fields["label"] == (["Ref($open, -2)/Ref($open, -1) - 1"], ["LABEL0"])
    expected_exprs, expected_names = Alpha158DL.get_feature_config()
    assert loader.fields["feature"] == (expected_exprs, expected_names)
    assert len(expected_names) == 158
    assert not any(re.search(r"Ref\(.*?,\s*-", expr) for expr in expected_exprs), (
        "features must not look forward"
    )
    custom = make_alpha158_price_loader("Ref($close, -1)/$close - 1")
    assert custom.fields["label"][0] == ["Ref($close, -1)/$close - 1"]
    with pytest.raises(ValueError, match="label_expr"):
        make_alpha158_price_loader("  ")
