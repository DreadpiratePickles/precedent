"""Research: the ablation ladder, its rung files, label horizon and run manifest."""

import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("qlib")

import json  # noqa: E402
import re  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from fixtures.synthetic_market import N_SESSIONS, START_DAY, write_planted_market  # noqa: E402
from qlib.config import C  # noqa: E402
from qlib.data.dataset.loader import DataLoader, QlibDataLoader  # noqa: E402

from precedent.spine.ablation import (  # noqa: E402
    Rung,
    code_commit,
    label_horizon_of,
    load_rungs,
    panel_sha256,
    run_ablation,
)
from precedent.spine.purged_splits import purged_walk_forward_segments  # noqa: E402
from precedent.spine.qlib_dataset import DEFAULT_LABEL_EXPR  # noqa: E402
from precedent.spine.qlib_provider import init_qlib  # noqa: E402

# Qlib 0.9.7 still passes `inplace=` to pandas, which pandas 3 deprecates. The warning comes from
# third-party code we do not control; it is filtered for this module only.
pytestmark = pytest.mark.filterwarnings("ignore:The inplace keyword in DataFrame:Warning")

SESSIONS = pd.bdate_range(START_DAY, periods=N_SESSIONS)
FOLDS = purged_walk_forward_segments(
    SESSIONS, train_len=100, valid_len=30, test_len=60, step=60, label_horizon=2
)
LGB_PARAMS = {
    "n_estimators": 40,
    "num_leaves": 7,
    "learning_rate": 0.1,
    "min_child_samples": 20,
    "n_jobs": 1,
    "random_state": 7,
    "verbose": -1,
}
SHIPPED_RUNGS_DIR = Path(__file__).resolve().parents[2] / "references" / "rungs"


@dataclass(frozen=True)
class SyntheticMarket:
    sessions: pd.DatetimeIndex
    panel: pd.DataFrame
    price_loader: DataLoader


@pytest.fixture
def synthetic_market(tmp_path, monkeypatch) -> SyntheticMarket:
    monkeypatch.chdir(tmp_path)
    planted = write_planted_market(tmp_path / "provider", seed=11)
    init_qlib(tmp_path / "provider", tmp_path / "work")
    # Qlib loads features through a worker pool; one kernel loads in-process, which is faster here.
    monkeypatch.setitem(C, "kernels", 1)
    loader = QlibDataLoader(
        config={
            "feature": (["$close/$open", "Ref($close, 1)/$open"], ["close_over_open", "prev_close"]),
            "label": ([DEFAULT_LABEL_EXPR], ["LABEL0"]),
        }
    )
    return SyntheticMarket(sessions=planted.sessions, panel=planted.panel, price_loader=loader)


def write_rung(rungs_dir: Path, filename: str, body: dict) -> None:
    rungs_dir.mkdir(parents=True, exist_ok=True)
    (rungs_dir / filename).write_text(yaml.safe_dump(body), encoding="utf-8")


@pytest.fixture
def rungs_dir(tmp_path) -> Path:
    """The test's own two-rung ladder, so a track adding a shipped rung cannot change the counts."""
    directory = tmp_path / "rungs"
    write_rung(directory, "m0_price.yaml", {"name": "m0_price", "panel_prefixes": []})
    write_rung(directory, "m1_news.yaml", {"name": "m1_news", "panel_prefixes": ["news_"]})
    return directory


def run(market: SyntheticMarket, out_dir: Path, rungs_dir: Path, folds=FOLDS) -> pd.DataFrame:
    return run_ablation(
        load_rungs(rungs_dir),
        market.price_loader,
        {"news_": market.panel},
        folds,
        out_dir,
        LGB_PARAMS,
        parser_version="rules-test",
        label_expr=DEFAULT_LABEL_EXPR,
        rungs_dir=rungs_dir,
    )


def test_label_horizon_is_derived_from_the_label_expression():
    assert label_horizon_of("Ref($open, -2)/Ref($open, -1) - 1") == 2
    assert label_horizon_of("Ref($open, -6)/Ref($open, -1) - 1") == 6


def test_label_horizon_reads_nested_references_and_ignores_look_backs():
    assert label_horizon_of("Ref(Mean($close, 5), -3) / Ref($close, 2) - 1") == 3
    assert label_horizon_of("Ref($close, 1) / $open") == 0
    assert label_horizon_of("$close / $open - 1") == 0


def test_label_horizon_rejects_a_shift_that_is_not_an_integer_literal():
    with pytest.raises(ValueError, match="Ref"):
        label_horizon_of("Ref($open, -$volume)")


def test_the_shipped_rungs_load_and_validate():
    rungs = {rung.name: rung for rung in load_rungs(SHIPPED_RUNGS_DIR)}
    assert rungs["m0_price"] == Rung(name="m0_price", panel_prefixes=[])
    assert rungs["m1_news"] == Rung(name="m1_news", panel_prefixes=["news_"])


def test_load_rungs_rejects_a_malformed_or_duplicate_rung_file(tmp_path):
    write_rung(tmp_path / "bad", "a.yaml", {"name": "a"})
    with pytest.raises(ValueError, match="a.yaml"):
        load_rungs(tmp_path / "bad")
    write_rung(tmp_path / "dup", "a.yaml", {"name": "same", "panel_prefixes": []})
    write_rung(tmp_path / "dup", "b.yaml", {"name": "same", "panel_prefixes": []})
    with pytest.raises(ValueError, match="same"):
        load_rungs(tmp_path / "dup")
    with pytest.raises(ValueError, match="no rung files"):
        load_rungs(tmp_path / "empty")


def test_panel_hash_ignores_column_and_row_order_but_not_values():
    panel = pd.DataFrame(
        {"b": [1.0, 2.0], "a": [3.0, 4.0]},
        index=pd.MultiIndex.from_tuples(
            [(pd.Timestamp("2024-01-03"), "X"), (pd.Timestamp("2024-01-02"), "X")],
            names=["datetime", "instrument"],
        ),
    )
    assert panel_sha256(panel) == panel_sha256(panel[["a", "b"]].iloc[::-1])
    assert panel_sha256(panel) != panel_sha256(panel.assign(a=[3.0, 4.5]))
    assert re.fullmatch(r"[0-9a-f]{64}", panel_sha256(panel))


def test_code_commit_is_unknown_outside_a_git_repository(tmp_path):
    assert code_commit(tmp_path) == "unknown"
    assert re.fullmatch(r"[0-9a-f]{40}", code_commit(Path(__file__).parent))


def test_ablation_records_every_trial(tmp_path, synthetic_market, rungs_dir):
    out_dir = tmp_path / "out"
    frame = run(synthetic_market, out_dir, rungs_dir)
    manifest = json.loads((out_dir / "run_manifest.json").read_text())
    assert manifest["trial_count"] == len(FOLDS) * 2 and len(manifest["panel_sha256"]["news_"]) == 64
    assert list(frame.columns) == ["rung", "fold", "rank_ic_mean", "icir", "n_days"]
    assert sorted(frame["rung"].unique()) == ["m0_price", "m1_news"]
    assert len(frame) == len(FOLDS) * 2
    assert (frame["n_days"] > 0).all()
    assert pd.read_csv(out_dir / "ablation.csv")["rung"].tolist() == frame["rung"].tolist()


def test_the_manifest_carries_the_run_provenance(tmp_path, synthetic_market, rungs_dir):
    run(synthetic_market, tmp_path, rungs_dir)
    manifest = json.loads((tmp_path / "run_manifest.json").read_text())
    assert manifest["parser_version"] == "rules-test"
    assert manifest["label_expr"] == DEFAULT_LABEL_EXPR
    assert manifest["lgb_params"] == LGB_PARAMS
    assert manifest["events_without_price_row"] == 0
    assert re.fullmatch(r"[0-9a-f]{40}|unknown", manifest["code_commit"])
    assert manifest["folds"][0]["train"] == [day.isoformat() for day in FOLDS[0]["train"]]
    assert sorted(manifest["rung_files_sha256"]) == ["m0_price.yaml", "m1_news.yaml"]


def test_a_purge_gap_shorter_than_the_label_horizon_is_refused(
    tmp_path, synthetic_market, rungs_dir
):
    too_short = purged_walk_forward_segments(
        SESSIONS, train_len=100, valid_len=30, test_len=60, step=60, label_horizon=1
    )
    with pytest.raises(ValueError, match="purge"):
        run(synthetic_market, tmp_path, rungs_dir, folds=too_short)
    assert not (tmp_path / "ablation.csv").exists()


def test_a_rung_that_selects_no_supplied_panel_is_refused(tmp_path, synthetic_market):
    directory = tmp_path / "rungs"
    write_rung(directory, "m9.yaml", {"name": "m9_macro", "panel_prefixes": ["macro_"]})
    with pytest.raises(ValueError, match="macro_"):
        run(synthetic_market, tmp_path / "out", directory)
