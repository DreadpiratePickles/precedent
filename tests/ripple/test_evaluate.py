"""Research: the Ripple graph ladder, its placebo pairing, the diffusion table and the M1 script."""

import pytest

pytest.importorskip("lightgbm")
pytest.importorskip("qlib")
pytest.importorskip("scipy")

import datetime as dt  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from fakes import make_event_record  # noqa: E402
from fixtures.tiny_provider import write_tiny_provider  # noqa: E402
from qlib.config import C  # noqa: E402
from qlib.data.dataset.loader import QlibDataLoader  # noqa: E402

from precedent.ripple.control_graphs import shuffle_preserving_degree  # noqa: E402
from precedent.ripple.evaluate import (  # noqa: E402
    LADDER_COLUMNS,
    diffusion_table,
    evaluate_graph_ladder,
    placebo_pairs,
    render_report,
)
from precedent.ripple.link_store import Link, LinkStore, Relation, graph_asof  # noqa: E402
from precedent.spine.event_record import ResolvedImpact  # noqa: E402
from precedent.spine.event_store import EventStore  # noqa: E402
from precedent.spine.pit_alignment import build_session_cutoffs  # noqa: E402
from precedent.spine.purged_splits import purged_walk_forward_segments  # noqa: E402
from precedent.spine.qlib_provider import init_qlib  # noqa: E402
from precedent.spine.session_calendar import load_nyse_calendar, weekday_sessions  # noqa: E402

# Qlib 0.9.7 still passes `inplace=` to pandas, which pandas 3 deprecates. The warning comes from
# third-party code we do not control; it is filtered for this module only.
pytestmark = pytest.mark.filterwarnings("ignore:The inplace keyword in DataFrame:Warning")

N_PAIRS = 30
SESSIONS = weekday_sessions(dt.date(2022, 1, 3), dt.date(2023, 12, 29), load_nyse_calendar())[:260]
CUTOFFS = build_session_cutoffs(SESSIONS, dt.time(16, 0))
SOURCES = [f"S{i:02d}" for i in range(N_PAIRS)]
RECEIVERS = [f"R{i:02d}" for i in range(N_PAIRS)]
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
RUNGS_DIR = Path(__file__).resolve().parents[2] / "references" / "rungs"
SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_ripple_m1.py"
LADDER_FEATURES = (["$close/$open", "Ref($close, 1)/$open"], ["close_over_open", "prev_close"])
SPILLOVER = 0.5
SHOCK_SIZE = 0.04
FIRST_EVENT_POSITION = 150  # abnormal_return_at needs 120 usable sessions before the event


@dataclass(frozen=True)
class Planted:
    """Prices and shocks in which each receiver follows its own source, one session later."""

    shocks: pd.Series
    abnormal_returns: pd.DataFrame
    prices: dict[str, pd.DataFrame]
    shock_matrix: np.ndarray


def planted_data(seed=5, market_sd=0.0) -> Planted:
    """Source S_i has a shock on session t with probability 0.5; its close-to-close return on t
    is that shock. Receiver R_i's return from open[t+1] to open[t+2], the label of session t,
    is `SPILLOVER * shock_i(t)` plus noise. With `market_sd == 0` the market never moves, so
    abnormal returns equal raw returns; a positive `market_sd` adds a MKT instrument."""
    rng = np.random.default_rng(seed)
    n = len(SESSIONS)
    has_shock = rng.random((n, N_PAIRS)) < 0.5
    shock = np.where(has_shock, rng.choice([-SHOCK_SIZE, SHOCK_SIZE], size=(n, N_PAIRS)), 0.0)
    open_return = 0.005 * rng.standard_normal((n, N_PAIRS))  # open[k] -> open[k+1]
    open_return[1:] += SPILLOVER * shock[:-1]
    receiver_open = 100.0 * np.vstack(
        [np.ones((1, N_PAIRS)), np.cumprod(1.0 + open_return[:-1], axis=0)]
    )
    receiver_close = receiver_open * (1.0 + 0.002 * rng.standard_normal((n, N_PAIRS)))
    source_open = 100.0 * np.cumprod(1.0 + 0.005 * rng.standard_normal((n, N_PAIRS)), axis=0)
    source_close = 100.0 * np.cumprod(1.0 + shock, axis=0)
    prices = {
        **{
            name: pd.DataFrame({"open": source_open[:, i], "close": source_close[:, i]}, SESSIONS)
            for i, name in enumerate(SOURCES)
        },
        **{
            name: pd.DataFrame(
                {"open": receiver_open[:, i], "close": receiver_close[:, i]}, SESSIONS
            )
            for i, name in enumerate(RECEIVERS)
        },
    }
    if market_sd > 0:
        market_open = 100.0 * np.cumprod(1.0 + market_sd * rng.standard_normal(n))
        prices["MKT"] = pd.DataFrame(
            {"open": market_open, "close": market_open * (1 + market_sd * rng.standard_normal(n))},
            SESSIONS,
        )
    realised = np.vstack([np.zeros((1, N_PAIRS)), open_return[:-1]])  # open-to-open on session k
    abnormal = pd.DataFrame(
        np.hstack([shock, realised]), SESSIONS, columns=[*SOURCES, *RECEIVERS]
    )
    rows, columns = np.nonzero(shock)
    shocks = pd.Series(
        shock[rows, columns],
        index=pd.MultiIndex.from_arrays(
            [SESSIONS[rows].as_unit("ns"), pd.Index([SOURCES[c] for c in columns], dtype=str)],
            names=["datetime", "instrument"],
        ),
        name="shock",
    ).sort_index()
    return Planted(shocks, abnormal, prices, shock)


def peer_link(source, receiver, doc="planted"):
    return Link(
        source_node=source,
        receiver_node=receiver,
        relation=Relation.PEER_OF,
        weight=1.0,
        valid_from_utc=CUTOFFS.iloc[0] - pd.Timedelta(days=1),
        valid_to_utc=CUTOFFS.iloc[-1] + pd.Timedelta(days=1),
        source_doc_id=doc,
        evidence_span=None,
        extractor_version="rules-planted",
    )


def graph_function(links, root: Path):
    """A `graph_for_cutoff` that reads the links through the real `graph_asof` visibility rule."""
    store = LinkStore(root)
    store.append(links)
    return lambda cutoff: graph_asof(store, cutoff)


def price_loader_for_horizon(horizon: int) -> QlibDataLoader:
    return QlibDataLoader(
        config={
            "feature": LADDER_FEATURES,
            "label": ([f"Ref($open, -{horizon + 1})/Ref($open, -1) - 1"], ["LABEL0"]),
        }
    )


@dataclass(frozen=True)
class PlantedMarket:
    true_graph: object
    shuffled_graph: object
    kwargs: dict
    planted: Planted


@pytest.fixture
def planted_market(tmp_path, monkeypatch) -> PlantedMarket:
    monkeypatch.chdir(tmp_path)
    planted = planted_data()
    write_tiny_provider(tmp_path / "provider", planted.prices)
    init_qlib(tmp_path / "provider", tmp_path / "work")
    # Qlib loads features through a worker pool; one kernel loads in-process, which is faster here.
    monkeypatch.setitem(C, "kernels", 1)
    true_links = [peer_link(source, receiver) for source, receiver in zip(SOURCES, RECEIVERS, strict=True)]
    true_graph = graph_function(true_links, tmp_path / "true_links")
    true_frame = true_graph(CUTOFFS.iloc[-1])
    shuffled_frame = shuffle_preserving_degree(true_frame, seed=7)
    moved = (shuffled_frame["receiver_node"].to_numpy() != true_frame["receiver_node"].to_numpy())
    assert moved.mean() > 0.8, "the placebo must move most links"
    shuffled_links = [
        peer_link(row.source_node, row.receiver_node, "shuffled")
        for row in shuffled_frame.itertuples()
    ]
    shuffled_graph = graph_function(shuffled_links, tmp_path / "shuffled_links")
    kwargs = {
        "shocks": planted.shocks,
        "price_loader_for_horizon": price_loader_for_horizon,
        "folds": FOLDS,
        "out_dir": tmp_path / "ladder",
        "session_cutoffs_utc": CUTOFFS,
        "sessions": SESSIONS,
        "abnormal_returns": planted.abnormal_returns,
        "lgb_params": LGB_PARAMS,
        "rungs_dir": RUNGS_DIR,
        "parser_version": "rules-test",
    }
    return PlantedMarket(true_graph, shuffled_graph, kwargs, planted)


def test_true_graph_beats_its_shuffle_on_planted_spillover(planted_market):
    # each receiver's next-session return is 0.5 times its source's shock, along the true links only
    table = evaluate_graph_ladder(
        {"true": planted_market.true_graph, "placebo": planted_market.shuffled_graph},
        **planted_market.kwargs,
        horizons=(1,),
    )
    ic = table.set_index("graph")["rank_ic_mean"]
    print("planted ICs:", ic.to_dict())
    assert ic["true"] > 0.1 and abs(ic["placebo"]) < 0.05
    assert list(table.columns) == LADDER_COLUMNS
    assert sorted(table["graph"]) == ["placebo", "true"] and (table["horizon"] == 1).all()
    assert (table["baseline"] == "m0_price").all()
    increments = table.set_index("graph")["incremental_rank_ic_vs_m1"]
    assert increments["true"] > 0.1 and abs(increments["placebo"]) < 0.05
    written = pd.read_csv(planted_market.kwargs["out_dir"] / "ladder.csv")
    assert list(written.columns) == LADDER_COLUMNS
    assert written["rank_ic_mean"].to_numpy() == pytest.approx(table["rank_ic_mean"].to_numpy())


def test_ladder_records_every_rung_with_the_horizon_label_and_calls_folds_per_horizon(
    planted_market,
):
    calls = []

    def folds_for(label_horizon):
        calls.append(label_horizon)
        return FOLDS

    kwargs = {**planted_market.kwargs, "folds": folds_for}
    table = evaluate_graph_ladder({"g1": planted_market.true_graph}, **kwargs, horizons=(1,))
    assert calls == [2]  # label horizon is h + 1
    manifest = json.loads(
        (planted_market.kwargs["out_dir"] / "g1_h1" / "run_manifest.json").read_text("utf-8")
    )
    assert manifest["label_expr"] == "Ref($open, -2)/Ref($open, -1) - 1"
    assert manifest["label_horizon"] == 2
    assert manifest["trial_count"] == 2 * len(FOLDS)
    rungs = {rung["name"]: rung["panel_prefixes"] for rung in manifest["rungs"]}
    # no baseline panel: the baseline is price-only, and the shipped m3_ripple_g1 rung (which adds
    # news_ too) cannot run, so the graph rung is built in memory with ripple_ only
    assert rungs == {"m0_price": [], "m3_ripple_g1": ["ripple_"]}
    assert table.loc[0, "baseline"] == "m0_price"


def test_a_news_baseline_makes_the_baseline_m1_news_and_uses_the_shipped_graph_rung(
    planted_market,
):
    rng = np.random.default_rng(2)
    index = pd.MultiIndex.from_product(
        [SESSIONS[100:200].as_unit("ns"), RECEIVERS[:5]], names=["datetime", "instrument"]
    )
    news = pd.DataFrame({"news_signed_1d": rng.normal(0, 1, len(index))}, index=index)
    table = evaluate_graph_ladder(
        {"g1": planted_market.true_graph},
        **planted_market.kwargs,
        baseline_panels={"news_": news},
        horizons=(1,),
    )
    assert table.loc[0, "baseline"] == "m1_news"
    manifest = json.loads(
        (planted_market.kwargs["out_dir"] / "g1_h1" / "run_manifest.json").read_text("utf-8")
    )
    rungs = {rung["name"]: rung["panel_prefixes"] for rung in manifest["rungs"]}
    assert rungs == {"m1_news": ["news_"], "m3_ripple_g1": ["news_", "ripple_"]}
    assert set(manifest["panel_sha256"]) == {"news_", "ripple_"}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"horizons": ()}, "horizon"),
        ({"horizons": (0,)}, "positive"),
        ({"horizons": (5, -1)}, "positive"),
        ({"horizons": (1.5,)}, "positive"),
        ({"horizons": (True,)}, "positive"),
    ],
)
def test_ladder_rejects_bad_horizons_before_fitting(planted_market, overrides, message):
    with pytest.raises(ValueError, match=message):
        evaluate_graph_ladder(
            {"g1": planted_market.true_graph}, **{**planted_market.kwargs, **overrides}
        )
    assert not planted_market.kwargs["out_dir"].exists()


def test_ladder_rejects_bad_graph_sets_before_fitting(planted_market):
    kwargs = planted_market.kwargs
    with pytest.raises(ValueError, match="at least one graph"):
        evaluate_graph_ladder({}, **kwargs, horizons=(1,))
    with pytest.raises(ValueError, match="graph name"):
        evaluate_graph_ladder({"../evil": planted_market.true_graph}, **kwargs, horizons=(1,))
    with pytest.raises(ValueError, match="ripple_"):
        evaluate_graph_ladder(
            {"g1": planted_market.true_graph},
            **kwargs,
            baseline_panels={"ripple_": pd.DataFrame()},
            horizons=(1,),
        )
    assert not kwargs["out_dir"].exists()


def ladder_frame(horizons=(1, 5, 20, 60), graphs=("g1", "g1_placebo", "g2", "g2_placebo")):
    rows = [
        {
            "graph": graph,
            "horizon": horizon,
            "rank_ic_mean": 0.01 * (index + 1) / horizon,
            "icir": 0.1,
            "incremental_rank_ic_vs_m1": 0.001 * (index + 1),
            "baseline": "m0_price",
        }
        for horizon in horizons
        for index, graph in enumerate(graphs)
    ]
    return pd.DataFrame(rows, columns=LADDER_COLUMNS)


def test_diffusion_table_leads_with_20_and_has_a_column_per_graph():
    text = diffusion_table(ladder_frame())
    lines = text.splitlines()
    head_at = [i for i, line in enumerate(lines) if line.startswith("| horizon")]
    assert len(head_at) == 2
    for position in head_at:
        assert lines[position] == "| horizon | g1 | g1_placebo | g2 | g2_placebo |"
    first_table = lines[head_at[0] + 2 : head_at[1] - 2]
    horizons = [line.split("|")[1].strip() for line in first_table if line.startswith("|")]
    assert horizons == ["20", "60", "5", "1"]
    assert "0.0005" in text  # g1 at horizon 20: 0.01 * 1 / 20, to 4 decimals
    assert "m0_price" in text  # the second table says what was subtracted


def test_diffusion_table_without_a_20_session_horizon_keeps_every_horizon():
    text = diffusion_table(ladder_frame(horizons=(1, 5)))
    horizons = [
        line.split("|")[1].strip()
        for line in text.splitlines()
        if line.startswith("|") and line.split("|")[1].strip().isdigit()
    ]
    assert horizons[:2] == ["5", "1"]


def test_placebo_pairs_put_each_graph_beside_its_placebo_and_demand_one():
    pairs = placebo_pairs(ladder_frame(horizons=(20,)))
    assert list(pairs["graph"]) == ["g1", "g2"]
    assert list(pairs.columns) == [
        "graph", "horizon", "rank_ic_mean", "placebo_rank_ic_mean", "margin_over_placebo",
    ]
    assert pairs["margin_over_placebo"].to_numpy() == pytest.approx(
        pairs["rank_ic_mean"] - pairs["placebo_rank_ic_mean"]
    )
    with pytest.raises(ValueError, match="g3"):
        placebo_pairs(ladder_frame(graphs=("g1", "g1_placebo", "g3")))


def test_report_says_plainly_when_no_graph_beats_its_placebo():
    ladder = ladder_frame(graphs=("g1", "g1_placebo"))
    ladder.loc[ladder["graph"] == "g1_placebo", "rank_ic_mean"] += 1.0  # placebo wins everywhere
    text = render_report(ladder, evidence_label="SYNTHETIC", counts={"shocks": 3})
    assert text.splitlines()[0].startswith("SYNTHETIC —")
    assert "No graph beats its placebo at any horizon." in text
    assert "EXPLORATORY (survivorship-biased prices)" in text
    assert "monthly" in text and "leads with 20" in text
    assert "shocks: 3" in text


def test_report_label_line_for_a_live_run_and_an_unknown_label():
    text = render_report(ladder_frame(), evidence_label="LIVE-DATA", counts={})
    assert text.splitlines()[0].startswith("LIVE —")
    with pytest.raises(ValueError, match="evidence_label"):
        render_report(ladder_frame(), evidence_label="PROBABLY-FINE", counts={})


# ---------------------------------------------------------------------------------------------
# scripts/run_ripple_m1.py on a generated market (the real run is an owner gate and is not made)
# ---------------------------------------------------------------------------------------------


def load_script():
    spec = importlib.util.spec_from_file_location("run_ripple_m1", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass(frozen=True)
class ScriptInputs:
    root: Path
    prices_dir: Path
    qlib_dir: Path
    events_dir: Path
    sector_map: Path
    data_dir: Path
    start: str
    end: str

    def arguments(self, *extra: str) -> list[str]:
        return [
            "--provider-dir", str(self.prices_dir),
            "--qlib-dir", str(self.qlib_dir),
            "--events-dir", str(self.events_dir),
            "--parser-version", "rules-test",
            "--sector-map", str(self.sector_map),
            "--start", self.start,
            "--end", self.end,
            "--market", "MKT",
            "--corr-window", "60",
            "--corr-top-k", "5",
            "--train-len", "100",
            "--valid-len", "30",
            "--test-len", "60",
            "--step", "60",
            *extra,
        ]


@pytest.fixture
def script_inputs(tmp_path, monkeypatch) -> ScriptInputs:
    monkeypatch.chdir(tmp_path)
    data_dir = tmp_path / "data"
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(data_dir))
    planted = planted_data(market_sd=0.001)
    qlib_dir, prices_dir = tmp_path / "qlib_us", tmp_path / "prices"
    write_tiny_provider(qlib_dir, planted.prices)
    prices_dir.mkdir()
    for name, frame in planted.prices.items():
        frame.rename_axis("date").reset_index().to_parquet(prices_dir / f"{name}.parquet")
    events_dir = tmp_path / "events"
    events_dir.mkdir()
    records = [
        make_event_record(
            article_id=f"test:{position}:{column}",
            parser_version="rules-test",
            available_utc=(CUTOFFS.iloc[position] - pd.Timedelta(hours=1)).to_pydatetime(),
            effective_session=SESSIONS[position].date(),
            resolved_impacts=[
                ResolvedImpact(
                    instrument=SOURCES[column], kind="company", direction=1.0, match_method="test"
                )
            ],
        )
        for position, column in zip(*np.nonzero(planted.shock_matrix[:, :10]), strict=True)
        if position >= FIRST_EVENT_POSITION
    ]
    EventStore(events_dir).append(records)
    sector_map = tmp_path / "sectors.json"
    sector_map.write_text(
        json.dumps(
            {
                name: "ABC"[i % 3]
                for i, name in enumerate([*SOURCES, *RECEIVERS])
            }
        ),
        encoding="utf-8",
    )
    return ScriptInputs(
        tmp_path, prices_dir, qlib_dir, events_dir, sector_map, data_dir,
        SESSIONS[0].date().isoformat(), SESSIONS[-1].date().isoformat(),
    )


def files_under(root: Path) -> set[str]:
    return {str(path.relative_to(root)) for path in root.rglob("*")}


def run_script(*arguments, cwd):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *arguments],
        capture_output=True, text=True, cwd=cwd, check=False, timeout=120,
    )


def test_script_dry_run_validates_inputs_and_writes_nothing(script_inputs):
    before = files_under(script_inputs.root)
    out_dir = script_inputs.root / "out"
    result = run_script(
        *script_inputs.arguments("--dry-run", "--out-dir", str(out_dir)), cwd=script_inputs.root
    )
    assert result.returncode == 0, result.stderr
    assert "dry run" in result.stdout.lower()
    assert "rules-test" in result.stdout and "g1" in result.stdout
    assert files_under(script_inputs.root) == before
    assert not out_dir.exists() and not script_inputs.data_dir.exists()


def test_script_dry_run_reports_a_missing_input_on_stderr_with_exit_1(script_inputs):
    arguments = script_inputs.arguments("--dry-run")
    arguments[arguments.index("--events-dir") + 1] = str(script_inputs.root / "no_events")
    result = run_script(*arguments, cwd=script_inputs.root)
    assert result.returncode == 1
    assert "events" in result.stderr.lower()


def test_script_usage_errors_exit_2(script_inputs):
    result = run_script("--dry-run", cwd=script_inputs.root)  # --start and --end are required
    assert result.returncode == 2
    result = run_script(*script_inputs.arguments("--dry-run", "--horizons", "1,zero"), cwd=script_inputs.root)
    assert result.returncode == 2


def test_script_main_writes_the_report_beside_the_placebo_and_publishes_only_when_asked(
    script_inputs, monkeypatch
):
    script = load_script()

    def small_loader_factory(horizon):
        monkeypatch.setitem(C, "kernels", 1)  # called after Qlib is initialised
        return price_loader_for_horizon(horizon)

    out_dir = script_inputs.root / "run"
    exit_code = script.main(
        script_inputs.arguments(
            "--evidence-label", "SYNTHETIC", "--horizons", "1", "--graphs", "g1,g2",
            "--out-dir", str(out_dir),
        ),
        price_loader_factory=small_loader_factory,
        lgb_params=LGB_PARAMS,
    )
    assert exit_code == 0
    report = (out_dir / "report.md").read_text(encoding="utf-8")
    assert report.splitlines()[0].startswith("SYNTHETIC —")
    assert "g1_placebo" in report and "EXPLORATORY (survivorship-biased prices)" in report
    ladder = pd.read_csv(out_dir / "ladder.csv")
    assert sorted(ladder["graph"]) == ["g1", "g1_placebo", "g2", "g2_placebo"]
    assert (out_dir / "links" / "g1" / "links.jsonl").exists()
    assert (out_dir / "links" / "g2" / "links.jsonl").exists()
    assert not script_inputs.data_dir.exists()  # nothing is written to the ledger by default

    published = script_inputs.root / "run_published"
    exit_code = script.main(
        script_inputs.arguments(
            "--evidence-label", "SYNTHETIC", "--horizons", "1", "--graphs", "g1",
            "--out-dir", str(published), "--publish",
        ),
        price_loader_factory=small_loader_factory,
        lgb_params=LGB_PARAMS,
    )
    assert exit_code == 0
    ripple_dir = script_inputs.data_dir / "ripple"
    assert {path.name for path in ripple_dir.iterdir()} == {
        "shocks.parquet", "abnormal_returns.parquet", "MANIFEST.json",
    }
    manifest = json.loads((ripple_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["evidence_label"] == "SYNTHETIC"
    shocks = pd.read_parquet(ripple_dir / "shocks.parquet")
    assert len(shocks) > 0 and set(shocks["instrument"]) <= set(SOURCES[:10])
    assert (shocks["datetime"] >= SESSIONS[FIRST_EVENT_POSITION]).all()
