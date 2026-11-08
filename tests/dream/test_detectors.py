import json
import time as wall_clock
from datetime import UTC, date, datetime, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fakes import make_event_record

from precedent.dream import detectors as detectors_module
from precedent.dream.claim_schema import Claim, claim_id_for, link_id_for
from precedent.dream.claim_store import (
    ClaimStore,
    ResolutionStore,
    ShiftStore,
    TensionStore,
    claims_asof,
)
from precedent.dream.detectors import (
    DETECTOR_VERSION,
    DetectReport,
    abnormal_returns_panel,
    detect_asymmetric_links,
    detect_stale_reliance,
    detect_text_vs_price,
    detect_value_conflicts,
    run_detectors,
)
from precedent.dream.inputs import DreamStores, documents_per_session, load_detector_params
from precedent.ripple.link_store import GRAPH_COLUMNS, Link, LinkStore, graph_asof
from precedent.spine.abnormal_returns import abnormal_return_at
from precedent.spine.pit_alignment import build_session_cutoffs

EST_WINDOW = (-120, -21)
MIN_OBS = 60
SESSIONS = pd.bdate_range("2023-01-02", periods=320)  # naive session dates
CUTOFFS = build_session_cutoffs(SESSIONS, time(16, 0))
SESSION = SESSIONS[250].date()
PARAMS = {
    "text_vs_price": {"z_threshold": 2.0},
    "abnormal_returns": {"est_window": EST_WINDOW, "min_obs": MIN_OBS},
    "stale_reliance": {"max_age_sessions": 300},
}


def utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00")).astimezone(UTC)


def claim(
    subject="ACME",
    attribute="capex_direction",
    value="up",
    period="fiscal_year",
    doc="doc-1",
    valid_from="2023-06-01T21:30:00Z",
) -> Claim:
    return Claim(
        subject=subject,
        attribute=attribute,
        period=period,
        value=value,
        valid_from_utc=utc(valid_from),
        recorded_utc=utc(valid_from),
        source_doc_id=doc,
        evidence_span="stated in the filing",
        extractor_version="rules-test-v1",
    )


def frame(claims: list[Claim]) -> pd.DataFrame:
    """The claims as `claims_asof` would show them (enum values as strings)."""
    return pd.DataFrame(
        [c.model_dump(mode="json") for c in claims], columns=list(Claim.model_fields)
    )


def returns_frame(
    planted: dict[tuple[int, str], float] | None = None,
    instruments: tuple[str, ...] = ("ACME",),
    seed: int = 7,
    n_sessions: int = len(SESSIONS),
) -> pd.DataFrame:
    """A market column and instruments with beta 1 and 1% residual noise; planted shocks added."""
    rng = np.random.default_rng(seed)
    market = rng.normal(0.0, 0.01, n_sessions)
    data = {"MKT": market}
    for name in instruments:
        data[name] = 0.0001 + market + rng.normal(0.0, 0.01, n_sessions)
    for (position, name), shock in (planted or {}).items():
        data[name][position] += shock
    return pd.DataFrame(data, index=SESSIONS[:n_sessions])


def abnormal_frame(rows: dict[tuple[date, str], tuple[float, float]]) -> pd.DataFrame:
    index = pd.MultiIndex.from_tuples(
        [(pd.Timestamp(s), i) for (s, i) in rows], names=["session", "instrument"]
    )
    return pd.DataFrame(
        list(rows.values()), index=index, columns=["abnormal_return", "residual_sd"]
    )


POSITIVE = [claim("ACME", "event_direction", "positive", period="unspecified", doc="gdelt:1")]
CONFLICT_FRAME = frame(
    [
        claim(value="up", doc="10K-2023"),
        claim(value="down", doc="8K-2024-03"),
    ]
)


# --- value conflicts ---------------------------------------------------------------


def test_conflicting_values_from_two_documents_open_one_tension():
    claims = frame(
        [
            claim("ACME", "capex_direction", "up", doc="10K-2023", period="fiscal_year"),
            claim("ACME", "capex_direction", "down", doc="8K-2024-03", period="fiscal_year"),
        ]
    )
    [t] = detect_value_conflicts(claims, SESSION)
    assert t.kind == "value_conflict" and set(t.claim_ids) == set(claims.claim_id)
    assert t.claim_ids == sorted(t.claim_ids) and t.subject == "ACME"
    assert t.opened_session == SESSION and t.detector_version == DETECTOR_VERSION
    assert t.details == {
        "attribute": "capex_direction",
        "period": "fiscal_year",
        "values": "down,up",
    }


def test_different_periods_are_not_a_conflict():
    claims = frame(
        [
            claim("ACME", "guidance_direction", "up", period="next_quarter"),
            claim("ACME", "guidance_direction", "down", period="fiscal_year", doc="doc-2"),
        ]
    )
    assert detect_value_conflicts(claims, SESSION) == []


def test_same_document_cannot_conflict_with_itself():
    claims = frame(
        [
            claim("ACME", "capex_direction", "up", doc="D"),
            claim("ACME", "capex_direction", "down", doc="D"),
        ]
    )
    assert detect_value_conflicts(claims, SESSION) == []


def test_non_conflicting_pairs_and_other_subjects_are_quiet():
    claims = frame(
        [
            claim("ACME", "capex_direction", "up", doc="A"),
            claim("ACME", "capex_direction", "flat", doc="B"),
            claim("ACME", "capex_direction", "up", doc="C"),
            claim("OTHER", "capex_direction", "down", doc="D"),
        ]
    )
    assert detect_value_conflicts(claims, SESSION) == []
    assert detect_value_conflicts(claims.iloc[0:0], SESSION) == []


def test_a_group_with_several_conflicting_claims_opens_one_tension_with_all_of_them():
    claims = frame(
        [
            claim(value="up", doc="A"),
            claim(value="up", doc="B"),
            claim(value="down", doc="C"),
            claim(value="flat", doc="D"),
        ]
    )
    [t] = detect_value_conflicts(claims, SESSION)
    expected = {
        claim_id
        for claim_id, value in zip(claims.claim_id, claims.value, strict=True)
        if value != "flat"
    }
    assert set(t.claim_ids) == expected and len(t.claim_ids) == 3


def test_started_ended_conflict_but_event_direction_takes_no_part():
    # D72: positive/negative news about one name is not a value conflict (text_vs_price and
    # direction_drift own event_direction)
    claims = frame(
        [
            claim("ACME", "supplier_relationship", "started", doc="A", period="unspecified"),
            claim("ACME", "supplier_relationship", "ended", doc="B", period="unspecified"),
            claim("ACME", "event_direction", "positive", doc="C", period="unspecified"),
            claim("ACME", "event_direction", "negative", doc="D", period="unspecified"),
        ]
    )
    tensions = detect_value_conflicts(claims, SESSION)
    assert [t.details["attribute"] for t in tensions] == ["supplier_relationship"]


def test_tension_id_is_stable_across_sessions():
    a = detect_value_conflicts(
        CONFLICT_FRAME, SESSIONS[10].date(), recorded_utc=CUTOFFS.iloc[10].to_pydatetime()
    )
    b = detect_value_conflicts(
        CONFLICT_FRAME, SESSIONS[11].date(), recorded_utc=CUTOFFS.iloc[11].to_pydatetime()
    )
    assert a[0].tension_id == b[0].tension_id and a[0].opened_session == SESSIONS[10].date()


# --- text versus price -------------------------------------------------------------


@pytest.fixture
def planted_prices() -> pd.DataFrame:
    return abnormal_frame({(SESSION, "ACME"): (-0.04, 0.01)})


def test_text_vs_price_fires_on_a_planted_opposite_move(planted_prices):
    [t] = detect_text_vs_price(frame(POSITIVE), planted_prices, SESSION)
    assert t.kind == "text_vs_price" and t.details["z"] < -2
    assert t.claim_ids == [POSITIVE[0].claim_id] and t.subject == "ACME"
    assert t.details == {
        "claim_value": "positive",
        "abnormal_return": pytest.approx(-0.04),
        "z": pytest.approx(-4.0),
        "residual_sd": pytest.approx(0.01),
    }


def test_text_vs_price_fires_for_a_negative_claim_and_a_positive_move():
    negative = [claim("ACME", "event_direction", "negative", period="unspecified", doc="g:2")]
    prices = abnormal_frame({(SESSION, "ACME"): (0.03, 0.01)})
    [t] = detect_text_vs_price(frame(negative), prices, SESSION)
    assert t.details["z"] == pytest.approx(3.0) and t.details["claim_value"] == "negative"


def test_same_sign_large_move_does_not_fire(planted_prices):
    prices = abnormal_frame({(SESSION, "ACME"): (0.05, 0.01)})
    assert detect_text_vs_price(frame(POSITIVE), prices, SESSION) == []


def test_small_opposite_move_does_not_fire():
    prices = abnormal_frame({(SESSION, "ACME"): (-0.019, 0.01)})
    assert detect_text_vs_price(frame(POSITIVE), prices, SESSION) == []


def test_neutral_claims_never_fire():
    neutral = [claim("ACME", "event_direction", "neutral", period="unspecified", doc="g:3")]
    prices = abnormal_frame({(SESSION, "ACME"): (-0.09, 0.01)})
    assert detect_text_vs_price(frame(neutral), prices, SESSION) == []


def test_other_attributes_missing_rows_nan_and_zero_sd_never_fire():
    other = [claim("ACME", "capex_direction", "up", doc="g:4")]
    prices = abnormal_frame({(SESSION, "ACME"): (-0.09, 0.01)})
    assert detect_text_vs_price(frame(other), prices, SESSION) == []
    assert detect_text_vs_price(frame(POSITIVE), prices, SESSIONS[251].date()) == []
    no_instrument = abnormal_frame({(SESSION, "ZZZ"): (-0.09, 0.01)})
    assert detect_text_vs_price(frame(POSITIVE), no_instrument, SESSION) == []
    for ar, sd in [(float("nan"), 0.01), (-0.09, float("nan")), (-0.09, 0.0)]:
        assert (
            detect_text_vs_price(
                frame(POSITIVE), abnormal_frame({(SESSION, "ACME"): (ar, sd)}), SESSION
            )
            == []
        )


def test_text_vs_price_is_quiet_on_stationary_data():
    names = tuple(f"S{i:02d}" for i in range(50))
    returns = returns_frame(instruments=names, seed=0)
    started = wall_clock.perf_counter()
    panel = abnormal_returns_panel(returns, "MKT", SESSIONS, EST_WINDOW, MIN_OBS)
    rng = np.random.default_rng(5)
    scored = SESSIONS[-200:]
    fired = 0
    for session in scored:
        values = rng.choice(["positive", "negative"], size=len(names))
        claims = pd.DataFrame(
            {
                "claim_id": [f"{session:%Y%m%d}-{n}" for n in names],
                "subject": list(names),
                "attribute": "event_direction",
                "period": "unspecified",
                "value": values,
                "source_doc_id": [f"doc-{session:%Y%m%d}-{n}" for n in names],
            }
        )
        recorded = CUTOFFS[session].to_pydatetime()  # the synthetic axis keeps market holidays
        fired += len(detect_text_vs_price(claims, panel, session.date(), recorded_utc=recorded))
    elapsed = wall_clock.perf_counter() - started
    rate = fired / (200 * 50)
    print(f"quiet-case rate {rate:.4f} ({fired}/10000) in {elapsed:.1f}s")
    assert rate < 0.03
    assert elapsed < 60


def test_panel_agrees_with_abnormal_return_at():
    names = ("AAA", "BBB", "CCC")
    returns = returns_frame(planted={(200, "BBB"): -0.05}, instruments=names, seed=3)
    returns.iloc[150:155, returns.columns.get_loc("CCC")] = np.nan  # a gap in one instrument
    panel = abnormal_returns_panel(returns, "MKT", SESSIONS, EST_WINDOW, MIN_OBS)
    assert list(panel.columns) == ["abnormal_return", "residual_sd"]
    assert panel.index.names == ["session", "instrument"]
    checked = 0
    for position in (5, 40, 130, 156, 200, 260, 319):
        for name in names:
            expected = abnormal_return_at(
                returns[name], returns["MKT"], position, EST_WINDOW, MIN_OBS
            )
            got = panel.loc[(SESSIONS[position], name), "abnormal_return"]
            assert np.isclose(got, expected, atol=1e-9, equal_nan=True), (position, name)
            checked += 1
    assert checked == 21
    assert np.isfinite(panel.loc[(SESSIONS[200], "BBB"), "abnormal_return"])
    assert "MKT" not in panel.index.get_level_values("instrument")


def test_panel_residual_sd_is_the_window_residual_standard_deviation():
    returns = returns_frame(instruments=("AAA",), seed=9)
    panel = abnormal_returns_panel(returns, "MKT", SESSIONS, EST_WINDOW, MIN_OBS)
    position = 200
    window = slice(position + EST_WINDOW[0], position + EST_WINDOW[1] + 1)
    x, y = returns["MKT"].to_numpy()[window], returns["AAA"].to_numpy()[window]
    beta, alpha = np.polyfit(x, y, 1)
    expected = np.std(y - alpha - beta * x, ddof=1)
    assert panel.loc[(SESSIONS[position], "AAA"), "residual_sd"] == pytest.approx(
        expected, abs=1e-9
    )


def test_panel_is_nan_without_enough_history():
    returns = returns_frame(instruments=("AAA",))
    panel = abnormal_returns_panel(returns, "MKT", SESSIONS, EST_WINDOW, MIN_OBS)
    assert np.isnan(panel.loc[(SESSIONS[30], "AAA"), "abnormal_return"])
    assert np.isnan(panel.loc[(SESSIONS[30], "AAA"), "residual_sd"])


def test_panel_planted_move_is_detected_end_to_end():
    returns = returns_frame(planted={(250, "ACME"): -0.04}, seed=21)
    panel = abnormal_returns_panel(returns, "MKT", SESSIONS, EST_WINDOW, MIN_OBS)
    [t] = detect_text_vs_price(frame(POSITIVE), panel, SESSION)
    assert t.details["z"] < -2


# --- asymmetric links ----------------------------------------------------------------


def link(source, receiver, relation, doc, valid_from="2023-03-01T21:30:00Z") -> Link:
    return Link(
        source_node=source,
        receiver_node=receiver,
        relation=relation,
        weight=0.5,
        valid_from_utc=utc(valid_from),
        valid_to_utc=utc(valid_from).replace(year=utc(valid_from).year + 3),
        source_doc_id=doc,
        evidence_span=None,
        extractor_version="rules-test-v1",
    )


def link_claim(source, receiver, relation, doc, valid_from="2023-03-01T21:30:00Z") -> Claim:
    attribute = {"customer_of": "customer_relationship", "supplier_of": "supplier_relationship"}[
        relation
    ]
    return Claim(
        subject=link_id_for(source, receiver, relation, doc),
        attribute=attribute,
        period="unspecified",
        value="ongoing",
        valid_from_utc=utc(valid_from),
        recorded_utc=utc(valid_from),
        source_doc_id=doc,
        evidence_span=None,
        extractor_version="rules-links-v1",
    )


def graph_of(tmp_path: Path, links: list[Link], at: str = "2023-12-01T21:30:00Z") -> pd.DataFrame:
    store = LinkStore(tmp_path / "links")
    store.path.parent.mkdir(parents=True, exist_ok=True)
    store.append(links)
    return graph_asof(store, pd.Timestamp(utc(at)))


def test_asymmetric_link_fires_on_a_one_sided_customer_link(tmp_path):
    graph = graph_of(tmp_path, [link("ACME", "BIGCO", "customer_of", "10K-ACME")])
    claims = frame([link_claim("ACME", "BIGCO", "customer_of", "10K-ACME")])
    [t], skipped = detect_asymmetric_links(graph, claims, SESSION)
    assert skipped == 0 and t.kind == "asymmetric_link"
    assert t.subject == t.group_key == "ACME|customer_of|BIGCO"  # the ordered pair (D72)
    assert t.claim_ids == [claims.claim_id.iloc[0]]
    assert t.details["source_node"] == "ACME" and t.details["receiver_node"] == "BIGCO"
    assert "ASC 280" in str(t.details["note"])


def test_asymmetric_link_is_quiet_when_the_reciprocal_supplier_link_exists(tmp_path):
    graph = graph_of(
        tmp_path,
        [
            link("ACME", "BIGCO", "customer_of", "10K-ACME"),
            link("BIGCO", "ACME", "supplier_of", "10K-BIGCO"),
        ],
    )
    claims = frame([link_claim("ACME", "BIGCO", "customer_of", "10K-ACME")])
    assert detect_asymmetric_links(graph, claims, SESSION) == ([], 0)


def test_asymmetric_link_skips_and_counts_when_the_link_claim_is_absent(tmp_path):
    graph = graph_of(tmp_path, [link("ACME", "BIGCO", "customer_of", "10K-ACME")])
    assert detect_asymmetric_links(graph, frame([]), SESSION) == ([], 1)


def test_asymmetric_link_on_an_empty_graph_is_quiet():
    empty = pd.DataFrame(columns=GRAPH_COLUMNS)
    assert detect_asymmetric_links(empty, frame([]), SESSION) == ([], 0)


# --- stale reliance ------------------------------------------------------------------


OLD = "2021-06-01T21:30:00Z"  # more than 300 sessions before SESSIONS[250]... see below
NEWER = "2023-02-01T21:30:00Z"


def long_axis_sessions() -> pd.DatetimeIndex:
    return pd.bdate_range("2021-01-04", "2024-12-31")


def stale_session() -> date:
    return date(2023, 6, 1)  # 2021-06-01 is ~500 business days before


def test_stale_reliance_fires_on_an_old_link_with_a_newer_filing_from_the_same_source(tmp_path):
    graph = graph_of(
        tmp_path,
        [
            link("ACME", "BIGCO", "supplier_of", "10K-2021", valid_from=OLD),
            link("ACME", "OTHERCO", "supplier_of", "10K-2023", valid_from=NEWER),
        ],
        at="2023-06-01T21:30:00Z",
    )
    [t] = detect_stale_reliance(graph, stale_session(), long_axis_sessions(), 300)
    assert t.kind == "stale_reliance"
    assert t.subject == link_id_for("ACME", "BIGCO", "supplier_of", "10K-2021")
    assert t.claim_ids == [
        claim_id_for(t.subject, "supplier_relationship", "unspecified", "ongoing", "10K-2021")
    ]
    assert t.details["newer_doc_id"] == "10K-2023" and t.details["age_sessions"] > 300


def test_stale_reliance_uses_the_customer_attribute_for_customer_links(tmp_path):
    graph = graph_of(
        tmp_path,
        [
            link("ACME", "BIGCO", "customer_of", "10K-2021", valid_from=OLD),
            link("ACME", "OTHERCO", "customer_of", "10K-2023", valid_from=NEWER),
        ],
        at="2023-06-01T21:30:00Z",
    )
    [t] = detect_stale_reliance(graph, stale_session(), long_axis_sessions(), 300)
    assert t.claim_ids == [
        claim_id_for(t.subject, "customer_relationship", "unspecified", "ongoing", "10K-2021")
    ]


def test_stale_reliance_is_quiet_when_the_newer_filing_restates_the_link(tmp_path):
    graph = graph_of(
        tmp_path,
        [
            link("ACME", "BIGCO", "supplier_of", "10K-2021", valid_from=OLD),
            link("ACME", "BIGCO", "supplier_of", "10K-2023", valid_from=NEWER),
        ],
        at="2023-06-01T21:30:00Z",
    )
    assert len(graph) == 1  # graph_asof shows only the restatement
    assert detect_stale_reliance(graph, stale_session(), long_axis_sessions(), 300) == []


def test_stale_reliance_is_quiet_without_a_newer_filing_or_when_young(tmp_path):
    only_old = graph_of(
        tmp_path / "a",
        [link("ACME", "BIGCO", "supplier_of", "10K-2021", valid_from=OLD)],
        at="2023-06-01T21:30:00Z",
    )
    assert detect_stale_reliance(only_old, stale_session(), long_axis_sessions(), 300) == []
    young = graph_of(
        tmp_path / "b",
        [
            link("ACME", "BIGCO", "supplier_of", "10K-2022", valid_from="2023-02-01T21:30:00Z"),
            link("ACME", "OTHERCO", "supplier_of", "10K-2023", valid_from="2023-05-01T21:30:00Z"),
        ],
        at="2023-06-01T21:30:00Z",
    )
    assert detect_stale_reliance(young, stale_session(), long_axis_sessions(), 300) == []


def test_stale_reliance_ignores_other_relations_and_a_different_source(tmp_path):
    graph = graph_of(
        tmp_path,
        [
            link("ACME", "BIGCO", "peer_of", "10K-2021", valid_from=OLD),
            link("ACME", "OTHERCO", "supplier_of", "10K-2023", valid_from=NEWER),
            link("ZETA", "BIGCO", "supplier_of", "10K-ZETA", valid_from=OLD),
        ],
        at="2023-06-01T21:30:00Z",
    )
    assert detect_stale_reliance(graph, stale_session(), long_axis_sessions(), 300) == []


def test_stale_reliance_rejects_a_session_off_the_axis():
    with pytest.raises(ValueError, match="session"):
        detect_stale_reliance(
            pd.DataFrame(columns=GRAPH_COLUMNS), date(2030, 1, 1), long_axis_sessions()
        )


# --- inputs ----------------------------------------------------------------------------


def test_documents_per_session_counts_and_zero_fills():
    on_session = make_event_record(article_id="a:1", available_utc=utc("2023-01-03T12:00:00Z"))
    same_session = make_event_record(article_id="a:2", available_utc=utc("2023-01-03T20:00:00Z"))
    next_session = make_event_record(article_id="a:3", available_utc=utc("2023-01-03T21:30:00Z"))
    undated = make_event_record(article_id="a:4", available_utc=None)
    past_end = make_event_record(article_id="a:5", available_utc=utc("2030-01-01T00:00:00Z"))
    counts = documents_per_session(
        [on_session, same_session, next_session, undated, past_end], CUTOFFS
    )
    assert len(counts) == len(CUTOFFS) and counts.index.equals(CUTOFFS.index)
    # the 2023-01-03 cutoff is 21:00Z (16:00 New York, winter): the first two land on it,
    # the 21:30Z record lands on 2023-01-04
    assert counts[pd.Timestamp("2023-01-03")] == 2
    assert counts[pd.Timestamp("2023-01-04")] == 1
    assert counts.sum() == 3 and counts.dtype.kind == "i"


def test_load_detector_params_reads_the_proposed_file_and_rejects_unknown_keys(tmp_path):
    params = load_detector_params()
    assert params["text_vs_price"]["z_threshold"] == 2.0
    assert params["stale_reliance"]["max_age_sessions"] == 300
    assert params["abnormal_returns"] == {"est_window": (-250, -21), "min_obs": 120}
    raw = json.loads(
        (Path(__file__).parents[2] / "references/dream/detector_params.json").read_text()
    )
    raw["text_vs_price"]["surprise"] = 1
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="surprise"):
        load_detector_params(bad)
    with pytest.raises(ValueError, match="cannot read"):
        load_detector_params(tmp_path / "missing.json")


# --- run_detectors -----------------------------------------------------------------------


def make_stores(
    tmp_path: Path,
    *,
    records=(),
    claims=(),
    links=(),
    returns: pd.DataFrame | None = None,
    with_link_store: bool = True,
) -> DreamStores:
    root = tmp_path / "dream"
    root.mkdir(exist_ok=True)
    claim_store = ClaimStore(root)
    claim_store.append(claims)
    link_store = LinkStore(root) if with_link_store else None
    if link_store is not None:
        link_store.append(links)
    return DreamStores(
        claim_store=claim_store,
        tension_store=TensionStore(root),
        shift_store=ShiftStore(root),
        resolution_store=ResolutionStore(root),
        event_records=list(records),
        link_store=link_store,
        returns=returns,
        market="MKT" if returns is not None else None,
        sessions=SESSIONS,
        session_cutoffs_utc=CUTOFFS,
    )


def record_on(session: date, article_id: str = "g:1"):
    cutoff = CUTOFFS[pd.Timestamp(session)]
    return make_event_record(
        article_id=article_id, available_utc=(cutoff - pd.Timedelta(hours=1)).to_pydatetime()
    )


def claim_valid_before(session: date, **kwargs) -> Claim:
    cutoff = CUTOFFS[pd.Timestamp(session)]
    return claim(valid_from=(cutoff - pd.Timedelta(days=2)).isoformat(), **kwargs)


def test_feed_gap_skips_price_detector(tmp_path):
    returns = returns_frame(planted={(250, "ACME"): -0.04}, seed=21)
    stores = make_stores(
        tmp_path,
        records=[record_on(SESSIONS[249].date())],
        claims=[
            claim_valid_before(
                SESSION,
                subject="ACME",
                attribute="event_direction",
                value="positive",
                period="unspecified",
                doc="g:9",
            )
        ],
        returns=returns,
    )
    tensions, report = run_detectors(stores, SESSION, CUTOFFS, PARAMS)
    assert report.feed_gap is True and report.documents_in_session == 0
    assert report.by_kind.get("text_vs_price", 0) == 0 and tensions == []
    assert report.skipped_no_prices == 0


def test_run_detectors_runs_every_detector_and_stamps_the_tensions(tmp_path):
    returns = returns_frame(planted={(250, "ACME"): -0.04}, seed=21)
    cutoff = CUTOFFS[pd.Timestamp(SESSION)]
    claims = [
        claim_valid_before(
            SESSION,
            subject="ACME",
            attribute="event_direction",
            value="positive",
            period="unspecified",
            doc="g:9",
        ),
        claim_valid_before(SESSION, value="up", doc="10K-2023"),
        claim_valid_before(SESSION, value="down", doc="8K-2024"),
    ]
    stores = make_stores(tmp_path, records=[record_on(SESSION)], claims=claims, returns=returns)
    tensions, report = run_detectors(stores, SESSION, CUTOFFS, PARAMS)
    assert isinstance(report, DetectReport) and not report.feed_gap
    assert report.documents_in_session == 1 and report.claims_seen == 3
    assert report.by_kind["text_vs_price"] == 1 and report.by_kind["value_conflict"] == 1
    assert len(tensions) == 2
    for t in tensions:
        assert t.opened_session == SESSION and t.recorded_utc == cutoff
        assert t.detector_version == DETECTOR_VERSION == "dream-detect-v1"
    assert stores.tension_store.read_all() == []  # the detectors never append


def test_run_detectors_with_no_returns_counts_skipped_no_prices(tmp_path):
    stores = make_stores(tmp_path, records=[record_on(SESSION)], with_link_store=False)
    tensions, report = run_detectors(stores, SESSION, CUTOFFS, PARAMS)
    assert tensions == [] and report.skipped_no_prices == 1 and not report.feed_gap
    assert report.by_kind.get("text_vs_price", 0) == 0


def test_run_detectors_reads_links_through_the_link_store(tmp_path):
    session = SESSIONS[250].date()
    cutoff = CUTOFFS[pd.Timestamp(session)]
    start = (cutoff - pd.Timedelta(days=30)).isoformat()
    one_sided = link("ACME", "BIGCO", "customer_of", "10K-ACME", valid_from=start)
    unclaimed = link("ZETA", "BIGCO", "customer_of", "10K-ZETA", valid_from=start)
    stores = make_stores(
        tmp_path,
        records=[record_on(session)],
        claims=[link_claim("ACME", "BIGCO", "customer_of", "10K-ACME", valid_from=start)],
        links=[one_sided, unclaimed],
    )
    tensions, report = run_detectors(stores, session, CUTOFFS, PARAMS)
    assert [t.kind.value for t in tensions] == ["asymmetric_link"]
    assert report.by_kind["asymmetric_link"] == 1 and report.skipped_without_claim == 1
    assert claims_asof(stores.claim_store, cutoff).shape[0] == 1


# --- carried from the Task 3 review (rulings D43 and D44) -----------------------------------


def test_run_detectors_uses_a_supplied_panel_and_never_rebuilds_one(tmp_path, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the panel must not be rebuilt when one is supplied")

    monkeypatch.setattr(detectors_module, "abnormal_returns_panel", refuse)
    planted = claim_valid_before(
        SESSION,
        subject="ACME",
        attribute="event_direction",
        value="positive",
        period="unspecified",
        doc="g:9",
    )
    stores = make_stores(tmp_path, records=[record_on(SESSION)], claims=[planted])
    panel = abnormal_frame({(SESSION, "ACME"): (-0.04, 0.01)})
    tensions, report = run_detectors(stores, SESSION, CUTOFFS, PARAMS, abnormal_returns=panel)
    assert [t.kind.value for t in tensions] == ["text_vs_price"]
    assert report.skipped_no_prices == 0  # stores.returns is None, the supplied panel is enough


def test_panel_raises_when_the_returns_share_no_session_with_the_axis():
    elsewhere = returns_frame()
    elsewhere.index = pd.bdate_range("2030-01-01", periods=len(elsewhere))
    with pytest.raises(ValueError, match="no session in common"):
        abnormal_returns_panel(elsewhere, "MKT", SESSIONS, EST_WINDOW, MIN_OBS)


def test_a_claim_on_a_conflicting_side_joins_the_tension_even_if_its_document_also_agrees():
    claims = frame(
        [claim(value="up", doc="A"), claim(value="down", doc="A"), claim(value="down", doc="B")]
    )
    [tension] = detect_value_conflicts(claims, SESSION)
    assert set(tension.claim_ids) == set(claims.claim_id)  # down(A) is on the conflicting side


def test_a_claim_outside_every_conflicting_pair_stays_out_of_the_tension():
    claims = frame(
        [
            claim(value="up", doc="A"),
            claim(value="down", doc="B"),
            claim(value="flat", doc="C"),
        ]
    )
    [tension] = detect_value_conflicts(claims, SESSION)
    assert len(tension.claim_ids) == 2 and set(tension.claim_ids) < set(claims.claim_id)


def test_a_pair_present_in_one_document_only_is_not_a_conflict_even_with_a_third_value():
    claims = frame(
        [claim(value="up", doc="A"), claim(value="down", doc="A"), claim(value="flat", doc="B")]
    )
    assert detect_value_conflicts(claims, SESSION) == []


def test_standalone_recorded_utc_is_the_sessions_cutoff_from_the_calendar():
    [tension] = detect_value_conflicts(CONFLICT_FRAME, SESSION)
    assert tension.recorded_utc == CUTOFFS[pd.Timestamp(SESSION)].to_pydatetime()
    assert tension.recorded_utc == utc("2023-12-18T21:00:00Z")  # 16:00 New York in winter
    given = utc("2023-12-18T22:00:00Z")
    [stamped] = detect_value_conflicts(CONFLICT_FRAME, SESSION, recorded_utc=given)
    assert stamped.recorded_utc == given


@pytest.mark.parametrize(
    "not_a_session", [date(2023, 12, 24), date(2023, 12, 25), date(2031, 1, 6)]
)
def test_standalone_default_refuses_a_date_that_is_not_a_calendar_session(not_a_session):
    with pytest.raises(ValueError, match=f"{not_a_session}.*recorded_utc"):
        detect_value_conflicts(CONFLICT_FRAME, not_a_session)


def test_run_detectors_stamps_the_cutoff_even_on_a_date_the_calendar_does_not_know(tmp_path):
    holiday = SESSIONS[10].date()  # 2023-01-16, a market holiday that this synthetic axis keeps
    stores = make_stores(
        tmp_path,
        records=[record_on(holiday)],
        claims=[
            claim_valid_before(holiday, value="up", doc="A"),
            claim_valid_before(holiday, doc="B", value="down"),
        ],
        with_link_store=False,
    )
    [tension], _ = run_detectors(stores, holiday, CUTOFFS, PARAMS)
    assert tension.recorded_utc == CUTOFFS[pd.Timestamp(holiday)].to_pydatetime()


def test_run_detectors_refuses_cutoffs_that_do_not_match_the_sessions_axis(tmp_path):
    stores = make_stores(tmp_path, records=[record_on(SESSION)], with_link_store=False)
    with pytest.raises(ValueError, match="cutoffs"):
        run_detectors(stores, SESSION, CUTOFFS.iloc[:-1], PARAMS)
