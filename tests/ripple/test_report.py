"""Research: the daily Ripple map, rendered from the dated link store and the ledger."""

import pytest

pytest.importorskip("scipy")

import datetime as dt  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from precedent import cli  # noqa: E402
from precedent.config import data_dir  # noqa: E402
from precedent.ripple.graph_builders import write_ripple_ledger  # noqa: E402
from precedent.ripple.link_store import Link, LinkStore, Relation  # noqa: E402
from precedent.ripple.report import DataError, _link_rows, run  # noqa: E402

MAP_DATE = "2024-06-03"
QUIET_DATE = "2024-06-04"
BANNER_PREFIXES = ("SYNTHETIC —", "LIVE —", "NO LIVE RUN —")


def _utc(text):
    return dt.datetime.fromisoformat(text).replace(tzinfo=dt.UTC)


def _link(source, receiver, relation, weight, doc, span, version, valid_from="2024-01-02T00:00:00"):
    return Link(
        source_node=source,
        receiver_node=receiver,
        relation=relation,
        weight=weight,
        valid_from_utc=_utc(valid_from),
        valid_to_utc=_utc("2025-01-01T00:00:00"),
        source_doc_id=doc,
        evidence_span=span,
        extractor_version=version,
    )


@pytest.fixture
def populated_ripple_stores(tmp_path, monkeypatch):
    """A link store plus a ledger under PRECEDENT_DATA_DIR/ripple, labelled SYNTHETIC."""
    monkeypatch.setenv("PRECEDENT_DATA_DIR", str(tmp_path / "data"))
    ripple_dir = data_dir() / "ripple"
    store = LinkStore(ripple_dir / "links")
    store.path.parent.mkdir(parents=True)
    store.append(
        [
            _link("AAA", "CCC", Relation.CUSTOMER_OF, 0.6, "sec-8k-0001",
                  "CCC buys | from AAA", "llm-v1"),
            _link("HHH", "CCC", Relation.CUSTOMER_OF, 0.2, "sec-8k-0002",
                  "HHH sells to CCC", "llm-v1"),
            _link("AAA", "DDD", Relation.SUPPLIER_OF, 0.5, "sec-8k-0003",
                  "DDD supplies AAA", "llm-v1"),
            _link("AAA", "EEE", Relation.PEER_OF, 0.4, "sector-map-1", None, "rules-sector-v1"),
            _link("BBB", "EEE", Relation.PEER_OF, 0.4, "sector-map-1", None, "rules-sector-v1"),
            _link("AAA", "III", Relation.INPUT_COMMODITY, 0.3, "sec-10k-0021",
                  "III burns the metal AAA mines", "llm-v1"),
            _link("AAA", "III", Relation.OUTPUT_COMMODITY, 0.1, "sec-10k-0022",
                  "III sells the grade AAA mines", "llm-v1"),
            _link("BBB", "III", Relation.INPUT_COMMODITY, 0.2, "sec-10k-0023",
                  "III also burns BBB's output", "llm-v1"),
            _link("AAA", "FFF", Relation.COUNTRY_EXPOSURE, 0.3, "sec-10k-0009",
                  "FFF earns most revenue where AAA operates", "llm-v1"),
            # known only after the 2024-06-03 cutoff (20:00 UTC): must not appear
            _link("AAA", "GGG", Relation.CUSTOMER_OF, 0.9, "sec-8k-0777",
                  "GGG signs a contract with AAA", "llm-v1", valid_from="2024-06-03T21:00:00"),
        ]
    )
    shocks = pd.Series(
        [0.08, -0.05, 0.01, 0.30],
        index=pd.MultiIndex.from_arrays(
            [
                pd.DatetimeIndex(["2024-06-03", "2024-06-03", "2024-06-03", "2024-05-31"]).as_unit("ns"),
                ["AAA", "BBB", "ZZZ", "YYY"],
            ],
            names=["datetime", "instrument"],
        ),
        name="shock",
    )
    sessions = pd.DatetimeIndex(["2024-06-03", "2024-06-04"]).as_unit("ns")
    abnormal = pd.DataFrame(
        {"CCC": [0.0123, 0.0], "DDD": [np.nan, 0.0], "EEE": [-0.0045, 0.0]}, index=sessions
    )
    write_ripple_ledger(ripple_dir, shocks, abnormal, {"evidence_label": "SYNTHETIC"})
    return ripple_dir


def _rewrite_label(ripple_dir, label):
    path = ripple_dir / "MANIFEST.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["evidence_label"] = label
    path.write_text(json.dumps(manifest), encoding="utf-8")


def _render(tmp_path, *extra, date=MAP_DATE):
    out = tmp_path / "map.md"
    assert run(["--date", date, "--out", str(out), *extra]) == 0
    return out.read_text(encoding="utf-8")


def test_every_link_line_shows_its_evidence(tmp_path, populated_ripple_stores):
    out = tmp_path / "map.md"
    assert run(["--date", "2024-06-03", "--out", str(out)]) == 0
    link_lines = [line for line in out.read_text().splitlines() if line.startswith("| ->")]
    assert link_lines and all("doc:" in line for line in link_lines)
    assert out.read_text().startswith(("SYNTHETIC —", "LIVE —", "NO LIVE RUN —"))  # evidence banner on line 1


def test_a_session_without_shocks_says_so(tmp_path, populated_ripple_stores):
    out = tmp_path / "map.md"
    assert run(["--date", "2024-06-04", "--out", str(out)]) == 0
    assert "No shocks on this session." in out.read_text()


def test_the_banner_is_line_one_and_the_largest_shock_comes_first(tmp_path, populated_ripple_stores):
    text = _render(tmp_path)
    lines = text.splitlines()
    assert lines[0].startswith("SYNTHETIC —")
    headings = [line for line in lines if line.startswith("## ")]
    assert headings == [
        "## AAA shock +0.0800",
        "## BBB shock -0.0500",
        "## ZZZ shock +0.0100",
    ]
    assert "YYY" not in text  # another session's shock


def test_link_rows_carry_size_move_and_evidence(tmp_path, populated_ripple_stores):
    text = _render(tmp_path)
    # hop-1 term alpha * A[receiver, source] * shock, A[CCC, AAA] = 0.6 / (0.6 + 0.2) = 0.75
    assert (
        '| -> CCC | customer_of | +0.0300 | +0.0123 | doc: sec-8k-0001 | "CCC buys \\| from AAA" |'
        in text
    )
    # a single supplier link: weight share 1, so alpha * 1 * 0.08; DDD's abnormal return is NaN
    assert '| -> DDD | supplier_of | +0.0400 | n/a | doc: sec-8k-0003 | "DDD supplies AAA" |' in text
    # rule-built link: the rule name stands in for a quote; EEE peers AAA and BBB
    assert "| -> EEE | peer_of | +0.0200 | -0.0045 | doc: sector-map-1 | rule: rules-sector-v1 |" in text
    assert "| -> EEE | peer_of | -0.0125 | -0.0045 | doc: sector-map-1 | rule: rules-sector-v1 |" in text
    assert "GGG" not in text and "sec-8k-0777" not in text  # not yet known at the cutoff
    assert "HHH" not in text  # HHH is not shocked, so its links are not on the map


def test_two_relations_of_one_channel_split_the_pair_term_by_weight(
    tmp_path, populated_ripple_stores
):
    text = _render(tmp_path)
    # commodity channel: A[III, AAA] = (0.3 + 0.1) / (0.3 + 0.1 + 0.2) = 2/3, pair term
    # 0.5 * 2/3 * 0.08 = 0.026667, split 0.3 : 0.1 -> 0.02 and 0.006667; BBB: 0.5 * 1/3 * -0.05
    assert "| -> III | input_commodity | +0.0200 | n/a | doc: sec-10k-0021 |" in text
    assert "| -> III | output_commodity | +0.0067 | n/a | doc: sec-10k-0022 |" in text
    assert "| -> III | input_commodity | -0.0083 | n/a | doc: sec-10k-0023 |" in text


def test_channels_are_ordered_and_country_exposure_has_no_size(tmp_path, populated_ripple_stores):
    text = _render(tmp_path)
    aaa = text.split("## AAA shock")[1].split("## BBB shock")[0]
    order = [
        aaa.index("customer"),
        aaa.index("supplier"),
        aaa.index("peer"),
        aaa.index("ripple_commodity"),
        aaa.index("country_exposure"),
    ]
    assert order == sorted(order)
    assert (
        '| -> FFF | country_exposure | n/a (no factor channel) | n/a | doc: sec-10k-0009 '
        '| "FFF earns most revenue where AAA operates" |'
    ) in text
    assert "pooled hop-2" in text


def test_top_limits_the_map_to_the_largest_absolute_shocks(tmp_path, populated_ripple_stores):
    text = _render(tmp_path, "--top", "1")
    assert "## AAA shock" in text and "## BBB shock" not in text


def test_output_is_byte_identical_across_runs(tmp_path, populated_ripple_stores):
    first = _render(tmp_path)
    second = _render(tmp_path)
    assert first == second


def test_no_price_columns_leak_into_the_tables(tmp_path, populated_ripple_stores):
    text = _render(tmp_path)
    cells = " ".join(line for line in text.splitlines() if line.startswith("|")).lower()
    assert not re.search(r"\b(open|close|price)\b", cells)


@pytest.mark.parametrize(
    ("label", "prefix"),
    [
        ("SYNTHETIC", "SYNTHETIC —"),
        ("LIVE-DATA", "LIVE —"),
        ("LIVE-MODEL", "LIVE —"),
        ("NO LIVE RUN", "NO LIVE RUN —"),
    ],
)
def test_the_banner_follows_the_manifest_label(tmp_path, populated_ripple_stores, label, prefix):
    _rewrite_label(populated_ripple_stores, label)
    first_line = _render(tmp_path).splitlines()[0]
    assert first_line.startswith(prefix)
    assert not re.search(r"\d{2}:\d{2}", first_line)  # no run time on the banner line


def test_an_unknown_label_stops_with_exit_1(tmp_path, populated_ripple_stores, capsys):
    _rewrite_label(populated_ripple_stores, "BOGUS")
    assert run(["--date", MAP_DATE, "--out", str(tmp_path / "map.md")]) == 1
    assert "BOGUS" in capsys.readouterr().err
    assert not (tmp_path / "map.md").exists()


def test_without_a_manifest_the_map_says_no_live_run(tmp_path, populated_ripple_stores):
    (populated_ripple_stores / "MANIFEST.json").unlink()
    text = _render(tmp_path)
    first_line = text.splitlines()[0]
    assert first_line.startswith("NO LIVE RUN —") and "/" not in first_line  # no absolute path
    assert "No Ripple data has been built" in text and str(populated_ripple_stores) in text
    assert "| ->" not in text


def test_a_missing_link_store_is_a_data_error_not_an_empty_graph(
    tmp_path, populated_ripple_stores, capsys
):
    (populated_ripple_stores / "links" / "links.jsonl").unlink()
    out = tmp_path / "map.md"
    assert run(["--date", MAP_DATE, "--out", str(out)]) == 1
    assert "links.jsonl" in capsys.readouterr().err
    assert not out.exists()


def test_an_unknown_relation_is_a_data_error():
    graph = pd.DataFrame(
        [
            {
                "source_node": "AAA", "receiver_node": "CCC", "relation": "mystery_of",
                "weight": 0.5, "source_doc_id": "d", "evidence_span": "x",
                "extractor_version": "llm-v1",
            }
        ]
    )
    with pytest.raises(DataError, match="mystery_of"):
        _link_rows(graph, "AAA", {}, pd.Series(dtype="float64"))


def test_an_unwritable_out_path_exits_1(tmp_path, populated_ripple_stores, capsys):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory", encoding="utf-8")
    assert run(["--date", MAP_DATE, "--out", str(blocker / "map.md")]) == 1
    assert "cannot write" in capsys.readouterr().err


def test_a_tampered_parquet_file_is_a_data_error(tmp_path, populated_ripple_stores, capsys):
    path = populated_ripple_stores / "shocks.parquet"
    path.write_bytes(path.read_bytes() + b"x")
    assert run(["--date", MAP_DATE, "--out", str(tmp_path / "map.md")]) == 1
    assert "shocks.parquet" in capsys.readouterr().err


def test_a_listed_parquet_file_that_is_missing_is_a_data_error(
    tmp_path, populated_ripple_stores, capsys
):
    (populated_ripple_stores / "abnormal_returns.parquet").unlink()
    assert run(["--date", MAP_DATE, "--out", str(tmp_path / "map.md")]) == 1
    assert "abnormal_returns.parquet" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        ["--date", "2024-06-08"],  # a Saturday
        ["--date", "2024-07-04"],  # an NYSE holiday
        ["--date", "2024-6-3"],  # not YYYY-MM-DD
        ["--date", MAP_DATE, "--top", "0"],
        [],
    ],
)
def test_usage_errors_exit_2(tmp_path, populated_ripple_stores, argv, capsys):
    assert run(argv) == 2
    assert capsys.readouterr().err


def test_a_non_session_date_is_named_in_the_message(populated_ripple_stores, capsys):
    assert run(["--date", "2024-06-08"]) == 2
    assert "2024-06-08" in capsys.readouterr().err


def test_without_out_the_map_goes_to_stdout(populated_ripple_stores, capsys):
    assert run(["--date", MAP_DATE]) == 0
    out = capsys.readouterr().out
    assert out.startswith("SYNTHETIC —") and "| -> CCC" in out


def test_the_console_script_reaches_the_report(tmp_path, populated_ripple_stores):
    out = tmp_path / "map.md"
    assert cli.main(["ripple", "--date", MAP_DATE, "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8").startswith("SYNTHETIC —")
