"""Tests for the dated index universe and the CIK-to-symbol resolution."""
from pathlib import Path

import pandas as pd
import pytest

from precedent.spine.universe import load_membership, members_asof, resolve_instrument

FIXTURE = Path(__file__).resolve().parent.parent / "fixtures" / "membership_sample.csv"


def test_membership_uses_only_changes_on_or_before_the_date():
    membership = load_membership(FIXTURE)
    assert members_asof(membership, pd.Timestamp("2015-05-29")) == frozenset(
        {"AAA", "CCC", "DDD-201001"}
    )
    assert members_asof(membership, pd.Timestamp("2015-06-01")) == frozenset({"BBB", "CCC"})


def test_membership_between_rows_carries_the_latest_earlier_row():
    membership = load_membership(FIXTURE)
    assert members_asof(membership, pd.Timestamp("2016-02-29")) == frozenset({"BBB", "CCC"})
    assert members_asof(membership, pd.Timestamp("2030-01-01")) == frozenset({"BBB", "CCC", "EEE"})


def test_date_before_the_first_row_is_an_error_not_an_empty_universe():
    membership = load_membership(FIXTURE)
    with pytest.raises(ValueError, match="before the first membership row"):
        members_asof(membership, pd.Timestamp("2014-12-31"))


def test_load_membership_rejects_a_file_without_the_expected_columns(tmp_path):
    bad = tmp_path / "bad.csv"
    bad.write_text("day,symbols\n2015-01-02,AAA\n")
    with pytest.raises(ValueError, match="date, tickers"):
        load_membership(bad)


def test_load_membership_rejects_duplicate_dates(tmp_path):
    bad = tmp_path / "dup.csv"
    bad.write_text('date,tickers\n2015-01-02,"AAA"\n2015-01-02,"BBB"\n')
    with pytest.raises(ValueError, match="duplicate"):
        load_membership(bad)


def test_unmapped_cik_is_unresolved_not_guessed():
    assert resolve_instrument(999, {320193: "AAPL"}, {"AAPL"}) is None
    assert resolve_instrument(320193, {320193: "AAPL"}, set()) is None


def test_mapped_cik_with_a_provider_symbol_resolves():
    assert resolve_instrument(320193, {320193: "AAPL"}, {"AAPL", "MSFT"}) == "AAPL"
