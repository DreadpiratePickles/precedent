from datetime import date

import pandas as pd
import pytest
from library_synthetic import ROW_ASSET, ROW_SEQ, rows_frame

from precedent.libraries.common import (
    LIBRARY_COLUMNS,
    OUTCOME_TYPES,
    LibrarySchemaError,
    empty_library_rows,
    validate_library_rows,
)


def test_valid_rows_pass_and_are_reordered():
    out = validate_library_rows(rows_frame([ROW_SEQ, ROW_ASSET])[list(reversed(LIBRARY_COLUMNS))])
    assert list(out.columns) == list(LIBRARY_COLUMNS)


@pytest.mark.parametrize(
    "bad",
    [
        {"outcome_type": "weather"},
        {"outcome_type": "asset", "is_binary": True},
        {"value": 0.5},
        {"value": float("nan")},
        {"response_complete_session": date(2020, 1, 1)},  # before effective_session 2020-01-10
    ],
)
def test_invalid_rows_raise(bad):
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(rows_frame([{**ROW_SEQ, **bad}]))


def test_extra_column_raises():
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(rows_frame([ROW_SEQ]).assign(extra=1))


def test_missing_column_raises():
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(rows_frame([ROW_SEQ]).drop(columns=["source_ref"]))


def test_a_binary_type_needs_is_binary_true():
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(rows_frame([{**ROW_SEQ, "is_binary": False}]))


def test_an_asset_value_may_be_any_finite_number_but_not_infinite():
    out = validate_library_rows(rows_frame([{**ROW_ASSET, "value": -0.4}]))
    assert out["value"].tolist() == [-0.4]
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(rows_frame([{**ROW_ASSET, "value": float("inf")}]))


def test_outcome_types_are_the_card_display_order():
    assert OUTCOME_TYPES == ("event_sequence", "market_resolution", "asset")


def test_dates_are_coerced_to_naive_midnight_datetimes_and_other_dtypes_fixed():
    frame = rows_frame([ROW_SEQ, ROW_ASSET])
    frame["horizon"] = frame["horizon"].astype(object)
    out = validate_library_rows(frame)
    assert str(out["effective_session"].dtype) == "datetime64[ns]"
    assert str(out["response_complete_session"].dtype) == "datetime64[ns]"
    assert out["effective_session"].tolist() == [pd.Timestamp("2020-01-10")] * 2
    assert str(out["horizon"].dtype) == "int64"
    assert str(out["value"].dtype) == "float64"
    assert str(out["is_binary"].dtype) == "bool"


def test_timestamps_are_accepted_and_row_order_is_kept():
    first = {**ROW_SEQ, "event_id": "b", "effective_session": pd.Timestamp("2020-01-10")}
    second = {**ROW_ASSET, "event_id": "a"}
    out = validate_library_rows(rows_frame([first, second]))
    assert out["event_id"].tolist() == ["b", "a"]


def test_a_tz_aware_or_non_midnight_session_raises():
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(
            rows_frame([{**ROW_SEQ, "effective_session": pd.Timestamp("2020-01-10 09:30")}])
        )
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(
            rows_frame([{**ROW_SEQ, "effective_session": pd.Timestamp("2020-01-10", tz="UTC")}])
        )


def test_a_non_integral_horizon_raises():
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(rows_frame([{**ROW_SEQ, "horizon": 2.5}]))


def test_a_missing_identifier_raises():
    with pytest.raises(LibrarySchemaError):
        validate_library_rows(rows_frame([{**ROW_SEQ, "source_ref": None}]))


def test_the_input_frame_is_not_modified():
    frame = rows_frame([ROW_SEQ])
    before = frame.copy()
    validate_library_rows(frame)
    pd.testing.assert_frame_equal(frame, before)


def test_empty_library_rows_has_the_columns_and_survives_validation():
    empty = empty_library_rows()
    assert list(empty.columns) == list(LIBRARY_COLUMNS) and len(empty) == 0
    pd.testing.assert_frame_equal(validate_library_rows(empty), empty)
    assert str(empty["effective_session"].dtype) == "datetime64[ns]"
    assert str(empty["horizon"].dtype) == "int64"
