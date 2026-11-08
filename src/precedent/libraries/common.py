"""The row schema shared by the three precedent libraries, and its validator."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "LIBRARY_COLUMNS",
    "OUTCOME_TYPES",
    "LibrarySchemaError",
    "empty_library_rows",
    "validate_library_rows",
]

OUTCOME_TYPES = ("event_sequence", "market_resolution", "asset")
"""The outcome types, in the order a precedent card displays them."""

LIBRARY_COLUMNS = (
    "event_id",
    "outcome_type",
    "outcome_key",
    "horizon",
    "value",
    "is_binary",
    "effective_session",
    "response_complete_session",
    "source_ref",
)

_STRING_COLUMNS = ("event_id", "outcome_type", "outcome_key", "source_ref")
_SESSION_COLUMNS = ("effective_session", "response_complete_session")
_EMPTY_DTYPES = {
    "event_id": "str",
    "outcome_type": "str",
    "outcome_key": "str",
    "horizon": "int64",
    "value": "float64",
    "is_binary": "bool",
    "effective_session": "datetime64[ns]",
    "response_complete_session": "datetime64[ns]",
    "source_ref": "str",
}


class LibrarySchemaError(ValueError):
    """A library rows frame does not match the shared row schema."""


def empty_library_rows() -> pd.DataFrame:
    """A zero-row frame with the library columns and their dtypes."""
    return pd.DataFrame(
        {column: pd.Series(dtype=_EMPTY_DTYPES[column]) for column in LIBRARY_COLUMNS}
    )


def validate_library_rows(rows: pd.DataFrame) -> pd.DataFrame:
    """Check `rows` against the schema; return a copy in `LIBRARY_COLUMNS` order, dtypes fixed.

    Sessions become naive midnight `datetime64[ns]` (dates and Timestamps are accepted),
    `horizon` int64, `value` float64, `is_binary` bool, identifiers str. Row order is kept.
    Raises `LibrarySchemaError` on missing or extra columns, a null identifier, an unknown
    `outcome_type`, an `is_binary` that does not match the type (False for `asset`, True
    otherwise), a non-finite or non-integral number, a binary value outside {0.0, 1.0}, a
    session that is not a naive midnight, or `response_complete_session < effective_session`.
    """
    _check_columns(rows)
    out = rows[list(LIBRARY_COLUMNS)].copy()
    for column in _STRING_COLUMNS:
        out[column] = _as_strings(out[column], column)
    for column in _SESSION_COLUMNS:
        out[column] = _as_sessions(out[column], column)
    out["horizon"] = _as_horizons(out["horizon"])
    out["value"] = _as_finite_floats(out["value"])
    out["is_binary"] = _as_bools(out["is_binary"])
    _check_outcome_types(out)
    if (out["response_complete_session"] < out["effective_session"]).any():
        raise LibrarySchemaError("response_complete_session is before effective_session")
    return out


def _check_columns(rows: pd.DataFrame) -> None:
    if not rows.columns.is_unique:
        raise LibrarySchemaError("library rows have duplicate column names")
    missing = [column for column in LIBRARY_COLUMNS if column not in rows.columns]
    extra = [column for column in rows.columns if column not in LIBRARY_COLUMNS]
    if missing or extra:
        raise LibrarySchemaError(
            f"library rows columns differ from the schema: missing {missing}, extra {extra}"
        )


def _as_strings(values: pd.Series, column: str) -> pd.Series:
    if values.isna().any():
        raise LibrarySchemaError(f"{column} has null values")
    return values.astype(str)


def _as_sessions(values: pd.Series, column: str) -> pd.Series:
    try:
        converted = pd.to_datetime(values)
    except (ValueError, TypeError) as error:
        raise LibrarySchemaError(f"{column} is not a date: {error}") from error
    if converted.isna().any():
        raise LibrarySchemaError(f"{column} has null values")
    if converted.dt.tz is not None:
        raise LibrarySchemaError(f"{column} must be timezone-naive")
    if (converted != converted.dt.normalize()).any():
        raise LibrarySchemaError(f"{column} must be at midnight")
    return converted.astype("datetime64[ns]")


def _as_numbers(values: pd.Series, column: str) -> pd.Series:
    try:
        numbers = pd.to_numeric(values)
    except (ValueError, TypeError) as error:
        raise LibrarySchemaError(f"{column} is not numeric: {error}") from error
    return numbers.astype("float64")


def _as_horizons(values: pd.Series) -> pd.Series:
    numbers = _as_numbers(values, "horizon")
    if not np.isfinite(numbers).all() or (numbers != numbers.round()).any():
        raise LibrarySchemaError("horizon must be whole numbers")
    return numbers.astype("int64")


def _as_finite_floats(values: pd.Series) -> pd.Series:
    numbers = _as_numbers(values, "value")
    if not np.isfinite(numbers).all():
        raise LibrarySchemaError("value must be finite (no NaN or infinity)")
    return numbers


def _as_bools(values: pd.Series) -> pd.Series:
    if not all(isinstance(item, bool | np.bool_) for item in values):
        raise LibrarySchemaError("is_binary must hold booleans")
    return values.astype(bool)


def _check_outcome_types(rows: pd.DataFrame) -> None:
    unknown = sorted(set(rows["outcome_type"]) - set(OUTCOME_TYPES))
    if unknown:
        raise LibrarySchemaError(f"unknown outcome_type {unknown}; expected one of {OUTCOME_TYPES}")
    expected_binary = rows["outcome_type"] != "asset"
    if (rows["is_binary"] != expected_binary).any():
        raise LibrarySchemaError("is_binary must be False for asset rows and True for the others")
    binary_values = rows.loc[rows["is_binary"], "value"]
    if (~binary_values.isin([0.0, 1.0])).any():
        raise LibrarySchemaError("a binary value must be exactly 0.0 or 1.0")
