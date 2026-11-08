"""Synthetic GDELT 2.0 export rows: 61 tab-separated fields, values invented for tests."""

import io
import zipfile

import pandas as pd

from precedent.spine.gdelt_events import read_export_csv

# 1-based positions of every named field in the 61-column export (codebook order, ADM2 included).
_POSITIONS = {
    "GLOBALEVENTID": 1,
    "SQLDATE": 2,
    "Actor1Code": 6,
    "Actor1Name": 7,
    "Actor1CountryCode": 8,
    "Actor1Type1Code": 13,
    "Actor2Code": 16,
    "Actor2Name": 17,
    "Actor2CountryCode": 18,
    "Actor2Type1Code": 23,
    "IsRootEvent": 26,
    "EventCode": 27,
    "EventBaseCode": 28,
    "EventRootCode": 29,
    "QuadClass": 30,
    "GoldsteinScale": 31,
    "NumMentions": 32,
    "NumSources": 33,
    "NumArticles": 34,
    "AvgTone": 35,
    "ActionGeo_CountryCode": 54,
    "DATEADDED": 60,
    "SOURCEURL": 61,
}
_DEFAULTS = {
    "GLOBALEVENTID": "1001",
    "SQLDATE": "20240101",
    "Actor1Code": "USAGOV",
    "Actor1Name": "UNITED STATES",
    "Actor1CountryCode": "USA",
    "Actor1Type1Code": "GOV",
    "Actor2Code": "RUS",
    "Actor2Name": "RUSSIA",
    "Actor2CountryCode": "RUS",
    "IsRootEvent": "1",
    "EventCode": "163",
    "EventBaseCode": "163",
    "EventRootCode": "16",
    "QuadClass": "4",
    "GoldsteinScale": "-8.0",
    "NumMentions": "7",
    "NumSources": "5",
    "NumArticles": "6",
    "AvgTone": "-3.5",
    "ActionGeo_CountryCode": "RS",
    "DATEADDED": "20240103141500",
    "SOURCEURL": "https://example.test/story",
}


def gdelt_row(**overrides: str) -> list[str]:
    """One export row as 61 strings; keyword overrides replace the named fields."""
    unknown = set(overrides) - set(_POSITIONS)
    if unknown:
        raise KeyError(f"not a named GDELT column: {sorted(unknown)}")
    fields = [""] * 61
    for name, value in {**_DEFAULTS, **overrides}.items():
        fields[_POSITIONS[name] - 1] = value
    return fields


def tsv(*rows: list[str]) -> str:
    return "".join("\t".join(row) + "\n" for row in rows)


def zip_bytes(text: str, member: str = "20240103141500.export.CSV") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(member, text)
    return buffer.getvalue()


def frame_of(*rows: list[str]) -> pd.DataFrame:
    """The named-column frame the reader would produce for these rows."""
    return read_export_csv(tsv(*rows).encode())
