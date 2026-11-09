from precedent.libraries import build, sample
from precedent.libraries.sources import CITATIONS, EVENT_SOURCES


def test_sources_are_public_and_shared_by_build_and_sample():
    """P-47 (I5): the citations and event sources live in libraries/sources.py."""
    assert set(CITATIONS) == {"gdelt2", "edgar_8k", "polymarket", "kalshi"}
    assert [name for name, _ in EVENT_SOURCES] == ["gdelt2", "edgar_8k"]
    assert not hasattr(build, "_CITATIONS") and not hasattr(build, "_EVENT_SOURCES")
    assert CITATIONS["gdelt2"] == sample.GDELT_CITATION
