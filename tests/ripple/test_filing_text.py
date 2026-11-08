import pytest

from precedent.ripple.filing_text import extract_business_section

TOC_AND_BODY = """
<html><body>
<h2>TABLE OF CONTENTS</h2>
<table>
<tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>9</td></tr>
<tr><td>Item 2.</td><td>Properties</td><td>15</td></tr>
</table>
<p>ITEM&nbsp;1. BUSINESS</p>
<p>We design and sell industrial valves &amp; pumps to refiners.</p>
<p>Our competitors include Acme Flow and Globex Pumps.</p>
<script>var hidden = "Item 1. Business should not appear";</script>
<style>.x { color: red }</style>
<p>ITEM 1A. RISK FACTORS</p>
<p>Demand for valves is cyclical.</p>
<p>ITEM 2. PROPERTIES</p>
<p>We lease offices in Houston.</p>
</body></html>
"""


def test_missing_business_section_is_none_not_empty_string():
    assert extract_business_section("<html><body>no items here</body></html>") is None


def test_the_real_section_wins_over_the_table_of_contents_line():
    text = extract_business_section(TOC_AND_BODY)
    assert text is not None
    assert text.startswith("We design and sell industrial valves & pumps to refiners.")
    assert "Acme Flow and Globex Pumps" in text
    assert "Risk Factors" not in text
    assert "Demand for valves" not in text


def test_tags_entities_scripts_and_styles_are_stripped():
    text = extract_business_section(TOC_AND_BODY)
    assert text is not None
    assert "<" not in text and "&amp;" not in text and "&nbsp;" not in text
    assert "hidden" not in text
    assert "color: red" not in text


def test_item_1a_alone_is_not_taken_as_item_1():
    html = "<p>Item 1A. Risk Factors</p><p>Many risks apply here.</p><p>Item 2. Properties</p>"
    assert extract_business_section(html) is None


def test_item_10_to_16_headings_are_not_item_1():
    html = "<p>Item 10. Directors</p><p>Board text goes here for a while.</p><p>Item 2. Properties</p>"
    assert extract_business_section(html) is None


def test_section_runs_to_item_2_when_there_is_no_item_1a():
    html = (
        "<p>Item 1. Business</p><p>We make small widgets for hobbyists.</p>"
        "<p>Item 2. Properties</p><p>One shed.</p>"
    )
    assert extract_business_section(html) == "We make small widgets for hobbyists."


def test_section_ends_at_item_1b_when_there_is_no_item_1a():
    html = (
        "<p>Item 1. Business</p><p>We make small widgets.</p>"
        "<p>Item 1B. Unresolved Staff Comments</p><p>None.</p>"
    )
    assert extract_business_section(html) == "We make small widgets."


@pytest.mark.parametrize("heading", ["Item 1.", "ITEM 1 -", "Item 1:", "Item 1.", "item 1 —"])
def test_heading_punctuation_variants_are_tolerated(heading):
    html = f"<p>{heading} Business</p><p>We build boats.</p><p>Item 1A. Risk Factors</p>"
    assert extract_business_section(html) == "We build boats."


def test_empty_span_is_none_not_empty_string():
    html = "<p>Item 1. Business</p><p>Item 1A. Risk Factors</p><p>Item 2. Properties</p>"
    assert extract_business_section(html) is None


def test_paragraph_breaks_are_kept_and_whitespace_collapsed():
    html = "<p>Item 1. Business</p><p>First   paragraph\n here.</p><p>Second one.</p><p>Item 2.</p>"
    assert extract_business_section(html) == "First paragraph here.\n\nSecond one."
