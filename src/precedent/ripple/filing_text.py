"""Item 1 (Business) of a 10-K, as plain text.

Core dependencies only: the standard library's `html.parser` and `re`. Nothing here fetches a
filing; the caller passes the HTML it already holds.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

_BLOCK_TAGS = frozenset(
    {
        "p", "div", "tr", "td", "th", "li", "ul", "ol", "table", "section", "article",
        "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "center", "body",
    }
)  # fmt: skip
_SKIPPED_TAGS = frozenset({"script", "style"})
_DASHES = "\\-\u2013\u2014"
_FLAGS = re.IGNORECASE | re.MULTILINE
_ITEM_1 = re.compile(rf"^[ \t]*item[ \t]+1(?![0-9A-Za-z])[ \t]*[.:{_DASHES}]*[ \t]*", _FLAGS)
_ITEM_1A = re.compile(r"^[ \t]*item[ \t]+1a(?![0-9A-Za-z])", _FLAGS)
_ITEM_1B = re.compile(r"^[ \t]*item[ \t]+1b(?![0-9A-Za-z])", _FLAGS)
_ITEM_2 = re.compile(r"^[ \t]*item[ \t]+2(?![0-9A-Za-z])", _FLAGS)
_LEADING_TITLE = re.compile(rf"\A[ \t]*business[ \t]*[.:{_DASHES}]*[ \t]*(?:\n|\Z)", re.IGNORECASE)


class _TextExtractor(HTMLParser):
    """Visible text of an HTML document; block tags become paragraph breaks."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth += 1
        elif tag == "br":
            self._parts.append("\n")
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIPPED_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self._parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self._parts.append(re.sub(r"\s+", " ", data))

    def text(self) -> str:
        joined = "".join(self._parts)
        lines = [line.strip() for line in joined.split("\n")]
        collapsed = "\n".join(lines)
        return re.sub(r"\n{3,}", "\n\n", collapsed).strip()


def _first_end(text: str, start: int) -> int | None:
    """Where Item 1 ends after `start`: the first Item 1A heading, else 1B, else Item 2."""
    for pattern in (_ITEM_1A, _ITEM_1B, _ITEM_2):
        match = pattern.search(text, start)
        if match:
            return match.start()
    return None


def extract_business_section(filing_html: str) -> str | None:
    """The text of Item 1 (Business) of a 10-K, or None when it cannot be found.

    The HTML is reduced to plain text (script and style dropped, entities unescaped, whitespace
    collapsed, paragraph breaks kept). An `Item 1` heading is a line that starts with "Item 1"
    (any case, non-breaking spaces tolerated, optionally followed by `.`, `:` or a dash) and is not
    `Item 1A`, `Item 1B` or `Item 10` to `Item 16`. The section runs to the next `Item 1A` heading,
    else the next `Item 1B`, else the next `Item 2`. A 10-K's table of contents lists the same
    headings, so every `Item 1` heading is a candidate and the candidate with the most text wins;
    the table-of-contents line holds only a title and a page number. A leading "Business" title
    line is dropped. None (never an empty string) is returned when no heading yields any text.
    """
    parser = _TextExtractor()
    parser.feed(filing_html)
    parser.close()
    text = parser.text()
    best = ""
    for heading in _ITEM_1.finditer(text):
        end = _first_end(text, heading.end())
        if end is None:
            continue
        section = _LEADING_TITLE.sub("", text[heading.end() : end]).strip()
        if len(section) > len(best):
            best = section
    return best or None
