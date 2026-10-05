"""
Step 1: PDF -> lines of text, each tagged with its page number and font info.

Font size and boldness are kept so chunk.py can tell headings from body text.
Repeated headers/footers (e.g. "Public Document", page numbers) are removed.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf


@dataclass
class Line:
    text: str
    page: int        # 1-based, matches what a person sees in a PDF viewer
    size: float      # largest font size on the line
    bold: bool


def _is_bold(span: dict) -> bool:
    return bool(span["flags"] & 16) or "bold" in span["font"].lower()


def parse_pdf(path: str | Path) -> list[Line]:
    """Return every non-empty line of the PDF, in reading order."""
    lines: list[Line] = []
    with pymupdf.open(path) as doc:
        for page_index, page in enumerate(doc):
            data = page.get_text("dict", sort=True)
            for block in data["blocks"]:
                if block.get("type") != 0:  # 0 = text, 1 = image
                    continue
                for line in block["lines"]:
                    spans = [s for s in line["spans"] if s["text"].strip()]
                    if not spans:
                        continue
                    text = " ".join(s["text"].strip() for s in spans)
                    lines.append(Line(
                        text=" ".join(text.split()),  # collapse whitespace
                        page=page_index + 1,
                        size=round(max(s["size"] for s in spans), 1),
                        bold=all(_is_bold(s) for s in spans),
                    ))
        n_pages = len(doc)

    return _remove_headers_footers(lines, n_pages)


def _remove_headers_footers(lines: list[Line], n_pages: int) -> list[Line]:
    """Drop bare page numbers and lines repeated on most pages."""
    cleaned = [l for l in lines if not _looks_like_page_number(l.text)]
    if n_pages < 3:
        return cleaned
    pages_per_text = Counter()
    for text in {(l.text, l.page) for l in cleaned}:
        pages_per_text[text[0]] += 1
    repeated = {t for t, n in pages_per_text.items() if n >= 0.6 * n_pages and len(t) < 80}
    return [l for l in cleaned if l.text not in repeated]


def _looks_like_page_number(text: str) -> bool:
    t = text.lower().replace("page", "").replace("of", " ").strip()
    return all(part.isdigit() for part in t.split()) and len(t) <= 12 and t != ""


def body_font_size(lines: list[Line]) -> float:
    """The most common font size, weighted by amount of text = body text size."""
    counts = Counter()
    for l in lines:
        counts[l.size] += len(l.text)
    return counts.most_common(1)[0][0] if counts else 0.0