"""
Step 2: lines -> sections -> chunks.

Why section-aware chunking: a fixed 500-character split can cut a fee table in
half, so "Annual fee" ends up in one chunk and "SGD 196.20" in the next. Here we
first group lines under their headings, then pack whole sections into chunks,
and only split a section when it is too long on its own.
"""

from dataclasses import dataclass, field

from ingest.parse import Line, body_font_size

MAX_CHARS = 1200      # target maximum chunk size
OVERLAP_LINES = 2     # lines repeated between pieces of an over-long section


@dataclass
class Section:
    heading: str
    lines: list[Line] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(l.text for l in self.lines)


@dataclass
class Chunk:
    text: str
    section: str
    page_start: int
    page_end: int


def is_heading(line: Line, body_size: float) -> bool:
    text = line.text
    if len(text) > 90 or text.endswith((".", ",", ";")):
        return False
    bigger = line.size >= body_size + 1.5
    bold_title = line.bold and line.size >= body_size and len(text) <= 70
    return bigger or bold_title


def split_sections(lines: list[Line]) -> list[Section]:
    body = body_font_size(lines)
    sections = [Section(heading="Introduction")]
    for line in lines:
        if is_heading(line, body):
            # Consecutive headings (e.g. a title then a subtitle) are joined
            if not sections[-1].lines:
                prev = sections[-1].heading
                sections[-1].heading = line.text if prev == "Introduction" else f"{prev} > {line.text}"
            else:
                sections.append(Section(heading=line.text))
        else:
            sections[-1].lines.append(line)
    return [s for s in sections if s.lines]


def make_chunks(sections: list[Section], max_chars: int = MAX_CHARS) -> list[Chunk]:
    chunks: list[Chunk] = []
    buffer: list[Section] = []

    def flush() -> None:
        if not buffer:
            return
        text = "\n\n".join(f"{s.heading}\n{s.text}" for s in buffer)
        all_lines = [l for s in buffer for l in s.lines]
        chunks.append(Chunk(
            text=text,
            section=" | ".join(s.heading for s in buffer),
            page_start=min(l.page for l in all_lines),
            page_end=max(l.page for l in all_lines),
        ))
        buffer.clear()

    for section in sections:
        size = len(section.heading) + len(section.text)
        if size > max_chars:
            flush()
            chunks.extend(_split_long_section(section, max_chars))
            continue
        current = sum(len(s.heading) + len(s.text) for s in buffer)
        if buffer and current + size > max_chars:
            flush()
        buffer.append(section)
    flush()
    return chunks


def _split_long_section(section: Section, max_chars: int) -> list[Chunk]:
    """Split by whole lines (never mid-line), repeating a little overlap."""
    pieces: list[list[Line]] = [[]]
    length = 0
    for line in section.lines:
        if pieces[-1] and length + len(line.text) > max_chars:
            overlap = pieces[-1][-OVERLAP_LINES:]
            pieces.append(list(overlap))
            length = sum(len(l.text) for l in overlap)
        pieces[-1].append(line)
        length += len(line.text)

    total = len(pieces)
    return [
        Chunk(
            text=f"{section.heading}" + (f" (part {i}/{total})" if total > 1 else "")
                 + "\n" + "\n".join(l.text for l in piece),
            section=section.heading,
            page_start=min(l.page for l in piece),
            page_end=max(l.page for l in piece),
        )
        for i, piece in enumerate(pieces, 1)
    ]