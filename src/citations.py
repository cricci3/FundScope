"""
citations.py — Structured citations: the model cites [Chunk N], the code resolves them

The model no longer writes source metadata by hand. Every chunk shown to it gets
a number; it cites facts as [Chunk 3] (or [Chunk 1, 4], [Chunks 2-3]) and
resolve_citations() replaces each tag with the real metadata of those chunks,
[ISIN | issuer | doc_type | year], taken from the index. So a citation can only
point to a document that was actually retrieved, and the resolved answer keeps
the format that evaluate.py, the judge and parse_citations() already read.

    book = ChunkBook()
    n = book.add(chunk)                      # → 1, 2, ... (same chunk → same number)
    resolved = resolve_citations("TER 0.20% [Chunk 1].", book)
    resolved.text           # "TER 0.20% [IE00B4L5Y983 | ishares | factsheet | 2026]."
    resolved.cited_sources  # [CitedSource(...)]
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ── Document-level citation (legacy format, still used in resolved answers) ───

CITATION_PATTERN = re.compile(
    r"\[([A-Z]{2}[A-Z0-9]{10})\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d{4})[^\]]*\]"
)


@dataclass
class CitedSource:
    etf_isin: str
    issuer: str
    doc_type: str
    year: int


def parse_citations(answer: str) -> list:
    """Extract [ISIN | issuer | doc_type | year] citations from answer text."""
    seen = set()
    sources = []
    for match in CITATION_PATTERN.finditer(answer):
        isin, issuer, doc_type, year = match.groups()
        key = (isin.upper(), issuer.lower(), doc_type.lower(), int(year))
        if key not in seen:
            seen.add(key)
            sources.append(CitedSource(*key))
    return sources


def source_label(chunk) -> str:
    return f"[{chunk.etf_isin} | {chunk.issuer} | {chunk.doc_type} | {chunk.year}]"


# ── Chunk numbering ────────────────────────────────────────────────────────────

class ChunkBook:
    """Numbers the chunks shown to the model (1, 2, ...); a chunk seen twice keeps its number."""

    def __init__(self):
        self.chunks: list = []                 # chunk N is self.chunks[N - 1]
        self._numbers: dict[str, int] = {}

    def add(self, chunk) -> tuple[int, bool]:
        """Return (number, is_new)."""
        n = self._numbers.get(chunk.chunk_id)
        if n is not None:
            return n, False
        self.chunks.append(chunk)
        n = self._numbers[chunk.chunk_id] = len(self.chunks)
        return n, True

    def get(self, n: int):
        return self.chunks[n - 1] if 1 <= n <= len(self.chunks) else None

    def __len__(self) -> int:
        return len(self.chunks)


def chunk_header(n: int, chunk) -> str:
    """Header shown to the model above each chunk's text."""
    quarter = f" {chunk.quarter}" if chunk.quarter else ""
    table   = " [TABLE]" if chunk.block_type in ("table", "scenario") else ""
    return (f"[Chunk {n}]{table} {chunk.etf_isin} | {chunk.issuer} | {chunk.doc_type} | "
            f"{chunk.year}{quarter} | section: {chunk.section_heading or 'unknown'}")


# ── Resolution ─────────────────────────────────────────────────────────────────

# [Chunk 3] · [chunk 1, 4] · [Chunks 2-3] · [Chunk 1 and 5] · [Chunk 2; Chunk 6]
CHUNK_TAG_RE = re.compile(r"\[\s*chunks?\s*\d[^\]]*\]", re.I)
_RANGE_RE    = re.compile(r"(\d+)\s*[-–]\s*(\d+)|(\d+)")


def _tag_numbers(tag: str) -> list[int]:
    numbers = []
    for a, b, single in _RANGE_RE.findall(tag):
        if single:
            numbers.append(int(single))
        else:
            lo, hi = int(a), int(b)
            if lo <= hi and hi - lo < 50:
                numbers.extend(range(lo, hi + 1))
    return list(dict.fromkeys(numbers))


@dataclass
class ResolvedAnswer:
    text: str                                          # answer with tags → [ISIN | ... | year]
    raw_text: str                                      # answer as written by the model
    cited_sources: list = field(default_factory=list)  # list[CitedSource], one per document
    cited_chunks: list = field(default_factory=list)   # chunk numbers cited, in order
    unknown_refs: list = field(default_factory=list)   # numbers that match no chunk


def resolve_citations(answer: str, book: ChunkBook) -> ResolvedAnswer:
    """Replace [Chunk N] tags with document citations taken from the chunks' metadata."""
    cited_chunks: list[int] = []
    unknown: list[int] = []

    def _replace(match: re.Match) -> str:
        labels = []
        for n in _tag_numbers(match.group(0)):
            chunk = book.get(n)
            if chunk is None:
                unknown.append(n)
                continue
            if n not in cited_chunks:
                cited_chunks.append(n)
            label = source_label(chunk)
            if label not in labels:
                labels.append(label)
        return "".join(labels)

    text = CHUNK_TAG_RE.sub(_replace, answer)
    text = re.sub(r"[ \t]+([.,;:)])", r"\1", text)      # " ." left by a dropped tag
    text = re.sub(r"[ \t]{2,}", " ", text).strip()

    return ResolvedAnswer(
        text=text,
        raw_text=answer,
        # also picks up citations the model wrote in the old format by hand
        cited_sources=parse_citations(text),
        cited_chunks=cited_chunks,
        unknown_refs=list(dict.fromkeys(unknown)),
    )


def book_from_chunks(chunks: list, book: Optional[ChunkBook] = None) -> ChunkBook:
    book = book or ChunkBook()
    for c in chunks:
        book.add(c)
    return book
