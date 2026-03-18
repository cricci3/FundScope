"""
ingest.py — Phase 1 & 2: PDF extraction, cleaning, chunking, and metadata tagging
ETF RAG Project

Usage:
    python ingest.py --pdf data/raw/SPY_prospectus_2023.pdf --ticker SPY --issuer SPDR \
                     --category sp500 --year 2023 --doc_type prospectus
"""

import re
import json
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

# ── PDF extraction ─────────────────────────────────────────────────────────────
# We try pdfplumber first (better on tables), fall back to pymupdf (faster on plain text).
# Install both: pip install pdfplumber pymupdf

try:
    import pdfplumber
    HAS_PDFPLUMBER = True
except ImportError:
    HAS_PDFPLUMBER = False

try:
    import fitz  # pymupdf
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

if not HAS_PDFPLUMBER and not HAS_PYMUPDF:
    raise ImportError("Install at least one PDF library: pip install pdfplumber pymupdf")


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class DocumentMetadata:
    etf_ticker: str          # e.g. "SPY"
    issuer: str              # e.g. "SPDR", "iShares", "Vanguard"
    category: str            # e.g. "sp500", "bond", "gold", "emerging_markets"
    year: int                # e.g. 2023
    doc_type: str            # "prospectus" or "annual_report"
    source_file: str         # original filename
    total_pages: int = 0


@dataclass
class Chunk:
    chunk_id: str            # deterministic hash for dedup / lookup
    text: str
    metadata: DocumentMetadata
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    char_start: int = 0
    char_end: int = 0


# ── Section heading detection ──────────────────────────────────────────────────
# ETF prospectuses follow predictable section structures.
# Heuristics: short lines, no trailing period, matches known sections or ALL CAPS.

KNOWN_SECTION_HEADINGS = [
    "fee table", "fees and expenses", "annual fund operating expenses",
    "risk factors", "principal risks", "investment objective",
    "principal investment strategies", "portfolio managers",
    "financial highlights", "distribution policy",
    "purchase and sale", "tax information",
    "management", "performance", "benchmark",
    "liquidity risk", "market risk", "tracking error",
]

def detect_section_heading(line: str) -> Optional[str]:
    """Return a normalised heading string if the line looks like a section title."""
    line = line.strip()
    if not line or len(line) > 120:
        return None
    if line.endswith(".") or line.endswith(","):
        return None

    line_lower = line.lower()
    for known in KNOWN_SECTION_HEADINGS:
        if known in line_lower:
            return line_lower.strip()

    # Catch ALL CAPS headings (common in SEC filings)
    if line.isupper() and len(line.split()) <= 8:
        return line.lower().strip()

    return None


# ── Text cleaning ──────────────────────────────────────────────────────────────

# Boilerplate patterns shared across ETF filings — these pollute retrieval.
# RESEARCH NOTE: logging how many lines match per document is Research Angle #2.
BOILERPLATE_PATTERNS = [
    r"this prospectus is not an offer to sell",
    r"you should consider the fund.s investment objectives",
    r"before you invest, you may want to review",
    r"investment involves risk",
    r"past performance (does not|is not) (guarantee|indicative)",
    r"sec file number",
    r"cusip number",
    r"not fdic insured",
    r"may lose value",
    r"no bank guarantee",
    r"©\s*\d{4}",
    r"page \d+ of \d+",
    r"^\s*\d+\s*$",           # standalone page numbers
    r"table of contents",
]

BOILERPLATE_RE = re.compile(
    "|".join(BOILERPLATE_PATTERNS),
    re.IGNORECASE | re.MULTILINE
)


def clean_text(raw: str) -> str:
    """
    Remove noise from PDF-extracted text:
    - Legal boilerplate
    - PDF ligature artifacts (fi, fl, ff)
    - Null bytes from bad encoding
    - Runs of blank lines (collapse to max 2)
    """
    text = raw
    # Fix common PDF ligature artifacts
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\ufb00", "ff")
    text = text.replace("\x00", "")

    # Remove boilerplate lines
    lines = text.splitlines()
    cleaned_lines = [line for line in lines if not BOILERPLATE_RE.search(line)]
    text = "\n".join(cleaned_lines)

    # Collapse runs of blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())

    return text.strip()


def compute_content_hash(text: str) -> str:
    """
    Aggressively normalise and hash text for near-duplicate detection.
    Two chunks with the same hash → likely boilerplate copies.
    """
    normalised = re.sub(r"\s+", " ", text.lower().strip())
    normalised = re.sub(r"[^a-z0-9 ]", "", normalised)
    return hashlib.md5(normalised.encode()).hexdigest()


# ── PDF extraction ─────────────────────────────────────────────────────────────

def extract_with_pdfplumber(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns list of (page_number, page_text). 1-indexed."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
            pages.append((i, text))
    return pages


def extract_with_pymupdf(pdf_path: Path) -> list[tuple[int, str]]:
    """Returns list of (page_number, page_text). 1-indexed."""
    pages = []
    doc = fitz.open(str(pdf_path))
    for i, page in enumerate(doc, start=1):
        text = page.get_text("text") or ""
        pages.append((i, text))
    doc.close()
    return pages


def extract_pdf(pdf_path: Path, backend: str = "auto") -> list[tuple[int, str]]:
    """
    Extract text page by page from a PDF.
    backend: "pdfplumber" | "pymupdf" | "auto" (prefers pdfplumber)
    """
    if backend in ("auto", "pdfplumber") and HAS_PDFPLUMBER:
        return extract_with_pdfplumber(pdf_path)
    if backend in ("auto", "pymupdf") and HAS_PYMUPDF:
        return extract_with_pymupdf(pdf_path)
    raise RuntimeError(f"Backend '{backend}' not available. Check your installations.")


# ── Chunking ───────────────────────────────────────────────────────────────────
# Recursive character splitter: tries paragraph breaks first, then sentences, then words.
# This keeps semantically coherent units together without requiring LangChain as a dependency.
# To swap in LangChain's splitter, replace split_text() with:
#   from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[tuple[str, int]]:
    """
    Split text into overlapping chunks.
    Returns list of (chunk_text, char_start_offset) tuples.
    """
    separators = ["\n\n", "\n", ". ", " "]
    results: list[tuple[str, int]] = []

    def _split(fragment: str, offset: int, seps: list[str]) -> None:
        if len(fragment) <= chunk_size:
            if fragment.strip():
                results.append((fragment.strip(), offset))
            return

        sep = seps[0] if seps else ""
        parts = fragment.split(sep) if sep else list(fragment)

        current = ""
        current_start = offset
        running_offset = offset

        for part in parts:
            connector = sep if current else ""
            candidate = current + connector + part

            if len(candidate) <= chunk_size:
                if not current:
                    current_start = running_offset
                current = candidate
            else:
                if current.strip():
                    results.append((current.strip(), current_start))
                    # Overlap: start next chunk with tail of previous
                    overlap = current[-chunk_overlap:] if len(current) > chunk_overlap else current
                    current = overlap + connector + part
                    current_start = running_offset - len(overlap)
                else:
                    # Part alone exceeds chunk_size — recurse with finer separator
                    if len(seps) > 1:
                        _split(part, running_offset, seps[1:])
                    else:
                        results.append((part.strip(), running_offset))
                    current = ""
                    current_start = running_offset + len(part)

            running_offset += len(part) + len(sep)

        if current.strip():
            results.append((current.strip(), current_start))

    _split(text, 0, separators)
    return results


# ── Section heading tracker ────────────────────────────────────────────────────

def assign_section_headings(
    pages: list[tuple[int, str]]
) -> list[tuple[int, str, Optional[str]]]:
    """
    Walk pages sequentially, tracking the most recently seen section heading.
    Returns list of (page_number, page_text, current_heading).
    """
    current_heading: Optional[str] = None
    result = []
    for page_num, page_text in pages:
        for line in page_text.splitlines():
            h = detect_section_heading(line)
            if h:
                current_heading = h
        result.append((page_num, page_text, current_heading))
    return result


# ── Deduplication ──────────────────────────────────────────────────────────────

def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    """
    Remove near-duplicate chunks using content hash.
    First occurrence wins. Returns (unique_chunks, removed_count).
    """
    seen: set[str] = set()
    unique: list[Chunk] = []
    removed = 0
    for chunk in chunks:
        h = compute_content_hash(chunk.text)
        if h not in seen:
            seen.add(h)
            unique.append(chunk)
        else:
            removed += 1
    if removed:
        print(f"  [dedup] Removed {removed} near-duplicate chunks (boilerplate)")
    return unique


# ── Main ingestion pipeline ────────────────────────────────────────────────────

def ingest_pdf(
    pdf_path: Path,
    metadata: DocumentMetadata,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    backend: str = "auto",
    deduplicate: bool = True,
) -> list[Chunk]:
    """
    Full ingestion pipeline for a single PDF:
      1. Extract text page-by-page
      2. Assign rolling section headings
      3. Clean each page (boilerplate removal, ligature repair)
      4. Chunk with overlap
      5. Attach metadata to every chunk
      6. Deduplicate near-identical chunks

    Returns a list of Chunk objects ready to be embedded.
    """
    print(f"\n[ingest] {pdf_path.name}")

    # 1. Extract
    pages = extract_pdf(pdf_path, backend=backend)
    metadata.total_pages = len(pages)
    print(f"  Pages extracted : {len(pages)}")

    # 2. Assign section headings
    pages_with_headings = assign_section_headings(pages)

    # 3 + 4. Clean and chunk
    chunks: list[Chunk] = []
    global_offset = 0

    for page_num, page_text, section_heading in pages_with_headings:
        cleaned = clean_text(page_text)
        if not cleaned:
            global_offset += len(page_text)
            continue

        page_chunks = split_text(cleaned, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        for chunk_text, local_offset in page_chunks:
            if len(chunk_text.strip()) < 50:   # skip tiny fragments
                continue

            # Deterministic ID: ticker + year + doc_type + content prefix
            chunk_id = hashlib.sha256(
                f"{metadata.etf_ticker}_{metadata.year}_{metadata.doc_type}_{chunk_text[:100]}"
                .encode()
            ).hexdigest()[:16]

            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=chunk_text,
                metadata=metadata,
                page_number=page_num,
                section_heading=section_heading,
                char_start=global_offset + local_offset,
                char_end=global_offset + local_offset + len(chunk_text),
            ))

        global_offset += len(cleaned)

    print(f"  Chunks (raw)    : {len(chunks)}")

    # 5. Deduplicate
    if deduplicate:
        chunks = deduplicate_chunks(chunks)

    print(f"  Chunks (final)  : {len(chunks)}")
    return chunks


# ── Serialisation ──────────────────────────────────────────────────────────────

def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    """Serialise chunks to JSON for inspection or downstream embedding."""
    records = [
        {
            "chunk_id": c.chunk_id,
            "text": c.text,
            "page_number": c.page_number,
            "section_heading": c.section_heading,
            "char_start": c.char_start,
            "char_end": c.char_end,
            "metadata": asdict(c.metadata),
        }
        for c in chunks
    ]
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"  Saved → {output_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Ingest ETF PDF filings into chunks")
    p.add_argument("--pdf",          type=Path,  required=True,  help="Path to PDF file")
    p.add_argument("--ticker",       required=True,              help="ETF ticker, e.g. SPY")
    p.add_argument("--issuer",       required=True,              help="Issuer name, e.g. SPDR")
    p.add_argument("--category",     required=True,              help="Category, e.g. sp500")
    p.add_argument("--year",         type=int,   required=True,  help="Filing year, e.g. 2023")
    p.add_argument("--doc_type",     required=True,
                   choices=["prospectus", "annual_report"])
    p.add_argument("--output_dir",   type=Path,  default=Path("data/processed"))
    p.add_argument("--chunk_size",   type=int,   default=500)
    p.add_argument("--chunk_overlap",type=int,   default=50)
    p.add_argument("--backend",      default="auto",
                   choices=["auto", "pdfplumber", "pymupdf"])
    p.add_argument("--no_dedup",     action="store_true",
                   help="Disable near-duplicate removal")
    return p


def main():
    args = build_parser().parse_args()

    meta = DocumentMetadata(
        etf_ticker=args.ticker.upper(),
        issuer=args.issuer,
        category=args.category,
        year=args.year,
        doc_type=args.doc_type,
        source_file=args.pdf.name,
    )

    chunks = ingest_pdf(
        pdf_path=args.pdf,
        metadata=meta,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        backend=args.backend,
        deduplicate=not args.no_dedup,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_file = args.output_dir / f"{args.ticker.upper()}_{args.year}_{args.doc_type}.json"
    save_chunks(chunks, out_file)


if __name__ == "__main__":
    main()
