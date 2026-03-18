"""
ingest.py — Ingestion for ETF Factsheets AND KIDs (Key Information Documents)
ETF RAG Project

Supports:
  - Issuer factsheets: iShares, Xtrackers, Amundi, UBS
  - PRIIPS KIDs: issuer-agnostic (legally standardised structure)

Document type is read from the metadata config JSON (field: "type").
KIDs and factsheets go through separate extraction paths that share
the same output schema (list of Chunk objects).

Usage:
    # Single file (type inferred from --type flag)
    python ingest.py --pdf data/raw/iShares_Core_MSCI_World_UCITS_ETF_USD_acc_kid.pdf \
                     --isin EUNL --issuer ishares --category "MSCI World" \
                     --year 2023 --type kid --currency USD --share_class acc

    # Batch with config JSON (recommended)
    python ingest.py --batch data/raw/ --config metadata.json --output_dir data/processed/
"""

import re
import json
import hashlib
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is required: pip install pdfplumber")


# ── Constants ──────────────────────────────────────────────────────────────────

SUPPORTED_ISSUERS = {"ishares", "xtrackers", "amundi", "ubs"}
SUPPORTED_TYPES   = {"factsheet", "kid"}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class KeyFacts:
    """
    Structured fields that can be extracted deterministically.
    Shared between factsheets and KIDs — KID-specific fields default to None
    when parsing a factsheet, and vice versa.
    """
    # ── Common ──
    isin: Optional[str] = None
    inception_date: Optional[str] = None
    benchmark: Optional[str] = None
    distribution_policy: Optional[str] = None   # Accumulating / Distributing
    replication_method: Optional[str] = None
    domicile: Optional[str] = None
    currency: Optional[str] = None

    # ── Factsheet-specific ──
    ter_ocf: Optional[str] = None
    aum: Optional[str] = None
    num_holdings: Optional[str] = None
    tracking_difference: Optional[str] = None
    morningstar_rating: Optional[str] = None

    # ── KID-specific ──
    # Costs are broken out in KIDs; TER is replaced by "ongoing costs"
    cost_entry: Optional[str] = None           # one-off entry cost
    cost_exit: Optional[str] = None            # one-off exit cost
    cost_ongoing: Optional[str] = None         # ongoing costs (≈ TER equivalent)
    cost_performance: Optional[str] = None     # performance fees
    cost_transaction: Optional[str] = None     # transaction costs
    srri: Optional[str] = None                 # Summary Risk Indicator (1–7)
    recommended_holding_period: Optional[str] = None
    target_market: Optional[str] = None


@dataclass
class DocumentMetadata:
    etf_isin: str
    issuer: str
    category: str
    year: int
    doc_type: str           # "factsheet" | "kid"  (renamed from "type" to avoid shadowing)
    currency: str
    share_class: str
    source_file: str
    quarter: Optional[str] = None       # factsheets only (Q1–Q4)
    reference_date: Optional[str] = None  # KIDs: "as of" date string if found
    total_pages: int = 0
    key_facts: KeyFacts = field(default_factory=KeyFacts)


@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: DocumentMetadata
    page_number: Optional[int] = None
    section_heading: Optional[str] = None
    block_type: str = "text"   # "text" | "table" | "key_facts" | "scenario"
    char_start: int = 0
    char_end: int = 0


# ── Section headings ───────────────────────────────────────────────────────────

# KID sections are legally mandated by PRIIPS — exact match is reliable.
# We normalise to lowercase keys used throughout the pipeline.
KID_SECTIONS = [
    "what is this product",
    "what are the risks and what could i get in return",
    "what happens if",           # "...unable to pay out" varies by issuer
    "what are the costs",
    "how long should i hold it",
    "how can i complain",
    "other relevant information",
    "performance scenarios",
    "composition of costs",
    "risk indicator",
]

ISHARES_SECTIONS = [
    "fund overview", "key facts", "performance",
    "calendar year performance", "top holdings",
    "sector breakdown", "country breakdown", "asset allocation",
    "risk and reward", "sustainability characteristics",
    "portfolio characteristics", "costs",
]

XTRACKERS_SECTIONS = [
    "fund details", "key data", "fund performance",
    "top 10 holdings", "sector allocation", "country allocation",
    "risk indicator", "ongoing charges", "portfolio data",
]

AMUNDI_SECTIONS = [
    "fund information", "key information", "performance overview",
    "asset breakdown", "geographical breakdown", "top holdings",
    "esg characteristics", "risk scale", "charges",
]

UBS_SECTIONS = [
    "fund facts", "key information", "fund performance",
    "top holdings", "sector weights", "country weights",
    "total expense ratio", "portfolio overview",
]

ALL_FACTSHEET_SECTIONS = set(
    ISHARES_SECTIONS + XTRACKERS_SECTIONS + AMUNDI_SECTIONS + UBS_SECTIONS
)

ISSUER_SECTIONS = {
    "ishares":   ISHARES_SECTIONS,
    "xtrackers": XTRACKERS_SECTIONS,
    "amundi":    AMUNDI_SECTIONS,
    "ubs":       UBS_SECTIONS,
}


def detect_section_heading(
    line: str,
    doc_type: str,
    issuer: Optional[str] = None,
) -> Optional[str]:
    """
    Return a normalised section heading string or None.
    KIDs use the standardised PRIIPS vocabulary.
    Factsheets use issuer-specific vocabulary with ALL-CAPS fallback.
    """
    line = line.strip()
    if not line or len(line) > 120:
        return None
    if line.endswith(","):
        return None

    line_lower = line.lower()

    if doc_type == "kid":
        for section in KID_SECTIONS:
            if section in line_lower:
                return section
        return None

    # Factsheet path
    sections = ISSUER_SECTIONS.get(issuer or "", []) or ALL_FACTSHEET_SECTIONS
    for section in sections:
        if section in line_lower:
            return section

    if line.isupper() and 2 <= len(line.split()) <= 6:
        return line.lower().strip()

    return None


# ── Key facts extraction ───────────────────────────────────────────────────────

# Factsheet patterns
_FACTSHEET_KF_PATTERNS = {
    "isin":                r"\bISIN\b[:\s]+([A-Z]{2}[A-Z0-9]{10})\b",
    "ter_ocf":             r"(?:TER|OCF|Ongoing [Cc]harge[s]?|Total [Ee]xpense [Rr]atio)[:\s]+([0-9]+\.[0-9]+\s*%)",
    "aum":                 r"(?:Fund [Ss]ize|AUM|Net [Aa]ssets|Total [Aa]ssets)[:\s]+((?:USD|EUR|GBP|CHF)?\s*[\d,\.]+\s*(?:bn|mn|m|b)?)",
    "inception_date":      r"(?:Inception [Dd]ate|Launch [Dd]ate|Fund [Ii]nception)[:\s]+(\d{1,2}[\./\-]\w+[\./\-]\d{2,4}|\w+ \d{1,2},? \d{4})",
    "benchmark":           r"(?:Benchmark|Index|Tracks)[:\s]+([A-Z][^\n]{5,80}?)(?:\n|$)",
    "distribution_policy": r"(?:Distribution|Dividend)[:\s]+(Accumulating|Distributing|Income|Acc|Dist)\b",
    "replication_method":  r"(?:Replication|Replication [Mm]ethod)[:\s]+(Physical|Synthetic|Optimised Physical|Sampled Physical)\b",
    "domicile":            r"(?:Domicile|Fund [Dd]omicile)[:\s]+([A-Za-z ]{3,30})(?:\n|$)",
    "currency":            r"(?:Base [Cc]urrency|Fund [Cc]urrency|Currency)[:\s]+([A-Z]{3})\b",
    "num_holdings":        r"(?:Number of [Hh]oldings|Holdings)[:\s]+(\d[\d,]+)\b",
    "tracking_difference": r"(?:Tracking [Dd]ifference|TD)[:\s]+(-?[0-9]+\.[0-9]+\s*%)",
    "morningstar_rating":  r"(?:Morningstar [Rr]ating)[:\s]+(\d(?:\.\d)?\s*(?:stars?)?|★+)",
}

# KID patterns — costs are itemised, SRRI replaces Morningstar
_KID_KF_PATTERNS = {
    "isin":                       r"\bISIN\b[:\s]+([A-Z]{2}[A-Z0-9]{10})\b",
    "inception_date":             r"(?:Inception [Dd]ate|Launch [Dd]ate)[:\s]+(\d{1,2}[\./\-]\w+[\./\-]\d{2,4}|\w+ \d{1,2},? \d{4})",
    "benchmark":                  r"(?:Benchmark|Index)[:\s]+([A-Z][^\n]{5,80}?)(?:\n|$)",
    "distribution_policy":        r"(Accumulating|Distributing|Income)\b",
    "replication_method":         r"(Physical|Synthetic|Optimised Physical|Sampled Physical)\b",
    "domicile":                   r"(?:Domicile)[:\s]+([A-Za-z ]{3,30})(?:\n|$)",
    "currency":                   r"(?:Currency of denomination|Currency)[:\s]+([A-Z]{3})\b",
    # Costs — KIDs present these as a table; we also try inline text
    "cost_entry":                 r"(?:Entry costs?|One-off entry)[:\s]+([0-9]+\.?[0-9]*\s*%)",
    "cost_exit":                  r"(?:Exit costs?|One-off exit)[:\s]+([0-9]+\.?[0-9]*\s*%)",
    "cost_ongoing":               r"(?:Ongoing costs?)[:\s]+([0-9]+\.?[0-9]*\s*%)",
    "cost_performance":           r"(?:Performance fees?)[:\s]+([0-9]+\.?[0-9]*\s*%|not applicable)",
    "cost_transaction":           r"(?:Transaction costs?)[:\s]+([0-9]+\.?[0-9]*\s*%)",
    "srri":                       r"(?:Summary Risk Indicator|SRI|Risk indicator)[:\s\n]+(\d)\b",
    "recommended_holding_period": r"(?:Recommended holding period)[:\s]+([^\n]{3,60})",
    "target_market":              r"(?:Intended retail investor|Target market)[:\s]+([^\n]{10,150})",
}

_FACTSHEET_KF_RE = {k: re.compile(v, re.IGNORECASE) for k, v in _FACTSHEET_KF_PATTERNS.items()}
_KID_KF_RE       = {k: re.compile(v, re.IGNORECASE) for k, v in _KID_KF_PATTERNS.items()}


def extract_key_facts(full_text: str, doc_type: str) -> KeyFacts:
    """
    Run the appropriate regex set over the full document text.
    Returns a populated KeyFacts dataclass.
    """
    kf = KeyFacts()
    patterns = _KID_KF_RE if doc_type == "kid" else _FACTSHEET_KF_RE
    for field_name, pattern in patterns.items():
        match = pattern.search(full_text)
        if match:
            setattr(kf, field_name, match.group(1).strip())
    return kf


# ── Table extraction ───────────────────────────────────────────────────────────

def table_to_text(table: list[list], context: str = "") -> str:
    """
    Convert a pdfplumber table to readable text.
    - 2-column tables  → "Key: Value" pairs
    - Wider tables     → tab-separated rows
    context hint ("costs", "scenarios") can be used for future specialisation.
    """
    if not table:
        return ""
    cleaned = []
    for row in table:
        row_clean = [str(c).strip() if c is not None else "" for c in row]
        if any(row_clean):
            cleaned.append(row_clean)
    if not cleaned:
        return ""

    if all(len(row) == 2 for row in cleaned):
        return "\n".join(f"{r[0]}: {r[1]}" for r in cleaned if r[0] or r[1])

    return "\n".join("\t".join(row) for row in cleaned)


# ── Text cleaning ──────────────────────────────────────────────────────────────

_FACTSHEET_BOILERPLATE = [
    r"capital at risk",
    r"past performance (is not|does not) (a reliable|guarantee)",
    r"this document is (a )?marketing material",
    r"for professional (clients?|investors?) only",
    r"not for retail distribution",
    r"please refer to the (prospectus|kiid|kid)",
    r"issued by (blackrock|xtrackers|amundi|dws|ubs)",
    r"all figures are (as of|as at)",
    r"source:\s*(blackrock|msci|ftse|bloomberg|ubs)",
    r"^\s*\d+\s*$",
    r"page \d+ of \d+",
]

# KID boilerplate: standard PRIIPS footer text — keep the substantive sections
_KID_BOILERPLATE = [
    r"this document is required by law",
    r"you are about to purchase a product",
    r"complaints can be directed",
    r"^\s*\d+\s*$",
    r"page \d+ of \d+",
]

_FACTSHEET_BOILERPLATE_RE = re.compile("|".join(_FACTSHEET_BOILERPLATE), re.IGNORECASE | re.MULTILINE)
_KID_BOILERPLATE_RE       = re.compile("|".join(_KID_BOILERPLATE),       re.IGNORECASE | re.MULTILINE)


def clean_text(raw: str, doc_type: str) -> str:
    text = raw
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl").replace("\ufb00", "ff")
    text = text.replace("\x00", "")

    pattern = _KID_BOILERPLATE_RE if doc_type == "kid" else _FACTSHEET_BOILERPLATE_RE
    lines = [l for l in text.splitlines() if not pattern.search(l)]
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return "\n".join(l.rstrip() for l in text.splitlines()).strip()


# ── Chunking ───────────────────────────────────────────────────────────────────
# KIDs: larger chunks (400 chars) — sections are short and self-contained,
#        splitting mid-section loses the regulatory context.
# Factsheets: smaller chunks (250 chars) — documents are denser.

CHUNK_SIZES = {"factsheet": 250, "kid": 400}
CHUNK_OVERLAPS = {"factsheet": 30, "kid": 40}


def split_text(text: str, chunk_size: int, chunk_overlap: int) -> list[tuple[str, int]]:
    """Recursive character splitter → list of (chunk_text, char_start)."""
    separators = ["\n\n", "\n", ". ", " "]
    results: list[tuple[str, int]] = []

    def _split(fragment: str, offset: int, seps: list[str]) -> None:
        if len(fragment) <= chunk_size:
            if fragment.strip():
                results.append((fragment.strip(), offset))
            return
        sep = seps[0] if seps else ""
        parts = fragment.split(sep) if sep else list(fragment)
        current, current_start, running_offset = "", offset, offset
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
                    overlap = current[-chunk_overlap:] if len(current) > chunk_overlap else current
                    current = overlap + connector + part
                    current_start = running_offset - len(overlap)
                else:
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


# ── Deduplication ──────────────────────────────────────────────────────────────

def compute_content_hash(text: str) -> str:
    n = re.sub(r"\s+", " ", text.lower().strip())
    n = re.sub(r"[^a-z0-9 ]", "", n)
    return hashlib.md5(n.encode()).hexdigest()


def deduplicate_chunks(chunks: list[Chunk]) -> list[Chunk]:
    seen: set[str] = set()
    unique, removed = [], 0
    for c in chunks:
        h = compute_content_hash(c.text)
        if h not in seen:
            seen.add(h)
            unique.append(c)
        else:
            removed += 1
    if removed:
        print(f"  [dedup] Removed {removed} near-duplicate chunks")
    return unique


# ── Chunk ID helper ────────────────────────────────────────────────────────────

def make_chunk_id(metadata: DocumentMetadata, prefix: str, text: str) -> str:
    key = f"{metadata.etf_isin}_{metadata.year}_{prefix}_{text[:80]}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


# ── Page-level extraction (shared) ────────────────────────────────────────────

def extract_page_blocks(
    page,
    page_num: int,
    metadata: DocumentMetadata,
    current_heading: Optional[str],
    global_offset: int,
) -> tuple[list[Chunk], Optional[str], int]:
    """
    Extract text + tables from one pdfplumber page.
    Works for both factsheets and KIDs — doc_type drives chunk size and
    section heading detection.
    """
    doc_type = metadata.doc_type
    chunk_size    = CHUNK_SIZES[doc_type]
    chunk_overlap = CHUNK_OVERLAPS[doc_type]
    chunks: list[Chunk] = []

    # ── Tables ──────────────────────────────────────────────────────────────
    tables = page.extract_tables() or []
    table_texts = []
    for table in tables:
        # Hint the context from the current section heading for future specialisation
        context = current_heading or ""
        tt = table_to_text(table, context=context)
        if tt and len(tt.strip()) > 20:
            table_texts.append(tt)

    # ── Plain text ──────────────────────────────────────────────────────────
    raw = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
    cleaned = clean_text(raw, doc_type)

    for line in cleaned.splitlines():
        h = detect_section_heading(line, doc_type, metadata.issuer)
        if h:
            current_heading = h

    # For KIDs: tag performance scenario tables with block_type="scenario"
    def _block_type_for_table(heading: Optional[str]) -> str:
        if heading and "scenario" in heading:
            return "scenario"
        if heading and "cost" in heading:
            return "table"
        return "table"

    text_parts = split_text(cleaned, chunk_size, chunk_overlap)
    for chunk_text, local_offset in text_parts:
        if len(chunk_text.strip()) < 40:
            continue
        chunks.append(Chunk(
            chunk_id=make_chunk_id(metadata, "text", chunk_text),
            text=chunk_text,
            metadata=metadata,
            page_number=page_num,
            section_heading=current_heading,
            block_type="text",
            char_start=global_offset + local_offset,
            char_end=global_offset + local_offset + len(chunk_text),
        ))

    global_offset += len(cleaned)

    for tt in table_texts:
        bt = _block_type_for_table(current_heading)
        chunks.append(Chunk(
            chunk_id=make_chunk_id(metadata, "table", tt),
            text=tt,
            metadata=metadata,
            page_number=page_num,
            section_heading=current_heading,
            block_type=bt,
            char_start=global_offset,
            char_end=global_offset + len(tt),
        ))
        global_offset += len(tt)

    return chunks, current_heading, global_offset


# ── Key-facts summary chunk ────────────────────────────────────────────────────

def build_key_facts_chunk(metadata: DocumentMetadata) -> Chunk:
    """
    Synthesise a single structured chunk summarising all extracted key facts.
    Always prepended to the chunk list so it is the first retrieval candidate
    for Type 1 (factual) and Type 2 (comparative) queries.

    Factsheets and KIDs expose different fields — we only emit what was found.
    """
    kf = metadata.key_facts
    lines = [
        f"ETF: {metadata.etf_isin}",
        f"Issuer: {metadata.issuer}",
        f"Category: {metadata.category}",
        f"Document type: {metadata.doc_type}",
        f"Currency: {metadata.currency}",
        f"Share class: {metadata.share_class}",
        f"Year: {metadata.year}" + (f", {metadata.quarter}" if metadata.quarter else ""),
    ]
    if metadata.reference_date:
        lines.append(f"Reference date: {metadata.reference_date}")

    # Fields shared by both doc types
    common_fields = [
        ("isin",                "ISIN"),
        ("inception_date",      "Inception Date"),
        ("benchmark",           "Benchmark"),
        ("distribution_policy", "Distribution Policy"),
        ("replication_method",  "Replication Method"),
        ("domicile",            "Domicile"),
        ("currency",            "Currency"),
    ]
    # Factsheet-only fields
    factsheet_fields = [
        ("ter_ocf",             "TER / OCF"),
        ("aum",                 "AUM"),
        ("num_holdings",        "Number of Holdings"),
        ("tracking_difference", "Tracking Difference"),
        ("morningstar_rating",  "Morningstar Rating"),
    ]
    # KID-only fields
    kid_fields = [
        ("cost_entry",                 "Entry Cost"),
        ("cost_exit",                  "Exit Cost"),
        ("cost_ongoing",               "Ongoing Cost"),
        ("cost_performance",           "Performance Fee"),
        ("cost_transaction",           "Transaction Cost"),
        ("srri",                       "Summary Risk Indicator (1–7)"),
        ("recommended_holding_period", "Recommended Holding Period"),
        ("target_market",              "Target Market"),
    ]

    active_extra = kid_fields if metadata.doc_type == "kid" else factsheet_fields
    for attr, label in common_fields + active_extra:
        val = getattr(kf, attr, None)
        if val:
            lines.append(f"{label}: {val}")

    text = "\n".join(lines)
    return Chunk(
        chunk_id=hashlib.sha256(text.encode()).hexdigest()[:16],
        text=text,
        metadata=metadata,
        page_number=None,
        section_heading="key facts summary",
        block_type="key_facts",
    )


# ── Main ingestion pipeline ────────────────────────────────────────────────────

def ingest_document(
    pdf_path: Path,
    metadata: DocumentMetadata,
    deduplicate: bool = True,
) -> list[Chunk]:
    """
    Unified ingestion pipeline for factsheets and KIDs.
    doc_type in metadata drives all branching internally.

      1. Full-text pass → extract key facts (regex)
      2. Page-by-page → tables + text, section heading tracking
      3. Prepend key-facts summary chunk
      4. Deduplicate

    Returns a list of Chunk objects ready for embedding.
    """
    doc_type = metadata.doc_type
    print(f"\n[ingest:{doc_type}] {pdf_path.name}")
    chunks: list[Chunk] = []

    with pdfplumber.open(pdf_path) as pdf:
        metadata.total_pages = len(pdf.pages)
        full_text = "\n".join((p.extract_text() or "") for p in pdf.pages)

        metadata.key_facts = extract_key_facts(full_text, doc_type)

        # For KIDs: try to find a reference date ("as of DD/MM/YYYY" or "Date: ...")
        if doc_type == "kid":
            ref_match = re.search(
                r"(?:as of|as at|dated?)[:\s]+(\d{1,2}[\/\.\-]\d{1,2}[\/\.\-]\d{2,4})",
                full_text, re.IGNORECASE
            )
            if ref_match:
                metadata.reference_date = ref_match.group(1).strip()

        # Print extraction summary
        print(f"  Pages           : {metadata.total_pages}")
        print(f"  ISIN            : {metadata.key_facts.isin or '—'}")
        if doc_type == "factsheet":
            print(f"  TER/OCF         : {metadata.key_facts.ter_ocf or '—'}")
            print(f"  AUM             : {metadata.key_facts.aum or '—'}")
        else:
            print(f"  Ongoing cost    : {metadata.key_facts.cost_ongoing or '—'}")
            print(f"  SRI             : {metadata.key_facts.srri or '—'}")
            print(f"  Reference date  : {metadata.reference_date or '—'}")
        print(f"  Benchmark       : {metadata.key_facts.benchmark or '—'}")

        # Prepend key-facts summary chunk
        chunks.append(build_key_facts_chunk(metadata))

        # Page-by-page extraction
        current_heading: Optional[str] = None
        global_offset = 0
        for page_num, page in enumerate(pdf.pages, start=1):
            page_chunks, current_heading, global_offset = extract_page_blocks(
                page, page_num, metadata, current_heading, global_offset
            )
            chunks.extend(page_chunks)

    print(f"  Chunks (raw)    : {len(chunks)}")
    if deduplicate:
        chunks = deduplicate_chunks(chunks)
    print(f"  Chunks (final)  : {len(chunks)}")
    return chunks


# ── Serialisation ──────────────────────────────────────────────────────────────

def save_chunks(chunks: list[Chunk], output_path: Path) -> None:
    records = [
        {
            "chunk_id":        c.chunk_id,
            "text":            c.text,
            "page_number":     c.page_number,
            "section_heading": c.section_heading,
            "block_type":      c.block_type,
            "char_start":      c.char_start,
            "char_end":        c.char_end,
            "metadata":        asdict(c.metadata),
        }
        for c in chunks
    ]
    output_path.write_text(json.dumps(records, indent=2, ensure_ascii=False))
    print(f"  Saved → {output_path}")


# ── Metadata config loading ────────────────────────────────────────────────────

def load_config(config_path: Path) -> dict:
    """
    Load the metadata JSON config.
    Accepts both "type" and "doc_type" as the document type key for flexibility.
    """
    raw = json.loads(config_path.read_text())
    config = {}
    for filename, entry in raw.items():
        # Normalise: support both "type" (your current JSON) and "doc_type"
        doc_type = entry.get("doc_type") or entry.get("type", "factsheet")
        config[filename] = {**entry, "doc_type": doc_type}
    return config


def metadata_from_config_entry(entry: dict, filename: str) -> DocumentMetadata:
    return DocumentMetadata(
        etf_isin=entry["isin"].upper(),
        issuer=entry["issuer"].lower(),
        category=entry["category"],
        year=int(entry["year"]),
        doc_type=entry["doc_type"],
        currency=entry.get("currency", "USD"),
        share_class=entry.get("share_class", "acc"),
        quarter=entry.get("quarter"),
        source_file=filename,
    )


# ── Output filename ────────────────────────────────────────────────────────────

def build_output_stem(meta: DocumentMetadata) -> str:
    parts = [meta.etf_isin, str(meta.year)]
    if meta.quarter:
        parts.append(meta.quarter)
    parts.append(meta.doc_type)
    return "_".join(parts)


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Ingest ETF factsheets and KIDs (iShares, Xtrackers, Amundi, UBS)"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--pdf",   type=Path, help="Single PDF to process")
    g.add_argument("--batch", type=Path, help="Folder of PDFs to process")

    p.add_argument("--isin",      help="ETF isin, e.g. EUNL")
    p.add_argument("--issuer",      choices=list(SUPPORTED_ISSUERS))
    p.add_argument("--category",    help="e.g. 'MSCI World'")
    p.add_argument("--year",        type=int)
    p.add_argument("--type",        dest="doc_type", default="factsheet",
                   choices=list(SUPPORTED_TYPES), help="Document type")
    p.add_argument("--currency",    default="USD")
    p.add_argument("--share_class", default="acc")
    p.add_argument("--quarter",     default=None)
    p.add_argument("--output_dir",  type=Path, default=Path("data/processed"))
    p.add_argument("--config",      type=Path,
                   help="JSON config mapping filenames to metadata (required for --batch)")
    p.add_argument("--no_dedup",    action="store_true")
    return p


def main():
    args = build_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.pdf:
        meta = DocumentMetadata(
            etf_isin=args.isin.upper(),
            issuer=args.issuer.lower(),
            category=args.category,
            year=args.year,
            doc_type=args.doc_type,
            currency=args.currency,
            share_class=args.share_class,
            quarter=args.quarter,
            source_file=args.pdf.name,
        )
        chunks = ingest_document(args.pdf, meta, deduplicate=not args.no_dedup)
        out = args.output_dir / f"{build_output_stem(meta)}.json"
        save_chunks(chunks, out)

    else:
        if not args.config:
            raise SystemExit("--config is required for --batch mode")
        config = load_config(args.config)
        pdf_files = sorted(args.batch.glob("*.pdf"))
        if not pdf_files:
            print(f"No PDFs found in {args.batch}")
            return

        skipped = []
        for pdf_path in pdf_files:
            fname = pdf_path.name
            if fname not in config:
                print(f"  [skip] No config entry for: {fname}")
                skipped.append(fname)
                continue
            entry = config[fname]
            meta = metadata_from_config_entry(entry, fname)
            chunks = ingest_document(pdf_path, meta, deduplicate=not args.no_dedup)
            out = args.output_dir / f"{build_output_stem(meta)}.json"
            save_chunks(chunks, out)

        if skipped:
            print(f"\n[warn] {len(skipped)} files skipped (no config entry):")
            for f in skipped:
                print(f"  {f}")


if __name__ == "__main__":
    main()