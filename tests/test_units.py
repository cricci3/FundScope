"""Pure functions: filters, citations, chunking, ids, fusion."""

from dataclasses import dataclass

from fundscope.citations import ChunkBook, parse_citations, resolve_citations
from fundscope.ingest import (DocumentMetadata, build_output_stem, compute_content_hash,
                              make_chunk_id, split_text)
from fundscope.retrieve import _build_where, _matches, apply_min_score, rrf_fuse, tokenize


# ── _build_where ───────────────────────────────────────────────────────────────

def test_build_where_empty_and_none_values():
    assert _build_where({}) is None
    assert _build_where({"doc_type": None, "year": ""}) is None


def test_build_where_single_field():
    assert _build_where({"etf_isin": "IE00B4L5Y983"}) == {"etf_isin": {"$eq": "IE00B4L5Y983"}}


def test_build_where_multi_field_drops_none():
    where = _build_where({"etf_isin": "IE00B4L5Y983", "doc_type": "kid", "year": None})
    assert where == {"$and": [{"etf_isin": {"$eq": "IE00B4L5Y983"}}, {"doc_type": {"$eq": "kid"}}]}


def test_matches_is_python_side_build_where():
    meta = {"etf_isin": "IE00B4L5Y983", "doc_type": "kid", "year": 2026}
    assert _matches(meta, {"doc_type": "kid", "year": None})
    assert not _matches(meta, {"doc_type": "factsheet"})


# ── Citations ──────────────────────────────────────────────────────────────────

def test_parse_citations_dedups_and_normalises():
    answer = ("TER 0.20% [IE00B4L5Y983 | iShares | factsheet | 2026 Q4]. SRI 4 "
              "[IE00B4L5Y983 | ishares | KID | 2026] and again [IE00B4L5Y983 | ishares | factsheet | 2026].")
    sources = parse_citations(answer)
    assert [(s.etf_isin, s.issuer, s.doc_type, s.year) for s in sources] == [
        ("IE00B4L5Y983", "ishares", "factsheet", 2026),
        ("IE00B4L5Y983", "ishares", "kid", 2026),
    ]


def test_parse_citations_ignores_other_brackets():
    assert parse_citations("See [Chunk 1] and [live data].") == []


@dataclass
class _C:
    chunk_id: str
    etf_isin: str
    issuer: str
    doc_type: str
    year: int = 2026


def test_resolve_citations_rewrites_tags_and_reports_unknown():
    book = ChunkBook()
    assert book.add(_C("a", "IE00B4L5Y983", "ishares", "factsheet")) == (1, True)
    assert book.add(_C("b", "IE00BD4TXV59", "ubs", "kid")) == (2, True)
    assert book.add(_C("a", "IE00B4L5Y983", "ishares", "factsheet")) == (1, False)

    resolved = resolve_citations("TER 0.20% [Chunk 1]. SRI 4 [Chunks 1-2]. X [Chunk 9].", book)
    assert resolved.text == ("TER 0.20% [IE00B4L5Y983 | ishares | factsheet | 2026]. SRI 4 "
                             "[IE00B4L5Y983 | ishares | factsheet | 2026][IE00BD4TXV59 | ubs | kid | 2026]. X.")
    assert resolved.cited_chunks == [1, 2]
    assert resolved.unknown_refs == [9]
    assert len(resolved.cited_sources) == 2


# ── Chunking ───────────────────────────────────────────────────────────────────

def test_split_text_short_text_is_one_chunk():
    assert split_text("  short text  ", 100, 10) == [("short text", 0)]


def test_split_text_respects_size_and_overlaps():
    paragraphs = [f"Paragraph {i} " + "word " * 30 for i in range(10)]
    text = "\n\n".join(paragraphs)
    chunks = split_text(text, 400, 60)
    assert len(chunks) > 1
    assert all(len(c) <= 400 + 60 for c, _ in chunks)
    # every paragraph survives somewhere
    for i in range(10):
        assert any(f"Paragraph {i} " in c for c, _ in chunks)
    # consecutive chunks overlap
    assert chunks[0][0][-20:] in chunks[1][0]


def test_split_text_falls_back_to_smaller_separators():
    text = "x" * 50 + " " + "y" * 50          # no paragraph or line breaks
    chunks = split_text(text, 60, 0)
    assert [c for c, _ in chunks] == ["x" * 50, "y" * 50]


# ── Ids and hashes ─────────────────────────────────────────────────────────────

def _meta(year=2026, quarter=None):
    return DocumentMetadata(etf_isin="IE00B4L5Y983", issuer="ishares", category="MSCI World",
                            year=year, doc_type="factsheet", currency="USD", share_class="acc", quarter=quarter,
                            source_file="ishares_factsheet.pdf")


def test_make_chunk_id_is_stable_and_year_independent():
    a = make_chunk_id(_meta(2026), "text", "TER 0.20%")
    assert a == make_chunk_id(_meta(2027, "Q1"), "text", "TER 0.20%")
    assert len(a) == 16
    assert a != make_chunk_id(_meta(), "table", "TER 0.20%")
    assert a != make_chunk_id(_meta(), "text", "TER 0.21%")


def test_content_hash_ignores_case_spacing_and_punctuation():
    assert compute_content_hash("TER:  0.20%\n") == compute_content_hash("ter 020")


def test_build_output_stem():
    assert build_output_stem(_meta()) == "IE00B4L5Y983_2026_factsheet"
    assert build_output_stem(_meta(quarter="Q4")) == "IE00B4L5Y983_2026_Q4_factsheet"


# ── Hybrid search helpers ──────────────────────────────────────────────────────

def test_tokenize_keeps_isins_and_percentages():
    assert tokenize("TER of IE00B4L5Y983 is 0.20%.") == ["ter", "of", "ie00b4l5y983", "is", "0.20%"]


def test_rrf_fuse_weights_rankings():
    fused = rrf_fuse([["a", "b"], ["b", "c"]], [1.0, 0.5], k=60)
    assert fused["b"] > fused["a"] > fused["c"]


@dataclass
class _Scored:
    text: str
    score: float
    block_type: str = "text"


def test_apply_min_score_keeps_key_facts_and_exact_identifiers():
    chunks = [_Scored("high", 0.9), _Scored("low", 0.1), _Scored("kf", 0.1, "key_facts"),
              _Scored("mentions IE00B4L5Y983", 0.1)]
    kept, dropped = apply_min_score(chunks, 0.5, query="TER of IE00B4L5Y983")
    assert [c.text for c in kept] == ["high", "kf", "mentions IE00B4L5Y983"]
    assert [c.text for c in dropped] == ["low"]
