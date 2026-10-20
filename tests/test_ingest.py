"""PDF → chunks: a generated sample PDF, plus the real corpus when data/raw/ is present."""

import json

import pytest

from fundscope.config import DATA_PROCESSED, DATA_RAW, METADATA_PATH
from fundscope.ingest import (build_output_stem, ingest_document, load_config,
                              metadata_from_config_entry)


def make_pdf(pages: list[list[str]]) -> bytes:
    """Minimal text-only PDF (Helvetica, one line per string) — no PDF library needed."""
    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")

    n_pages = len(pages)
    font_id = 3 + 2 * n_pages
    objects = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [" + " ".join(f"{3 + 2 * i} 0 R" for i in range(n_pages))
        + f"] /Count {n_pages} >>",
    ]
    for i, lines in enumerate(pages):
        stream = "BT /F1 10 Tf 14 TL 50 800 Td " + " ".join(f"({esc(line)}) Tj T*" for line in lines) + " ET"
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] "
                       f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {4 + 2 * i} 0 R >>")
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out, offsets = "%PDF-1.4\n", []
    for n, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += f"{n} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")


FACTSHEET_PAGES = [
    ["FUND OVERVIEW",
     "ISIN: IE00B4L5Y983",
     "Total Expense Ratio : 0.20%",
     "Benchmark : MSCI World Index",
     "Replication Method : Physical (Optimised)",
     *[f"The fund invests in large and mid cap companies, sentence {i}." for i in range(30)]],
    ["TOP HOLDINGS",
     *[f"Holding {i}: company number {i} with weight 0.{i}%" for i in range(40)]],
]


@pytest.fixture
def sample_factsheet(tmp_path):
    path = tmp_path / "sample_factsheet.pdf"
    path.write_bytes(make_pdf(FACTSHEET_PAGES))
    entry = {"isin": "IE00B4L5Y983", "issuer": "ishares", "category": "MSCI World",
             "doc_type": "factsheet", "year": 2026, "quarter": "Q4"}
    return path, metadata_from_config_entry(entry, path.name)


def test_ingest_sample_pdf(sample_factsheet):
    path, meta = sample_factsheet
    chunks = ingest_document(path, meta)

    assert meta.total_pages == 2
    assert meta.key_facts.isin == "IE00B4L5Y983"
    assert meta.key_facts.ter_ocf == "0.20%"
    assert meta.key_facts.replication_method == "Physical (Optimised)"

    key_facts = chunks[0]
    assert key_facts.block_type == "key_facts"
    assert "0.20%" in key_facts.text

    text_chunks = chunks[1:]
    assert len(text_chunks) >= 3                       # ~3.5k characters → several 1000-char chunks
    assert all(len(c.text) <= 1000 for c in text_chunks)
    assert {c.page_number for c in text_chunks} == {1, 2}
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    assert all(c.metadata.etf_isin == "IE00B4L5Y983" and c.metadata.doc_type == "factsheet"
               for c in chunks)
    assert "top holdings" in {c.section_heading for c in text_chunks}


def test_ingest_is_deterministic(sample_factsheet):
    path, meta = sample_factsheet
    first = [(c.chunk_id, c.text) for c in ingest_document(path, meta)]
    assert first == [(c.chunk_id, c.text) for c in ingest_document(path, meta)]


_CORPUS = [(name, entry) for name, entry in load_config(METADATA_PATH).items()
           if (DATA_RAW / name).exists()]


@pytest.mark.skipif(not _CORPUS, reason="corpus PDFs not in data/raw/ (they are not versioned)")
@pytest.mark.parametrize("filename,entry", _CORPUS, ids=[name for name, _ in _CORPUS])
def test_ingest_real_document_matches_processed(filename, entry):
    meta = metadata_from_config_entry(entry, filename)
    processed = DATA_PROCESSED / f"{build_output_stem(meta)}.json"
    if not processed.exists():
        pytest.skip(f"{processed.name} not built")
    chunks = ingest_document(DATA_RAW / filename, meta)
    expected = json.loads(processed.read_text(encoding="utf-8"))
    assert [c.chunk_id for c in chunks] == [c["chunk_id"] for c in expected]
