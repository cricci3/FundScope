"""
Shared fixtures. No test touches the network or a real LLM: the embedding model
is replaced by a hashing embedder and the LLM by scripted clients.
"""

import hashlib
import re

import numpy as np
import pytest

from fundscope.llm import SESSION, Usage

ISHARES = "IE00B4L5Y983"
UBS     = "IE00BD4TXV59"
YEAR    = 2026
FAKE_EMBEDDING_MODEL = "fake-hashing-embedder"


# ── Embeddings ─────────────────────────────────────────────────────────────────

class HashingEmbedder:
    """Stand-in for SentenceTransformer: bag of hashed tokens, L2-normalised."""

    dim = 256
    max_seq_length = 512

    def __init__(self, model_name: str = FAKE_EMBEDDING_MODEL):
        self.model_name = model_name

    def get_embedding_dimension(self) -> int:
        return self.dim

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in re.findall(r"[a-z0-9]+", text.lower()):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm else vec

    def encode(self, texts, **kwargs):
        if isinstance(texts, str):
            return self._vector(texts)
        return np.stack([self._vector(t) for t in texts])


# ── Corpus ─────────────────────────────────────────────────────────────────────

def _chunk(cid, isin, issuer, doc_type, text, block_type="text", section=""):
    """A chunk record in the format written by ingest.save_chunks()."""
    return {
        "chunk_id": cid,
        "text": text,
        "page_number": 1,
        "section_heading": section,
        "block_type": block_type,
        "char_start": 0,
        "char_end": len(text),
        "metadata": {
            "etf_isin": isin, "issuer": issuer, "category": "MSCI World",
            "year": YEAR, "doc_type": doc_type, "quarter": "Q4" if doc_type == "factsheet" else None,
            "source_file": f"{issuer}_{doc_type}.pdf",
        },
    }


SAMPLE_CHUNKS = [
    _chunk("ish-fs-kf", ISHARES, "ishares", "factsheet",
           "Key facts\nTotal Expense Ratio: 0.20%\nReplication method: Physical (Optimised)",
           block_type="key_facts", section="key facts"),
    _chunk("ish-fs-1", ISHARES, "ishares", "factsheet",
           "Top holdings of the fund: Apple, Microsoft, Nvidia, Amazon.", section="top holdings"),
    _chunk("ish-kid-kf", ISHARES, "ishares", "kid",
           "Key facts\nSummary Risk Indicator: 4\nOngoing costs: 0.20%",
           block_type="key_facts", section="key facts"),
    _chunk("ish-kid-1", ISHARES, "ishares", "kid",
           "Recommended holding period: 5 years. The risk indicator assumes you keep the product.",
           section="how long should i hold it"),
    _chunk("ubs-fs-kf", UBS, "ubs", "factsheet",
           "Key facts\nTotal Expense Ratio: 0.06%\nReplication method: Physical (Full replication)",
           block_type="key_facts", section="key facts"),
    _chunk("ubs-fs-1", UBS, "ubs", "factsheet",
           "Top holdings of the fund: Apple, Microsoft, Nvidia, Alphabet.", section="top holdings"),
    _chunk("ubs-kid-kf", UBS, "ubs", "kid",
           "Key facts\nSummary Risk Indicator: 4\nOngoing costs: 0.06%",
           block_type="key_facts", section="key facts"),
    _chunk("ubs-kid-1", UBS, "ubs", "kid",
           "Recommended holding period: 5 years. Performance scenarios: stress, unfavourable.",
           section="performance scenarios"),
]


@pytest.fixture(scope="session")
def index_path(tmp_path_factory):
    """A ChromaDB index of SAMPLE_CHUNKS built with the hashing embedder."""
    from fundscope import embed

    path = tmp_path_factory.mktemp("chroma")
    collection = embed.get_collection(path, rebuild=True, model_name=FAKE_EMBEDDING_MODEL)
    summary = embed.index_chunks(SAMPLE_CHUNKS, collection, HashingEmbedder())
    assert summary["indexed"] == len(SAMPLE_CHUNKS)
    return path


@pytest.fixture(scope="session")
def retriever(index_path):
    from fundscope import retrieve

    mp = pytest.MonkeyPatch()
    mp.setattr(retrieve, "SentenceTransformer", HashingEmbedder)
    try:
        yield retrieve.Retriever(db_path=index_path, model_name=FAKE_EMBEDDING_MODEL)
    finally:
        mp.undo()


# ── LLM ────────────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def fresh_session(monkeypatch):
    """Every test starts with an empty cost tracker and no budget."""
    monkeypatch.setattr(SESSION, "usage", Usage())
    monkeypatch.setattr(SESSION, "cost_usd", 0.0)
    monkeypatch.setattr(SESSION, "calls", 0)
    monkeypatch.setattr(SESSION, "unpriced_calls", 0)
    monkeypatch.setattr(SESSION, "max_cost_usd", None)
    return SESSION
