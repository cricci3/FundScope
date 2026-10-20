"""End-to-end retrieval on a temporary index (hashing embedder, no LLM)."""

import pytest

from conftest import FAKE_EMBEDDING_MODEL, ISHARES, SAMPLE_CHUNKS, UBS, HashingEmbedder
from fundscope import embed, retrieve


def test_single_respects_filters(retriever):
    hits = retriever.single("Total Expense Ratio", filters={"etf_isin": UBS, "doc_type": "factsheet"}, k=5)
    assert hits and all(h.etf_isin == UBS and h.doc_type == "factsheet" for h in hits)
    assert hits[0].chunk_id == "ubs-fs-kf"
    assert hits[0].fused is not None                   # hybrid search ranked it
    assert 0.0 <= hits[0].score <= 1.0


def test_comparative_returns_every_fund(retriever):
    hits = retriever.comparative("top holdings", isin_list=[ISHARES, UBS],
                                 filters={"doc_type": "factsheet"}, k_per_etf=1)
    assert {h.etf_isin for h in hits} == {ISHARES, UBS}
    assert {h.chunk_id for h in hits} == {"ish-fs-1", "ubs-fs-1"}


def test_cross_document_returns_every_doc_type(retriever):
    hits = retriever.cross_document("recommended holding period", etf_isin=ISHARES, k_per_doc_type=1)
    assert {h.doc_type for h in hits} == {"factsheet", "kid"}
    assert all(h.etf_isin == ISHARES for h in hits)


def test_key_facts_lookup(retriever):
    kf = retriever.key_facts(UBS, "kid")
    assert kf.chunk_id == "ubs-kid-kf" and kf.block_type == "key_facts"
    assert retriever.key_facts(UBS, "kid", year=1999) is None


def test_route_query_auto_modes(retriever):
    both = retrieve.route_query(retriever, "ongoing costs", isin_list=[ISHARES, UBS], k=6)
    assert {h.etf_isin for h in both} == {ISHARES, UBS}
    one = retrieve.route_query(retriever, "ongoing costs", isin_list=[ISHARES], k=4)
    assert {h.doc_type for h in one} == {"factsheet", "kid"}


def test_reindexing_is_idempotent(index_path):
    collection = embed.get_collection(index_path, model_name=FAKE_EMBEDDING_MODEL)
    summary = embed.index_chunks(SAMPLE_CHUNKS, collection, HashingEmbedder())
    assert summary == {"indexed": 0, "skipped": len(SAMPLE_CHUNKS), "total": len(SAMPLE_CHUNKS)}


def test_retriever_refuses_another_embedding_model(index_path, monkeypatch):
    monkeypatch.setattr(retrieve, "SentenceTransformer", HashingEmbedder)
    with pytest.raises(RuntimeError, match="embedding model"):
        retrieve.Retriever(db_path=index_path, model_name="another-model")


def test_retriever_on_missing_index(tmp_path):
    with pytest.raises(RuntimeError, match="not found"):
        retrieve.Retriever(db_path=tmp_path / "empty")
