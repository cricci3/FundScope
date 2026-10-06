"""
retrieve.py — Semantic search + metadata filtering over ChromaDB
ETF RAG Project — Phase 3

Three retrieval modes, matching the four query types in ground_truth.json:

  single(query, filters)
    → Type 1 (factual): one ETF, one doc_type, top-k chunks
    → Filters on etf_isin + doc_type + year (any combination)

  comparative(query, isin_list, filters)
    → Type 2 (comparative): retrieves top-k per ETF independently,
      then merges. This avoids one ETF dominating the result list.

  cross_document(query, etf_isin, filters)
    → T2_006–T2_009: same ETF, multiple doc_types (factsheet + KID)
      Retrieves top-k from each doc_type separately, then merges.

All three modes return a list of RetrievedChunk objects with score,
source attribution, and the full chunk text ready for the LLM prompt.

Each search is hybrid by default (config.HYBRID_SEARCH): the vector ranking
and a BM25 ranking of the same chunks are fused with Reciprocal Rank Fusion,
which helps on ISINs, acronyms (TER, OCF, SRI) and figures. `score` stays the
cosine similarity; apply_min_score() drops chunks below the per-model
threshold (config.MIN_SIMILARITY) — the agent uses it and tells the model.
The index records its embedding model; querying it with another model fails.

Usage (Python API — called from pipeline.py):
    from retrieve import Retriever

    retriever = Retriever()   # defaults to config.INDEX_PATH

    # Type 1 — factual
    results = retriever.single(
        query="What is the TER of IE00B4L5Y983?",
        filters={"etf_isin": "IE00B4L5Y983", "doc_type": "factsheet"},
        k=5,
    )

    # Type 2 — comparative
    results = retriever.comparative(
        query="Compare the ongoing charges of IE00B4L5Y983 and IE00BD4TXV59",
        isin_list=["IE00B4L5Y983", "IE00BD4TXV59"],
        filters={"doc_type": "factsheet"},
        k_per_etf=3,
    )

    # Cross-document (factsheet + KID, same ETF)
    results = retriever.cross_document(
        query="Does the KID cost figure match the factsheet OCF for IE00B4L5Y983?",
        etf_isin="IE00B4L5Y983",
        doc_types=["factsheet", "kid"],
        k_per_doc_type=3,
    )

Usage (CLI — useful for manual inspection during development):
    python src/retrieve.py --query "What is the TER of IE00B4L5Y983?" \
                       --etf_isin IE00B4L5Y983 --doc_type factsheet

    python src/retrieve.py --query "Compare ongoing charges" \
                       --isin_list IE00B4L5Y983 IE00BD4TXV59 \
                       --doc_type factsheet --mode comparative

Install:
    pip install chromadb sentence-transformers
"""

import re
import json
import argparse
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional, Union

import numpy as np

try:
    import chromadb
except ImportError:
    raise ImportError("pip install chromadb")

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("uv add rank_bm25")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")


from config import (COLLECTION_NAME, EMBEDDING_MODEL, HYBRID_SEARCH, INDEX_PATH,
                    embedding_prefixes, min_similarity)


# ── Result type ────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float            # cosine similarity (0–1, higher = more similar)
    etf_isin: str
    etf_ticker: str
    issuer: str
    doc_type: str
    year: int
    quarter: str
    section_heading: str
    block_type: str
    page_number: int
    source_file: str
    fused: Optional[float] = None   # hybrid search: Reciprocal Rank Fusion score (ranks results)

    def as_context_string(self) -> str:
        """
        Format for inclusion in an LLM prompt.
        Source attribution is embedded directly so the LLM can cite it.
        """
        source = (
            f"[{self.etf_isin} | {self.issuer} | {self.doc_type} | "
            f"{self.year}{' ' + self.quarter if self.quarter else ''} | "
            f"section: {self.section_heading or 'unknown'} | "
            f"type: {self.block_type}]"
        )
        return f"{source}\n{self.text}"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_where(filters: dict) -> Optional[dict]:
    """
    Convert a plain {field: value} dict into a ChromaDB $where clause.
    Handles single-field and multi-field cases.
    Ignores None values.

    ChromaDB $where syntax:
      Single:  {"field": {"$eq": value}}
      Multi:   {"$and": [{"f1": {"$eq": v1}}, {"f2": {"$eq": v2}}]}
    """
    clean = {k: v for k, v in filters.items() if v is not None and v != ""}
    if not clean:
        return None
    if len(clean) == 1:
        k, v = next(iter(clean.items()))
        return {k: {"$eq": v}}
    return {"$and": [{k: {"$eq": v}} for k, v in clean.items()]}


def _make_chunk(cid: str, doc: str, meta: dict, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=doc,
        score=round(score, 4),
        etf_isin=       meta.get("etf_isin",        ""),
        etf_ticker=     meta.get("etf_ticker",       ""),
        issuer=         meta.get("issuer",           ""),
        doc_type=       meta.get("doc_type",         ""),
        year=           int(meta.get("year",         0)),
        quarter=        meta.get("quarter",          ""),
        section_heading=meta.get("section_heading",  ""),
        block_type=     meta.get("block_type",       "text"),
        page_number=    int(meta.get("page_number",  -1)),
        source_file=    meta.get("source_file",      ""),
    )


def _parse_results(
    query_result: dict,
    query: str = "",
) -> list[RetrievedChunk]:
    """
    Parse a ChromaDB query() response into RetrievedChunk objects.
    ChromaDB returns parallel lists: ids[0], documents[0], metadatas[0], distances[0].
    """
    ids       = query_result["ids"][0]
    docs      = query_result["documents"][0]
    metas     = query_result["metadatas"][0]
    distances = query_result["distances"][0]   # cosine distance (0 = identical)

    # distance → similarity
    return [_make_chunk(cid, doc, meta, 1.0 - dist)
            for cid, doc, meta, dist in zip(ids, docs, metas, distances)]


# ── Lexical search (BM25) ──────────────────────────────────────────────────────
# Embeddings are weak on exact tokens: ISINs, acronyms (TER, OCF, SRI, KID) and
# figures. BM25 over the same chunks catches them; the two rankings are merged
# with Reciprocal Rank Fusion (RRF), which needs only ranks, not comparable scores.

RRF_K          = 60     # standard RRF constant: score = Σ 1 / (RRF_K + rank)
HYBRID_POOL    = 20     # candidates taken from each ranking before fusion
BM25_WEIGHT    = 0.5    # weight of the BM25 ranking in the fusion (vector ranking: 1.0;
                        # task 5.3: 0.5 beat 1.0, which let generic tokens push out vector hits)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,]\d+)?%?")


def tokenize(text: str) -> list[str]:
    """Lowercase word/number tokens; keeps 0.20% and IE00B4L5Y983 whole."""
    return _TOKEN_RE.findall(text.lower())


def _matches(meta: dict, filters: dict) -> bool:
    """Python-side equivalent of the _build_where equality filter."""
    return all(meta.get(k) == v for k, v in filters.items() if v is not None and v != "")


def rrf_fuse(rankings: list[list[str]], weights: Optional[list[float]] = None,
             k: int = RRF_K) -> dict[str, float]:
    """(Weighted) Reciprocal Rank Fusion of ranked id lists → {id: fused score}."""
    weights = weights or [1.0] * len(rankings)
    fused: dict[str, float] = {}
    for ranking, w in zip(rankings, weights):
        for rank, cid in enumerate(ranking, 1):
            fused[cid] = fused.get(cid, 0.0) + w / (k + rank)
    return fused


def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Remove duplicate chunk_ids, keeping the highest-scored copy."""
    seen: dict[str, RetrievedChunk] = {}
    for c in chunks:
        if c.chunk_id not in seen or c.score > seen[c.chunk_id].score:
            seen[c.chunk_id] = c
    return list(seen.values())


def apply_min_score(chunks: list[RetrievedChunk], min_score: float,
                    query: str = "") -> tuple[list[RetrievedChunk], list[RetrievedChunk]]:
    """
    Split chunks into (kept, dropped): a chunk is dropped when its cosine
    similarity is below `min_score` — unless it contains a token of the query
    that has a digit (an ISIN, a figure), an exact match that embeddings rank poorly.
    """
    identifiers = {t for t in tokenize(query) if any(ch.isdigit() for ch in t)}
    kept, dropped = [], []
    for c in chunks:
        exact = bool(identifiers & set(tokenize(c.text)))
        (kept if c.score >= min_score or exact else dropped).append(c)
    return kept, dropped


def _sort_by_score(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Fused (hybrid) score when present, else cosine similarity."""
    return sorted(chunks, key=lambda c: c.fused if c.fused is not None else c.score, reverse=True)


# ── Retriever class ────────────────────────────────────────────────────────────

class Retriever:
    """
    Wraps a ChromaDB collection + embedding model.
    Instantiate once, call many times (the model stays loaded in memory).
    """

    def __init__(
        self,
        db_path: Union[str, Path] = INDEX_PATH,
        model_name: str = EMBEDDING_MODEL,
        collection_name: str = COLLECTION_NAME,
        hybrid: bool = HYBRID_SEARCH,
    ):
        self._db_path    = Path(db_path)
        self._model_name = model_name
        self.hybrid      = hybrid

        print(f"[retrieve] Loading ChromaDB from {self._db_path}")
        self._client = chromadb.PersistentClient(path=str(self._db_path))

        try:
            self._collection = self._client.get_collection(collection_name)
        except Exception:
            raise RuntimeError(
                f"\n  Collection '{collection_name}' not found in ChromaDB.\n"
                f"  The index has not been built yet.\n\n"
                f"  Run setup first:\n"
                f"      python setup.py\n\n"
                f"  Or rebuild manually:\n"
                f"      python src/embed.py --rebuild\n"
            )

        count = self._collection.count()
        if count == 0:
            raise RuntimeError(
                f"\n  Collection '{collection_name}' exists but contains no chunks.\n"
                f"  Ingestion may have failed or produced no output.\n\n"
                f"  Check that data/processed/ contains .json files, then run:\n"
                f"      python setup.py --rebuild\n"
            )

        print(f"[retrieve] Collection '{collection_name}' — {count} chunks")

        # Query and passage vectors must come from the same model
        indexed_with = (self._collection.metadata or {}).get("embedding_model")
        if indexed_with != model_name:
            raise RuntimeError(
                f"\n  The index was built with embedding model {indexed_with!r}, "
                f"but queries would use {model_name!r}.\n"
                f"  Rebuild the index with the configured model:\n"
                f"      python setup.py --rebuild\n"
            )

        print(f"[retrieve] Loading model: {model_name}")
        self._model = SentenceTransformer(model_name)
        self._query_prefix = embedding_prefixes(model_name)[0]
        # Callers (the agent) drop chunks below this similarity: see apply_min_score
        self.min_score = min_similarity(model_name)

        # The whole corpus in memory for BM25 (a few hundred chunks at most) and
        # its vectors, to give lexical-only hits a cosine similarity too
        if self.hybrid:
            corpus = self._collection.get(include=["documents", "metadatas", "embeddings"])
            self._ids   = corpus["ids"]
            self._docs  = corpus["documents"]
            self._metas = corpus["metadatas"]
            self._vecs  = np.asarray(corpus["embeddings"], dtype=np.float32)
            self._bm25  = BM25Okapi([tokenize(d) for d in self._docs])
            print(f"[retrieve] Hybrid search: BM25 + vectors (RRF)")

    def _embed_query(self, query: str) -> list[float]:
        vec = self._model.encode(
            self._query_prefix + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return vec.tolist()

    def _query(self, query: str, q_vec: list[float], filters: dict, k: int) -> list[RetrievedChunk]:
        """
        Top-k chunks matching `filters` (equality on metadata fields). Vector
        search only, or — hybrid — vector and BM25 rankings fused with RRF.
        `score` is always the cosine similarity of the chunk to the query.
        """
        where = _build_where(filters)
        kwargs = dict(
            query_embeddings=[q_vec],
            n_results=max(k, HYBRID_POOL) if self.hybrid else k,
            include=["documents", "metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where
        vector_hits = _parse_results(self._collection.query(**kwargs), query)
        if not self.hybrid:
            return vector_hits

        candidates = [i for i, m in enumerate(self._metas) if _matches(m, filters)]
        bm25 = self._bm25.get_scores(tokenize(query))
        lexical = sorted((i for i in candidates if bm25[i] > 0), key=lambda i: -bm25[i])
        lexical_ids = [self._ids[i] for i in lexical[:HYBRID_POOL]]

        fused = rrf_fuse([[c.chunk_id for c in vector_hits], lexical_ids], [1.0, BM25_WEIGHT])
        by_id = {c.chunk_id: c for c in vector_hits}
        q = np.asarray(q_vec, dtype=np.float32)
        for i in lexical[:HYBRID_POOL]:
            cid = self._ids[i]
            if cid not in by_id:
                by_id[cid] = _make_chunk(cid, self._docs[i], self._metas[i],
                                         float(self._vecs[i] @ q))
        for cid, c in by_id.items():
            c.fused = round(fused[cid], 6)
        return _sort_by_score(list(by_id.values()))[:k]

    # ── Public retrieval methods ───────────────────────────────────────────────

    def single(
        self,
        query: str,
        filters: Optional[dict] = None,
        k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Semantic search with optional metadata filtering.
        Use for Type 1 (factual) queries and single-ETF Type 3 (temporal).

        filters: any subset of {etf_isin, doc_type, year, issuer,
                                 block_type, section_heading}
        """
        return self._query(query, self._embed_query(query), filters or {}, k)

    def comparative(
        self,
        query: str,
        isin_list: list[str],
        filters: Optional[dict] = None,
        k_per_etf: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-k chunks per ETF independently, then merge and re-rank.
        Use for Type 2 (comparative) queries across multiple ETFs.

        This prevents one ETF from monopolising the result list when its
        documents happen to match the query embedding more closely.

        filters: applied to ALL ETFs (e.g. {"doc_type": "factsheet"})
        """
        all_chunks: list[RetrievedChunk] = []
        q_vec = self._embed_query(query)

        for isin in isin_list:
            etf_filter = {"etf_isin": isin, **(filters or {})}
            try:
                all_chunks.extend(self._query(query, q_vec, etf_filter, k_per_etf))
            except Exception as e:
                print(f"  [warn] No results for {isin}: {e}")

        return _sort_by_score(_deduplicate(all_chunks))

    def cross_document(
        self,
        query: str,
        etf_isin: str,
        doc_types: Optional[list[str]] = None,
        filters: Optional[dict] = None,
        k_per_doc_type: int = 3,
    ) -> list[RetrievedChunk]:
        """
        Retrieve top-k chunks per doc_type for the same ETF, then merge.
        Use for T2_006–T2_009 (cross-document consistency queries):
        same ETF, comparing factsheet vs KID.

        doc_types defaults to ["factsheet", "kid"].
        """
        doc_types = doc_types or ["factsheet", "kid"]
        all_chunks: list[RetrievedChunk] = []
        q_vec = self._embed_query(query)

        for dt in doc_types:
            dt_filter = {"etf_isin": etf_isin, "doc_type": dt, **(filters or {})}
            try:
                all_chunks.extend(self._query(query, q_vec, dt_filter, k_per_doc_type))
            except Exception as e:
                print(f"  [warn] No results for {etf_isin}/{dt}: {e}")

        return _sort_by_score(_deduplicate(all_chunks))

    def key_facts(self, etf_isin: str, doc_type: str, year: int) -> Optional[RetrievedChunk]:
        """
        Fetch the key_facts summary chunk for a specific ETF document.
        This is a direct lookup by metadata — no embedding needed.
        Useful as a fast first step before semantic search.
        """
        where = _build_where({
            "etf_isin":   etf_isin,
            "doc_type":   doc_type,
            "year":       year,
            "block_type": "key_facts",
        })
        try:
            result = self._collection.query(
                query_embeddings=[self._embed_query(etf_isin)],
                n_results=1,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
            chunks = _parse_results(result)
            return chunks[0] if chunks else None
        except Exception:
            return None

    # ── Utility ────────────────────────────────────────────────────────────────

    def count(self) -> int:
        return self._collection.count()

    def collection_stats(self) -> dict:
        """Return a breakdown of what's in the collection."""
        from collections import Counter
        result = self._collection.get(include=["metadatas"])
        metas  = result["metadatas"]
        return {
            "total":      len(metas),
            "by_doc_type": dict(Counter(m.get("doc_type",  "?") for m in metas)),
            "by_isin":     dict(Counter(m.get("etf_isin",  "?") for m in metas)),
            "by_issuer":   dict(Counter(m.get("issuer",    "?") for m in metas)),
            "by_block":    dict(Counter(m.get("block_type","?") for m in metas)),
        }


# ── Query routing helper ───────────────────────────────────────────────────────
# Called by pipeline.py to auto-select the right retrieval mode.

def route_query(
    retriever: Retriever,
    query: str,
    metadata_filter_hint: Optional[dict] = None,
    isin_list: Optional[list[str]] = None,
    mode: Optional[str] = None,
    k: int = 5,
) -> list[RetrievedChunk]:
    """
    Route a query to the appropriate retrieval method based on mode or hints.

    mode:
      "single"       → retriever.single()
      "comparative"  → retriever.comparative()  (requires isin_list)
      "cross_doc"    → retriever.cross_document() (requires single isin + both doc_types)
      None           → auto-detect from isin_list length and filter hints

    metadata_filter_hint: from ground_truth.json, used as ChromaDB filters.
    """
    hint = metadata_filter_hint or {}

    # Auto-detect mode if not specified
    if mode is None:
        if isin_list and len(isin_list) > 1:
            mode = "comparative"
        elif isin_list and len(isin_list) == 1 and "doc_type" not in hint:
            mode = "cross_doc"
        else:
            mode = "single"

    if mode == "comparative":
        if not isin_list:
            raise ValueError("comparative mode requires isin_list")
        # Strip etf_isin from hint since comparative sets it per-ETF
        shared_filters = {k: v for k, v in hint.items() if k != "etf_isin"}
        return retriever.comparative(
            query=query,
            isin_list=isin_list,
            filters=shared_filters,
            k_per_etf=max(3, k // len(isin_list)),
        )

    if mode == "cross_doc":
        etf_isin = (isin_list or [hint.get("etf_isin", "")])[0]
        if not etf_isin:
            raise ValueError("cross_doc mode requires etf_isin")
        year = int(hint.get("year", 0)) or None
        extra = {"year": year} if year else {}
        return retriever.cross_document(
            query=query,
            etf_isin=etf_isin,
            k_per_doc_type=max(2, k // 2),
            filters=extra,
        )

    # Default: single
    return retriever.single(query=query, filters=hint, k=k)


# ── CLI (for manual inspection) ───────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Query the ETF ChromaDB index")
    p.add_argument("--query",      required=True, help="Natural language query")
    p.add_argument("--db_path",    type=Path, default=INDEX_PATH)
    p.add_argument("--mode",       default=None,
                   choices=["single", "comparative", "cross_doc"],
                   help="Retrieval mode (auto-detected if omitted)")
    p.add_argument("--etf_isin",   default=None, help="Single ETF ISIN filter")
    p.add_argument("--isin_list",  nargs="+",    help="Multiple ISINs for comparative mode")
    p.add_argument("--doc_type",   default=None, choices=["factsheet", "kid"])
    p.add_argument("--year",       type=int, default=None)
    p.add_argument("--k",          type=int, default=5, help="Chunks to retrieve")
    p.add_argument("--json",       action="store_true", help="Output results as JSON")
    p.add_argument("--vector_only", action="store_true",
                   help="Embeddings only, no BM25 fusion (default: config.HYBRID_SEARCH)")
    return p


def main():
    args = build_parser().parse_args()

    retriever = Retriever(db_path=args.db_path, hybrid=HYBRID_SEARCH and not args.vector_only)

    filters: dict = {}
    if args.etf_isin:  filters["etf_isin"]  = args.etf_isin
    if args.doc_type:  filters["doc_type"]  = args.doc_type
    if args.year:      filters["year"]       = args.year

    isin_list = args.isin_list or ([args.etf_isin] if args.etf_isin else None)

    results = route_query(
        retriever=retriever,
        query=args.query,
        metadata_filter_hint=filters,
        isin_list=isin_list,
        mode=args.mode,
        k=args.k,
    )

    if args.json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return

    print(f"\n{'═' * 64}")
    print(f"  Query : {args.query}")
    print(f"  Mode  : {args.mode or 'auto'}")
    print(f"  Hits  : {len(results)}")
    print(f"{'═' * 64}")
    for i, r in enumerate(results, 1):
        print(f"\n  [{i}] score={r.score:.4f}  "
              f"{r.etf_isin} | {r.doc_type} | {r.year} | "
              f"{r.section_heading or '—'} | {r.block_type}")
        # Print first 200 chars of text as preview
        preview = r.text[:200].replace("\n", " ")
        print(f"      {preview}{'…' if len(r.text) > 200 else ''}")
    print()


if __name__ == "__main__":
    main()
