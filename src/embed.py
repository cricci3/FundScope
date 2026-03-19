"""
embed.py — Embedding + ChromaDB indexing pipeline
ETF RAG Project — Phase 3

Loads processed chunk JSON files from data/processed/, embeds each chunk
using sentence-transformers (all-MiniLM-L6-v2, CPU-native), and stores
vectors + metadata in a persisted ChromaDB collection.

Design decisions:
  - One ChromaDB collection for the entire corpus ("etf_chunks").
    Filtering by etf_isin / doc_type / year happens at query time in retrieve.py.
  - text is stored inline in ChromaDB (as a Document) so retrieve.py
    never has to go back to the JSON files.
  - Embedding is batched to avoid OOM on large corpora.
  - Re-running embed.py is idempotent: existing chunk_ids are skipped,
    so you can add new documents without rebuilding the entire index.

Usage:
    # Index all processed chunks
    python src/embed.py --input_dir data/processed/ --db_path index/chroma_db/

    # Force full rebuild (wipes existing collection first)
    python src/embed.py --input_dir data/processed/ --db_path index/chroma_db/ --rebuild

    # Dry run: show what would be indexed without writing
    python src/embed.py --input_dir data/processed/ --db_path index/chroma_db/ --dry_run

Install:
    pip install chromadb sentence-transformers
"""

import json
import argparse
from pathlib import Path

try:
    import chromadb
    from chromadb.config import Settings
except ImportError:
    raise ImportError("pip install chromadb")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    raise ImportError("pip install sentence-transformers")


# ── Constants ──────────────────────────────────────────────────────────────────

COLLECTION_NAME  = "etf_chunks"
EMBEDDING_MODEL  = "all-MiniLM-L6-v2"   # 384d, ~80 MB, CPU-native
EMBED_BATCH_SIZE = 64                    # safe for CPU; increase if you have more RAM


# ── ChromaDB metadata schema ───────────────────────────────────────────────────
# ChromaDB metadata values must be str | int | float | bool.
# We flatten the nested metadata dict from ingest.py into these top-level fields.
# These are the fields retrieve.py can filter on with a $where clause.

METADATA_FIELDS = [
    "etf_isin",        # e.g. "IE00B4L5Y983"
    "etf_ticker",      # e.g. "EUNL"  (kept for convenience)
    "issuer",          # "ishares" | "ubs" | "xtrackers" | "amundi"
    "category",        # "MSCI World" etc.
    "doc_type",        # "factsheet" | "kid"
    "year",            # int
    "quarter",         # "Q1"–"Q4" or "" (empty string, not None)
    "section_heading", # e.g. "key facts"
    "block_type",      # "text" | "table" | "key_facts" | "scenario"
    "page_number",     # int or -1 if unknown
    "source_file",     # original PDF filename
]


def _extract_metadata(chunk: dict) -> dict:
    """
    Flatten a chunk dict (as written by ingest.py save_chunks()) into
    a flat metadata dict suitable for ChromaDB.

    Handles both old-style flat chunks and the nested {"metadata": {...}} format.
    All values are coerced to str/int/float/bool — no None allowed in ChromaDB.
    """
    # Nested format from ingest.py
    meta = chunk.get("metadata", {})

    # key_facts is a nested sub-dict — we don't index it into ChromaDB metadata
    # (it's already embedded in the key_facts chunk text, which is searchable).
    # Flatten the rest.
    flat = {
        "etf_isin":        str(meta.get("etf_isin")   or chunk.get("etf_isin", "")),
        "etf_ticker":      str(meta.get("etf_ticker") or chunk.get("etf_ticker", "")),
        "issuer":          str(meta.get("issuer")      or chunk.get("issuer", "")),
        "category":        str(meta.get("category")    or chunk.get("category", "")),
        "doc_type":        str(meta.get("doc_type")    or meta.get("type") or chunk.get("doc_type", "")),
        "year":            int(meta.get("year")        or chunk.get("year", 0)),
        "quarter":         str(meta.get("quarter")     or chunk.get("quarter") or ""),
        "section_heading": str(chunk.get("section_heading") or ""),
        "block_type":      str(chunk.get("block_type")      or "text"),
        "page_number":     int(chunk.get("page_number") or -1),
        "source_file":     str(meta.get("source_file") or chunk.get("source_file", "")),
    }
    return flat


# ── Loader ────────────────────────────────────────────────────────────────────

def load_chunks(input_dir: Path) -> list[dict]:
    """
    Load all chunk JSON files from input_dir.
    Each file is a JSON array of chunk dicts (as written by save_chunks()).
    """
    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {input_dir}")

    all_chunks = []
    for path in json_files:
        chunks = json.loads(path.read_text(encoding="utf-8-sig", errors="replace"))
        if not isinstance(chunks, list):
            print(f"  [skip] {path.name} — expected a JSON array")
            continue
        all_chunks.extend(chunks)
        print(f"  Loaded {len(chunks):>4} chunks  ←  {path.name}")

    print(f"\n  Total chunks loaded: {len(all_chunks)}")
    return all_chunks


# ── ChromaDB client + collection ──────────────────────────────────────────────

def get_collection(
    db_path: Path,
    rebuild: bool = False,
) -> "chromadb.Collection":
    """
    Open (or create) a persistent ChromaDB collection.
    If rebuild=True, the existing collection is deleted first.
    """
    db_path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(db_path))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
            print(f"  [rebuild] Deleted existing collection '{COLLECTION_NAME}'")
        except Exception:
            pass  # didn't exist yet

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # ChromaDB will use its own internal embedding if we pass None here,
        # but we compute embeddings ourselves for full control.
        # embedding_function=None means we supply vectors directly.
        metadata={"hnsw:space": "cosine"},  # cosine similarity
    )
    return collection


# ── Embedding ─────────────────────────────────────────────────────────────────

def load_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    print(f"\n  Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")
    return model


def embed_texts(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = EMBED_BATCH_SIZE,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Embed a list of strings in batches.
    Returns a list of float vectors (one per text).
    """
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine similarity = dot product after L2 norm
    )
    return embeddings.tolist()


# ── Indexing ──────────────────────────────────────────────────────────────────

def index_chunks(
    chunks: list[dict],
    collection: "chromadb.Collection",
    model: SentenceTransformer,
    batch_size: int = EMBED_BATCH_SIZE,
    dry_run: bool = False,
) -> dict:
    """
    Embed and insert chunks into the ChromaDB collection.
    Skips chunks whose chunk_id already exists (idempotent).

    Returns a summary dict with counts.
    """
    # Find which chunk_ids are already in the collection
    existing_ids: set[str] = set()
    total_existing = collection.count()
    if total_existing > 0:
        # Fetch all existing IDs (ChromaDB returns them in pages of 10k max)
        result = collection.get(include=[])  # IDs only, no vectors
        existing_ids = set(result["ids"])
        print(f"  Collection already has {total_existing} chunks — will skip duplicates")

    # Filter to only new chunks
    new_chunks = [c for c in chunks if c.get("chunk_id", "") not in existing_ids]
    skipped    = len(chunks) - len(new_chunks)

    if skipped:
        print(f"  Skipping {skipped} already-indexed chunks")
    if not new_chunks:
        print("  Nothing new to index.")
        return {"indexed": 0, "skipped": skipped, "total": total_existing}

    print(f"  Indexing {len(new_chunks)} new chunks...")

    if dry_run:
        print("  [dry_run] Would index the above — no writes performed.")
        return {"indexed": 0, "skipped": skipped, "total": total_existing, "dry_run": True}

    # Process in batches
    indexed = 0
    for batch_start in range(0, len(new_chunks), batch_size):
        batch = new_chunks[batch_start : batch_start + batch_size]

        ids       = [c["chunk_id"]  for c in batch]
        texts     = [c["text"]      for c in batch]
        metadatas = [_extract_metadata(c) for c in batch]

        vectors = embed_texts(texts, model, batch_size=batch_size, show_progress=False)

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,      # stored inline → no need to re-fetch at query time
            metadatas=metadatas,
        )
        indexed += len(batch)
        print(f"  [{indexed}/{len(new_chunks)}] batches written", end="\r")

    print(f"\n  Done. Indexed {indexed} chunks.")
    return {
        "indexed": indexed,
        "skipped": skipped,
        "total":   collection.count(),
    }


# ── Summary ───────────────────────────────────────────────────────────────────

def print_collection_summary(collection: "chromadb.Collection") -> None:
    """Print a breakdown of what's in the collection after indexing."""
    total = collection.count()
    print(f"\n{'═' * 48}")
    print(f"  Collection '{COLLECTION_NAME}' — {total} chunks total")
    print(f"{'═' * 48}")

    if total == 0:
        return

    # Sample to get metadata distribution — ChromaDB has no GROUP BY,
    # so we fetch everything and count in Python.
    result = collection.get(include=["metadatas"])
    metas  = result["metadatas"]

    from collections import Counter
    by_doc_type = Counter(m.get("doc_type", "?") for m in metas)
    by_issuer   = Counter(m.get("issuer",   "?") for m in metas)
    by_isin     = Counter(m.get("etf_isin", "?") for m in metas)
    by_block    = Counter(m.get("block_type","?") for m in metas)

    print("\n  By doc_type:")
    for k, v in sorted(by_doc_type.items()):
        print(f"    {k:<15} {v:>4}")

    print("\n  By issuer:")
    for k, v in sorted(by_issuer.items()):
        print(f"    {k:<15} {v:>4}")

    print("\n  By etf_isin:")
    for k, v in sorted(by_isin.items()):
        print(f"    {k:<20} {v:>4}")

    print("\n  By block_type:")
    for k, v in sorted(by_block.items()):
        print(f"    {k:<15} {v:>4}")

    print(f"{'═' * 48}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Embed ETF chunks and store in ChromaDB")
    p.add_argument("--input_dir",  type=Path, default=Path("data/processed/"),
                   help="Directory of chunk JSON files from ingest.py")
    p.add_argument("--db_path",    type=Path, default=Path("index/chroma_db/"),
                   help="Where to persist the ChromaDB database")
    p.add_argument("--model",      default=EMBEDDING_MODEL,
                   help="Sentence-transformers model name")
    p.add_argument("--batch_size", type=int, default=EMBED_BATCH_SIZE)
    p.add_argument("--rebuild",    action="store_true",
                   help="Delete and recreate the collection from scratch")
    p.add_argument("--dry_run",    action="store_true",
                   help="Load and count chunks without writing to ChromaDB")
    return p


def main():
    args = build_parser().parse_args()

    print(f"\n[embed] Input  : {args.input_dir}")
    print(f"[embed] DB     : {args.db_path}")
    print(f"[embed] Model  : {args.model}")
    print(f"[embed] Rebuild: {args.rebuild}")

    chunks     = load_chunks(args.input_dir)
    collection = get_collection(args.db_path, rebuild=args.rebuild)
    model      = load_model(args.model)

    summary = index_chunks(
        chunks, collection, model,
        batch_size=args.batch_size,
        dry_run=args.dry_run,
    )

    if not args.dry_run:
        print_collection_summary(collection)

    print(f"[embed] Summary: {summary}")


if __name__ == "__main__":
    main()
