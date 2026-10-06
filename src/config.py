"""
config.py — Centralised paths and constants for FundScope

Every script imports its paths from here, so they resolve to the same
locations regardless of the current working directory.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

PROJECT_ROOT   = Path(__file__).resolve().parent.parent
SRC_DIR        = PROJECT_ROOT / "src"
DATA_RAW       = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
INDEX_PATH     = PROJECT_ROOT / "index" / "chroma_db"
METADATA_PATH  = SRC_DIR / "metadata.json"
EVAL_DIR       = PROJECT_ROOT / "evaluation"

# ── Vector store ───────────────────────────────────────────────────────────────

COLLECTION_NAME = "etf_chunks"
# Task 5.2: multilingual (Italian questions over English documents), 768d,
# ~1.1 GB, 512-token window; needs the "query: "/"passage: " prefixes below.
# Retrieval-only comparison (chunk value MRR, no metadata filters): MiniLM 0.66
# (Italian 0.43), multilingual-e5-small 0.68 (0.53), multilingual-e5-base 0.74 (0.78).
EMBEDDING_MODEL = "intfloat/multilingual-e5-base"

# Models trained with instruction prefixes: (query prefix, passage prefix).
# The model name is stored in the collection metadata (embed.py) and
# retrieve.py refuses to query an index built with a different model.
EMBEDDING_PREFIXES = {
    "intfloat/multilingual-e5-small": ("query: ", "passage: "),
    "intfloat/multilingual-e5-base":  ("query: ", "passage: "),
}


def embedding_prefixes(model_name: str) -> tuple[str, str]:
    return EMBEDDING_PREFIXES.get(model_name, ("", ""))


# Hybrid retrieval (retrieve.py): BM25 + vector rankings fused with RRF
HYBRID_SEARCH = True

# Minimum cosine similarity for a chunk to reach the model (agent.py drops the
# rest and says so). Similarity ranges differ per model, so it is per model;
# unknown models keep every chunk.
# Task 5.4 calibration (top-20 hits per ground-truth question): chunks holding
# an expected value never scored below 0.768 with e5-base, while off-topic
# questions ("weather in Rome") top out at 0.756. e5 similarities are compressed
# (unrelated fund chunks still score ~0.76-0.80), so this only drops clear misses.
MIN_SIMILARITY = {
    "intfloat/multilingual-e5-base":  0.76,
    "intfloat/multilingual-e5-small": 0.76,
    "all-MiniLM-L6-v2":               0.05,
}


def min_similarity(model_name: str) -> float:
    return MIN_SIMILARITY.get(model_name, 0.0)


# Chunk size / overlap in characters, per doc type (used by ingest.py).
# Task 5.1 sweep (retrieval only, chunk value recall): 250/400 → 0.71,
# 500/700 → 0.89, 800/1000 → 0.93, 1000/1200 → 1.00; overlap 15%.
CHUNK_SIZES    = {"factsheet": 1000, "kid": 1200}
CHUNK_OVERLAPS = {"factsheet": 150, "kid": 180}

# ── LLM provider ───────────────────────────────────────────────────────────────
# Values come from .env (see .env.example); real environment variables win over
# .env, and CLI flags (--provider / --model) win over both.

import os
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# Default model per provider and role. A provider missing here (e.g. groq) has
# no default: LLM_MODEL / JUDGE_MODEL / --model must name one.
DEFAULT_MODELS = {
    "anthropic": {"generator": "claude-haiku-4-5", "judge": "claude-sonnet-5-5"},
}

LLM_PROVIDER   = os.getenv("LLM_PROVIDER") or "anthropic"
LLM_MODEL      = os.getenv("LLM_MODEL") or None
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER") or "anthropic"
JUDGE_MODEL    = os.getenv("JUDGE_MODEL") or None
JUDGE_EFFORT   = os.getenv("JUDGE_EFFORT") or "medium"   # reasoning effort hint for the judge

LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S") or 0)     # throttle for free tiers
LLM_MAX_COST_USD   = float(os.getenv("LLM_MAX_COST_USD") or 0.50)    # per-run budget
