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
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # 384d, ~80 MB, CPU-native

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
