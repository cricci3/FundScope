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

LLM_MIN_INTERVAL_S = float(os.getenv("LLM_MIN_INTERVAL_S") or 0)     # throttle for free tiers
LLM_MAX_COST_USD   = float(os.getenv("LLM_MAX_COST_USD") or 0.50)    # per-run budget
