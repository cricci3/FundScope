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
