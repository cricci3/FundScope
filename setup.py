"""
setup.py — One-command setup for FundScope
Runs all steps required before ask.py can be used:
  1. Checks prerequisites (dependencies, PDFs, LLM provider key + a tiny ping)
  2. Ingests PDFs from data/raw/ using metadata.json
  3. Embeds chunks and builds the ChromaDB index

The LLM provider is configured in .env (cp .env.example .env). If the key is
missing or the ping fails, the index is still built and FundScope works in
retrieval-only mode.

Usage:
    python setup.py

    # Force a full rebuild of the index (e.g. after adding new documents)
    python setup.py --rebuild

    # Skip ingestion if data/processed/ already has files
    python setup.py --skip_ingest

    # Skip the LLM check (no API call at all)
    python setup.py --skip_llm_check
"""

import sys
import json
import argparse
import subprocess
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from config import (  # noqa: E402
    PROJECT_ROOT as ROOT,
    DATA_RAW,
    DATA_PROCESSED as DATA_PROC,
    INDEX_PATH,
    METADATA_PATH as METADATA,
    SRC_DIR as SRC,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def header(text: str) -> None:
    print(f"\n{'─' * 56}")
    print(f"  {text}")
    print(f"{'─' * 56}")


def success(text: str) -> None:
    print(f"  ✓  {text}")


def warn(text: str) -> None:
    print(f"  ⚠  {text}")


def error(text: str) -> None:
    print(f"  ✗  {text}")


def abort(text: str) -> None:
    error(text)
    sys.exit(1)


def run(cmd: list, description: str) -> bool:
    """Run a subprocess command, stream output, return True on success."""
    print(f"\n  Running: {' '.join(str(c) for c in cmd)}\n")
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode != 0:
        error(f"{description} failed (exit code {result.returncode})")
        return False
    return True


# ── Prerequisite checks ────────────────────────────────────────────────────────

def check_python_deps() -> bool:
    header("Checking Python dependencies")
    missing = []
    for pkg, import_name in [
        ("pdfplumber",            "pdfplumber"),
        ("sentence-transformers", "sentence_transformers"),
        ("chromadb",              "chromadb"),
        ("anthropic",             "anthropic"),
        ("openai",                "openai"),
        ("python-dotenv",         "dotenv"),
    ]:
        try:
            __import__(import_name)
            success(pkg)
        except ImportError:
            error(f"{pkg} not installed")
            missing.append(pkg)

    if missing:
        print(f"\n  Install missing packages with:")
        print(f"    uv sync\n")
        return False
    return True


def check_llm() -> bool:
    header("Checking LLM provider")
    from llm import LLMError, resolve
    from llm.cli import ping

    try:
        provider, model = resolve("generator")
    except LLMError as e:
        warn(str(e))
        return False

    print(f"  Provider: {provider}  |  Model: {model}  (from .env / defaults)")
    if not (ROOT / ".env").exists():
        warn("No .env file found — create one with:  cp .env.example .env")

    if not ping("generator"):
        warn("LLM ping failed (see the message above).")
        return False
    success(f"{provider} / {model} answered")
    return True


def check_pdfs() -> bool:
    header("Checking PDF documents")

    if not METADATA.exists():
        abort(
            f"metadata.json not found at {METADATA}\n"
            "  Create it to register your PDF documents.\n"
            "  See README.md for the required format."
        )

    config = json.loads(METADATA.read_text(encoding="utf-8"))
    if not config:
        abort("metadata.json is empty. Add at least one PDF entry.")

    missing_pdfs = []
    for filename in config:
        pdf_path = DATA_RAW / filename
        if not pdf_path.exists():
            missing_pdfs.append(filename)

    if missing_pdfs:
        error(f"{len(missing_pdfs)} PDF(s) listed in metadata.json not found in data/raw/:")
        for f in missing_pdfs:
            print(f"    • {f}")
        print(f"\n  Place the PDF files in:  {DATA_RAW}\n")
        return False

    success(f"Found {len(config)} PDF(s) registered in metadata.json")
    return True


# ── Setup steps ────────────────────────────────────────────────────────────────

def step_ingest() -> bool:
    header("Step 1 — Ingesting PDFs")
    DATA_PROC.mkdir(parents=True, exist_ok=True)
    return run(
        [sys.executable, str(SRC / "ingest.py"),
         "--batch",      str(DATA_RAW),
         "--config",     str(METADATA),
         "--output_dir", str(DATA_PROC)],
        "Ingestion"
    )


def step_embed(rebuild: bool) -> bool:
    header("Step 2 — Building vector index")
    cmd = [
        sys.executable, str(SRC / "embed.py"),
        "--input_dir", str(DATA_PROC),
        "--db_path",   str(INDEX_PATH),
    ]
    if rebuild:
        cmd.append("--rebuild")
    return run(cmd, "Embedding")


# ── Main ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FundScope setup — run this once before using ask.py"
    )
    p.add_argument("--rebuild",      action="store_true",
                   help="Wipe and rebuild the vector index from scratch")
    p.add_argument("--skip_ingest",  action="store_true",
                   help="Skip PDF ingestion (use if data/processed/ is already populated)")
    p.add_argument("--skip_llm_check", action="store_true",
                   help="Skip the LLM provider check (no API call)")
    return p


def main():
    args = build_parser().parse_args()

    print("\n" + "═" * 56)
    print("  FundScope — Setup")
    print("═" * 56)

    # ── Prerequisites ──────────────────────────────────────────────────────────
    if not check_python_deps():
        abort("Fix missing dependencies and re-run setup.py")

    if not check_pdfs():
        abort("Fix missing PDF files and re-run setup.py")

    llm_ok = True
    if not args.skip_llm_check:
        llm_ok = check_llm()
        if not llm_ok:
            print("  Continuing setup without an LLM.")
            print("  You can use retrieval-only mode after setup completes.\n")

    # ── Ingest ─────────────────────────────────────────────────────────────────
    if args.skip_ingest:
        existing = list(DATA_PROC.glob("*.json"))
        if existing:
            header("Step 1 — Ingestion skipped")
            success(f"Using {len(existing)} existing chunk file(s) in data/processed/")
        else:
            warn("--skip_ingest passed but data/processed/ is empty — running ingest anyway")
            if not step_ingest():
                abort("Ingestion failed. Check the error above.")
    else:
        if not step_ingest():
            abort("Ingestion failed. Check the error above.")

    # ── Embed ──────────────────────────────────────────────────────────────────
    if not step_embed(rebuild=args.rebuild):
        abort("Embedding failed. Check the error above.")

    # ── Done ───────────────────────────────────────────────────────────────────
    print("\n" + "═" * 56)
    print("  Setup complete!")
    print("═" * 56)

    if llm_ok:
        print("""
  You can now ask questions:

    python src/ask.py                         ← interactive mode
    python src/ask.py --query "your question" ← single question
""")
    else:
        print("""
  Setup complete, but no LLM provider is working (retrieval-only mode).
  The index is built — you can run retrieval-only evaluation:

    python evaluation/evaluate.py --retrieval_only ...

  To enable answer generation:
    cp .env.example .env        then set ANTHROPIC_API_KEY (or another provider)
    python src/llm/cli.py --ping
  Then ask questions with:
    python src/ask.py
""")


if __name__ == "__main__":
    main()
