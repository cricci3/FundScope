"""
ask.py — Simple interactive entrypoint for FundScope
ETF RAG Project

The user types a question in plain English.
The system figures out the query type and retrieval mode automatically.
No flags, no ISINs, no doc_type — just a question.

Usage:
    # Interactive mode (keeps asking until you type 'exit')
    python src/ask.py

    # Single question mode
    python src/ask.py --query "What is the TER of the iShares MSCI World ETF?"

    # Show the retrieved chunks alongside the answer
    python src/ask.py --show_chunks
"""

import re
import sys
import argparse
from pathlib import Path

# Allow imports from src/
ROOT    = Path(__file__).resolve().parent
DB_PATH = ROOT / "index" / "chroma_db"

from retrieve import Retriever, route_query
from generate import Generator, build_context

# ── Config ─────────────────────────────────────────────────────────────────────

K_DEFAULT  = 6

# Known ISINs in the corpus — used for automatic query routing
KNOWN_ISINS = {
    "IE00B4L5Y983": {"issuer": "ishares", "name": "iShares Core MSCI World"},
    "IE00BD4TXV59": {"issuer": "ubs",     "name": "UBS Core MSCI World"},
}

# Keywords that suggest a comparative query
COMPARATIVE_KEYWORDS = [
    "compare", "versus", "vs", "difference", "both", "which",
    "better", "higher", "lower", "more", "less", "between",
]

# Keywords that suggest the user wants KID information
KID_KEYWORDS = [
    "kid", "key information", "risk indicator", "sri", "holding period",
    "entry cost", "exit cost", "ongoing cost", "performance scenario",
    "target investor", "target market",
]


# ── Query router ───────────────────────────────────────────────────────────────

def detect_isins(query: str) -> list:
    """Extract any ISINs mentioned explicitly in the query."""
    return [isin for isin in KNOWN_ISINS if isin in query.upper()]


def detect_doc_type(query: str) -> str:
    """Guess whether the user wants factsheet or KID information."""
    q = query.lower()
    if any(kw in q for kw in KID_KEYWORDS):
        return "kid"
    return "factsheet"


def detect_query_type(query: str, isin_list: list) -> int:
    """
    Guess the query type from the question text.
    1 = factual, 2 = comparative, 4 = synthetic/reasoning
    """
    q = query.lower()
    if any(kw in q for kw in COMPARATIVE_KEYWORDS) or len(isin_list) > 1:
        return 2
    if any(phrase in q for phrase in ["suitable", "recommend", "best", "summarise", "summary", "differences"]):
        return 4
    return 1


def build_filters(query: str, isin_list: list) -> dict:
    """Build a metadata filter hint from the query."""
    filters = {}
    doc_type = detect_doc_type(query)
    if doc_type:
        filters["doc_type"] = doc_type
    if len(isin_list) == 1:
        filters["etf_isin"] = isin_list[0]
    return filters


def resolve_isins(query: str) -> list:
    """
    Return the list of ISINs relevant to the query.
    If the user didn't mention any ISIN explicitly, use all known ISINs
    for comparative/synthetic queries, or prompt for clarification.
    """
    mentioned = detect_isins(query)
    if mentioned:
        return mentioned

    q = query.lower()

    # Map common name references to ISINs
    found = []
    for isin, meta in KNOWN_ISINS.items():
        if meta["issuer"] in q or meta["name"].lower() in q:
            found.append(isin)

    if found:
        return found

    # For comparative or synthetic queries default to all ISINs
    if any(kw in q for kw in COMPARATIVE_KEYWORDS + ["suitable", "summarise", "differences"]):
        return list(KNOWN_ISINS.keys())

    # For factual queries with no ETF mentioned, use all (retriever will find best match)
    return list(KNOWN_ISINS.keys())


# ── Answer printer ─────────────────────────────────────────────────────────────

def print_answer(query: str, answer: str, cited_sources: list, show_chunks: bool, chunks: list) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  Q: {query}")
    print(f"{'─' * width}")

    if show_chunks and chunks:
        print(f"\n  Retrieved {len(chunks)} chunk(s):\n")
        for i, c in enumerate(chunks, 1):
            print(f"  [{i}] score={c.score:.3f}  "
                  f"{c.etf_isin} | {c.doc_type} | {c.year} | "
                  f"{c.section_heading or '—'} | {c.block_type}")
            preview = c.text[:150].replace("\n", " ")
            print(f"      {preview}{'…' if len(c.text) > 150 else ''}\n")
        print(f"{'─' * width}")

    print(f"\n  {answer}\n")

    if cited_sources:
        print(f"  Sources:")
        for s in cited_sources:
            name = KNOWN_ISINS.get(s.etf_isin, {}).get("name", s.etf_isin)
            print(f"    • {name} ({s.etf_isin}) | {s.doc_type} | {s.year}")

    print(f"{'─' * width}\n")


# ── Main ask function ──────────────────────────────────────────────────────────

def ask(
    query: str,
    retriever: Retriever,
    generator: Generator,
    show_chunks: bool = False,
    k: int = K_DEFAULT,
) -> None:
    """Process a single question end-to-end and print the answer."""

    isin_list   = resolve_isins(query)
    query_type  = detect_query_type(query, isin_list)
    filters     = build_filters(query, isin_list)

    # Determine retrieval mode
    if len(isin_list) > 1:
        mode = "comparative"
    elif len(isin_list) == 1 and "doc_type" not in filters:
        mode = "cross_doc"
    else:
        mode = "single"

    chunks = route_query(
        retriever=retriever,
        query=query,
        metadata_filter_hint=filters,
        isin_list=isin_list,
        mode=mode,
        k=k,
    )

    result = generator.answer(
        query=query,
        chunks=chunks,
        query_type=query_type,
    )

    print_answer(
        query=query,
        answer=result.answer,
        cited_sources=result.cited_sources,
        show_chunks=show_chunks,
        chunks=chunks,
    )


# ── Interactive loop ───────────────────────────────────────────────────────────

HELP_TEXT = """
  Commands:
    <any question>   Ask anything about the ETFs in the corpus
    chunks           Toggle showing retrieved chunks (default: off)
    funds            List the ETFs available in the corpus
    help             Show this message
    exit / quit      Exit
"""

FUNDS_TEXT = "\n".join(
    f"  • {meta['name']}  ({isin})  [{meta['issuer']}]"
    for isin, meta in KNOWN_ISINS.items()
)


def interactive_loop(retriever: Retriever, generator: Generator, show_chunks: bool) -> None:
    print("\n" + "═" * 64)
    print("  FundScope — ETF Research Assistant")
    print("  Type a question, 'funds' to list ETFs, or 'exit' to quit.")
    print("═" * 64)

    while True:
        try:
            raw = input("\n  Ask> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n  Goodbye.")
            break

        if not raw:
            continue

        cmd = raw.lower()

        if cmd in ("exit", "quit", "q"):
            print("  Goodbye.")
            break
        elif cmd == "help":
            print(HELP_TEXT)
        elif cmd == "funds":
            print(f"\n  ETFs in corpus:\n{FUNDS_TEXT}\n")
        elif cmd == "chunks":
            show_chunks = not show_chunks
            print(f"  Show chunks: {'on' if show_chunks else 'off'}")
        else:
            try:
                ask(raw, retriever, generator, show_chunks=show_chunks)
            except Exception as e:
                print(f"\n  [error] {e}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FundScope — Ask questions about ETFs in plain English"
    )
    p.add_argument("--query",       default=None,
                   help="Single question (omit for interactive mode)")
    p.add_argument("--show_chunks", action="store_true",
                   help="Print retrieved chunks alongside the answer")
    p.add_argument("--db_path",     type=Path, default=DB_PATH)
    p.add_argument("--model",       default="llama3.2:3b",
                   help="Ollama model name (run 'ollama list' to see available)")
    p.add_argument("--k",           type=int, default=K_DEFAULT,
                   help="Number of chunks to retrieve (default 6)")
    return p


def main():
    args      = build_parser().parse_args()
    retriever = Retriever(db_path=args.db_path)
    generator = Generator(model=args.model)

    if args.query:
        ask(args.query, retriever, generator,
            show_chunks=args.show_chunks, k=args.k)
    else:
        interactive_loop(retriever, generator, show_chunks=args.show_chunks)


if __name__ == "__main__":
    main()
