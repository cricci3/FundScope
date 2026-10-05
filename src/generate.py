"""
generate.py — Prompt construction + LLM call
ETF RAG Project — Phase 4

The model is reached through the provider-neutral interface in src/llm/
(provider and model come from .env — see .env.example).

Usage:
    # Test generate.py standalone (pipe in chunks from retrieve.py)
    python src/retrieve.py --query "What is the TER of IE00B4L5Y983?" \\
                           --etf_isin IE00B4L5Y983 --doc_type factsheet \\
                           --json > /tmp/chunks.json

    python src/generate.py --query "What is the TER of IE00B4L5Y983?" \\
                           --chunks_file /tmp/chunks.json --query_type 1
"""

import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from llm import PROVIDERS, SESSION, LLMClient, Message, get_llm
from citations import book_from_chunks, chunk_header, resolve_citations
from citations import CITATION_PATTERN, CitedSource, parse_citations  # noqa: F401 (re-exported)

try:
    from retrieve import RetrievedChunk
except ImportError:
    @dataclass
    class RetrievedChunk:
        chunk_id: str = ""
        text: str = ""
        score: float = 0.0
        etf_isin: str = ""
        etf_ticker: str = ""
        issuer: str = ""
        doc_type: str = ""
        year: int = 0
        quarter: str = ""
        section_heading: str = ""
        block_type: str = "text"
        page_number: int = -1
        source_file: str = ""

        def as_context_string(self) -> str:
            source = (
                f"[{self.etf_isin} | {self.issuer} | {self.doc_type} | "
                f"{self.year}{' ' + self.quarter if self.quarter else ''} | "
                f"section: {self.section_heading or 'unknown'} | "
                f"type: {self.block_type}]"
            )
            return f"{source}\n{self.text}"


# ── Constants ──────────────────────────────────────────────────────────────────

MAX_TOKENS       = 1024
TEMPERATURE_FACT = 0.0   # Types 1, 2, 3 — deterministic
TEMPERATURE_SYN  = 0.2   # Type 4 — slight variation for reasoning

# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    query: str
    query_type: int
    answer: str
    cited_sources: list = field(default_factory=list)   # list[CitedSource]
    raw_answer: str = ""                                # before [Chunk N] resolution
    cited_chunk_ids: list = field(default_factory=list)
    chunks_used: int = 0
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Optional[float] = None


# ── Context builder ────────────────────────────────────────────────────────────

def build_context(chunks: list) -> str:
    """
    Format retrieved chunks into a numbered CONTEXT block.
    Each chunk gets a [Chunk N] header with its source metadata: the model cites
    the number and resolve_citations() maps it back. Table chunks get a [TABLE] marker.
    """
    if not chunks:
        return "No relevant context found."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        lines.append(chunk_header(i, chunk))
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a financial document analyst for ETF factsheets and KIDs.
Rules:
- Answer ONLY from the CONTEXT provided. Never use outside knowledge.
- Quote numbers exactly as they appear.
- Cite every fact with the number of the chunk it comes from, e.g. \
"The TER is 0.20% [Chunk 2]." Several chunks: [Chunk 1, 4]. \
Only cite chunk numbers that appear in the CONTEXT.
- Be concise and direct.
- If the context lacks the answer, say: \
"The provided documents do not contain sufficient information."\
"""


# ── Prompt templates ───────────────────────────────────────────────────────────

def _prompt_type1(query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer in 1-2 sentences. Cite the source as [Chunk N]."
    )


def _prompt_type2(query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Compare each ETF or document in turn. "
        "Cite each fact as [Chunk N]. "
        "End with a one-sentence summary of the key difference."
    )


def _prompt_type3(query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Present findings in chronological order. "
        "Cite each year's value as [Chunk N]. "
        "State whether the value changed and by how much."
    )


def _prompt_type4(query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Reason across all context chunks. Cover: costs (TER/OCF or ongoing costs), "
        "replication method, and risk profile (SRI if available). "
        "Cite every fact as [Chunk N]. "
        "Conclude with a clear evidence-based answer."
    )


PROMPT_BUILDERS = {
    1: _prompt_type1,
    2: _prompt_type2,
    3: _prompt_type3,
    4: _prompt_type4,
}


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator:
    """Builds the RAG prompt and calls the configured LLM (see src/llm/)."""

    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        llm: Optional[LLMClient] = None,
    ):
        self._llm = llm or get_llm("generator", provider, model)
        print(f"[generate] Provider: {self._llm.provider}  |  Model: {self._llm.model}")

    @property
    def provider(self) -> str:
        return self._llm.provider

    @property
    def model(self) -> str:
        return self._llm.model

    def answer(
        self,
        query: str,
        chunks: list,
        query_type: int = 1,
    ) -> GenerationResult:
        """
        Build prompt → call the LLM → parse citations → return GenerationResult.
        """
        if query_type not in PROMPT_BUILDERS:
            raise ValueError(f"query_type must be 1–4, got {query_type}")

        context     = build_context(chunks)
        user_prompt = PROMPT_BUILDERS[query_type](query, context)
        temperature = TEMPERATURE_SYN if query_type == 4 else TEMPERATURE_FACT

        response = self._llm.chat(
            [Message("user", user_prompt)],
            system=SYSTEM_PROMPT,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
        )

        if response.stop_reason == "max_tokens":
            print(f"[generate] [warn] answer truncated at max_tokens={MAX_TOKENS}")

        book     = book_from_chunks(chunks)
        resolved = resolve_citations(response.text, book)
        if resolved.unknown_refs:
            print(f"[generate] [warn] cited unknown chunk(s): {resolved.unknown_refs}")

        return GenerationResult(
            query=query,
            query_type=query_type,
            answer=resolved.text,
            raw_answer=resolved.raw_text,
            cited_sources=resolved.cited_sources,
            cited_chunk_ids=[book.get(n).chunk_id for n in resolved.cited_chunks],
            chunks_used=len(chunks),
            provider=response.provider,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=response.usage.cache_read_tokens,
            cache_write_tokens=response.usage.cache_write_tokens,
            cost_usd=response.cost_usd,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate an ETF answer with the configured LLM"
    )
    p.add_argument("--query",        required=True)
    p.add_argument("--chunks_file",  type=Path,
                   help="JSON file of retrieved chunks (from retrieve.py --json)")
    p.add_argument("--query_type",   type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--provider",     default=None, choices=PROVIDERS,
                   help="LLM provider (default: LLM_PROVIDER from .env)")
    p.add_argument("--model",        default=None,
                   help="Model id (default: LLM_MODEL from .env, else provider default)")
    p.add_argument("--show_context", action="store_true",
                   help="Print the full context sent to the model")
    return p


def main():
    args = build_parser().parse_args()

    if args.chunks_file:
        raw    = json.loads(Path(args.chunks_file).read_text(encoding="utf-8"))
        chunks = [RetrievedChunk(**c) for c in raw]
    else:
        print("[warn] No --chunks_file provided. Running with empty context.")
        chunks = []

    if args.show_context:
        print("\n── CONTEXT ───────────────────────────────────────────")
        print(build_context(chunks))
        print("──────────────────────────────────────────────────────\n")

    gen    = Generator(provider=args.provider, model=args.model)
    result = gen.answer(query=args.query, chunks=chunks, query_type=args.query_type)

    print(f"\n── ANSWER (type {result.query_type}) ───────────────────────────────")
    print(result.answer)
    print(f"\n── SOURCES CITED ({len(result.cited_sources)}) ─────────────────────")
    for s in result.cited_sources:
        print(f"  {s.etf_isin} | {s.issuer} | {s.doc_type} | {s.year}")
    print(f"\n── COST  {SESSION.summary()} ──\n")


if __name__ == "__main__":
    main()
