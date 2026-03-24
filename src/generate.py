"""
generate.py — Prompt construction + Ollama API call
ETF RAG Project — Phase 4

Setup (one time):
    1. Download and install Ollama: https://ollama.com
    2. Pull the model:  ollama pull llama3.2:3b

    From powershell, you can run the following commands to perform these 2 steps:
    1. irm https://ollama.com/install.ps1 | iex
    2. ollama pull llama3.2:3b

    3. Ollama starts automatically as a background service.

Usage:
    # Test generate.py standalone (pipe in chunks from retrieve.py)
    python src/retrieve.py --query "What is the TER of IE00B4L5Y983?" \\
                           --etf_isin IE00B4L5Y983 --doc_type factsheet \\
                           --json > /tmp/chunks.json

    python src/generate.py --query "What is the TER of IE00B4L5Y983?" \\
                           --chunks_file /tmp/chunks.json --query_type 1
"""

import re
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

try:
    from openai import OpenAI          # pip install openai>=1.0
except ImportError:
    raise ImportError(
        "pip install openai\n"
        "The openai package is used as the HTTP client for Ollama's "
        "OpenAI-compatible API. No OpenAI account or key is needed."
    )

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

OLLAMA_BASE_URL  = "http://localhost:11434/v1"
DEFAULT_MODEL    = "llama3.2:3b"
MAX_TOKENS       = 1024
TEMPERATURE_FACT = 0.0   # Types 1, 2, 3 — deterministic
TEMPERATURE_SYN  = 0.2   # Type 4 — slight variation for reasoning

CITATION_PATTERN = re.compile(
    r"\[([A-Z]{2}[A-Z0-9]{10})\s*\|\s*(\w+)\s*\|\s*(\w+)\s*\|\s*(\d{4})[^\]]*\]"
)


# ── Result types ───────────────────────────────────────────────────────────────

@dataclass
class CitedSource:
    etf_isin: str
    issuer: str
    doc_type: str
    year: int


@dataclass
class GenerationResult:
    query: str
    query_type: int
    answer: str
    cited_sources: list = field(default_factory=list)   # list[CitedSource]
    chunks_used: int = 0
    model: str = DEFAULT_MODEL
    input_tokens: int = 0
    output_tokens: int = 0


# ── Context builder ────────────────────────────────────────────────────────────

def build_context(chunks: list) -> str:
    """
    Format retrieved chunks into a numbered CONTEXT block.
    Each chunk gets a source header for citation.
    Table chunks get a [TABLE] marker.
    """
    if not chunks:
        return "No relevant context found."

    lines = []
    for i, chunk in enumerate(chunks, 1):
        source_header = (
            f"[{chunk.etf_isin} | {chunk.issuer} | {chunk.doc_type} | "
            f"{chunk.year}{' ' + chunk.quarter if chunk.quarter else ''} | "
            f"section: {chunk.section_heading or 'unknown'}]"
        )
        block_marker = " [TABLE]" if chunk.block_type in ("table", "scenario") else ""
        lines.append(f"--- Chunk {i}{block_marker} ---")
        lines.append(source_header)
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()


# ── System prompt ──────────────────────────────────────────────────────────────
# Kept concise for small models — 3b models struggle with very long system prompts.

SYSTEM_PROMPT = """\
You are a financial document analyst for ETF factsheets and KIDs.
Rules:
- Answer ONLY from the CONTEXT provided. Never use outside knowledge.
- Quote numbers exactly as they appear.
- Cite every fact as [ISIN | issuer | doc_type | year], \
e.g. [IE00B4L5Y983 | ishares | factsheet | 2023].
- Be concise and direct.
- If the context lacks the answer, say: \
"The provided documents do not contain sufficient information."\
"""


# ── Prompt templates ───────────────────────────────────────────────────────────
# Small models do better with shorter, more direct prompts.

def _prompt_type1(query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Answer in 1-2 sentences. Cite the source as [ISIN | issuer | doc_type | year]."
    )


def _prompt_type2(query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Compare each ETF or document in turn. "
        "Cite each fact as [ISIN | issuer | doc_type | year]. "
        "End with a one-sentence summary of the key difference."
    )


def _prompt_type3(query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Present findings in chronological order. "
        "Cite each year's value as [ISIN | issuer | doc_type | year]. "
        "State whether the value changed and by how much."
    )


def _prompt_type4(query: str, context: str) -> str:
    return (
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {query}\n\n"
        "Reason across all context chunks. Cover: costs (TER/OCF or ongoing costs), "
        "replication method, and risk profile (SRI if available). "
        "Cite every fact as [ISIN | issuer | doc_type | year]. "
        "Conclude with a clear evidence-based answer."
    )


PROMPT_BUILDERS = {
    1: _prompt_type1,
    2: _prompt_type2,
    3: _prompt_type3,
    4: _prompt_type4,
}


# ── Citation parser ────────────────────────────────────────────────────────────

def parse_citations(answer: str) -> list:
    """Extract [ISIN | issuer | doc_type | year] citations from answer text."""
    seen = set()
    sources = []
    for match in CITATION_PATTERN.finditer(answer):
        isin, issuer, doc_type, year = match.groups()
        key = (isin.upper(), issuer.lower(), doc_type.lower(), int(year))
        if key not in seen:
            seen.add(key)
            sources.append(CitedSource(
                etf_isin=isin.upper(),
                issuer=issuer.lower(),
                doc_type=doc_type.lower(),
                year=int(year),
            ))
    return sources


# ── Generator ─────────────────────────────────────────────────────────────────

class Generator:
    """
    Wraps the Ollama local API via the openai-compatible client.
    No API key required. Ollama must be running (it starts automatically
    on install, or run `ollama serve` manually).
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        base_url: str = OLLAMA_BASE_URL,
    ):
        self._model  = model
        self._client = OpenAI(
            base_url=base_url,
            api_key="ollama",          # required by the client but ignored by Ollama
        )
        print(f"[generate] Model: {model}  |  Endpoint: {base_url}")
        self._check_connection()

    def _check_connection(self) -> None:
        """Verify Ollama is reachable and the model is available."""
        try:
            models = self._client.models.list()
            available = [m.id for m in models.data]
            if self._model not in available:
                print(
                    f"[warn] Model '{self._model}' not found in Ollama.\n"
                    f"       Available: {available}\n"
                    f"       Run: ollama pull {self._model}"
                )
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {OLLAMA_BASE_URL}.\n"
                f"Make sure Ollama is installed and running.\n"
                f"Download: https://ollama.com\n"
                f"Error: {e}"
            )

    def answer(
        self,
        query: str,
        chunks: list,
        query_type: int = 1,
    ) -> GenerationResult:
        """
        Build prompt → call Ollama → parse citations → return GenerationResult.
        """
        if query_type not in PROMPT_BUILDERS:
            raise ValueError(f"query_type must be 1–4, got {query_type}")

        context     = build_context(chunks)
        user_prompt = PROMPT_BUILDERS[query_type](query, context)
        temperature = TEMPERATURE_SYN if query_type == 4 else TEMPERATURE_FACT

        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature,
            max_tokens=MAX_TOKENS,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
        )

        answer_text   = response.choices[0].message.content.strip()
        cited_sources = parse_citations(answer_text)

        # Ollama may not always return token counts
        usage        = response.usage or type("U", (), {"prompt_tokens": 0, "completion_tokens": 0})()
        input_tokens  = getattr(usage, "prompt_tokens",     0)
        output_tokens = getattr(usage, "completion_tokens", 0)

        return GenerationResult(
            query=query,
            query_type=query_type,
            answer=answer_text,
            cited_sources=cited_sources,
            chunks_used=len(chunks),
            model=self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate an ETF answer using a local Ollama model"
    )
    p.add_argument("--query",        required=True)
    p.add_argument("--chunks_file",  type=Path,
                   help="JSON file of retrieved chunks (from retrieve.py --json)")
    p.add_argument("--query_type",   type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--model",        default=DEFAULT_MODEL,
                   help=f"Ollama model name (default: {DEFAULT_MODEL})")
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

    gen    = Generator(model=args.model)
    result = gen.answer(query=args.query, chunks=chunks, query_type=args.query_type)

    print(f"\n── ANSWER (type {result.query_type}) ───────────────────────────────")
    print(result.answer)
    print(f"\n── SOURCES CITED ({len(result.cited_sources)}) ─────────────────────")
    for s in result.cited_sources:
        print(f"  {s.etf_isin} | {s.issuer} | {s.doc_type} | {s.year}")
    print(f"\n── TOKENS  in={result.input_tokens}  out={result.output_tokens} ──\n")


if __name__ == "__main__":
    main()
