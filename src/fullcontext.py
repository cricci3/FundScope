"""
fullcontext.py — "Full context" engine: no retrieval, every document in the prompt

The whole 2026 corpus (four factsheets/KIDs) is a few tens of thousands of
tokens, so the model can read all of it. This engine answers from the complete
documents, to measure what retrieval adds (or loses) compared with giving the
model everything (task 5.5):

    python src/pipeline.py --run_eval --engine full
    python src/fullcontext.py --query "What is the TER of IE00B4L5Y983?" --show_cost

The documents go in the system prompt, which the Anthropic backend marks for
prompt caching: the first question writes the cache (1.25× input price), the
following ones read it at a tenth of the input price. Each document is one
numbered source ([Chunk N], see citations.py), so citations resolve to
[ISIN | issuer | doc_type | year] like those of the other engines.
"""

import argparse

try:
    import pdfplumber
except ImportError:
    raise ImportError("pdfplumber is required: pip install pdfplumber")

from agent import AgentResult, strip_preamble
from citations import ChunkBook, chunk_header, resolve_citations
from config import DATA_RAW, METADATA_PATH
from ingest import clean_text, load_config, metadata_from_config_entry
from llm import PROVIDERS, SESSION, LLMClient, Message, get_llm
from registry import FUNDS
from retrieve import RetrievedChunk

MAX_TOKENS = 1024

SYSTEM_PROMPT = """\
You are FundScope, an ETF research assistant. Answer the user's question using \
ONLY the fund documents below (factsheets and KIDs, one per [Chunk N] block).

Funds in the corpus: {funds}.

Rules:
- State fund facts ONLY from the documents, never from outside knowledge. Quote numbers \
exactly as they appear. Do not add your own calculations, worked examples or general \
claims about markets, and never carry a fact over from one fund to the other: if a \
document does not state something for a fund, say so for that fund.
- Cite every fact with the document it comes from, e.g. "The TER is 0.20% [Chunk 2]." \
Facts repeated in a summary or conclusion need their citation too.
- If the documents do not contain the answer, say: "The provided documents do not \
contain sufficient information."
- Comparisons: cover each fund (or document) in turn and end with a one-sentence summary \
of the key difference.
- Suitability or summary questions: cover costs (TER/ongoing charges), replication method \
and risk profile (SRI) of each fund, then conclude with a clear evidence-based answer.
- Be concise. Write only the answer.

=== DOCUMENTS ===

{documents}
"""


def load_documents() -> list[RetrievedChunk]:
    """Every document registered in metadata.json, as one 'chunk' holding its full cleaned text."""
    docs = []
    for filename, entry in load_config(METADATA_PATH).items():
        meta = metadata_from_config_entry(entry, filename)
        with pdfplumber.open(DATA_RAW / filename) as pdf:
            pages = [clean_text(p.extract_text(x_tolerance=2, y_tolerance=2) or "", meta.doc_type)
                     for p in pdf.pages]
        docs.append(RetrievedChunk(
            chunk_id=f"doc:{filename}", text="\n\n".join(p for p in pages if p),
            score=1.0, etf_isin=meta.etf_isin, etf_ticker="", issuer=meta.issuer,
            doc_type=meta.doc_type, year=meta.year, quarter=meta.quarter or "",
            section_heading="full document", block_type="document", page_number=-1,
            source_file=filename,
        ))
    return docs


class FullContextAnswerer:
    """Same interface as agent.Agent for pipeline.py: .provider, .model, .run(question)."""

    engine = "full"
    reference_data = ""

    def __init__(self, provider: str = None, model: str = None, llm: LLMClient = None):
        self._llm  = llm or get_llm("generator", provider, model)
        self._docs = load_documents()
        self._book = ChunkBook()
        blocks = []
        for d in self._docs:
            n, _ = self._book.add(d)
            blocks.append(f"{chunk_header(n, d)}\n{d.text}")
        funds = "; ".join(f"{f.name} = {f.isin} ({f.issuer})" for f in FUNDS.values())
        self._system = SYSTEM_PROMPT.format(funds=funds, documents="\n\n".join(blocks))
        print(f"[full] Provider: {self._llm.provider}  |  Model: {self._llm.model}  |  "
              f"{len(self._docs)} documents, {len(self._system):,} characters in the prompt")

    @property
    def provider(self) -> str:
        return self._llm.provider

    @property
    def model(self) -> str:
        return self._llm.model

    def run(self, question: str) -> AgentResult:
        response = self._llm.chat([Message("user", question)], system=self._system,
                                  max_tokens=MAX_TOKENS)
        resolved = resolve_citations(strip_preamble(response.text), self._book)
        return AgentResult(
            question=question,
            answer=resolved.text,
            raw_answer=resolved.raw_text,
            chunks=list(self._book.chunks),
            cited_sources=resolved.cited_sources,
            cited_chunks=resolved.cited_chunks,
            iterations=1,
            stop_reason="answer" if response.text else "empty",
            provider=self._llm.provider,
            model=response.model or self._llm.model,
            usage=response.usage,
            cost_usd=response.cost_usd,
        )


def main():
    p = argparse.ArgumentParser(description="Answer from the full documents (no retrieval)")
    p.add_argument("--query", required=True)
    p.add_argument("--provider", default=None, choices=PROVIDERS)
    p.add_argument("--model", default=None)
    p.add_argument("--show_cost", action="store_true")
    args = p.parse_args()

    result = FullContextAnswerer(args.provider, args.model).run(args.query)
    print(f"\n  {result.answer}\n")
    if args.show_cost:
        u = result.usage
        print(f"  [cost] in={u.input_tokens} out={u.output_tokens} "
              f"cache_read={u.cache_read_tokens} cache_write={u.cache_write_tokens}")
    print(f"  [session] {SESSION.summary()}")


if __name__ == "__main__":
    main()
