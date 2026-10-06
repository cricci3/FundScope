"""
agent.py — Agentic loop for FundScope
ETF Research Assistant with tool-calling

This is the only entrypoint: the LLM decides which tools to call and in what
order (ask.py is a thin alias of this script; pipeline.py --engine agent
evaluates it on the ground truth).

How it works:
    1. User asks a question (after the last few remembered turns)
    2. LLM receives the question + tool definitions
    3. LLM returns either a tool call or a final answer
    4. If tool call → execute it, feed result back to LLM → repeat
    5. If final answer → resolve its [Chunk N] citations, print and stop

Tools available to the agent (ISIN/issuer/doc_type/year enums come from the
fund registry, src/registry.py):
    - search_etf_docs    : semantic search over the ChromaDB corpus
                           (isins, issuer, doc_type, year, mode single|comparative|cross_doc)
    - get_live_data      : fetch current price, returns, AUM via yfinance
    - list_available_etfs: list all ETFs in the corpus (no args needed)

Interactive commands: funds, chunks, calls, reset, help, exit.

Requirements:
    A tool-capable LLM configured in .env (default: Claude Haiku 4.5 via the
    Anthropic API — see .env.example). The loop only uses the neutral types
    in src/llm/, so any provider registered there works.

Usage:
    # Interactive agent mode
    python src/agent.py

    # Single question
    python src/agent.py --query "How has the iShares MSCI World performed this year?"

    # Show the tool calls the agent made, and the cost of each question
    python src/agent.py --show_calls --show_cost

    # Use a different provider / model
    python src/agent.py --provider groq --model <model-id>
"""

import re
import json
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Local imports ──────────────────────────────────────────────────────────────
from config import INDEX_PATH as DB_PATH
from citations import ChunkBook, chunk_header, resolve_citations
from llm import PROVIDERS, SESSION, LLMClient, Message, ToolSpec, Usage, get_llm
from retrieve import Retriever, RetrievedChunk, apply_min_score
from live_data import get_etf_live_data, format_for_prompt
from registry import (FUNDS, all_doc_types, all_isins, all_issuers, all_years, get_fund,
                      resolve_isins)


# ── Config ─────────────────────────────────────────────────────────────────────

MAX_ITERATIONS  = 6              # safety cap on the agent loop
HISTORY_TURNS   = 5              # past question/answer pairs kept in interactive mode

# Appended to the last tool result before the final allowed call
FINAL_CALL_NOTICE = (
    "\n\n[Search limit reached: do not call any more tools. Answer now from the results "
    "above, and say which facts the documents did not provide.]"
)
MAX_TOKENS      = 1024
# Chunks are ~1000 characters (config.CHUNK_SIZES), so a few per search suffice
K_SINGLE        = 5              # chunks for a single-fund search
K_PER_ETF       = 3              # chunks per fund in comparative mode
K_PER_DOC_TYPE  = 2              # chunks per document type in cross_doc mode
KEY_FACTS_FIRST = True           # prepend each searched document's key-facts chunk
MAX_KEY_FACTS   = 4              # ... unless the search spans more documents than this

SEARCH_MODES = ["single", "comparative", "cross_doc"]


# ── Tool definitions (sent to the LLM) ────────────────────────────────────────
# These tell the LLM what tools exist and what arguments they take.
# The LLM decides when and how to call them. ISINs, issuers, document types and
# years are enums generated from the fund registry (metadata.json).

def _fund_list() -> str:
    return "; ".join(f"{f.name} = {f.isin} ({f.issuer})" for f in FUNDS.values())


def build_tools() -> list[ToolSpec]:
    isins = all_isins()
    live_isins = [f.isin for f in FUNDS.values() if f.yahoo_ticker]
    return [
        ToolSpec(
            name="search_etf_docs",
            description=(
                "Search the ETF document corpus (factsheets and KIDs) for information "
                "about costs, fees, risk indicators, replication method, index, holdings, "
                "performance scenarios, target investors or any other content of the PDF "
                "documents. Factsheets hold TER/ongoing charges, index, replication, "
                "holdings and past performance; KIDs hold the Summary Risk Indicator, "
                "recommended holding period, entry/exit/ongoing costs, performance "
                "scenarios and the target investor. Funds: " + _fund_list() + "."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query, e.g. 'ongoing charges'. "
                                       "Rephrase and search again if the first results "
                                       "do not contain the answer.",
                    },
                    "isins": {
                        "type": "array",
                        "items": {"type": "string", "enum": isins},
                        "description": "Funds to search. Omit to search every fund.",
                    },
                    "issuer": {
                        "type": "string",
                        "enum": all_issuers(),
                        "description": "Restrict to the funds of one issuer (alternative to isins).",
                    },
                    "doc_type": {
                        "type": "string",
                        "enum": all_doc_types(),
                        "description": "Restrict to one document type. Omit to search both.",
                    },
                    "year": {
                        "type": "integer",
                        "enum": all_years(),
                        "description": "Restrict to documents of one year. Omit for all years.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": SEARCH_MODES,
                        "description": (
                            "single: best chunks for one fund (or the whole corpus); "
                            "comparative: top chunks from EACH selected fund, so every fund "
                            "is represented; cross_doc: top chunks from EACH document type "
                            "(factsheet and KID) of the selected fund(s). Omit to choose "
                            "automatically (several funds → comparative, one fund without "
                            "doc_type → cross_doc, otherwise single)."
                        ),
                    },
                },
                "required": ["query"],
            },
        ),
        ToolSpec(
            name="get_live_data",
            description=(
                "Fetch real-time market data for an ETF: current price, 1-month, "
                "3-month, and year-to-date returns, AUM, and average daily volume. "
                "Use this for questions about recent performance, price, or liquidity. "
                "Do NOT use this for questions about fund documents (TER, SRI, etc.)."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {
                    "isin": {
                        "type": "string",
                        "enum": live_isins,
                        "description": "ISIN of the ETF. Funds: " + _fund_list(),
                    },
                },
                "required": ["isin"],
            },
        ),
        ToolSpec(
            name="list_available_etfs",
            description=(
                "Return a list of all ETFs available in the corpus with their ISINs, "
                "issuers, names and available documents. Call this if the user asks what "
                "funds are available or if you are unsure which ISIN to use."
            ),
            parameters_json_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


TOOLS = build_tools()


# ── Tool executor ──────────────────────────────────────────────────────────────
# This is where tool calls from the LLM are actually run.
# Returns a plain string — the LLM will read this as the tool result.

class ToolExecutor:
    def __init__(self, retriever: Retriever, key_facts_first: bool = KEY_FACTS_FIRST):
        self._retriever = retriever
        self.key_facts_first = key_facts_first
        self.new_question()

    def new_question(self) -> None:
        """Restart chunk numbering: [Chunk N] is only valid within one question."""
        self.book = ChunkBook()
        self.retrieved: list[RetrievedChunk] = []

    def execute(self, name: str, arguments: dict, show_calls: bool = False) -> str:
        if show_calls:
            print(f"\n  [tool call] {name}({json.dumps(arguments)})")

        if name == "search_etf_docs":
            return self._search_etf_docs(**arguments)
        elif name == "get_live_data":
            return self._get_live_data(**arguments)
        elif name == "list_available_etfs":
            return self._list_available_etfs()
        else:
            raise ValueError(f"Unknown tool: {name}")

    def search(
        self,
        query: str,
        isins: Optional[list[str]] = None,
        issuer: Optional[str] = None,
        doc_type: Optional[str] = None,
        year: Optional[int] = None,
        mode: Optional[str] = None,
    ) -> list[RetrievedChunk]:
        """Resolve the tool arguments into a retrieval call (raises ValueError on bad args)."""
        isin_list = resolve_isins(isins, issuer)
        if not isin_list:
            raise ValueError(f"No fund matches isins={isins} issuer={issuer!r}.")
        if mode is not None and mode not in SEARCH_MODES:
            raise ValueError(f"mode must be one of {SEARCH_MODES}, got {mode!r}")
        if mode is None:
            if len(isin_list) > 1:
                mode = "comparative"
            elif doc_type is None:
                mode = "cross_doc"
            else:
                mode = "single"

        chunks = self._semantic_search(query, isin_list, doc_type, year, mode)
        if not self.key_facts_first:
            return chunks
        # The key-facts chunk of every searched document comes first: it holds the
        # values most questions ask for (TER, replication, SRI, costs), and short
        # queries like "replication method" often rank it below generic prose
        doc_types = [doc_type] if doc_type else all_doc_types()
        if len(isin_list) * len(doc_types) > MAX_KEY_FACTS:
            return chunks           # a corpus-wide search: too many documents to summarise
        facts = [kf for isin in isin_list for dt in doc_types
                 if (kf := self._retriever.key_facts(isin, dt, year)) is not None]
        seen = {c.chunk_id for c in facts}
        return facts + [c for c in chunks if c.chunk_id not in seen]

    def _semantic_search(self, query: str, isin_list: list[str], doc_type: Optional[str],
                         year: Optional[int], mode: str) -> list[RetrievedChunk]:
        filters = {"doc_type": doc_type, "year": year}

        if mode == "cross_doc":
            doc_types = [doc_type] if doc_type else all_doc_types()
            year_filter = {"year": year} if year else {}
            chunks = []
            for isin in isin_list:
                chunks.extend(self._retriever.cross_document(
                    query=query, etf_isin=isin, doc_types=doc_types,
                    filters=year_filter, k_per_doc_type=K_PER_DOC_TYPE,
                ))
            return chunks

        if mode == "comparative" and len(isin_list) > 1:
            return self._retriever.comparative(
                query=query, isin_list=isin_list, filters=filters, k_per_etf=K_PER_ETF,
            )

        # single: one fund, or every fund when no restriction was given
        if len(isin_list) == 1:
            filters["etf_isin"] = isin_list[0]
        elif len(isin_list) < len(FUNDS):
            # a subset of funds cannot be expressed as one equality filter
            return self._retriever.comparative(
                query=query, isin_list=isin_list, filters=filters, k_per_etf=K_PER_ETF,
            )
        return self._retriever.single(query=query, filters=filters, k=K_SINGLE)

    def _search_etf_docs(self, query: str, **kwargs) -> str:
        """Run semantic search and return the chunks, numbered for citation, as text."""
        min_score = self._retriever.min_score
        chunks, dropped = apply_min_score(self.search(query, **kwargs), min_score, query)
        self.retrieved.extend(chunks)
        if not chunks:
            best = f" (best similarity {max(c.score for c in dropped):.2f}, " \
                   f"threshold {min_score:.2f})" if dropped else ""
            return (f"No relevant document chunks found for this query{best}. "
                    "Rephrase the query or change the filters.")

        # Numbers run across every search of the same question; a chunk already
        # shown keeps its number and is not repeated.
        lines = []
        for c in chunks:
            n, is_new = self.book.add(c)
            if is_new:
                lines.append(chunk_header(n, c))
                lines.append(c.text)
                lines.append("")
            else:
                lines.append(f"[Chunk {n}] (already shown above)")
        if dropped:
            lines.append(f"[{len(dropped)} more result(s) discarded as not relevant enough "
                         f"(similarity below {min_score:.2f}).]")
        return "\n".join(lines).strip()

    def _get_live_data(self, isin: str) -> str:
        """Fetch live market data and return as a formatted string."""
        data = get_etf_live_data(isin)
        return format_for_prompt(data)

    def _list_available_etfs(self) -> str:
        lines = ["ETFs available in the corpus:"]
        for fund in FUNDS.values():
            lines.append(f"  - {fund.describe()}")
        return "\n".join(lines)


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are FundScope, an ETF research assistant.

Funds in the corpus: {funds}.

You have access to two sources of information:
1. ETF document corpus (factsheets and KIDs) — search with search_etf_docs
2. Live market data (prices, returns, AUM) — fetch with get_live_data

Rules:
- State fund facts ONLY from tool results, never from outside knowledge. Quote numbers \
exactly as they appear. Do not add your own calculations, worked examples or general \
claims about markets, and never carry a fact over from one fund to the other: if a \
document does not state something for a fund, say so for that fund.
- Every document chunk is labelled [Chunk N]. Cite every document fact with the chunk \
it comes from, e.g. "The TER is 0.20% [Chunk 2]." Several chunks: [Chunk 1, 4]. Only \
cite chunk numbers you have seen. Facts repeated in a summary or conclusion need their \
citation too. Cite live market data as [live data].
- If the results do not contain the answer, search again with other wording (e.g. \
"Total Expense Ratio", "ongoing charges", "costs over time") or another doc_type before \
concluding. Only then say: "The provided documents do not contain sufficient information."
- For questions about performance or price, use get_live_data; for costs, risk or fund \
structure, use search_etf_docs; for broad questions, use both.
- Comparisons: cover each fund (or document) in turn and end with a one-sentence summary \
of the key difference.
- Suitability or summary questions: cover costs (TER/ongoing charges), replication method \
and risk profile (SRI) of each fund, then conclude with a clear evidence-based answer.
- Be concise. Do not repeat tool output verbatim — synthesize it. Write only the answer, \
with no preamble about your searches (e.g. no "Now I have the information I need").
"""


# "Perfect! Now I have all the information I need." — narration about the
# searches that the model sometimes puts before the answer despite the prompt.
# Italian too ("Perfetto! Ora ho la risposta completa."), for questions in Italian.
_PREAMBLE_RE = re.compile(
    r"^\s*(?:perfect|great|excellent|perfetto|ottimo|eccellente)\b"
    r"|\b(?:i now have|now i have|let me|ora ho|adesso ho|lasciami|fammi)\b", re.I)


def strip_preamble(text: str) -> str:
    """
    Drop leading paragraphs that narrate the search ("Perfect! ... Let me verify ..."),
    as long as an answer follows them.
    """
    paragraphs = re.split(r"\n\s*\n", text.strip())
    while len(paragraphs) > 1 and _PREAMBLE_RE.search(paragraphs[0]):
        paragraphs.pop(0)
    return "\n\n".join(paragraphs)


def build_system_prompt() -> str:
    return SYSTEM_PROMPT.format(funds=_fund_list())


# ── Agent loop ─────────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    question: str
    answer: str                    # [Chunk N] tags resolved to [ISIN | issuer | doc_type | year]
    raw_answer: str = ""           # as written by the model
    chunks: list = field(default_factory=list)          # list[RetrievedChunk], numbered 1..N
    cited_sources: list = field(default_factory=list)   # list[CitedSource]
    cited_chunks: list = field(default_factory=list)    # chunk numbers cited
    tool_calls: list = field(default_factory=list)      # [{"name", "arguments", "error"}]
    iterations: int = 0
    stop_reason: str = ""          # answer | max_iterations | empty
    provider: str = ""
    model: str = ""
    usage: Usage = field(default_factory=Usage)
    cost_usd: Optional[float] = None


class Agent:
    """
    The core agent loop.

    Each call to .run() answers one question. The loop:
      1. Sends the user message + tool definitions to the LLM
      2. If the LLM returns tool calls → execute them, append results, loop
      3. If the LLM returns a text message → that is the final answer
      4. Hard stop after MAX_ITERATIONS to prevent runaway loops
    The [Chunk N] citations of the final answer are then resolved to the
    metadata of the chunks the tools returned (citations.py).

    Conversational memory: the last `history_turns` question/answer pairs are
    sent before the new question, so follow-ups ("and the UBS one?") work.
    Only the final answers are kept (not the tool calls and chunks), which keeps
    the prompt small; reset() forgets everything. history_turns=0 makes every
    question independent (used by the evaluation).
    """

    def __init__(
        self,
        retriever: Retriever,
        provider: str = None,
        model: str = None,
        show_calls: bool = False,
        llm: LLMClient = None,
        history_turns: int = HISTORY_TURNS,
    ):
        self._llm      = llm or get_llm("generator", provider, model)
        self._executor = ToolExecutor(retriever)
        self._system   = build_system_prompt()
        self._show_calls = show_calls
        self._history_turns = history_turns
        self._history: list[tuple[str, str]] = []     # (question, resolved answer)
        print(f"[agent] Provider: {self._llm.provider}  |  Model: {self._llm.model}")

    @property
    def provider(self) -> str:
        return self._llm.provider

    @property
    def model(self) -> str:
        return self._llm.model

    @property
    def reference_data(self) -> str:
        """Facts the system prompt gives the model (the fund registry), for the judge."""
        return f"Funds in the corpus: {_fund_list()}."

    @property
    def history_len(self) -> int:
        return len(self._history)

    def reset(self) -> None:
        """Forget the conversation so far."""
        self._history.clear()

    def _history_messages(self) -> list[Message]:
        messages = []
        for question, answer in self._history:
            messages += [Message("user", question), Message("assistant", answer)]
        return messages

    def run(self, question: str) -> AgentResult:
        """Run the agent loop for a single question (after the remembered turns)."""
        self._executor.new_question()
        result   = AgentResult(question=question, answer="",
                               provider=self._llm.provider, model=self._llm.model)
        priced   = True
        cost     = 0.0
        messages = self._history_messages() + [Message("user", question)]
        text     = ""

        for iteration in range(MAX_ITERATIONS):
            response = self._llm.chat(
                messages,
                system=self._system,
                tools=TOOLS,
                max_tokens=MAX_TOKENS,
            )
            result.iterations += 1
            result.model = response.model or result.model
            result.usage = result.usage + response.usage
            if response.cost_usd is None:
                priced = False
            else:
                cost += response.cost_usd

            # ── Case 1: LLM produced a final answer ───────────────────────────
            if not response.tool_calls:
                text = response.text
                result.stop_reason = "answer" if text else "empty"
                if not text:
                    text = f"[Agent stopped: empty answer ({response.stop_reason}).]"
                break

            # ── Case 2: LLM wants to call tools ───────────────────────────────
            messages.append(response.to_message())

            for tc in response.tool_calls:
                is_error = False
                if tc.error:
                    # Malformed arguments: tell the model so it can retry
                    output, is_error = f"[tool error] {tc.error}", True
                else:
                    try:
                        output = self._executor.execute(
                            tc.name, tc.arguments, show_calls=self._show_calls
                        )
                    except Exception as e:
                        output, is_error = f"[tool error] {type(e).__name__}: {e}", True

                result.tool_calls.append({"name": tc.name, "arguments": tc.arguments,
                                          "error": output if is_error else None})
                messages.append(Message("tool", output, tool_call_id=tc.id,
                                        is_error=is_error))

            # The next call is the last one: ask for the answer instead of more searches
            if iteration == MAX_ITERATIONS - 2:
                messages[-1].content += FINAL_CALL_NOTICE

            # Loop — LLM will now read the tool results and decide next step
        else:
            # Safety fallback
            result.stop_reason = "max_iterations"
            text = "[Agent stopped: maximum iterations reached without a final answer.]"

        book     = self._executor.book
        resolved = resolve_citations(strip_preamble(text), book)
        if resolved.unknown_refs:
            print(f"[agent] [warn] cited unknown chunk(s): {resolved.unknown_refs}")

        result.answer        = resolved.text
        result.raw_answer    = resolved.raw_text
        result.chunks        = list(book.chunks)
        result.cited_sources = resolved.cited_sources
        result.cited_chunks  = resolved.cited_chunks
        result.cost_usd      = round(cost, 6) if priced else None

        if self._history_turns > 0 and result.stop_reason == "answer":
            self._history.append((question, result.answer))
            del self._history[:-self._history_turns]
        return result


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_answer(result: AgentResult, show_chunks: bool = False) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  Q: {result.question}")
    print(f"{'─' * width}")

    if show_chunks and result.chunks:
        print(f"\n  Retrieved {len(result.chunks)} chunk(s):\n")
        for i, c in enumerate(result.chunks, 1):
            cited = "  ← cited" if i in result.cited_chunks else ""
            print(f"  [{i}] score={c.score:.3f}  {c.etf_isin} | {c.doc_type} | {c.year} | "
                  f"{c.section_heading or '—'} | {c.block_type}{cited}")
            preview = c.text[:150].replace("\n", " ")
            print(f"      {preview}{'…' if len(c.text) > 150 else ''}\n")
        print(f"{'─' * width}")

    print(f"\n  {result.answer}\n")

    if result.cited_sources:
        print("  Sources:")
        for s in result.cited_sources:
            fund = get_fund(s.etf_isin)
            name = fund.name if fund else s.etf_isin
            print(f"    • {name} ({s.etf_isin}) | {s.doc_type} | {s.year}")
    print(f"{'─' * width}\n")


def print_cost(result: AgentResult) -> None:
    u = result.usage
    cost = f"${result.cost_usd:.5f}" if result.cost_usd is not None else "unknown"
    print(f"  [cost] {result.provider}/{result.model}  {result.iterations} call(s)  "
          f"in={u.input_tokens} out={u.output_tokens} cache_read={u.cache_read_tokens} "
          f"| est. {cost}\n")


def ask_agent(agent: Agent, question: str, show_cost: bool = False,
              show_chunks: bool = False) -> AgentResult:
    result = agent.run(question)
    print_answer(result, show_chunks=show_chunks)
    if show_cost:
        print_cost(result)
    return result


# ── Interactive loop ───────────────────────────────────────────────────────────

HELP_TEXT = """
  Commands:
    <any question>   Ask anything about the ETFs
    funds            List ETFs in the corpus
    calls            Toggle showing tool calls (default: off)
    chunks           Toggle showing the retrieved chunks (default: off)
    reset            Forget the conversation (follow-up questions use the last {turns} turns)
    help             Show this message
    exit / quit      Exit
"""


def interactive_loop(agent: Agent, show_cost: bool = False, show_chunks: bool = False) -> None:
    print("\n" + "═" * 64)
    print("  FundScope Agent — ETF Research Assistant")
    print("  Powered by tool-calling. The LLM decides what to look up.")
    print("  Type a question or 'help' for commands.")
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
            print(HELP_TEXT.format(turns=HISTORY_TURNS))
        elif cmd == "funds":
            funds = "\n".join(f"  • {f.describe()}" for f in FUNDS.values())
            print(f"\n  ETFs in corpus:\n{funds}\n")
        elif cmd == "reset":
            agent.reset()
            print("  Conversation cleared.")
        elif cmd == "chunks":
            show_chunks = not show_chunks
            print(f"  Show chunks: {'on' if show_chunks else 'off'}")
        elif cmd == "calls":
            agent._show_calls = not agent._show_calls
            print(f"  Show tool calls: {'on' if agent._show_calls else 'off'}")
        else:
            try:
                ask_agent(agent, raw, show_cost, show_chunks)
            except Exception as e:
                print(f"\n  [error] {e}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FundScope Agent — ETF assistant with tool-calling"
    )
    p.add_argument("--query",       default=None,
                   help="Single question (omit for interactive mode)")
    p.add_argument("--provider",    default=None, choices=PROVIDERS,
                   help="LLM provider (default: LLM_PROVIDER from .env)")
    p.add_argument("--model",       default=None,
                   help="Model id; must support tool calling "
                        "(default: LLM_MODEL from .env, else provider default)")
    p.add_argument("--db_path",     type=Path, default=DB_PATH)
    p.add_argument("--show_calls",  action="store_true",
                   help="Print each tool call as it happens")
    p.add_argument("--show_chunks", action="store_true",
                   help="Print the retrieved chunks alongside the answer")
    p.add_argument("--show_cost",   action="store_true",
                   help="Print tokens and estimated cost of each question")
    return p


def main():
    args      = build_parser().parse_args()
    retriever = Retriever(db_path=args.db_path)
    agent     = Agent(retriever=retriever, provider=args.provider, model=args.model,
                      show_calls=args.show_calls)

    if args.query:
        ask_agent(agent, args.query, show_cost=args.show_cost, show_chunks=args.show_chunks)
    else:
        interactive_loop(agent, show_cost=args.show_cost, show_chunks=args.show_chunks)
    print(f"  [session] {SESSION.summary()}")


if __name__ == "__main__":
    main()
