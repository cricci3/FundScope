"""
agent.py — Agentic loop for FundScope
ETF Research Assistant with tool-calling

This is the real agent: the LLM decides which tools to call and in what order.
It replaces the hardcoded query-type detection in ask.py with dynamic reasoning.

How it works:
    1. User asks a question
    2. LLM receives the question + tool definitions
    3. LLM returns either a tool call or a final answer
    4. If tool call → execute it, feed result back to LLM → repeat
    5. If final answer → print and stop

Tools available to the agent:
    - search_etf_docs    : semantic search over your ChromaDB corpus
    - get_live_data      : fetch current price, returns, AUM via yfinance
    - list_available_etfs: list all ETFs in the corpus (no args needed)

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

import json
import argparse
from pathlib import Path

# ── Local imports ──────────────────────────────────────────────────────────────
from config import INDEX_PATH as DB_PATH
from llm import PROVIDERS, SESSION, LLMClient, Message, ToolSpec, get_llm
from retrieve import Retriever, route_query
from live_data import get_etf_live_data, format_for_prompt
from registry import FUNDS


# ── Config ─────────────────────────────────────────────────────────────────────

MAX_ITERATIONS  = 6              # safety cap on the agent loop
MAX_TOKENS      = 1024
K_RETRIEVE      = 6              # chunks per retrieval call

KNOWN_ISINS = {isin: {"issuer": f.issuer, "name": f.name} for isin, f in FUNDS.items()}


# ── Tool definitions (sent to the LLM) ────────────────────────────────────────
# These tell the LLM what tools exist and what arguments they take.
# The LLM decides when and how to call them.

TOOLS = [
    ToolSpec(
        name="search_etf_docs",
        description=(
            "Search the ETF document corpus (factsheets and KIDs) for information "
            "about costs, fees, risk indicators, replication method, holdings, "
            "performance scenarios, or any other content from the PDF documents. "
            "Use this for questions that can be answered from the fund documents."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language search query, e.g. 'ongoing charges iShares'",
                },
                "issuer": {
                    "type": "string",
                    "description": "Filter by fund issuer. One of: ishares, ubs. Omit to search all.",
                    "enum": ["ishares", "ubs"],
                },
                "doc_type": {
                    "type": "string",
                    "description": "Filter by document type. One of: factsheet, kid. Omit to search both.",
                    "enum": ["factsheet", "kid"],
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
                    "description": (
                        "ISIN of the ETF. "
                        "iShares Core MSCI World = IE00B4L5Y983, "
                        "UBS Core MSCI World = IE00BD4TXV59"
                    ),
                },
            },
            "required": ["isin"],
        },
    ),
    ToolSpec(
        name="list_available_etfs",
        description=(
            "Return a list of all ETFs available in the corpus with their ISINs, "
            "issuers, and names. Call this if the user asks what funds are available "
            "or if you are unsure which ISIN to use."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
            "required": [],
        },
    ),
]


# ── Tool executor ──────────────────────────────────────────────────────────────
# This is where tool calls from the LLM are actually run.
# Returns a plain string — the LLM will read this as the tool result.

class ToolExecutor:
    def __init__(self, retriever: Retriever):
        self._retriever = retriever

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

    def _search_etf_docs(
        self,
        query: str,
        issuer: str = None,
        doc_type: str = None,
    ) -> str:
        """Run semantic search and return formatted chunks as a string."""
        # Build metadata filter from agent-supplied args
        filter_hint = {}
        if doc_type:
            filter_hint["doc_type"] = doc_type

        # Resolve ISINs from issuer filter
        if issuer:
            isin_list = [
                isin for isin, meta in KNOWN_ISINS.items()
                if meta["issuer"] == issuer
            ]
        else:
            isin_list = list(KNOWN_ISINS.keys())

        # Use comparative mode when searching across multiple ETFs
        mode = "comparative" if len(isin_list) > 1 else "single"
        if len(isin_list) == 1:
            filter_hint["etf_isin"] = isin_list[0]

        chunks = route_query(
            retriever=self._retriever,
            query=query,
            metadata_filter_hint=filter_hint,
            isin_list=isin_list,
            mode=mode,
            k=K_RETRIEVE,
        )

        if not chunks:
            return "No relevant document chunks found for this query."

        # Format chunks as plain text for the LLM to read
        lines = []
        for i, c in enumerate(chunks, 1):
            lines.append(
                f"[Chunk {i} | {c.etf_isin} | {c.issuer} | {c.doc_type} | "
                f"{c.year} | section: {c.section_heading or 'unknown'}]"
            )
            lines.append(c.text)
            lines.append("")
        return "\n".join(lines)

    def _get_live_data(self, isin: str) -> str:
        """Fetch live market data and return as a formatted string."""
        data = get_etf_live_data(isin)
        return format_for_prompt(data)

    def _list_available_etfs(self) -> str:
        lines = ["ETFs available in the corpus:"]
        for isin, meta in KNOWN_ISINS.items():
            lines.append(f"  - {meta['name']} | ISIN: {isin} | Issuer: {meta['issuer']}")
        return "\n".join(lines)


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are FundScope, an ETF research assistant.

You have access to two sources of information:
1. ETF document corpus (factsheets and KIDs) — search with search_etf_docs
2. Live market data (prices, returns, AUM) — fetch with get_live_data

Rules:
- Always cite document facts as [ISIN | issuer | doc_type | year].
- For questions about performance or price, always use get_live_data.
- For questions about costs, risk, or fund structure, always use search_etf_docs.
- For broad questions, use both tools.
- Be concise. Do not repeat tool output verbatim — synthesize it.
- If you cannot answer from the available tools, say so clearly.
"""


# ── Agent loop ─────────────────────────────────────────────────────────────────

class Agent:
    """
    The core agent loop.

    Each call to .run() starts a fresh conversation. The loop:
      1. Sends the user message + tool definitions to the LLM
      2. If the LLM returns tool calls → execute them, append results, loop
      3. If the LLM returns a text message → that is the final answer
      4. Hard stop after MAX_ITERATIONS to prevent runaway loops
    """

    def __init__(
        self,
        retriever: Retriever,
        provider: str = None,
        model: str = None,
        show_calls: bool = False,
        llm: LLMClient = None,
    ):
        self._llm      = llm or get_llm("generator", provider, model)
        self._executor = ToolExecutor(retriever)
        self._show_calls = show_calls
        print(f"[agent] Provider: {self._llm.provider}  |  Model: {self._llm.model}")

    def run(self, question: str) -> str:
        """
        Run the agent loop for a single question.
        Returns the final answer string.
        """
        messages = [Message("user", question)]

        for iteration in range(MAX_ITERATIONS):
            response = self._llm.chat(
                messages,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                max_tokens=MAX_TOKENS,
            )

            # ── Case 1: LLM produced a final answer ───────────────────────────
            if not response.tool_calls:
                return response.text or f"[Agent stopped: empty answer ({response.stop_reason}).]"

            # ── Case 2: LLM wants to call tools ───────────────────────────────
            messages.append(response.to_message())

            for tc in response.tool_calls:
                is_error = False
                if tc.error:
                    # Malformed arguments: tell the model so it can retry
                    result, is_error = f"[tool error] {tc.error}", True
                else:
                    try:
                        result = self._executor.execute(
                            tc.name, tc.arguments, show_calls=self._show_calls
                        )
                    except Exception as e:
                        result, is_error = f"[tool error] {type(e).__name__}: {e}", True

                messages.append(Message("tool", result, tool_call_id=tc.id,
                                        is_error=is_error))

            # Loop — LLM will now read the tool results and decide next step

        # Safety fallback
        return "[Agent stopped: maximum iterations reached without a final answer.]"


# ── Pretty printer ─────────────────────────────────────────────────────────────

def print_answer(question: str, answer: str) -> None:
    width = 64
    print(f"\n{'─' * width}")
    print(f"  Q: {question}")
    print(f"{'─' * width}")
    print(f"\n  {answer}\n")
    print(f"{'─' * width}\n")


def print_cost(before: tuple) -> None:
    """Print the tokens and estimated cost spent since the `before` snapshot."""
    usage0, cost0 = before
    u = SESSION.usage
    print(f"  [cost] in={u.input_tokens - usage0.input_tokens} "
          f"out={u.output_tokens - usage0.output_tokens} "
          f"cache_read={u.cache_read_tokens - usage0.cache_read_tokens} "
          f"| est. ${SESSION.cost_usd - cost0:.5f}\n")


def ask_agent(agent: Agent, question: str, show_cost: bool = False) -> None:
    before = SESSION.snapshot()
    answer = agent.run(question)
    print_answer(question, answer)
    if show_cost:
        print_cost(before)


# ── Interactive loop ───────────────────────────────────────────────────────────

HELP_TEXT = """
  Commands:
    <any question>   Ask anything about the ETFs
    funds            List ETFs in the corpus
    calls            Toggle showing tool calls (default: off)
    help             Show this message
    exit / quit      Exit
"""


def interactive_loop(agent: Agent, show_cost: bool = False) -> None:
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
            print(HELP_TEXT)
        elif cmd == "funds":
            funds = "\n".join(
                f"  • {meta['name']}  ({isin})  [{meta['issuer']}]"
                for isin, meta in KNOWN_ISINS.items()
            )
            print(f"\n  ETFs in corpus:\n{funds}\n")
        elif cmd == "calls":
            agent._show_calls = not agent._show_calls
            print(f"  Show tool calls: {'on' if agent._show_calls else 'off'}")
        else:
            try:
                ask_agent(agent, raw, show_cost)
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
    p.add_argument("--show_cost",   action="store_true",
                   help="Print tokens and estimated cost of each question")
    return p


def main():
    args      = build_parser().parse_args()
    retriever = Retriever(db_path=args.db_path)
    agent     = Agent(retriever=retriever, provider=args.provider, model=args.model,
                      show_calls=args.show_calls)

    if args.query:
        ask_agent(agent, args.query, show_cost=args.show_cost)
    else:
        interactive_loop(agent, show_cost=args.show_cost)
    print(f"  [session] {SESSION.summary()}")


if __name__ == "__main__":
    main()
