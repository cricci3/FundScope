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
    pip install yfinance
    Ollama running with a tool-capable model — recommended: mistral:7b or llama3.1:8b
    (llama3.2:3b does NOT support tool-calling reliably)

Usage:
    # Interactive agent mode
    python src/agent.py

    # Single question
    python src/agent.py --query "How has the iShares MSCI World performed this year?"

    # Show the tool calls the agent made
    python src/agent.py --show_calls

    # Use a different model
    python src/agent.py --model mistral:7b
"""

import json
import argparse
from pathlib import Path
from typing import Any

from openai import OpenAI

# ── Local imports ──────────────────────────────────────────────────────────────
ROOT    = Path(__file__).resolve().parent
DB_PATH = ROOT / "index" / "chroma_db"

from retrieve import Retriever, route_query
from live_data import get_etf_live_data, format_for_prompt, ISIN_TO_NAME


# ── Config ─────────────────────────────────────────────────────────────────────

OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_MODEL   = "mistral:7b"   # needs tool-calling support
MAX_ITERATIONS  = 6              # safety cap on the agent loop
K_RETRIEVE      = 6              # chunks per retrieval call

KNOWN_ISINS = {
    "IE00B4L5Y983": {"issuer": "ishares", "name": "iShares Core MSCI World"},
    "IE00BD4TXV59": {"issuer": "ubs",     "name": "UBS Core MSCI World"},
}


# ── Tool definitions (sent to the LLM) ────────────────────────────────────────
# These tell the LLM what tools exist and what arguments they take.
# The LLM decides when and how to call them.

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_etf_docs",
            "description": (
                "Search the ETF document corpus (factsheets and KIDs) for information "
                "about costs, fees, risk indicators, replication method, holdings, "
                "performance scenarios, or any other content from the PDF documents. "
                "Use this for questions that can be answered from the fund documents."
            ),
            "parameters": {
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
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_live_data",
            "description": (
                "Fetch real-time market data for an ETF: current price, 1-month, "
                "3-month, and year-to-date returns, AUM, and average daily volume. "
                "Use this for questions about recent performance, price, or liquidity. "
                "Do NOT use this for questions about fund documents (TER, SRI, etc.)."
            ),
            "parameters": {
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
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_available_etfs",
            "description": (
                "Return a list of all ETFs available in the corpus with their ISINs, "
                "issuers, and names. Call this if the user asks what funds are available "
                "or if you are unsure which ISIN to use."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
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
            return f"[error] Unknown tool: {name}"

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
        model: str = DEFAULT_MODEL,
        show_calls: bool = False,
    ):
        self._client   = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        self._model    = model
        self._executor = ToolExecutor(retriever)
        self._show_calls = show_calls
        print(f"[agent] Model: {model}  |  Endpoint: {OLLAMA_BASE_URL}")
        self._check_model()

    def _check_model(self):
        try:
            available = [m.id for m in self._client.models.list().data]
            if self._model not in available:
                print(
                    f"[warn] '{self._model}' not found in Ollama.\n"
                    f"       Available: {available}\n"
                    f"       Run: ollama pull {self._model}\n"
                    f"       Note: llama3.2:3b does NOT support tool-calling reliably."
                )
        except Exception as e:
            raise RuntimeError(
                f"Cannot connect to Ollama. Make sure it is running.\n"
                f"Error: {e}"
            )

    def run(self, question: str) -> str:
        """
        Run the agent loop for a single question.
        Returns the final answer string.
        """
        # Start conversation with system prompt + user question
        messages = [
            {"role": "system",  "content": SYSTEM_PROMPT},
            {"role": "user",    "content": question},
        ]

        for iteration in range(MAX_ITERATIONS):
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",   # LLM decides: call a tool or answer directly
            )

            choice  = response.choices[0]
            message = choice.message

            # ── Case 1: LLM wants to call tools ───────────────────────────────
            if choice.finish_reason == "tool_calls" and message.tool_calls:
                # Append the assistant's tool-call message
                messages.append({
                    "role":       "assistant",
                    "content":    message.content or "",
                    "tool_calls": [
                        {
                            "id":       tc.id,
                            "type":     "function",
                            "function": {
                                "name":      tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in message.tool_calls
                    ],
                })

                # Execute each tool call and append results
                for tc in message.tool_calls:
                    try:
                        args   = json.loads(tc.function.arguments)
                        result = self._executor.execute(
                            tc.function.name, args, show_calls=self._show_calls
                        )
                    except Exception as e:
                        result = f"[tool error] {e}"

                    messages.append({
                        "role":         "tool",
                        "tool_call_id": tc.id,
                        "content":      result,
                    })

                # Loop — LLM will now read the tool results and decide next step

            # ── Case 2: LLM produced a final answer ───────────────────────────
            else:
                return message.content.strip()

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


# ── Interactive loop ───────────────────────────────────────────────────────────

HELP_TEXT = """
  Commands:
    <any question>   Ask anything about the ETFs
    funds            List ETFs in the corpus
    calls            Toggle showing tool calls (default: off)
    help             Show this message
    exit / quit      Exit
"""


def interactive_loop(agent: Agent) -> None:
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
                answer = agent.run(raw)
                print_answer(raw, answer)
            except Exception as e:
                print(f"\n  [error] {e}\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FundScope Agent — ETF assistant with tool-calling"
    )
    p.add_argument("--query",       default=None,
                   help="Single question (omit for interactive mode)")
    p.add_argument("--model",       default=DEFAULT_MODEL,
                   help=f"Ollama model (default: {DEFAULT_MODEL}). "
                        "Must support tool-calling. Recommended: mistral:7b or llama3.1:8b")
    p.add_argument("--db_path",     type=Path, default=DB_PATH)
    p.add_argument("--show_calls",  action="store_true",
                   help="Print each tool call as it happens")
    return p


def main():
    args      = build_parser().parse_args()
    retriever = Retriever(db_path=args.db_path)
    agent     = Agent(retriever=retriever, model=args.model, show_calls=args.show_calls)

    if args.query:
        answer = agent.run(args.query)
        print_answer(args.query, answer)
    else:
        interactive_loop(agent)


if __name__ == "__main__":
    main()
