"""
backend.py — Everything the web UI needs that is not Streamlit code

Pure functions (formatting, evaluation reports) and the builders of the
retriever and the agent, kept here so they can be tested without a browser.
"""

import json
import re
from typing import Callable, Optional

from fundscope.citations import CITATION_PATTERN
from fundscope.config import EVAL_DIR
from fundscope.registry import FUNDS, get_fund

# ── Answer formatting ──────────────────────────────────────────────────────────

# Streamlit badge colours, one per fund in registry order (a 7th fund repeats gray)
_BADGE_COLORS = ["blue", "orange", "violet", "green", "red", "gray"]

DOC_LABELS = {"factsheet": "factsheet", "kid": "KID"}


def fund_short_name(isin: str) -> str:
    """'iShares Core MSCI World UCITS ETF' → 'iShares'."""
    fund = get_fund(isin)
    return fund.name.split()[0] if fund else isin


def _badge_color(isin: str) -> str:
    isins = list(FUNDS)
    i = isins.index(isin.upper()) if isin.upper() in isins else len(_BADGE_COLORS) - 1
    return _BADGE_COLORS[min(i, len(_BADGE_COLORS) - 1)]


def source_badge(isin: str, doc_type: str, year) -> str:
    label = f"{fund_short_name(isin)} · {DOC_LABELS.get(doc_type.lower(), doc_type)} {year}"
    return f":{_badge_color(isin)}-badge[{label}]"


def render_answer(text: str) -> str:
    """
    Markdown for st.markdown: [ISIN | issuer | doc_type | year] citations become
    coloured badges, and "$" is escaped (Streamlit would read "$5 ... $10" as LaTeX).
    """
    text = text.replace("$", r"\$")
    return CITATION_PATTERN.sub(lambda m: " " + source_badge(m.group(1), m.group(3), m.group(4)), text)


# Questions (or answers) about which fund suits an investor get the disclaimer (task 9.3)
_SUITABILITY_RE = re.compile(
    r"suitab|\bbetter\b|\bbest\b|recommend|should i|which (?:etf|fund|one)|prefer|cheaper|"
    r"adatt|miglior|conviene|consigli|quale (?:etf|fondo)|scegli|preferi", re.I)

DISCLAIMER = ("FundScope is a research demo built on public fund documents. "
              "It is not financial advice: check the official documents and your own situation before investing.")


def needs_disclaimer(question: str, answer: str = "") -> bool:
    return bool(_SUITABILITY_RE.search(question) or _SUITABILITY_RE.search(answer[:400]))


def tool_call_label(name: str, arguments: dict) -> str:
    """One line describing a tool call, for the live status of an answer."""
    if name == "search_etf_docs":
        parts = [f"“{arguments.get('query', '')}”"]
        isins = arguments.get("isins") or []
        if isins:
            parts.append(" + ".join(fund_short_name(i) for i in isins))
        elif arguments.get("issuer"):
            parts.append(arguments["issuer"])
        if arguments.get("doc_type"):
            parts.append(DOC_LABELS.get(arguments["doc_type"], arguments["doc_type"]))
        return "Searching the documents: " + " · ".join(parts)
    if name == "get_live_data":
        return f"Fetching live market data for {fund_short_name(arguments.get('isin', '?'))}"
    if name == "list_available_etfs":
        return "Listing the funds in the corpus"
    return f"Calling {name}"


# ── Retriever and agent ────────────────────────────────────────────────────────

def load_retriever():
    from fundscope.retrieve import Retriever
    return Retriever()


def build_agent(retriever, provider: Optional[str], model: Optional[str],
                on_tool_call: Optional[Callable[[str, dict], None]] = None):
    """Raises fundscope.llm.LLMError when the provider is not usable (e.g. no key)."""
    from fundscope.agent import Agent
    return Agent(retriever, provider=provider or None, model=model or None,
                 on_tool_call=on_tool_call)


# ── Evaluation reports ─────────────────────────────────────────────────────────

# (report file, label, engine description) — the reference runs shown in the UI
REPORTS = [
    ("baseline_cloud.json",      "Baseline (phase 3)", "fixed RAG pipeline"),
    ("report_phase4_agent.json", "Agent (phase 4)",    "tool-calling agent"),
    ("report_phase5_agent.json", "Agent (phase 5)",    "agent + hybrid search"),
    ("report_phase5_full.json",  "Full context",       "every document in the prompt"),
]


def eval_rows() -> list[dict]:
    """One row per reference report that exists in evaluation/."""
    rows = []
    for filename, label, engine in REPORTS:
        path = EVAL_DIR / filename
        if not path.exists():
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        overall = report["summary"]["overall"]
        run = report.get("pipeline_run") or {}
        meta = report.get("meta") or {}
        run_cost = (run.get("totals") or {}).get("cost_usd")
        rows.append({
            "run": label,
            "engine": engine,
            "questions": overall.get("n"),
            "pass": overall.get("judge_pass_pct"),
            "correctness": overall.get("judge_correct_mean"),
            "grounded": overall.get("judge_grounded_pct"),
            "attribution": overall.get("attribution_score_mean"),
            "retrieval_precision": overall.get("retrieval_precision_mean"),
            "retrieval_recall": overall.get("retrieval_recall_mean"),
            "cost_usd": round((run_cost or 0) + (meta.get("judge_cost_usd") or 0), 3),
            "model": meta.get("model") or run.get("model"),
            "judge": meta.get("judge_model"),
        })
    return rows
