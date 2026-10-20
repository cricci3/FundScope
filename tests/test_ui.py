"""Web UI: formatting helpers and the Streamlit app run headless (AppTest), with no network."""

import pytest
from streamlit.testing.v1 import AppTest

from conftest import ISHARES, UBS
from fundscope import live_data
from fundscope.agent import Agent
from fundscope.llm import LLMClient, LLMResponse, ToolCall, Usage
from fundscope.ui import backend
from fundscope.ui.launch import APP


# ── Helpers ────────────────────────────────────────────────────────────────────

def test_render_answer_turns_citations_into_badges_and_escapes_dollars():
    text = f"TER 0.20% [{ISHARES} | ishares | factsheet | 2026] vs 0.06% [{UBS} | ubs | kid | 2026]; costs $5."
    out = backend.render_answer(text)
    assert ":blue-badge[iShares · factsheet 2026]" in out
    assert ":orange-badge[UBS · KID 2026]" in out
    assert r"\$5" in out and ISHARES not in out


def test_needs_disclaimer():
    assert backend.needs_disclaimer("Which ETF is better for a long-term investor?")
    assert backend.needs_disclaimer("Quale ETF è più adatto a me?")
    assert not backend.needs_disclaimer("What is the TER of the UBS ETF?", "The TER is 0.06%.")


def test_tool_call_label():
    label = backend.tool_call_label("search_etf_docs", {"query": "TER", "isins": [ISHARES, UBS], "doc_type": "kid"})
    assert label == "Searching the documents: “TER” · iShares + UBS · KID"
    assert backend.tool_call_label("get_live_data", {"isin": UBS}) == "Fetching live market data for UBS"


def test_eval_rows_read_the_reference_reports():
    rows = backend.eval_rows()
    assert [r["run"] for r in rows] == [label for _, label, _ in backend.REPORTS]
    baseline = rows[0]
    assert baseline["questions"] == 17 and baseline["pass"] == pytest.approx(0.353)
    assert all(0 <= r["pass"] <= 1 and r["cost_usd"] > 0 for r in rows)


# ── App ────────────────────────────────────────────────────────────────────────

class _OneSearchLLM(LLMClient):
    provider = "anthropic"

    def __init__(self):
        super().__init__("claude-haiku-4-5")

    def _chat(self, messages, **kwargs):
        if not any(m.role == "tool" for m in messages):
            call = ToolCall("c1", "search_etf_docs", {"query": "Total Expense Ratio", "isins": [UBS],
                                                      "doc_type": "factsheet"})
            return LLMResponse("", [call], Usage(1000, 20), self.model, self.provider, "tool_use")
        return LLMResponse("The TER is 0.06% [Chunk 1].", [], Usage(1500, 30), self.model, self.provider, "end_turn")


LIVE = {"name": "UBS Core MSCI World UCITS ETF", "isin": UBS, "ticker": "TEST.DE", "as_of": "2026-10-20",
        "current_price": 30.5, "currency": "EUR", "week52_high": 31.0, "week52_low": 25.0,
        "return_1m_pct": 1.2, "return_3m_pct": 3.4, "return_ytd": 12.0, "total_assets": 5e9,
        "avg_volume": 120_000}


@pytest.fixture
def app(retriever, monkeypatch):
    monkeypatch.setattr(backend, "load_retriever", lambda: retriever)
    monkeypatch.setattr(backend, "build_agent", lambda retriever, provider, model, on_tool_call=None:
                        Agent(retriever, llm=_OneSearchLLM(), on_tool_call=on_tool_call))
    monkeypatch.setattr(live_data, "get_etf_live_data", lambda isin, use_cache=True: dict(LIVE, isin=isin))
    monkeypatch.setattr(live_data, "get_price_history",
                        lambda isin: [("2026-10-17", 30.1), ("2026-10-20", 30.5)])
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.run()
    assert not at.exception
    return at


def test_app_renders_every_tab(app):
    assert [t.label for t in app.tabs] == [":material/forum: Ask", ":material/show_chart: Live data",
                                           ":material/fact_check: Evaluation", ":material/schema: How it works"]
    assert any(m.label == "Price" and m.value == "30.50 EUR" for m in app.metric)
    assert any(m.label.startswith("Judge pass rate") for m in app.metric)
    assert len(app.button) >= 4                        # example questions


def test_app_answers_with_resolved_citation(app):
    app.chat_input[0].set_value("What is the TER of the UBS ETF?").run()
    assert not app.exception
    answer = [m.value for m in app.markdown if "0.06%" in m.value]
    assert answer and ":orange-badge[UBS · factsheet 2026]" in answer[0]
    assert any("Estimated cost" == m.label for m in app.metric)
