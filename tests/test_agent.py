"""The agent loop with a scripted LLM: tool calls → answer, no network."""

import copy

import pytest

from conftest import UBS
from fundscope import agent as agent_mod
from fundscope.agent import Agent
from fundscope.llm import LLMClient, LLMResponse, ToolCall, Usage


class ScriptedLLM(LLMClient):
    """Returns the scripted responses in order and records every request."""

    provider = "anthropic"

    def __init__(self, script):
        super().__init__("claude-haiku-4-5")
        self.script = list(script)
        self.requests = []

    def _chat(self, messages, *, system, tools, **kwargs):
        self.requests.append({"messages": copy.deepcopy(messages), "system": system, "tools": tools})
        text, calls = self.script.pop(0)
        return LLMResponse(text=text, tool_calls=calls, usage=Usage(1000, 50),
                           model=self.model, provider=self.provider,
                           stop_reason="tool_use" if calls else "end_turn")


def _search(call_id, **arguments):
    return ToolCall(call_id, "search_etf_docs", arguments)


def test_tool_call_then_cited_answer(retriever):
    llm = ScriptedLLM([
        ("", [_search("c1", query="Total Expense Ratio", isins=[UBS], doc_type="factsheet")]),
        ("Perfect! Now I have the information I need.\n\nThe TER is 0.06% [Chunk 1].", []),
    ])
    result = Agent(retriever, llm=llm, history_turns=0).run("What is the TER of the UBS ETF?")

    assert result.stop_reason == "answer" and result.iterations == 2
    assert result.answer == "The TER is 0.06% [IE00BD4TXV59 | ubs | factsheet | 2026]."
    assert [(s.etf_isin, s.doc_type) for s in result.cited_sources] == [(UBS, "factsheet")]
    assert result.chunks[0].chunk_id == "ubs-fs-kf"            # key facts come first
    assert result.tool_calls == [{"name": "search_etf_docs", "error": None, "arguments": {
        "query": "Total Expense Ratio", "isins": [UBS], "doc_type": "factsheet"}}]
    assert result.usage == Usage(2000, 100)
    assert result.cost_usd == pytest.approx(2 * (1000 * 1.0 + 50 * 5.0) / 1e6)

    # the second request carries the assistant turn and the numbered chunks
    second = llm.requests[1]["messages"]
    assert [m.role for m in second] == ["user", "assistant", "tool"]
    assert second[2].tool_call_id == "c1" and not second[2].is_error
    assert second[2].content.startswith("[Chunk 1] IE00BD4TXV59 | ubs | factsheet | 2026 Q4")
    assert {t.name for t in llm.requests[0]["tools"]} == {"search_etf_docs", "get_live_data",
                                                          "list_available_etfs"}


def test_bad_tool_arguments_are_returned_as_errors(retriever):
    llm = ScriptedLLM([
        ("", [_search("c1", query="TER", isins=["XX0000000000"]),
              ToolCall("c2", "search_etf_docs", {}, error="malformed JSON arguments")]),
        ("The documents do not contain sufficient information.", []),
    ])
    result = Agent(retriever, llm=llm, history_turns=0).run("TER?")
    tool_msgs = llm.requests[1]["messages"][2:]
    assert all(m.is_error for m in tool_msgs)
    assert "ValueError" in tool_msgs[0].content and "malformed JSON" in tool_msgs[1].content
    assert result.stop_reason == "answer" and result.cited_sources == []


def test_iteration_cap(retriever, monkeypatch):
    monkeypatch.setattr(agent_mod, "MAX_ITERATIONS", 3)
    llm = ScriptedLLM([("", [_search(f"c{i}", query="costs", isins=[UBS])]) for i in range(3)])
    result = Agent(retriever, llm=llm, history_turns=0).run("costs?")
    assert result.stop_reason == "max_iterations" and result.iterations == 3
    # before the last allowed call the model is told to stop searching
    assert llm.requests[2]["messages"][-1].content.endswith(agent_mod.FINAL_CALL_NOTICE)


def test_history_keeps_previous_answers(retriever):
    llm = ScriptedLLM([("First answer.", []), ("Second answer.", [])])
    agent = Agent(retriever, llm=llm, history_turns=1)
    agent.run("first question")
    agent.run("follow-up")
    assert [m.content for m in llm.requests[1]["messages"]] == ["first question", "First answer.", "follow-up"]
    agent.reset()
    assert agent.history_len == 0
