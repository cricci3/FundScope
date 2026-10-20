"""
Provider backends with canned SDK responses (no network, no keys), cost
estimates and the budget stop. Both directions of each format conversion are
covered, so switching provider stays verified without credit.
"""

import json

import anthropic.types
import openai.types.chat
import pytest

from fundscope import llm
from fundscope.llm import BudgetExceededError, LLMError, Message, ToolCall, ToolSpec, Usage
from fundscope.llm.anthropic_backend import AnthropicClient
from fundscope.llm.openai_compat import OpenAICompatClient
from fundscope.llm.pricing import estimate_cost

TOOL = ToolSpec("search_etf_docs", "Search the documents.",
                {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]})

# A tool round trip in neutral form: question → tool call → tool result
CONVERSATION = [
    Message("user", "What is the TER?"),
    Message("assistant", "Searching.", tool_calls=[ToolCall("call_1", "search_etf_docs", {"query": "TER"})]),
    Message("tool", "[Chunk 1] TER 0.20%", tool_call_id="call_1"),
    Message("tool", "bad args", tool_call_id="call_2", is_error=True),
]


class _Recorder:
    """Stands in for the SDK method: records the kwargs, returns a canned response."""

    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def __call__(self, **kwargs):
        self.kwargs = kwargs
        return self.response


# ── Anthropic ──────────────────────────────────────────────────────────────────

def _anthropic_client(model="claude-haiku-4-5", content=None, stop_reason="tool_use"):
    client = AnthropicClient(model, api_key="test-key")
    response = anthropic.types.Message.model_validate({
        "id": "msg_1", "type": "message", "role": "assistant", "model": model,
        "content": content or [
            {"type": "text", "text": "Let me search."},
            {"type": "tool_use", "id": "toolu_1", "name": "search_etf_docs", "input": {"query": "TER"}},
        ],
        "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 1000, "output_tokens": 100,
                  "cache_read_input_tokens": 2000, "cache_creation_input_tokens": 0},
    })
    recorder = _Recorder(response)
    client._client.messages.create = recorder
    return client, recorder


def test_anthropic_neutral_to_api():
    api = AnthropicClient._to_api_messages(CONVERSATION)
    assert api[0] == {"role": "user", "content": "What is the TER?"}
    assert api[1] == {"role": "assistant", "content": [
        {"type": "text", "text": "Searching."},
        {"type": "tool_use", "id": "call_1", "name": "search_etf_docs", "input": {"query": "TER"}},
    ]}
    # both tool results go in one user message
    assert api[2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_1", "content": "[Chunk 1] TER 0.20%"},
        {"type": "tool_result", "tool_use_id": "call_2", "content": "bad args", "is_error": True},
    ]}
    assert len(api) == 3
    assert AnthropicClient._to_api_tools([TOOL])[0]["input_schema"] == TOOL.parameters_json_schema


def test_anthropic_api_to_neutral_and_request():
    client, recorder = _anthropic_client()
    resp = client.chat([Message("user", "What is the TER?")], system="Be brief.", tools=[TOOL],
                       temperature=0.0, max_tokens=50)

    assert resp.text == "Let me search."
    assert resp.tool_calls == [ToolCall("toolu_1", "search_etf_docs", {"query": "TER"})]
    assert resp.usage == Usage(1000, 100, 2000, 0)
    assert resp.provider == "anthropic" and resp.stop_reason == "tool_use"
    # Haiku 4.5: 1000 × 1 $ + 100 × 5 $ + 2000 × 0.10 $ per million tokens
    assert resp.cost_usd == pytest.approx(0.0017)

    sent = recorder.kwargs
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["tools"][0]["name"] == "search_etf_docs"
    assert sent["extra_body"] == {"temperature": 0.0}
    assert "output_config" not in sent                 # Haiku 4.5 rejects effort


def test_anthropic_replays_native_content():
    client, _ = _anthropic_client()
    turn = client.chat([Message("user", "q")]).to_message()
    api = AnthropicClient._to_api_messages([Message("user", "q"), turn])
    assert api[1]["content"][1] == {"type": "tool_use", "id": "toolu_1", "name": "search_etf_docs",
                                    "input": {"query": "TER"}}


def test_anthropic_structured_output_and_effort():
    client, recorder = _anthropic_client("claude-sonnet-5-5",
                                         content=[{"type": "text", "text": '{"verdict": "pass"}'}],
                                         stop_reason="end_turn")
    resp = client.chat([Message("user", "judge")], response_schema={"type": "object"}, effort="medium")
    assert json.loads(resp.text) == {"verdict": "pass"}
    assert recorder.kwargs["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}},
                                                "effort": "medium"}
    assert "extra_body" not in recorder.kwargs        # Sonnet 5.5 rejects temperature


# ── OpenAI-compatible ──────────────────────────────────────────────────────────

def _openai_client(arguments='{"query": "TER"}'):
    client = OpenAICompatClient("groq", "llama-test", api_key="test-key")
    response = openai.types.chat.ChatCompletion.model_validate({
        "id": "c1", "object": "chat.completion", "created": 0, "model": "llama-test",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
            "role": "assistant", "content": None,
            "tool_calls": [{"id": "call_9", "type": "function",
                            "function": {"name": "search_etf_docs", "arguments": arguments}}],
        }}],
        "usage": {"prompt_tokens": 300, "completion_tokens": 20, "total_tokens": 320,
                  "prompt_tokens_details": {"cached_tokens": 100}},
    })
    recorder = _Recorder(response)
    client._client.chat.completions.create = recorder
    return client, recorder


def test_openai_neutral_to_api():
    api = OpenAICompatClient._to_api_messages(CONVERSATION, system="Be brief.")
    assert api[0] == {"role": "system", "content": "Be brief."}
    assert api[1] == {"role": "user", "content": "What is the TER?"}
    assert api[2]["tool_calls"] == [{"id": "call_1", "type": "function", "function": {
        "name": "search_etf_docs", "arguments": '{"query": "TER"}'}}]
    assert api[3] == {"role": "tool", "tool_call_id": "call_1", "content": "[Chunk 1] TER 0.20%"}
    assert api[4] == {"role": "tool", "tool_call_id": "call_2", "content": "[error] bad args"}
    tool = OpenAICompatClient._to_api_tools([TOOL])[0]
    assert tool["type"] == "function" and tool["function"]["parameters"] == TOOL.parameters_json_schema


def test_openai_api_to_neutral():
    client, recorder = _openai_client()
    resp = client.chat([Message("user", "q")], tools=[TOOL])
    assert resp.text == ""
    assert resp.tool_calls[0].arguments == {"query": "TER"} and resp.tool_calls[0].error is None
    assert resp.usage == Usage(input_tokens=200, output_tokens=20, cache_read_tokens=100)
    assert resp.stop_reason == "tool_use"
    assert resp.cost_usd is None                       # unknown model → no price, no error
    assert recorder.kwargs["tool_choice"] == "auto"


def test_openai_malformed_arguments_become_tool_error():
    client, _ = _openai_client(arguments='{"query": ')
    call = client.chat([Message("user", "q")]).tool_calls[0]
    assert call.arguments == {} and "malformed JSON" in call.error
    assert call.raw_arguments == '{"query": '


# ── Provider resolution, costs, budget ─────────────────────────────────────────

def test_resolve_precedence(monkeypatch):
    monkeypatch.setattr(llm.config, "LLM_PROVIDER", "anthropic")
    monkeypatch.setattr(llm.config, "LLM_MODEL", None)
    assert llm.resolve("generator") == ("anthropic", "claude-haiku-4-5")
    assert llm.resolve("judge", model="x") == ("anthropic", "x")
    with pytest.raises(LLMError):
        llm.resolve("generator", provider="groq")      # no default model for groq
    with pytest.raises(LLMError):
        llm.resolve("generator", provider="nope")


def test_get_llm_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError, match="ANTHROPIC_API_KEY"):
        llm.get_llm("generator", provider="anthropic", model="claude-haiku-4-5")


def test_estimate_cost_prefix_match():
    usage = Usage(input_tokens=1_000_000, output_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert estimate_cost("claude-haiku-4-5-20251001", usage) == pytest.approx(1.0 + 5.0 + 1.25)
    assert estimate_cost("unknown-model", usage) is None


def test_budget_stops_further_calls(fresh_session):
    client, recorder = _anthropic_client()
    fresh_session.max_cost_usd = 0.003
    client.chat([Message("user", "q")])                # 0.0017
    client.chat([Message("user", "q")])                # 0.0034 ≥ budget after this call
    recorder.kwargs = None
    with pytest.raises(BudgetExceededError):
        client.chat([Message("user", "q")])
    assert recorder.kwargs is None                     # the third request was never sent
    assert fresh_session.calls == 2
    assert fresh_session.cost_usd == pytest.approx(0.0034)
