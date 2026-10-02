"""
base.py — Provider-neutral types and the LLMClient interface

Everything outside src/llm/ talks to the model only through these types,
so it never needs to know which provider (Anthropic, Groq, ...) is underneath.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from llm.pricing import estimate_cost


# ── Neutral types ──────────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    name: str
    description: str
    parameters_json_schema: dict


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    # Set when the provider returned arguments that are not valid JSON:
    # the agent sends this back to the model as the tool result.
    error: Optional[str] = None
    raw_arguments: Optional[str] = None


@dataclass
class Message:
    role: str                                   # "user" | "assistant" | "tool"
    content: str = ""
    tool_calls: list = field(default_factory=list)   # list[ToolCall], assistant only
    tool_call_id: Optional[str] = None          # role="tool" only
    is_error: bool = False                      # role="tool": result is an error
    # Opaque provider-native content of an assistant turn (e.g. Anthropic
    # thinking blocks). Backends replay it verbatim when it is theirs.
    raw: Any = field(default=None, repr=False)


@dataclass
class Usage:
    input_tokens: int = 0          # uncached input tokens
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
            self.cache_read_tokens + other.cache_read_tokens,
            self.cache_write_tokens + other.cache_write_tokens,
        )


@dataclass
class LLMResponse:
    text: str
    tool_calls: list               # list[ToolCall]
    usage: Usage
    model: str
    provider: str
    stop_reason: str
    cost_usd: Optional[float] = None    # None when the model has no known price
    raw: Any = field(default=None, repr=False)

    def to_message(self) -> Message:
        """The assistant turn to append to the history before sending tool results."""
        return Message(role="assistant", content=self.text,
                       tool_calls=list(self.tool_calls), raw=self.raw)


# ── Errors ─────────────────────────────────────────────────────────────────────

class LLMError(RuntimeError):
    """Non-retryable provider error (bad key, no credit, bad request, ...)."""


class BudgetExceededError(LLMError):
    """The run's estimated cost reached LLM_MAX_COST_USD."""


# ── Session cost tracker ───────────────────────────────────────────────────────

class CostTracker:
    """Accumulates tokens and estimated cost over every call in this process."""

    def __init__(self):
        self.usage = Usage()
        self.cost_usd = 0.0
        self.calls = 0
        self.unpriced_calls = 0
        self.max_cost_usd: Optional[float] = None

    def record(self, response: LLMResponse) -> None:
        self.calls += 1
        self.usage = self.usage + response.usage
        if response.cost_usd is None:
            self.unpriced_calls += 1
        else:
            self.cost_usd += response.cost_usd

    def check_budget(self) -> None:
        if self.max_cost_usd is not None and self.cost_usd >= self.max_cost_usd:
            raise BudgetExceededError(
                f"Estimated cost ${self.cost_usd:.4f} reached the budget "
                f"${self.max_cost_usd:g} (LLM_MAX_COST_USD)."
            )

    def snapshot(self) -> tuple:
        return self.usage, self.cost_usd

    def summary(self) -> str:
        u = self.usage
        cost = f"${self.cost_usd:.5f}"
        if self.unpriced_calls:
            cost += f" (+{self.unpriced_calls} unpriced call(s))"
        return (f"{self.calls} call(s) | in={u.input_tokens} out={u.output_tokens} "
                f"cache_read={u.cache_read_tokens} cache_write={u.cache_write_tokens} "
                f"| est. cost {cost}")


SESSION = CostTracker()


# ── Interface ──────────────────────────────────────────────────────────────────

class LLMClient(ABC):
    """
    One method: chat(). Subclasses implement _chat() for their provider;
    this wrapper adds throttling, the budget check and cost accounting.
    """

    provider: str = ""

    def __init__(self, model: str, min_interval_s: float = 0.0):
        self.model = model
        self._min_interval_s = min_interval_s
        self._last_call = 0.0

    def chat(
        self,
        messages: list,
        *,
        system: Optional[str] = None,
        tools: Optional[list] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        response_schema: Optional[dict] = None,
        effort: Optional[str] = None,
    ) -> LLMResponse:
        """
        response_schema: JSON Schema the reply text must follow (structured
            output, enforced by the provider where supported). Parse resp.text.
        effort: reasoning-effort hint ("low" | "medium" | "high"), ignored by
            providers/models that do not support it.
        """
        SESSION.check_budget()

        wait = self._min_interval_s - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

        response = self._chat(messages, system=system, tools=tools or [],
                              temperature=temperature, max_tokens=max_tokens,
                              response_schema=response_schema, effort=effort)
        response.cost_usd = estimate_cost(response.model or self.model, response.usage)
        SESSION.record(response)
        return response

    @abstractmethod
    def _chat(self, messages, *, system, tools, temperature, max_tokens,
              response_schema, effort) -> LLMResponse:
        ...
