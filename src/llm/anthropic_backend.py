"""
anthropic_backend.py — Native Anthropic Messages API backend (default provider)

Conversions to the Messages API format:
  - system goes in the separate `system` parameter, not in messages
  - tools are declared with `input_schema`
  - assistant tool calls are `tool_use` blocks; tool results go back as
    `tool_result` blocks inside a single `user` message

Retries: the SDK retries 408/409/429/5xx (529 overloaded included) with
exponential backoff; we raise max_retries. Auth, permission, bad-request and
exhausted-credit errors are not retried and surface as LLMError.

Prompt caching: a cache_control breakpoint on the system block caches tools +
system (render order is tools → system → messages). Prefixes shorter than the
model minimum (4096 tokens for Haiku 4.5) silently are not cached — no error,
no extra cost — so this only pays off once the prompt grows.
"""

import anthropic

from llm.base import LLMClient, LLMError, LLMResponse, ToolCall, Usage

MAX_RETRIES = 5

# Models that reject sampling parameters (temperature/top_p) with a 400.
# SDK 1.x dropped `temperature` from messages.create(); for models that still
# honour it (Haiku 4.5, the 4.6/4.5 line) we send it via extra_body, since the
# evaluation relies on temperature 0 for repeatable answers.
_NO_SAMPLING_PREFIXES = ("claude-opus-4-7", "claude-opus-4-8", "claude-opus-5",
                         "claude-sonnet-5", "claude-fable", "claude-mythos")

# Models that reject output_config.effort (400).
_NO_EFFORT_PREFIXES = ("claude-haiku-4-5", "claude-sonnet-4-5")


class AnthropicClient(LLMClient):
    provider = "anthropic"

    def __init__(self, model: str, api_key: str, min_interval_s: float = 0.0):
        super().__init__(model, min_interval_s)
        self._client = anthropic.Anthropic(api_key=api_key, max_retries=MAX_RETRIES)

    # ── Format conversion ──────────────────────────────────────────────────────

    @staticmethod
    def _to_api_messages(messages: list) -> list:
        out = []
        for m in messages:
            if m.role == "tool":
                block = {"type": "tool_result", "tool_use_id": m.tool_call_id,
                         "content": m.content}
                if m.is_error:
                    block["is_error"] = True
                # All results of one assistant turn go in a single user message
                if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list) \
                        and out[-1]["content"] and out[-1]["content"][0].get("type") == "tool_result":
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
            elif m.role == "assistant":
                if m.raw and m.raw[0] == "anthropic":
                    out.append({"role": "assistant", "content": m.raw[1]})
                    continue
                content = []
                if m.content:
                    content.append({"type": "text", "text": m.content})
                for tc in m.tool_calls:
                    content.append({"type": "tool_use", "id": tc.id,
                                    "name": tc.name, "input": tc.arguments})
                out.append({"role": "assistant", "content": content or m.content})
            else:
                out.append({"role": "user", "content": m.content})
        return out

    @staticmethod
    def _to_api_tools(tools: list) -> list:
        return [{"name": t.name, "description": t.description,
                 "input_schema": t.parameters_json_schema} for t in tools]

    # ── Call ───────────────────────────────────────────────────────────────────

    def _chat(self, messages, *, system, tools, temperature, max_tokens,
              response_schema, effort) -> LLMResponse:
        kwargs = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": self._to_api_messages(messages),
        }
        if system:
            kwargs["system"] = [{"type": "text", "text": system,
                                 "cache_control": {"type": "ephemeral"}}]
        if tools:
            kwargs["tools"] = self._to_api_tools(tools)
        if not self.model.startswith(_NO_SAMPLING_PREFIXES):
            kwargs["extra_body"] = {"temperature": temperature}

        output_config = {}
        if response_schema:
            output_config["format"] = {"type": "json_schema", "schema": response_schema}
        if effort and not self.model.startswith(_NO_EFFORT_PREFIXES):
            output_config["effort"] = effort
        if output_config:
            kwargs["output_config"] = output_config

        try:
            resp = self._client.messages.create(**kwargs)
        except anthropic.AuthenticationError as e:
            raise LLMError("Anthropic: invalid API key (check ANTHROPIC_API_KEY in .env).") from e
        except anthropic.PermissionDeniedError as e:
            raise LLMError(f"Anthropic: permission denied — {e.message}") from e
        except anthropic.NotFoundError as e:
            raise LLMError(f"Anthropic: model '{self.model}' not found — {e.message}") from e
        except anthropic.BadRequestError as e:
            if "credit balance" in str(e.message).lower():
                raise LLMError("Anthropic: credit exhausted — top up at console.anthropic.com "
                               "(Plans & Billing).") from e
            raise LLMError(f"Anthropic: bad request — {e.message}") from e
        except (anthropic.RateLimitError, anthropic.InternalServerError,
                anthropic.APIConnectionError) as e:
            raise LLMError(f"Anthropic: still failing after {MAX_RETRIES} retries — {e}") from e
        except anthropic.APIStatusError as e:
            raise LLMError(f"Anthropic: API error {e.status_code} — {e.message}") from e

        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name,
                                           arguments=dict(block.input or {})))

        u = resp.usage
        usage = Usage(
            input_tokens=u.input_tokens or 0,
            output_tokens=u.output_tokens or 0,
            cache_read_tokens=u.cache_read_input_tokens or 0,
            cache_write_tokens=u.cache_creation_input_tokens or 0,
        )
        return LLMResponse(
            text="".join(text_parts).strip(),
            tool_calls=tool_calls,
            usage=usage,
            model=resp.model,
            provider=self.provider,
            stop_reason=resp.stop_reason or "",
            raw=("anthropic", [b.model_dump(exclude_none=True) for b in resp.content]),
        )
