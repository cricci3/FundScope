"""
openai_compat.py — Generic backend for OpenAI-compatible providers

Adding a provider = one line in OPENAI_COMPAT_PROVIDERS; its key is read from
<NAME>_API_KEY (e.g. GROQ_API_KEY) and its model from LLM_MODEL / --model.

Tool calls are detected with `if message.tool_calls:` rather than
finish_reason (providers are inconsistent about it), and `content` may be None.
"""

import json

import openai

from llm.base import LLMClient, LLMError, LLMResponse, ToolCall, Usage

OPENAI_COMPAT_PROVIDERS = {
    "groq":       "https://api.groq.com/openai/v1",
    "gemini":     "https://generativelanguage.googleapis.com/v1beta/openai/",
    "openrouter": "https://openrouter.ai/api/v1",
}

MAX_RETRIES = 5


class OpenAICompatClient(LLMClient):

    def __init__(self, provider: str, model: str, api_key: str, min_interval_s: float = 0.0):
        super().__init__(model, min_interval_s)
        self.provider = provider
        self._client = openai.OpenAI(base_url=OPENAI_COMPAT_PROVIDERS[provider],
                                     api_key=api_key, max_retries=MAX_RETRIES)

    # ── Format conversion ──────────────────────────────────────────────────────

    @staticmethod
    def _to_api_messages(messages: list, system) -> list:
        out = [{"role": "system", "content": system}] if system else []
        for m in messages:
            if m.role == "tool":
                content = f"[error] {m.content}" if m.is_error else m.content
                out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": content})
            elif m.role == "assistant" and m.tool_calls:
                out.append({
                    "role": "assistant",
                    "content": m.content or None,
                    "tool_calls": [
                        {"id": tc.id, "type": "function",
                         "function": {"name": tc.name,
                                      "arguments": tc.raw_arguments or json.dumps(tc.arguments)}}
                        for tc in m.tool_calls
                    ],
                })
            else:
                out.append({"role": m.role, "content": m.content})
        return out

    @staticmethod
    def _to_api_tools(tools: list) -> list:
        return [{"type": "function",
                 "function": {"name": t.name, "description": t.description,
                              "parameters": t.parameters_json_schema}} for t in tools]

    # ── Call ───────────────────────────────────────────────────────────────────

    def _chat(self, messages, *, system, tools, temperature, max_tokens,
              response_schema, effort) -> LLMResponse:
        # `effort` is ignored: reasoning controls differ per provider.
        kwargs = {
            "model": self.model,
            "messages": self._to_api_messages(messages, system),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = self._to_api_tools(tools)
            kwargs["tool_choice"] = "auto"
        if response_schema:
            # Not every provider enforces the schema; callers must still validate.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema, "strict": True},
            }

        try:
            resp = self._client.chat.completions.create(**kwargs)
        except openai.AuthenticationError as e:
            raise LLMError(f"{self.provider}: invalid API key "
                           f"(check {self.provider.upper()}_API_KEY in .env).") from e
        except openai.PermissionDeniedError as e:
            raise LLMError(f"{self.provider}: permission denied — {e.message}") from e
        except openai.NotFoundError as e:
            raise LLMError(f"{self.provider}: model '{self.model}' not found — {e.message}") from e
        except openai.BadRequestError as e:
            raise LLMError(f"{self.provider}: bad request — {e.message}") from e
        except (openai.RateLimitError, openai.InternalServerError,
                openai.APIConnectionError) as e:
            raise LLMError(f"{self.provider}: still failing after {MAX_RETRIES} retries — {e}") from e
        except openai.APIStatusError as e:
            raise LLMError(f"{self.provider}: API error {e.status_code} — {e.message}") from e

        choice  = resp.choices[0]
        message = choice.message

        tool_calls = []
        if message.tool_calls:
            for tc in message.tool_calls:
                raw = tc.function.arguments or "{}"
                try:
                    args, err = json.loads(raw), None
                    if not isinstance(args, dict):
                        args, err = {}, f"arguments must be a JSON object, got: {raw}"
                except json.JSONDecodeError as e:
                    args, err = {}, f"malformed JSON arguments ({e}): {raw}"
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name,
                                           arguments=args, error=err, raw_arguments=raw))

        u = resp.usage
        cached = 0
        if u is not None and getattr(u, "prompt_tokens_details", None):
            cached = u.prompt_tokens_details.cached_tokens or 0
        usage = Usage(
            input_tokens=((u.prompt_tokens or 0) - cached) if u else 0,
            output_tokens=(u.completion_tokens or 0) if u else 0,
            cache_read_tokens=cached,
        )
        return LLMResponse(
            text=(message.content or "").strip(),
            tool_calls=tool_calls,
            usage=usage,
            model=resp.model or self.model,
            provider=self.provider,
            stop_reason="tool_use" if tool_calls else (choice.finish_reason or ""),
        )
