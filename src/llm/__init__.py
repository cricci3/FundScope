"""
llm — Provider-neutral LLM access for FundScope

    from llm import get_llm, Message, ToolSpec, SESSION

    llm  = get_llm("generator")                    # provider/model from .env
    resp = llm.chat([Message("user", "Hi")], system="Be brief.")
    print(resp.text, resp.usage, resp.cost_usd, SESSION.summary())

Precedence for provider and model: explicit arguments (CLI --provider/--model)
> environment / .env (LLM_PROVIDER, LLM_MODEL, JUDGE_*) > defaults in config.py.
"""

import os

import config
from llm.base import (
    SESSION, BudgetExceededError, CostTracker, LLMClient, LLMError, LLMResponse,
    Message, ToolCall, ToolSpec, Usage,
)
from llm.openai_compat import OPENAI_COMPAT_PROVIDERS

PROVIDERS = ["anthropic", *OPENAI_COMPAT_PROVIDERS]

__all__ = [
    "get_llm", "resolve", "PROVIDERS", "SESSION", "BudgetExceededError", "CostTracker",
    "LLMClient", "LLMError", "LLMResponse", "Message", "ToolCall", "ToolSpec", "Usage",
]

# The budget applies to everything this process spends.
SESSION.max_cost_usd = config.LLM_MAX_COST_USD


def resolve(role: str = "generator", provider: str | None = None,
            model: str | None = None) -> tuple[str, str]:
    """Return the (provider, model) pair that get_llm() would use."""
    if role not in ("generator", "judge"):
        raise ValueError(f"role must be 'generator' or 'judge', got {role!r}")
    env_provider = config.LLM_PROVIDER if role == "generator" else config.JUDGE_PROVIDER
    env_model    = config.LLM_MODEL    if role == "generator" else config.JUDGE_MODEL

    provider = (provider or env_provider).lower()
    if provider not in PROVIDERS:
        raise LLMError(f"Unknown provider '{provider}'. Known: {', '.join(PROVIDERS)}")

    # The env model belongs to the env provider: ignore it if --provider changed it.
    if not model and provider == env_provider.lower():
        model = env_model
    model = model or config.DEFAULT_MODELS.get(provider, {}).get(role)
    if not model:
        var = "LLM_MODEL" if role == "generator" else "JUDGE_MODEL"
        raise LLMError(f"No default model for provider '{provider}': set {var} in .env "
                       f"or pass --model.")
    return provider, model


def get_llm(role: str = "generator", provider: str | None = None,
            model: str | None = None) -> LLMClient:
    provider, model = resolve(role, provider, model)

    key_var = f"{provider.upper()}_API_KEY"
    api_key = os.getenv(key_var)
    if not api_key:
        raise LLMError(f"{key_var} is not set. Copy .env.example to .env and add the key.")

    if provider == "anthropic":
        from llm.anthropic_backend import AnthropicClient
        return AnthropicClient(model, api_key, config.LLM_MIN_INTERVAL_S)

    from llm.openai_compat import OpenAICompatClient
    return OpenAICompatClient(provider, model, api_key, config.LLM_MIN_INTERVAL_S)
