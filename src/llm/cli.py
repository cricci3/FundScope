"""
cli.py — Check the configured LLM provider

Usage:
    python src/llm/cli.py --ping
    python src/llm/cli.py --ping --provider groq --model llama-3.3-70b-versatile
    python src/llm/cli.py --ping --role judge
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # src/

from llm import PROVIDERS, SESSION, LLMError, Message, get_llm  # noqa: E402


def ping(role: str = "generator", provider: str | None = None, model: str | None = None) -> bool:
    """Minimal round trip (a few tokens). Prints model, tokens and cost; returns success."""
    try:
        llm  = get_llm(role, provider, model)
        resp = llm.chat([Message("user", "Reply with the single word: pong")], max_tokens=10)
    except LLMError as e:
        print(f"[ping] FAILED: {e}")
        return False
    cost = f"${resp.cost_usd:.6f}" if resp.cost_usd is not None else "unknown"
    print(f"[ping] {resp.provider} / {resp.model} → {resp.text!r}")
    print(f"[ping] tokens in={resp.usage.input_tokens} out={resp.usage.output_tokens}  cost={cost}")
    return True


def main():
    p = argparse.ArgumentParser(description="FundScope LLM provider check")
    p.add_argument("--ping", action="store_true", help="Send a minimal request")
    p.add_argument("--role", default="generator", choices=["generator", "judge"])
    p.add_argument("--provider", default=None, choices=PROVIDERS)
    p.add_argument("--model", default=None)
    args = p.parse_args()
    if not args.ping:
        p.print_help()
        return
    ok = ping(args.role, args.provider, args.model)
    print(f"[ping] session: {SESSION.summary()}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
