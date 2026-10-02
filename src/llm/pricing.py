"""
pricing.py — USD prices per million tokens, used for cost estimates

Update the table when prices change. A model that is not listed gets cost
None (and one warning), never an error. Keys are matched as prefixes, so
"claude-haiku-4-5" also covers the dated id "claude-haiku-4-5-20251001".
"""

# model prefix: (input, output, cache_read, cache_write_5m)  — $ per 1M tokens
PRICES = {
    "claude-haiku-4-5":  (1.00,  5.00, 0.10, 1.25),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20, 2.50),
    "claude-opus-5-5":   (4.00, 20.00, 0.20, 5.00),
}

_warned: set = set()


def _lookup(model: str):
    for prefix in sorted(PRICES, key=len, reverse=True):
        if model.startswith(prefix):
            return PRICES[prefix]
    return None


def estimate_cost(model: str, usage) -> float | None:
    price = _lookup(model)
    if price is None:
        if model not in _warned:
            _warned.add(model)
            print(f"[llm] [warn] No price known for model '{model}' — cost not estimated "
                  f"(add it to src/llm/pricing.py)")
        return None
    p_in, p_out, p_read, p_write = price
    return (usage.input_tokens * p_in
            + usage.output_tokens * p_out
            + usage.cache_read_tokens * p_read
            + usage.cache_write_tokens * p_write) / 1_000_000
