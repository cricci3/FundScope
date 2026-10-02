"""
judge.py — LLM-as-judge for FundScope answers

One call per question to the judge model (get_llm(role="judge"), default
Claude Sonnet 5.5 — a different model from the generator). The reply is
structured JSON (schema enforced by the provider where supported, and always
re-validated here). Used by evaluate.py --judge llm.
"""

import json
from typing import Optional

from llm import LLMClient, LLMError, Message

VERDICTS    = ["pass", "partial", "fail"]
CORRECTNESS = ["correct", "partially_correct", "incorrect", "abstained", "not_applicable"]

_ITEM_MET = {
    "type": "object",
    "properties": {"item": {"type": "string"}, "met": {"type": "boolean"},
                   "note": {"type": "string"}},
    "required": ["item", "met", "note"],
    "additionalProperties": False,
}
_ITEM_VIOLATED = {
    "type": "object",
    "properties": {"item": {"type": "string"}, "violated": {"type": "boolean"},
                   "note": {"type": "string"}},
    "required": ["item", "violated", "note"],
    "additionalProperties": False,
}

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness":        {"type": "string", "enum": CORRECTNESS},
        "grounded":           {"type": "boolean"},
        "unsupported_claims": {"type": "array", "items": {"type": "string"}},
        "must_mention":       {"type": "array", "items": _ITEM_MET},
        "should_mention":     {"type": "array", "items": _ITEM_MET},
        "must_not_contain":   {"type": "array", "items": _ITEM_VIOLATED},
        "verdict":            {"type": "string", "enum": VERDICTS},
        "reasoning":          {"type": "string"},
    },
    "required": ["correctness", "grounded", "unsupported_claims", "must_mention",
                 "should_mention", "must_not_contain", "verdict", "reasoning"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """\
You are a strict evaluator of a retrieval-augmented assistant that answers questions \
about ETF factsheets and KIDs (Key Information Documents). For each case you receive \
the question, the context chunks the assistant retrieved, the assistant's answer, and \
the evaluation reference (an expected answer, or a rubric for open-ended questions).

Judge two things independently:

1. correctness — does the answer agree with the reference?
   - "correct": all key facts of the reference are present and right (wording may differ; \
0.2% and 0.20% are the same value).
   - "partially_correct": some key facts right, others missing or wrong.
   - "incorrect": the key facts are wrong, or the conclusion contradicts the reference.
   - "abstained": the answer says the documents do not contain the information, \
without giving a wrong value.
   - "not_applicable": only for rubric-based questions (no expected answer); then judge \
the rubric items instead.

2. grounded — is every factual claim in the answer supported by the retrieved context? \
Judge only against the context shown, never against your own knowledge or the reference. \
List each claim that the context does not support in unsupported_claims (empty list if none). \
Citation tags like [ISIN | issuer | doc_type | year] are not claims. A statement that the \
documents lack some information is not an unsupported claim.

Rubric (only when a rubric is given; otherwise return empty lists for must_mention, \
should_mention and must_not_contain):
- must_mention / should_mention: one entry per item, met=true only if the answer \
covers it with correct content.
- must_not_contain: one entry per item, violated=true if the answer does it.

verdict:
- "pass": correctness is "correct" (or, for rubric questions, every must_mention item \
is met and no must_not_contain item is violated) AND grounded is true.
- "fail": correctness is "incorrect" or "abstained", or (rubric) most must_mention items \
are not met or a must_not_contain item is violated.
- "partial": everything else.

reasoning: two to four sentences explaining the verdict, naming the specific facts \
that are right, wrong or missing. Copy rubric item texts verbatim into "item".\
"""


def _format_context(entry: dict) -> str:
    chunks = entry.get("retrieved_chunks", [])
    if not chunks:
        return "(no context retrieved)"
    return "\n\n".join(
        f"--- Chunk {i} [{c.get('etf_isin')} | {c.get('issuer')} | {c.get('doc_type')} | "
        f"{c.get('year')} | section: {c.get('section_heading') or 'unknown'}]\n{c.get('text', '')}"
        for i, c in enumerate(chunks, 1)
    )


def _rubric_texts(items: list) -> list:
    return [i["item"] if isinstance(i, dict) else str(i) for i in items or []]


def build_prompt(question: dict, entry: dict) -> str:
    rubric = question.get("answer_rubric")
    if rubric:
        reference = (
            "RUBRIC (no single expected answer):\n"
            + "must_mention:\n" + "\n".join(f"- {t}" for t in _rubric_texts(rubric.get("must_mention"))) + "\n"
            + "should_mention:\n" + "\n".join(f"- {t}" for t in _rubric_texts(rubric.get("should_mention"))) + "\n"
            + "must_not_contain:\n" + "\n".join(f"- {t}" for t in _rubric_texts(rubric.get("must_not_contain")))
        )
    else:
        reference = f"EXPECTED ANSWER (reference):\n{question.get('expected_answer')}"

    return (
        f"QUESTION:\n{question['question']}\n\n"
        f"{reference}\n\n"
        f"RETRIEVED CONTEXT:\n{_format_context(entry)}\n\n"
        f"ASSISTANT ANSWER:\n{entry.get('generated_answer', '')}"
    )


def validate(data: dict) -> Optional[str]:
    """Return an error message if `data` does not follow JUDGE_SCHEMA, else None."""
    if not isinstance(data, dict):
        return "not a JSON object"
    missing = [k for k in JUDGE_SCHEMA["required"] if k not in data]
    if missing:
        return f"missing keys: {missing}"
    if data["verdict"] not in VERDICTS:
        return f"bad verdict: {data['verdict']!r}"
    if data["correctness"] not in CORRECTNESS:
        return f"bad correctness: {data['correctness']!r}"
    if not isinstance(data["grounded"], bool):
        return "grounded is not a boolean"
    for key in ("unsupported_claims", "must_mention", "should_mention", "must_not_contain"):
        if not isinstance(data[key], list):
            return f"{key} is not a list"
    return None


def judge_question(llm: LLMClient, question: dict, entry: dict,
                   effort: Optional[str] = None, max_tokens: int = 8000) -> dict:
    """
    Judge one answer. Returns the validated judge JSON plus model/cost fields,
    or {"error": "..."} (with model/cost) when the reply cannot be used.
    LLMError / BudgetExceededError propagate to the caller.
    """
    if entry.get("generated_answer", "").startswith("ERROR:") or not entry.get("generated_answer"):
        return {"error": "no answer to judge", "model": llm.model, "cost_usd": 0.0}

    resp = llm.chat(
        [Message("user", build_prompt(question, entry))],
        system=SYSTEM_PROMPT,
        temperature=0.0,
        max_tokens=max_tokens,
        response_schema=JUDGE_SCHEMA,
        effort=effort,
    )
    meta = {"model": resp.model, "cost_usd": resp.cost_usd,
            "input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}

    if resp.stop_reason not in ("end_turn", "stop", "stop_sequence"):
        return {"error": f"judge stopped with stop_reason={resp.stop_reason}", **meta}
    try:
        data = json.loads(resp.text)
    except json.JSONDecodeError as e:
        return {"error": f"invalid JSON from judge: {e}", "raw": resp.text[:500], **meta}
    err = validate(data)
    if err:
        return {"error": f"judge output does not match schema: {err}", "raw": data, **meta}
    return {**data, **meta}


__all__ = ["JUDGE_SCHEMA", "build_prompt", "judge_question", "validate", "LLMError"]
