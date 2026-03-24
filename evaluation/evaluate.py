"""
evaluate.py — Evaluation scaffold for the ETF RAG pipeline
ETF RAG Project — Phase 5

Measures three things independently:
  1. Retrieval quality    — are the right chunks retrieved?
  2. Answer faithfulness  — does the answer contradict the source chunks?
  3. Attribution accuracy — are sources correctly cited?

Identifier note: ground truth uses etf_isin (e.g. "IE00B4L5Y983") as the
primary ETF identifier. source_matches_chunk() matches on (etf_isin, doc_type, year).

Usage:
    # Full evaluation
    python evaluation/evaluate.py --ground_truth evaluation/ground_truth.json \
                       --pipeline_output evaluation/pipeline_output.json \
                       --report evaluation/report.json

    # Retrieval-only (before generation is wired up)
    python evaluation/evaluate.py --ground_truth evaluation/ground_truth.json \
                       --pipeline_output evaluation/pipeline_output.json \
                       --retrieval_only

Pipeline output format — one entry per question:
    [
      {
        "qid": "T1_001",
        "question": "...",
        "retrieved_chunks": [
          {
            "chunk_id": "...",
            "etf_isin": "IE00B4L5Y983",
            "doc_type": "factsheet",
            "year": 2026,
            "section_heading": "key facts",
            "text": "..."
          }
        ],
        "generated_answer": "The TER of IE00B4L5Y983 is 0.20%.",
        "cited_sources": [
          {"etf_isin": "IE00B4L5Y983", "doc_type": "factsheet", "year": 2026}
        ]
      },
      ...
    ]
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional


# ── Result data structures ─────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    qid: str
    query_type: int
    precision_at_k: float
    recall_at_k: float
    any_relevant_retrieved: bool


@dataclass
class FaithfulnessResult:
    qid: str
    query_type: int
    answer_contains_expected: Optional[bool]  # None for Type 4
    rubric_scores: Optional[dict]             # Type 4 only
    has_unsupported_claims: bool


@dataclass
class AttributionResult:
    qid: str
    query_type: int
    correct_sources_cited: bool
    wrong_sources_cited: bool
    attribution_score: float


@dataclass
class QuestionEval:
    qid: str
    query_type: int
    query_type_label: str
    difficulty: str
    retrieval: RetrievalResult
    faithfulness: FaithfulnessResult
    attribution: AttributionResult


# ── Source matching ────────────────────────────────────────────────────────────

def _get(obj: dict, *keys: str, default="") -> str:
    """
    Try multiple key names in order, return the first match as a string.
    Handles both flat chunk dicts and nested {"metadata": {...}} dicts.
    """
    for key in keys:
        val = obj.get(key)
        if val is not None:
            return str(val)
    # Also check inside a nested "metadata" sub-dict (from ingest.py output)
    meta = obj.get("metadata", {})
    for key in keys:
        val = meta.get(key)
        if val is not None:
            return str(val)
    return default


def source_matches_chunk(source: dict, chunk: dict) -> bool:
    """
    Check if a ground-truth source spec matches a retrieved chunk.
    Primary identifier: etf_isin.
    Hard match on (etf_isin, doc_type, year).
    section_heading is informational only — not matched here.
    """
    isin_match = (
        source.get("etf_isin", "").upper() ==
        _get(chunk, "etf_isin").upper()
    )
    doc_type_match = (
        source.get("doc_type", "") ==
        _get(chunk, "doc_type")
    )
    year_match = (
        str(source.get("year", "")) ==
        str(_get(chunk, "year"))
    )
    return isin_match and doc_type_match and year_match


# ── Retrieval evaluation ───────────────────────────────────────────────────────

def evaluate_retrieval(question: dict, pipeline_entry: dict) -> RetrievalResult:
    """
    Precision@k = |retrieved ∩ relevant| / |retrieved|
    Recall@k    = |required sources covered| / |required sources|
    """
    required  = question.get("sources", [])
    retrieved = pipeline_entry.get("retrieved_chunks", [])

    if not retrieved:
        return RetrievalResult(
            qid=question["qid"],
            query_type=question["query_type"],
            precision_at_k=0.0,
            recall_at_k=0.0,
            any_relevant_retrieved=False,
        )

    relevant_retrieved = [
        c for c in retrieved
        if any(source_matches_chunk(s, c) for s in required)
    ]
    covered_sources = [
        s for s in required
        if any(source_matches_chunk(s, c) for c in retrieved)
    ]

    precision = len(relevant_retrieved) / len(retrieved)
    recall    = len(covered_sources) / len(required) if required else 0.0

    return RetrievalResult(
        qid=question["qid"],
        query_type=question["query_type"],
        precision_at_k=round(precision, 3),
        recall_at_k=round(recall, 3),
        any_relevant_retrieved=len(relevant_retrieved) > 0,
    )


# ── Faithfulness evaluation ────────────────────────────────────────────────────

def _normalise(text: str) -> str:
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s%\.\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _answer_contains_expected(answer: str, expected: str) -> bool:
    return _normalise(expected) in _normalise(answer)


def _check_unsupported_claims(answer: str, retrieved: list[dict]) -> bool:
    """
    Heuristic: flag if the answer contains numerical figures that do not
    appear in any retrieved chunk. Proxy for hallucination detection.
    Upgrade to LLM-as-judge for production eval.
    """
    import re
    answer_numbers = set(re.findall(r"\d+\.?\d*\s*%|\d[\d,]+", answer))
    if not answer_numbers:
        return False
    all_text = " ".join(_get(c, "text") for c in retrieved)
    return any(num not in all_text for num in answer_numbers)


def _score_rubric(answer: str, rubric: dict) -> dict:
    """Score a Type 4 answer against its rubric (keyword heuristic)."""
    answer_lower = answer.lower()

    def _check_items(items):
        return {
            item: any(kw.lower() in answer_lower for kw in item.split())
            for item in items
        }

    must_mention     = rubric.get("must_mention", [])
    should_mention   = rubric.get("should_mention", [])
    must_not_contain = rubric.get("must_not_contain", [])

    must_results     = _check_items(must_mention)
    should_results   = _check_items(should_mention)
    must_not_results = {item: (item.lower() not in answer_lower) for item in must_not_contain}

    must_score   = sum(must_results.values())   / len(must_mention)   if must_mention   else 1.0
    should_score = sum(should_results.values()) / len(should_mention) if should_mention else 1.0
    must_not_ok  = all(must_not_results.values())

    return {
        "must_mention":     must_results,
        "should_mention":   should_results,
        "must_not_contain": must_not_results,
        "summary": {
            "must_mention_score":   round(must_score, 2),
            "should_mention_score": round(should_score, 2),
            "must_not_violated":    must_not_ok,
            "overall_pass":         must_score == 1.0 and must_not_ok,
        },
    }


def evaluate_faithfulness(question: dict, pipeline_entry: dict) -> FaithfulnessResult:
    query_type = question["query_type"]
    answer     = pipeline_entry.get("generated_answer", "")
    retrieved  = pipeline_entry.get("retrieved_chunks", [])
    expected   = question.get("expected_answer")
    rubric     = question.get("answer_rubric")

    contains_expected = None
    rubric_scores     = None

    if query_type in (1, 2, 3) and expected and "<FILL" not in str(expected):
        contains_expected = _answer_contains_expected(answer, expected)
    elif query_type == 4 and rubric:
        rubric_scores = _score_rubric(answer, rubric)

    unsupported = _check_unsupported_claims(answer, retrieved)

    return FaithfulnessResult(
        qid=question["qid"],
        query_type=query_type,
        answer_contains_expected=contains_expected,
        rubric_scores=rubric_scores,
        has_unsupported_claims=unsupported,
    )


# ── Attribution evaluation ─────────────────────────────────────────────────────

def evaluate_attribution(question: dict, pipeline_entry: dict) -> AttributionResult:
    required = question.get("sources", [])
    cited    = pipeline_entry.get("cited_sources", [])

    if not cited:
        return AttributionResult(
            qid=question["qid"],
            query_type=question["query_type"],
            correct_sources_cited=False,
            wrong_sources_cited=False,
            attribution_score=0.0,
        )

    correctly_cited = [
        s for s in required
        if any(source_matches_chunk(s, c) for c in cited)
    ]
    wrongly_cited = [
        c for c in cited
        if not any(source_matches_chunk(s, c) for s in required)
    ]

    score = len(correctly_cited) / len(required) if required else 0.0

    return AttributionResult(
        qid=question["qid"],
        query_type=question["query_type"],
        correct_sources_cited=(len(correctly_cited) == len(required)),
        wrong_sources_cited=(len(wrongly_cited) > 0),
        attribution_score=round(score, 3),
    )


# ── Aggregate metrics ──────────────────────────────────────────────────────────

def aggregate(results: list[QuestionEval]) -> dict:
    def _mean(vals):
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    overall = {
        "n": len(results),
        "retrieval_precision_mean": _mean([r.retrieval.precision_at_k for r in results]),
        "retrieval_recall_mean":    _mean([r.retrieval.recall_at_k    for r in results]),
        "any_relevant_pct":         _mean([float(r.retrieval.any_relevant_retrieved) for r in results]),
        "attribution_score_mean":   _mean([r.attribution.attribution_score for r in results]),
        "faithfulness_pass_pct":    _mean([
            float(r.faithfulness.answer_contains_expected)
            for r in results
            if r.faithfulness.answer_contains_expected is not None
        ]),
        "unsupported_claims_pct": _mean([
            float(r.faithfulness.has_unsupported_claims) for r in results
        ]),
    }

    by_type = {}
    for qt in (1, 2, 3, 4):
        subset = [r for r in results if r.query_type == qt]
        if not subset:
            continue
        by_type[f"type_{qt}"] = {
            "n":                        len(subset),
            "retrieval_precision_mean": _mean([r.retrieval.precision_at_k for r in subset]),
            "retrieval_recall_mean":    _mean([r.retrieval.recall_at_k    for r in subset]),
            "attribution_score_mean":   _mean([r.attribution.attribution_score for r in subset]),
        }

    by_difficulty = {}
    for diff in ("easy", "medium", "hard"):
        subset = [r for r in results if r.difficulty == diff]
        if not subset:
            continue
        by_difficulty[diff] = {
            "n":                        len(subset),
            "retrieval_precision_mean": _mean([r.retrieval.precision_at_k for r in subset]),
            "retrieval_recall_mean":    _mean([r.retrieval.recall_at_k    for r in subset]),
        }

    return {"overall": overall, "by_query_type": by_type, "by_difficulty": by_difficulty}


# ── LLM-as-judge stub ─────────────────────────────────────────────────────────
# Uncomment once generation is wired up. Replaces heuristic rubric scoring
# for Type 4 questions with a grounded Claude judgment.

# def evaluate_faithfulness_llm(question: dict, pipeline_entry: dict) -> dict:
#     import anthropic
#     client = anthropic.Anthropic()
#     context = "\n\n---\n\n".join(
#         f"[{_get(c, 'etf_isin')} | {_get(c, 'doc_type')} | {_get(c, 'year')}]\n{_get(c, 'text')}"
#         for c in pipeline_entry.get("retrieved_chunks", [])
#     )
#     rubric_str = json.dumps(question.get("answer_rubric", {}), indent=2)
#     prompt = f"""You are evaluating a RAG system answer about ETF documents.
#
# QUESTION: {question['question']}
#
# RETRIEVED CONTEXT:
# {context}
#
# GENERATED ANSWER:
# {pipeline_entry.get('generated_answer', '')}
#
# RUBRIC:
# {rubric_str}
#
# Score the answer:
# 1. Does it cover all must_mention items? (yes/no + list any missing)
# 2. Does it violate any must_not_contain items? (yes/no)
# 3. Is every factual claim grounded in the retrieved context? (yes/no)
# Respond in JSON only."""
#     response = client.messages.create(
#         model="claude-sonnet-4-20250514",
#         max_tokens=1000,
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return json.loads(response.content[0].text)


# ── Main eval loop ─────────────────────────────────────────────────────────────

def run_evaluation(
    ground_truth_path: Path,
    pipeline_output_path: Path,
    retrieval_only: bool = False,
) -> tuple[list[QuestionEval], dict]:

    gt_data = json.loads(ground_truth_path.read_text())
    po_data = json.loads(pipeline_output_path.read_text())

    # Only evaluate questions in the active "questions" list — not _parked_type3
    questions    = {q["qid"]: q for q in gt_data["questions"]}
    pipeline_out = {e["qid"]: e for e in po_data}

    results: list[QuestionEval] = []
    missing_qids = []

    _empty_faithful = lambda qid, qt: FaithfulnessResult(
        qid=qid, query_type=qt,
        answer_contains_expected=None, rubric_scores=None,
        has_unsupported_claims=False,
    )
    _empty_attr = lambda qid, qt: AttributionResult(
        qid=qid, query_type=qt,
        correct_sources_cited=False, wrong_sources_cited=False,
        attribution_score=0.0,
    )

    for qid, question in questions.items():
        if qid not in pipeline_out:
            missing_qids.append(qid)
            continue

        entry = pipeline_out[qid]
        qt    = question["query_type"]

        retrieval    = evaluate_retrieval(question, entry)
        faithfulness = _empty_faithful(qid, qt) if retrieval_only else evaluate_faithfulness(question, entry)
        attribution  = _empty_attr(qid, qt)     if retrieval_only else evaluate_attribution(question, entry)

        results.append(QuestionEval(
            qid=qid,
            query_type=qt,
            query_type_label=question.get("query_type_label", ""),
            difficulty=question.get("difficulty", ""),
            retrieval=retrieval,
            faithfulness=faithfulness,
            attribution=attribution,
        ))

    if missing_qids:
        print(f"[warn] {len(missing_qids)} questions missing from pipeline output: {missing_qids}")

    return results, aggregate(results)


# ── Console summary ────────────────────────────────────────────────────────────

def print_summary(summary: dict) -> None:
    print("\n" + "═" * 56)
    print("  EVALUATION SUMMARY")
    print("═" * 56)

    ov = summary["overall"]
    print(f"\n  Overall  (n={ov['n']})")
    print(f"  Retrieval Precision@k  : {ov['retrieval_precision_mean']:.1%}")
    print(f"  Retrieval Recall@k     : {ov['retrieval_recall_mean']:.1%}")
    print(f"  Any Relevant Retrieved : {ov['any_relevant_pct']:.1%}")
    print(f"  Attribution Score      : {ov['attribution_score_mean']:.1%}")
    print(f"  Faithfulness Pass      : {ov.get('faithfulness_pass_pct', 0):.1%}")
    print(f"  Unsupported Claims     : {ov['unsupported_claims_pct']:.1%}")

    print("\n  By Query Type")
    for label, stats in summary.get("by_query_type", {}).items():
        print(f"  {label}  (n={stats['n']}):  "
              f"P={stats['retrieval_precision_mean']:.1%}  "
              f"R={stats['retrieval_recall_mean']:.1%}  "
              f"Attr={stats['attribution_score_mean']:.1%}")

    print("\n  By Difficulty")
    for diff, stats in summary.get("by_difficulty", {}).items():
        print(f"  {diff:6s}  (n={stats['n']}):  "
              f"P={stats['retrieval_precision_mean']:.1%}  "
              f"R={stats['retrieval_recall_mean']:.1%}")

    print("═" * 56 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate ETF RAG pipeline")
    p.add_argument("--ground_truth",    type=Path, required=True)
    p.add_argument("--pipeline_output", type=Path, required=True,
                   help="JSON list of pipeline outputs, one entry per qid")
    p.add_argument("--report",          type=Path,
                   default=Path("evaluation/report.json"),
                   help="Where to write the full per-question report")
    p.add_argument("--retrieval_only",  action="store_true",
                   help="Skip faithfulness and attribution — useful before generation is wired up")
    return p


def main():
    args = build_parser().parse_args()
    results, summary = run_evaluation(
        args.ground_truth,
        args.pipeline_output,
        retrieval_only=args.retrieval_only,
    )
    print_summary(summary)

    report = {
        "summary": summary,
        "per_question": [
            {
                "qid":              r.qid,
                "query_type":       r.query_type,
                "query_type_label": r.query_type_label,
                "difficulty":       r.difficulty,
                "retrieval":        asdict(r.retrieval),
                "faithfulness":     asdict(r.faithfulness),
                "attribution":      asdict(r.attribution),
            }
            for r in results
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"Full report → {args.report}")


if __name__ == "__main__":
    main()