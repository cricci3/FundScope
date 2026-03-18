"""
evaluate.py — Evaluation scaffold for the ETF RAG pipeline
ETF RAG Project — Phase 5

Measures three things independently:
  1. Retrieval quality  — are the right chunks retrieved?
  2. Answer faithfulness — does the answer contradict the source chunks?
  3. Attribution accuracy — are sources correctly cited?

Each metric can be run standalone. The full eval loop runs all three.

Usage:
    # Full evaluation against ground truth
    python evaluate.py --ground_truth evaluation/ground_truth.json \
                       --pipeline_output evaluation/pipeline_output.json \
                       --report evaluation/report.json

    # Retrieval-only (useful during Phase 3 before generation is wired up)
    python evaluate.py --ground_truth evaluation/ground_truth.json \
                       --pipeline_output evaluation/pipeline_output.json \
                       --retrieval_only

Pipeline output format (one entry per question):
    [
      {
        "qid": "T1_001",
        "question": "...",
        "retrieved_chunks": [
          {"chunk_id": "...", "etf_ticker": "EUNL", "doc_type": "factsheet",
           "year": 2023, "section_heading": "key facts", "text": "..."}
        ],
        "generated_answer": "The ongoing charge for EUNL is 0.20%.",
        "cited_sources": [
          {"etf_ticker": "EUNL", "doc_type": "factsheet", "year": 2023}
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
    precision_at_k: float       # fraction of retrieved chunks that are relevant
    recall_at_k: float          # fraction of required sources that were retrieved
    any_relevant_retrieved: bool  # loose pass/fail


@dataclass
class FaithfulnessResult:
    qid: str
    query_type: int
    answer_contains_expected: Optional[bool]  # None for Type 4 (rubric-based)
    rubric_scores: Optional[dict]             # for Type 4 only
    has_unsupported_claims: bool              # True if answer goes beyond retrieved chunks


@dataclass
class AttributionResult:
    qid: str
    query_type: int
    correct_sources_cited: bool    # all required sources present in cited_sources
    wrong_sources_cited: bool      # any cited source doesn't match ground truth
    attribution_score: float       # 0.0–1.0


@dataclass
class QuestionEval:
    qid: str
    query_type: int
    query_type_label: str
    difficulty: str
    retrieval: RetrievalResult
    faithfulness: FaithfulnessResult
    attribution: AttributionResult


# ── Retrieval evaluation ───────────────────────────────────────────────────────

def source_matches_chunk(source: dict, chunk: dict) -> bool:
    """
    Check if a ground-truth source specification matches a retrieved chunk.
    Matching is on (etf_ticker, doc_type, year) — section_heading is a soft hint,
    not a hard requirement, because retrieval may surface the right content
    from an adjacent chunk.
    """
    ticker_match = (
        source.get("etf_ticker", "").upper() ==
        chunk.get("etf_ticker", chunk.get("metadata", {}).get("etf_ticker", "")).upper()
    )
    doc_type_match = (
        source.get("doc_type", "") ==
        chunk.get("doc_type", chunk.get("metadata", {}).get("doc_type", ""))
    )
    year_match = (
        int(source.get("year", 0)) ==
        int(chunk.get("year", chunk.get("metadata", {}).get("year", 0)))
    )
    return ticker_match and doc_type_match and year_match


def evaluate_retrieval(question: dict, pipeline_entry: dict) -> RetrievalResult:
    """
    Precision@k and Recall@k against ground truth sources.

    Precision@k = |retrieved ∩ relevant| / |retrieved|
    Recall@k    = |retrieved ∩ relevant| / |required sources|

    A retrieved chunk is "relevant" if it matches any required source.
    """
    required_sources = question.get("sources", [])
    retrieved_chunks = pipeline_entry.get("retrieved_chunks", [])

    if not retrieved_chunks:
        return RetrievalResult(
            qid=question["qid"],
            query_type=question["query_type"],
            precision_at_k=0.0,
            recall_at_k=0.0,
            any_relevant_retrieved=False,
        )

    # For each retrieved chunk, check if it matches any required source
    relevant_retrieved = [
        chunk for chunk in retrieved_chunks
        if any(source_matches_chunk(src, chunk) for src in required_sources)
    ]

    # For each required source, check if at least one retrieved chunk matches
    covered_sources = [
        src for src in required_sources
        if any(source_matches_chunk(src, chunk) for chunk in retrieved_chunks)
    ]

    precision = len(relevant_retrieved) / len(retrieved_chunks)
    recall    = len(covered_sources)    / len(required_sources) if required_sources else 0.0

    return RetrievalResult(
        qid=question["qid"],
        query_type=question["query_type"],
        precision_at_k=round(precision, 3),
        recall_at_k=round(recall, 3),
        any_relevant_retrieved=len(relevant_retrieved) > 0,
    )


# ── Faithfulness evaluation ────────────────────────────────────────────────────

def normalise(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s%\.\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def answer_contains_expected(answer: str, expected: str) -> bool:
    """
    Fuzzy check: does the generated answer contain the expected answer string?
    Normalises both sides before comparing.
    Uses substring match. For numerical answers (e.g. "0.20%"), this is precise enough.
    For longer prose answers, consider upgrading to an LLM-as-judge call.
    """
    return normalise(expected) in normalise(answer)


def check_unsupported_claims(answer: str, retrieved_chunks: list[dict]) -> bool:
    """
    Heuristic: flag if the answer contains numerical figures (percentages, currency amounts)
    that do not appear in any retrieved chunk text.
    This is a lightweight proxy for hallucination detection.
    For a rigorous check, use an LLM-as-judge prompt (see evaluate_faithfulness_llm below).
    """
    import re
    # Extract numbers from the answer
    answer_numbers = set(re.findall(r"\d+\.?\d*\s*%|\d[\d,]+", answer))
    if not answer_numbers:
        return False  # no numerical claims to check

    # Collect all text from retrieved chunks
    all_chunk_text = " ".join(c.get("text", "") for c in retrieved_chunks)

    unsupported = [
        num for num in answer_numbers
        if num not in all_chunk_text
    ]
    return len(unsupported) > 0


def score_rubric(answer: str, rubric: dict) -> dict:
    """
    Score a Type 4 answer against its rubric.
    Returns a dict with pass/fail per rubric item.
    This is intentionally simple — upgrade to LLM-as-judge for the real eval.
    """
    must_mention  = rubric.get("must_mention", [])
    should_mention = rubric.get("should_mention", [])
    must_not_contain = rubric.get("must_not_contain", [])

    answer_lower = answer.lower()

    results = {
        "must_mention":  {item: any(kw.lower() in answer_lower for kw in item.split()) for item in must_mention},
        "should_mention": {item: any(kw.lower() in answer_lower for kw in item.split()) for item in should_mention},
        "must_not_contain": {item: (item.lower() not in answer_lower) for item in must_not_contain},
    }
    must_score    = sum(results["must_mention"].values()) / len(must_mention) if must_mention else 1.0
    should_score  = sum(results["should_mention"].values()) / len(should_mention) if should_mention else 1.0
    must_not_pass = all(results["must_not_contain"].values())

    results["summary"] = {
        "must_mention_score":   round(must_score, 2),
        "should_mention_score": round(should_score, 2),
        "must_not_violated":    must_not_pass,
        "overall_pass":         must_score == 1.0 and must_not_pass,
    }
    return results


def evaluate_faithfulness(question: dict, pipeline_entry: dict) -> FaithfulnessResult:
    query_type = question["query_type"]
    answer     = pipeline_entry.get("generated_answer", "")
    retrieved  = pipeline_entry.get("retrieved_chunks", [])
    expected   = question.get("expected_answer")
    rubric     = question.get("answer_rubric")

    contains_expected = None
    rubric_scores     = None

    if query_type in (1, 2, 3) and expected and "<FILL" not in expected:
        contains_expected = answer_contains_expected(answer, expected)
    elif query_type == 4 and rubric:
        rubric_scores = score_rubric(answer, rubric)

    unsupported = check_unsupported_claims(answer, retrieved)

    return FaithfulnessResult(
        qid=question["qid"],
        query_type=query_type,
        answer_contains_expected=contains_expected,
        rubric_scores=rubric_scores,
        has_unsupported_claims=unsupported,
    )


# ── Attribution evaluation ─────────────────────────────────────────────────────

def evaluate_attribution(question: dict, pipeline_entry: dict) -> AttributionResult:
    """
    Check whether cited_sources in the pipeline output align with ground-truth sources.

    correct_sources_cited: every required source was cited
    wrong_sources_cited:   any cited source is not in the required set
    attribution_score:     |correct ∩ cited| / |required|
    """
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
        src for src in required
        if any(source_matches_chunk(src, cite) for cite in cited)
    ]
    wrongly_cited = [
        cite for cite in cited
        if not any(source_matches_chunk(src, cite) for src in required)
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
    """
    Compute summary statistics sliced by query_type and difficulty.
    """
    def _mean(vals):
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    def _slice(items, key, val):
        return [r for r in items if getattr(r, key, None) == val]

    # Overall
    overall = {
        "n":                       len(results),
        "retrieval_precision_mean": _mean([r.retrieval.precision_at_k for r in results]),
        "retrieval_recall_mean":    _mean([r.retrieval.recall_at_k    for r in results]),
        "any_relevant_pct":         _mean([float(r.retrieval.any_relevant_retrieved) for r in results]),
        "attribution_score_mean":   _mean([r.attribution.attribution_score for r in results]),
        "faithfulness_pass_pct":    _mean([
            float(r.faithfulness.answer_contains_expected)
            for r in results
            if r.faithfulness.answer_contains_expected is not None
        ]),
        "unsupported_claims_pct":   _mean([float(r.faithfulness.has_unsupported_claims) for r in results]),
    }

    # By query type
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

    # By difficulty
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


# ── LLM-as-judge stub ──────────────────────────────────────────────────────────
# Uncomment and wire up once your generation pipeline is working.
# This replaces the heuristic faithfulness check for Type 4 questions.

# def evaluate_faithfulness_llm(question: dict, pipeline_entry: dict) -> dict:
#     """
#     Use Claude to judge faithfulness and rubric satisfaction.
#     Pass retrieved chunks as context so the judge can verify grounding.
#     """
#     import anthropic
#     client = anthropic.Anthropic()
#     context = "\n\n---\n\n".join(
#         f"[{c.get('etf_ticker','?')} | {c.get('doc_type','?')} | {c.get('year','?')}]\n{c.get('text','')}"
#         for c in pipeline_entry.get("retrieved_chunks", [])
#     )
#     rubric_str = json.dumps(question.get("answer_rubric", {}), indent=2)
#     prompt = f"""You are evaluating a RAG system answer.
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
# 1. Does it cover all must_mention items? (yes/no + which are missing)
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

    questions     = {q["qid"]: q  for q in gt_data["questions"]}
    pipeline_out  = {e["qid"]: e  for e in po_data}

    results: list[QuestionEval] = []
    missing_qids = []

    for qid, question in questions.items():
        if qid not in pipeline_out:
            missing_qids.append(qid)
            continue

        entry = pipeline_out[qid]

        retrieval    = evaluate_retrieval(question, entry)
        faithfulness = (
            FaithfulnessResult(qid=qid, query_type=question["query_type"],
                               answer_contains_expected=None,
                               rubric_scores=None,
                               has_unsupported_claims=False)
            if retrieval_only
            else evaluate_faithfulness(question, entry)
        )
        attribution = (
            AttributionResult(qid=qid, query_type=question["query_type"],
                              correct_sources_cited=False,
                              wrong_sources_cited=False,
                              attribution_score=0.0)
            if retrieval_only
            else evaluate_attribution(question, entry)
        )

        results.append(QuestionEval(
            qid=qid,
            query_type=question["query_type"],
            query_type_label=question.get("query_type_label", ""),
            difficulty=question.get("difficulty", ""),
            retrieval=retrieval,
            faithfulness=faithfulness,
            attribution=attribution,
        ))

    if missing_qids:
        print(f"[warn] {len(missing_qids)} questions missing from pipeline output: {missing_qids}")

    summary = aggregate(results)
    return results, summary


def print_summary(summary: dict) -> None:
    print("\n" + "═" * 56)
    print("  EVALUATION SUMMARY")
    print("═" * 56)

    ov = summary["overall"]
    print(f"\n  Overall ({ov['n']} questions)")
    print(f"  Retrieval Precision@k : {ov['retrieval_precision_mean']:.1%}")
    print(f"  Retrieval Recall@k    : {ov['retrieval_recall_mean']:.1%}")
    print(f"  Any Relevant Retrieved: {ov['any_relevant_pct']:.1%}")
    print(f"  Attribution Score     : {ov['attribution_score_mean']:.1%}")
    print(f"  Faithfulness Pass     : {ov.get('faithfulness_pass_pct', 0):.1%}")
    print(f"  Unsupported Claims    : {ov['unsupported_claims_pct']:.1%}")

    print("\n  By Query Type")
    for label, stats in summary.get("by_query_type", {}).items():
        print(f"  {label} (n={stats['n']}): "
              f"P={stats['retrieval_precision_mean']:.1%}  "
              f"R={stats['retrieval_recall_mean']:.1%}  "
              f"Attr={stats['attribution_score_mean']:.1%}")

    print("\n  By Difficulty")
    for diff, stats in summary.get("by_difficulty", {}).items():
        print(f"  {diff:6s} (n={stats['n']}): "
              f"P={stats['retrieval_precision_mean']:.1%}  "
              f"R={stats['retrieval_recall_mean']:.1%}")

    print("═" * 56 + "\n")


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate ETF RAG pipeline")
    p.add_argument("--ground_truth",    type=Path, required=True)
    p.add_argument("--pipeline_output", type=Path, required=True,
                   help="JSON list of pipeline outputs (one entry per qid)")
    p.add_argument("--report",          type=Path, default=Path("evaluation/report.json"),
                   help="Where to write the full per-question report")
    p.add_argument("--retrieval_only",  action="store_true",
                   help="Skip faithfulness and attribution (use before generation is wired up)")
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
                "qid":             r.qid,
                "query_type":      r.query_type,
                "query_type_label":r.query_type_label,
                "difficulty":      r.difficulty,
                "retrieval":       asdict(r.retrieval),
                "faithfulness":    asdict(r.faithfulness),
                "attribution":     asdict(r.attribution),
            }
            for r in results
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2))
    print(f"Full report written → {args.report}")


if __name__ == "__main__":
    main()