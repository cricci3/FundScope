"""
evaluate.py — Evaluation of the ETF RAG pipeline

Measures independently:
  1. Retrieval quality    — are the right documents retrieved? (precision/recall@k)
  2. Answer correctness   — heuristic: expected key values (Type 1–2) / rubric keywords (Type 4)
  3. Unsupported numbers  — heuristic: figures in the answer that are absent from the context
  4. Attribution accuracy — are the right sources cited?
  5. LLM judge (--judge llm) — correctness, groundedness, rubric and verdict with a
     motivation, from a different model than the generator (default Claude Sonnet 5.5)

Identifier note: ground truth uses etf_isin (e.g. "IE00B4L5Y983") as the
primary ETF identifier. source_matches_chunk() matches on (etf_isin, doc_type, year).

Usage:
    # Heuristic evaluation (no API calls)
    python evaluation/evaluate.py --ground_truth evaluation/ground_truth.json \\
                       --pipeline_output evaluation/runs/<run>.json \\
                       --report evaluation/report.json

    # With the LLM judge (costs credit: use --limit / --qids while developing)
    python evaluation/evaluate.py ... --judge llm [--judge_provider anthropic --judge_model claude-sonnet-5-5]

    # Retrieval-only (output of run_retrieval.py)
    python evaluation/evaluate.py ... --retrieval_only

    # Only a subset (must match the subset the pipeline ran on)
    python evaluation/evaluate.py ... --limit 3
    python evaluation/evaluate.py ... --qids T1_001 T2_003

    # Print metric deltas against an earlier report (e.g. the baseline)
    python evaluation/evaluate.py ... --compare evaluation/baseline_cloud.json

Pipeline output format — {"run": {...metadata...}, "results": [...]} as written
by pipeline.py --run_eval or run_retrieval.py (a bare list of results is accepted
too), one result per question:
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

import re
import sys
import json
import argparse
from datetime import datetime, timezone
from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import config  # noqa: E402


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
    answer_contains_expected: Optional[bool]  # None when there are no expected_values (Type 4, T2_008)
    rubric_scores: Optional[dict]             # Type 4 only
    has_unsupported_claims: bool
    expected_details: Optional[dict] = None   # which expected values were found
    unsupported_numbers: list = field(default_factory=list)
    abstained: bool = False                   # answer says the documents lack the information


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
    judge: Optional[dict] = None              # --judge llm only


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


# ── Text and number normalisation ──────────────────────────────────────────────

CITATION_RE = re.compile(r"\[[A-Z]{2}[A-Z0-9]{10}\s*\|[^\]]*\]")
ISIN_RE     = re.compile(r"\b[A-Z]{2}[A-Z0-9]{9}\d\b")
# 1,309 · 10 000 · 0.20 · 0,20 — optionally followed by %
NUMBER_RE   = re.compile(r"(?<![\w.,])(\d{1,3}(?:[ ,]\d{3})+(?!\d)|\d+(?:[.,]\d+)?)(\s*%)?")
ABSTAIN_RE  = re.compile(
    r"(do(?:es)? not (?:contain|include|specify|provide|state|mention|describe|disclose)\b|"
    r"\bnot (?:explicitly |clearly )?(?:stated|specified|disclosed|provided|included|mentioned|"
    r"described|available)\b|cannot (?:adequately |fully )?(?:answer|determine|compare)|"
    r"insufficient information|no (?:information|details) (?:about|on|regarding))", re.I)


def strip_citations(text: str) -> str:
    return CITATION_RE.sub(" ", text)


def drop_abstentions(text: str) -> str:
    """
    Remove sentences that say information is missing: they often list candidate
    values ("full replication, sampling or synthetic") that must not count as found.
    """
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return " ".join(s for s in sentences if not ABSTAIN_RE.search(s))


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _parse_number(token: str) -> float:
    if re.fullmatch(r"\d{1,3}(?:[ ,]\d{3})+", token):      # thousands separators
        return float(re.sub(r"[ ,]", "", token))
    return float(token.replace(",", "."))                  # decimal comma


def extract_numbers(text: str, skip_trivial: bool = False) -> list[tuple[float, bool, str]]:
    """
    Numbers in `text` as (value, is_percent, original). ISINs are removed first.
    skip_trivial drops years (1900–2099) and bare integers 0–10 (list markers,
    "4 out of 7"-style scales), which make the unsupported-numbers check noisy.
    """
    text = ISIN_RE.sub(" ", text)
    out = []
    for m in NUMBER_RE.finditer(text):
        token, pct = m.group(1), bool(m.group(2))
        value = _parse_number(token)
        if skip_trivial and not pct:
            is_int = value.is_integer() and not re.search(r"[.,]\d", token)
            if is_int and (1900 <= value <= 2099 or value <= 10):
                continue
        out.append((value, pct, m.group(0).strip()))
    return out


def _is_numeric_value(value: str) -> Optional[tuple[float, bool]]:
    m = re.fullmatch(r"\s*(\d+(?:[.,]\d+)?)\s*(%)?\s*", value)
    return (float(m.group(1).replace(",", ".")), bool(m.group(2))) if m else None


def value_present(value, text: str) -> bool:
    """`value` is a string or a list of alternatives; numbers are compared numerically."""
    alternatives = value if isinstance(value, list) else [value]
    numbers = None
    for alt in alternatives:
        num = _is_numeric_value(str(alt))
        if num is not None:
            if numbers is None:
                numbers = extract_numbers(text)
            if any(abs(v - num[0]) < 1e-9 and p == num[1] for v, p, _ in numbers):
                return True
        elif _norm_text(str(alt)) in _norm_text(text):
            return True
    return False


# ── Faithfulness evaluation (heuristic) ────────────────────────────────────────

def _entity_segments(text: str, label: str, aliases: dict) -> str:
    """
    Part of `text` about the ETF `label` (an ISIN): from each mention of it (ISIN or
    issuer alias) up to the next mention of another ETF. Whole text if never mentioned.
    """
    mentions = []
    for isin, names in aliases.items():
        for name in [isin, *names]:
            for m in re.finditer(re.escape(name), text, re.I):
                mentions.append((m.start(), isin))
    mentions.sort()
    if not any(isin == label for _, isin in mentions):
        return text
    parts = []
    for i, (pos, isin) in enumerate(mentions):
        if isin != label:
            continue
        end = next((p for p, other in mentions[i + 1:] if other != label), len(text))
        parts.append(text[pos:end])
    return " ".join(parts)


def _answer_contains_expected(question: dict, answer: str) -> tuple[Optional[bool], Optional[dict]]:
    """
    Type 1: every value in expected_values must appear in the answer.
    Type 2: expected_values is {label: [values]}; labels that are ISINs are looked up
    in the part of the answer about that ETF, other labels in the whole answer.
    Returns (all found, {label: {value: found}}), or (None, None) without expected_values.
    """
    expected = question.get("expected_values")
    if not expected:
        return None, None
    text = drop_abstentions(strip_citations(answer))
    groups = expected if isinstance(expected, dict) else {"answer": expected}
    aliases = {s["etf_isin"]: [s["issuer"]] for s in question.get("sources", [])
               if s.get("etf_isin") and s.get("issuer")}

    details = {}
    for label, values in groups.items():
        scope = _entity_segments(text, label, aliases) if label in aliases else text
        details[label] = {json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v:
                          value_present(v, scope) for v in values}
    return all(all(d.values()) for d in details.values()), details


def _unsupported_numbers(answer: str, retrieved: list[dict]) -> list[str]:
    """
    Figures in the answer (citations, ISINs, years and small integers removed) whose
    value appears in no retrieved chunk. 0.2% matches 0.20%; 1,309 matches 1309.
    """
    context_values = {round(v, 6) for c in retrieved
                      for v, _, _ in extract_numbers(_get(c, "text"))}
    missing = []
    for value, _, original in extract_numbers(strip_citations(answer), skip_trivial=True):
        if round(value, 6) not in context_values and original not in missing:
            missing.append(original)
    return missing


def _score_rubric(answer: str, rubric: dict) -> dict:
    """
    Heuristic Type 4 rubric score. Items are {"item", "keywords", "min_match"}: an item
    is met when at least min_match keyword groups (a string or a list of alternatives)
    appear in the answer. Plain-string items and must_not_contain cannot be checked
    by keywords: they are reported as None and left to the LLM judge.
    """
    answer = drop_abstentions(answer)

    def _check(items):
        out = {}
        for it in items:
            if isinstance(it, dict) and it.get("keywords"):
                hits = sum(value_present(kw, answer) for kw in it["keywords"])
                out[it["item"]] = hits >= it.get("min_match", len(it["keywords"]))
            else:
                out[it["item"] if isinstance(it, dict) else it] = None
        return out

    def _score(results):
        checked = [v for v in results.values() if v is not None]
        return round(sum(checked) / len(checked), 2) if checked else None

    must_results   = _check(rubric.get("must_mention", []))
    should_results = _check(rubric.get("should_mention", []))
    must_not       = {(it["item"] if isinstance(it, dict) else it): None
                      for it in rubric.get("must_not_contain", [])}

    must_score = _score(must_results)
    return {
        "must_mention":     must_results,
        "should_mention":   should_results,
        "must_not_contain": must_not,
        "summary": {
            "must_mention_score":   must_score,
            "should_mention_score": _score(should_results),
            "must_not_violated":    None,        # judge only
            "overall_pass":         must_score == 1.0 if must_score is not None else None,
        },
    }


def evaluate_faithfulness(question: dict, pipeline_entry: dict) -> FaithfulnessResult:
    query_type = question["query_type"]
    answer     = pipeline_entry.get("generated_answer", "")
    retrieved  = pipeline_entry.get("retrieved_chunks", [])
    rubric     = question.get("answer_rubric")

    contains_expected, details, rubric_scores = None, None, None
    if query_type in (1, 2, 3):
        contains_expected, details = _answer_contains_expected(question, answer)
    elif query_type == 4 and rubric:
        rubric_scores = _score_rubric(answer, rubric)

    unsupported = _unsupported_numbers(answer, retrieved)

    return FaithfulnessResult(
        qid=question["qid"],
        query_type=query_type,
        answer_contains_expected=contains_expected,
        rubric_scores=rubric_scores,
        has_unsupported_claims=bool(unsupported),
        expected_details=details,
        unsupported_numbers=unsupported,
        abstained=bool(ABSTAIN_RE.search(answer)),
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

_CORRECTNESS_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0, "abstained": 0.0}


def _mean(vals):
    vals = list(vals)
    return round(sum(vals) / len(vals), 3) if vals else None


def _judge_metrics(results: list[QuestionEval]) -> dict:
    judged = [r.judge for r in results if r.judge and "error" not in r.judge]
    if not judged:
        return {}
    scored = [_CORRECTNESS_SCORE[j["correctness"]] for j in judged if j["correctness"] in _CORRECTNESS_SCORE]
    return {
        "judge_n":             len(judged),
        "judge_pass_pct":      _mean(float(j["verdict"] == "pass") for j in judged),
        "judge_score_mean":    _mean({"pass": 1.0, "partial": 0.5, "fail": 0.0}[j["verdict"]] for j in judged),
        "judge_correct_mean":  _mean(scored),
        "judge_grounded_pct":  _mean(float(j["grounded"]) for j in judged),
        "judge_abstained_pct": _mean(float(j["correctness"] == "abstained") for j in judged),
    }


def _metrics(results: list[QuestionEval]) -> dict:
    m = {
        "n": len(results),
        "retrieval_precision_mean": _mean(r.retrieval.precision_at_k for r in results),
        "retrieval_recall_mean":    _mean(r.retrieval.recall_at_k    for r in results),
        "any_relevant_pct":         _mean(float(r.retrieval.any_relevant_retrieved) for r in results),
        "attribution_score_mean":   _mean(r.attribution.attribution_score for r in results),
        "expected_values_pass_pct": _mean(
            float(r.faithfulness.answer_contains_expected) for r in results
            if r.faithfulness.answer_contains_expected is not None),
        "rubric_must_mention_mean": _mean(
            r.faithfulness.rubric_scores["summary"]["must_mention_score"] for r in results
            if r.faithfulness.rubric_scores
            and r.faithfulness.rubric_scores["summary"]["must_mention_score"] is not None),
        "unsupported_numbers_pct":  _mean(float(r.faithfulness.has_unsupported_claims) for r in results),
        "abstained_pct":            _mean(float(r.faithfulness.abstained) for r in results),
    }
    m.update(_judge_metrics(results))
    return m


def aggregate(results: list[QuestionEval]) -> dict:
    by_type = {f"type_{qt}": _metrics([r for r in results if r.query_type == qt])
               for qt in (1, 2, 3, 4) if any(r.query_type == qt for r in results)}
    by_difficulty = {d: _metrics([r for r in results if r.difficulty == d])
                     for d in ("easy", "medium", "hard") if any(r.difficulty == d for r in results)}
    return {"overall": _metrics(results), "by_query_type": by_type, "by_difficulty": by_difficulty}


# ── Main eval loop ─────────────────────────────────────────────────────────────

def select_questions(questions: list, qids: Optional[list] = None,
                     limit: Optional[int] = None) -> list:
    """Same selection as pipeline.py --qids / --limit."""
    if qids:
        unknown = set(qids) - {q["qid"] for q in questions}
        if unknown:
            raise SystemExit(f"Unknown qids: {sorted(unknown)}")
        questions = [q for q in questions if q["qid"] in set(qids)]
    if limit is not None:
        questions = questions[:limit]
    return questions


def load_pipeline_output(path: Path) -> tuple[list, dict]:
    """Return (results, run metadata). Older outputs are a bare list without metadata."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data, {}
    return data["results"], data.get("run", {})


def run_evaluation(
    ground_truth_path: Path,
    pipeline_output_path: Path,
    retrieval_only: bool = False,
    qids: Optional[list] = None,
    limit: Optional[int] = None,
    judge_llm=None,
    judge_effort: Optional[str] = None,
) -> tuple[list[QuestionEval], dict, str]:
    """Returns (per-question results, summary, judge status)."""

    gt_data = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    po_data, _ = load_pipeline_output(pipeline_output_path)

    # Only evaluate questions in the active "questions" list — not _parked_type3
    questions    = {q["qid"]: q for q in select_questions(gt_data["questions"], qids, limit)}
    pipeline_out = {e["qid"]: e for e in po_data}

    results: list[QuestionEval] = []
    missing_qids = []
    judge_status = "off" if judge_llm is None else "complete"

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

    if judge_llm is not None:
        from judge import judge_question
        from llm import BudgetExceededError, LLMError

    for qid, question in questions.items():
        if qid not in pipeline_out:
            missing_qids.append(qid)
            continue

        entry = pipeline_out[qid]
        qt    = question["query_type"]

        retrieval    = evaluate_retrieval(question, entry)
        faithfulness = _empty_faithful(qid, qt) if retrieval_only else evaluate_faithfulness(question, entry)
        attribution  = _empty_attr(qid, qt)     if retrieval_only else evaluate_attribution(question, entry)

        verdict = None
        if judge_llm is not None and judge_status == "complete":
            try:
                verdict = judge_question(judge_llm, question, entry, effort=judge_effort)
                label = verdict.get("verdict") or f"error: {verdict.get('error')}"
                print(f"  [judge] {qid}: {label}")
            except BudgetExceededError as e:
                judge_status = "partial (budget)"
                print(f"  [judge] STOPPED: {e}")
            except LLMError as e:
                verdict = {"error": str(e)}
                print(f"  [judge] {qid}: error: {e}")

        results.append(QuestionEval(
            qid=qid,
            query_type=qt,
            query_type_label=question.get("query_type_label", ""),
            difficulty=question.get("difficulty", ""),
            retrieval=retrieval,
            faithfulness=faithfulness,
            attribution=attribution,
            judge=verdict,
        ))

    if missing_qids:
        print(f"[warn] {len(missing_qids)} questions missing from pipeline output: {missing_qids}")

    return results, aggregate(results), judge_status


# ── Console output ─────────────────────────────────────────────────────────────

def _pct(v) -> str:
    return "  n/a" if v is None else f"{v:.1%}"


def print_summary(summary: dict) -> None:
    print("\n" + "═" * 56)
    print("  EVALUATION SUMMARY")
    print("═" * 56)

    ov = summary["overall"]
    print(f"\n  Overall  (n={ov['n']})")
    print(f"  Retrieval Precision@k  : {_pct(ov['retrieval_precision_mean'])}")
    print(f"  Retrieval Recall@k     : {_pct(ov['retrieval_recall_mean'])}")
    print(f"  Any Relevant Retrieved : {_pct(ov['any_relevant_pct'])}")
    print(f"  Attribution Score      : {_pct(ov['attribution_score_mean'])}")
    print(f"  Expected Values Found  : {_pct(ov['expected_values_pass_pct'])}   (heuristic, Type 1–2)")
    print(f"  Rubric must_mention    : {_pct(ov['rubric_must_mention_mean'])}   (heuristic, Type 4)")
    print(f"  Unsupported Numbers    : {_pct(ov['unsupported_numbers_pct'])}   (heuristic)")
    print(f"  Abstained              : {_pct(ov['abstained_pct'])}   (heuristic)")
    if "judge_n" in ov:
        print(f"\n  LLM judge  (n={ov['judge_n']})")
        print(f"  Verdict pass           : {_pct(ov['judge_pass_pct'])}")
        print(f"  Verdict score (p=1,½)  : {_pct(ov['judge_score_mean'])}")
        print(f"  Correctness            : {_pct(ov['judge_correct_mean'])}")
        print(f"  Grounded               : {_pct(ov['judge_grounded_pct'])}")
        print(f"  Abstained              : {_pct(ov['judge_abstained_pct'])}")

    print("\n  By Query Type")
    for label, s in summary.get("by_query_type", {}).items():
        judge = f"  Judge={_pct(s.get('judge_pass_pct'))}" if "judge_n" in s else ""
        print(f"  {label}  (n={s['n']}):  P={_pct(s['retrieval_precision_mean'])}  "
              f"R={_pct(s['retrieval_recall_mean'])}  Attr={_pct(s['attribution_score_mean'])}{judge}")

    print("\n  By Difficulty")
    for diff, s in summary.get("by_difficulty", {}).items():
        judge = f"  Judge={_pct(s.get('judge_pass_pct'))}" if "judge_n" in s else ""
        print(f"  {diff:6s}  (n={s['n']}):  P={_pct(s['retrieval_precision_mean'])}  "
              f"R={_pct(s['retrieval_recall_mean'])}{judge}")

    print("═" * 56 + "\n")


def print_run_cost(run: dict) -> None:
    """Tokens and cost of the pipeline run being scored."""
    if not run:
        print("  Pipeline run: no metadata (old output format)")
        return
    t = run.get("totals", {})
    cost = t.get("cost_usd")
    cost = f"${cost:.4f}" if cost is not None else "unknown"
    print(f"  Pipeline run: {run.get('provider')}/{run.get('model')}  "
          f"[{run.get('status')}, {run.get('n_questions')} q, {run.get('started_at')}]")
    print(f"  Tokens in={t.get('input_tokens')} out={t.get('output_tokens')} "
          f"cache_read={t.get('cache_read_tokens')}  |  est. cost {cost}")


def _flatten(summary: dict) -> dict:
    flat = {f"overall.{k}": v for k, v in summary.get("overall", {}).items()}
    for group in ("by_query_type", "by_difficulty"):
        for label, metrics in summary.get(group, {}).items():
            flat.update({f"{label}.{k}": v for k, v in metrics.items()})
    return flat


def print_comparison(current: dict, reference_path: Path) -> dict:
    """Print metric deltas (current − reference) and return them."""
    ref = json.loads(reference_path.read_text(encoding="utf-8"))
    ref_flat, cur_flat = _flatten(ref.get("summary", {})), _flatten(current)
    # Older reports used different names for the same heuristics
    renamed = {"faithfulness_pass_pct": "expected_values_pass_pct",
               "unsupported_claims_pct": "unsupported_numbers_pct"}
    ref_flat = {next((k.replace(o, n) for o, n in renamed.items() if k.endswith(o)), k): v
                for k, v in ref_flat.items()}

    print(f"  Comparison with {reference_path}")
    print(f"  {'metric':44s} {'ref':>8s} {'now':>8s} {'delta':>8s}")
    deltas = {}
    for key, now in cur_flat.items():
        if key.endswith(".n") or key.endswith("judge_n"):
            continue
        before = ref_flat.get(key)
        if isinstance(now, (int, float)) and isinstance(before, (int, float)):
            deltas[key] = round(now - before, 3)
            mark = "" if abs(deltas[key]) < 1e-9 else ("  ▲" if deltas[key] > 0 else "  ▼")
            print(f"  {key:44s} {before:8.3f} {now:8.3f} {deltas[key]:+8.3f}{mark}")
        elif before is None and isinstance(now, (int, float)) and key.startswith("overall."):
            print(f"  {key:44s} {'—':>8s} {now:8.3f}      new")
    print()
    return deltas


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Evaluate ETF RAG pipeline")
    p.add_argument("--ground_truth",    type=Path, required=True)
    p.add_argument("--pipeline_output", type=Path, required=True,
                   help="Run file from pipeline.py / run_retrieval.py")
    p.add_argument("--report",          type=Path,
                   default=Path("evaluation/report.json"),
                   help="Where to write the full per-question report")
    p.add_argument("--retrieval_only",  action="store_true",
                   help="Skip answer scoring and attribution (output of run_retrieval.py)")
    p.add_argument("--judge",           choices=["heuristic", "llm"], default="heuristic",
                   help="heuristic: no API calls (default). llm: add the LLM judge (costs credit)")
    p.add_argument("--judge_provider",  default=None,
                   help="Judge provider (default: JUDGE_PROVIDER from .env)")
    p.add_argument("--judge_model",     default=None,
                   help="Judge model (default: JUDGE_MODEL from .env, else provider default)")
    p.add_argument("--qids",            nargs="+", default=None,
                   help="Only evaluate these question ids")
    p.add_argument("--limit",           type=int, default=None,
                   help="Only evaluate the first N (selected) questions")
    p.add_argument("--compare",         type=Path, default=None,
                   help="Earlier report (e.g. evaluation/baseline_cloud.json) to print deltas against")
    return p


def main():
    args = build_parser().parse_args()

    judge_llm = None
    if args.judge == "llm":
        if args.retrieval_only:
            raise SystemExit("--judge llm needs generated answers (not --retrieval_only)")
        from llm import get_llm
        judge_llm = get_llm("judge", args.judge_provider, args.judge_model)
        print(f"[judge] {judge_llm.provider} / {judge_llm.model}  effort={config.JUDGE_EFFORT}")

    results, summary, judge_status = run_evaluation(
        args.ground_truth,
        args.pipeline_output,
        retrieval_only=args.retrieval_only,
        qids=args.qids,
        limit=args.limit,
        judge_llm=judge_llm,
        judge_effort=config.JUDGE_EFFORT,
    )
    print_summary(summary)
    _, run = load_pipeline_output(args.pipeline_output)
    print_run_cost(run)

    judge_cost = None
    if judge_llm is not None:
        from llm import SESSION
        judge_cost = round(SESSION.cost_usd, 6)
        print(f"  Judge: {SESSION.summary()}  [{judge_status}]")
        run_cost = (run.get("totals") or {}).get("cost_usd") or 0.0
        print(f"  Total (pipeline + judge): est. ${run_cost + judge_cost:.4f}")
    print()

    deltas = print_comparison(summary, args.compare) if args.compare else None

    report = {
        "meta": {
            "created_at":       datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "ground_truth":     str(args.ground_truth),
            "pipeline_output":  str(args.pipeline_output),
            "provider":         run.get("provider"),
            "model":            run.get("model"),
            "judge":            args.judge,
            "judge_provider":   judge_llm.provider if judge_llm else None,
            "judge_model":      judge_llm.model if judge_llm else None,
            "judge_effort":     config.JUDGE_EFFORT if judge_llm else None,
            "judge_status":     judge_status,
            "judge_cost_usd":   judge_cost,
            "embedding_model":  config.EMBEDDING_MODEL,
            "chunk_sizes":      config.CHUNK_SIZES,
            "chunk_overlaps":   config.CHUNK_OVERLAPS,
            "subset":           {"qids": args.qids, "limit": args.limit},
            "retrieval_only":   args.retrieval_only,
        },
        "pipeline_run": run,
        "summary": summary,
        "compared_with": {"report": str(args.compare), "deltas": deltas} if args.compare else None,
        "per_question": [
            {
                "qid":              r.qid,
                "query_type":       r.query_type,
                "query_type_label": r.query_type_label,
                "difficulty":       r.difficulty,
                "retrieval":        asdict(r.retrieval),
                "faithfulness":     asdict(r.faithfulness),
                "attribution":      asdict(r.attribution),
                "judge":            r.judge,
            }
            for r in results
        ],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Full report → {args.report}")


if __name__ == "__main__":
    main()
