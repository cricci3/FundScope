"""
pipeline.py — End-to-end query handler
ETF RAG Project — Phase 4

Wires retrieve.py → generate.py into a single entry point.
Accepts a question, routes to the correct retrieval mode,
calls the LLM, and returns a structured response.

Batch output (--run_eval) is what evaluate.py expects:
    {
      "run":     {provider, model, timestamps, status, totals (tokens + cost)},
      "results": [ one entry per question, as below ]
    }
    {
      "qid":               "T1_001",
      "question":          "...",
      "query_type":        1,
      "retrieved_chunks":  [...],   ← list of chunk dicts with metadata
      "generated_answer":  "...",
      "cited_sources":     [...],   ← parsed from answer text
      "provider", "model", "input_tokens", "output_tokens",
      "cache_read_tokens", "cache_write_tokens", "cost_usd"
    }

Usage:

    # Single query (interactive)
    python src/pipeline.py --query "What is the TER of IE00B4L5Y983?" \
                       --etf_isin IE00B4L5Y983 --doc_type factsheet \
                       --query_type 1

    # Comparative query
    python src/pipeline.py --query "Compare the ongoing charges of IE00B4L5Y983 and IE00BD4TXV59" \
                       --isin_list IE00B4L5Y983 IE00BD4TXV59 \
                       --doc_type factsheet --query_type 2

    # Run all ground truth questions; output goes to evaluation/runs/<timestamp>_<model>.json
    python src/pipeline.py --run_eval

    # Development: only a subset (keeps the API cost low)
    python src/pipeline.py --run_eval --limit 3
    python src/pipeline.py --run_eval --qids T1_001 T2_003

    # Explicit output path
    python src/pipeline.py --run_eval --output evaluation/pipeline_output.json

    # Then score:
    python evaluate.py --ground_truth evaluation/ground_truth.json \
                       --pipeline_output evaluation/pipeline_output.json \
                       --report evaluation/report.json
"""

import json
import argparse
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from config import INDEX_PATH, EVAL_DIR
from retrieve import Retriever, RetrievedChunk, route_query
from generate import Generator, GenerationResult
from llm import PROVIDERS, SESSION, BudgetExceededError


# ── Config defaults ────────────────────────────────────────────────────────────

DB_PATH    = INDEX_PATH
K_DEFAULT  = 5          # chunks to retrieve for single/temporal queries
K_PER_ETF  = 3          # chunks per ETF for comparative queries


# ── Pipeline result ────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    qid: str
    question: str
    query_type: int
    retrieved_chunks: list[dict]     # serialisable dicts for evaluate.py
    generated_answer: str
    cited_sources: list[dict]        # serialisable dicts for evaluate.py
    chunks_used: int = 0
    provider: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost_usd: Optional[float] = None


# ── Query type inference ───────────────────────────────────────────────────────

def infer_retrieval_mode(
    query_type: int,
    isin_list: Optional[list[str]],
    metadata_filter_hint: Optional[dict],
) -> str:
    """
    Map ground_truth query_type + available ISINs to a retrieval mode.

    Type 1 → single
    Type 2 → comparative (multiple ISINs) or cross_doc (single ISIN, no doc_type filter)
    Type 3 → single (temporal, filtered by etf_isin; year filtering done post-retrieval)
    Type 4 → comparative (needs all ETFs)
    """
    if query_type in (1, 3):
        return "single"
    if query_type == 4:
        return "comparative"
    # Type 2: check whether it's cross-ETF or cross-document
    if isin_list and len(isin_list) > 1:
        return "comparative"
    if isin_list and len(isin_list) == 1:
        hint = metadata_filter_hint or {}
        if "doc_type" not in hint:
            return "cross_doc"
    return "single"


# ── Chunk → serialisable dict ──────────────────────────────────────────────────

def chunk_to_dict(chunk: RetrievedChunk) -> dict:
    """
    Convert a RetrievedChunk to a flat dict for JSON serialisation.
    evaluate.py reads etf_isin, doc_type, year directly from this dict.
    """
    return {
        "chunk_id":        chunk.chunk_id,
        "text":            chunk.text,
        "score":           chunk.score,
        "etf_isin":        chunk.etf_isin,
        "etf_ticker":      chunk.etf_ticker,
        "issuer":          chunk.issuer,
        "doc_type":        chunk.doc_type,
        "year":            chunk.year,
        "quarter":         chunk.quarter,
        "section_heading": chunk.section_heading,
        "block_type":      chunk.block_type,
        "page_number":     chunk.page_number,
        "source_file":     chunk.source_file,
    }


# ── Core pipeline function ─────────────────────────────────────────────────────

def run_query(
    question: str,
    retriever: Retriever,
    generator: Generator,
    qid: str = "manual",
    query_type: int = 1,
    isin_list: Optional[list[str]] = None,
    metadata_filter_hint: Optional[dict] = None,
    k: int = K_DEFAULT,
) -> PipelineResult:
    """
    Full pipeline for a single question:
      1. Infer retrieval mode from query_type and available ISINs
      2. Retrieve relevant chunks
      3. Generate answer with citations
      4. Return structured PipelineResult
    """
    hint = metadata_filter_hint or {}
    mode = infer_retrieval_mode(query_type, isin_list, hint)

    # ── Retrieval ──────────────────────────────────────────────────────────────
    chunks: list[RetrievedChunk] = route_query(
        retriever=retriever,
        query=question,
        metadata_filter_hint=hint,
        isin_list=isin_list,
        mode=mode,
        k=k,
    )

    # ── Generation ────────────────────────────────────────────────────────────
    result: GenerationResult = generator.answer(
        query=question,
        chunks=chunks,
        query_type=query_type,
    )

    # ── Serialise ─────────────────────────────────────────────────────────────
    return PipelineResult(
        qid=qid,
        question=question,
        query_type=query_type,
        retrieved_chunks=[chunk_to_dict(c) for c in chunks],
        generated_answer=result.answer,
        cited_sources=[asdict(s) for s in result.cited_sources],
        chunks_used=result.chunks_used,
        provider=result.provider,
        model=result.model,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
        cache_read_tokens=result.cache_read_tokens,
        cache_write_tokens=result.cache_write_tokens,
        cost_usd=result.cost_usd,
    )


# ── Batch evaluation runner ────────────────────────────────────────────────────

def select_questions(
    questions: list[dict],
    qids: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[dict]:
    """Restrict the ground-truth questions to --qids (in file order) and/or the first --limit."""
    if qids:
        known   = {q["qid"] for q in questions}
        unknown = [qid for qid in qids if qid not in known]
        if unknown:
            raise SystemExit(f"Unknown qids: {unknown}")
        questions = [q for q in questions if q["qid"] in set(qids)]
    if limit is not None:
        questions = questions[:limit]
    return questions


def default_output_path(generator: Generator) -> Path:
    """evaluation/runs/<UTC timestamp>_<model>.json — never overwrites a previous run."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model = generator.model.replace("/", "-").replace(":", "-")
    return EVAL_DIR / "runs" / f"{stamp}_{model}.json"


def _run_metadata(generator: Generator, started: str, results: list, status: str) -> dict:
    priced = [r.cost_usd for r in results if r.cost_usd is not None]
    return {
        "provider":    generator.provider,
        "model":       generator.model,
        "started_at":  started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "status":      status,     # running | complete | partial (budget) | partial (interrupted)
        "n_questions": len(results),
        "totals": {
            "input_tokens":       sum(r.input_tokens for r in results),
            "output_tokens":      sum(r.output_tokens for r in results),
            "cache_read_tokens":  sum(r.cache_read_tokens for r in results),
            "cache_write_tokens": sum(r.cache_write_tokens for r in results),
            "cost_usd":           round(sum(priced), 6) if priced else None,
            "unpriced_questions": len(results) - len(priced),
        },
        "budget_usd": SESSION.max_cost_usd,
    }


def _save(output_path: Path, generator: Generator, started: str,
          results: list, status: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run":     _run_metadata(generator, started, results, status),
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def run_eval_batch(
    ground_truth_path: Path,
    output_path: Path,
    retriever: Retriever,
    generator: Generator,
    k: int = K_DEFAULT,
    qids: Optional[list[str]] = None,
    limit: Optional[int] = None,
) -> list[PipelineResult]:
    """
    Run the pipeline over the active questions in ground_truth.json
    (optionally only --qids / the first --limit).

    The output file is rewritten after every question, so a run stopped by the
    budget (LLM_MAX_COST_USD) or by Ctrl+C keeps every answer obtained so far.
    """
    gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    questions = select_questions(gt["questions"], qids, limit)   # not _parked_type3

    results: list[PipelineResult] = []
    total   = len(questions)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    status  = "complete"

    try:
        for i, q in enumerate(questions, 1):
            qid        = q["qid"]
            question   = q["question"]
            query_type = q["query_type"]
            hint       = q.get("metadata_filter_hint") or {}

            # Derive isin_list from sources or hint
            isin_list = _extract_isin_list(q)

            print(f"\n[{i}/{total}] {qid} (type {query_type}) — {question[:60]}...")

            try:
                result = run_query(
                    question=question,
                    retriever=retriever,
                    generator=generator,
                    qid=qid,
                    query_type=query_type,
                    isin_list=isin_list,
                    metadata_filter_hint=hint,
                    k=k,
                )
                results.append(result)
                cost = f"${result.cost_usd:.5f}" if result.cost_usd is not None else "n/a"
                print(f"  ✓ {len(result.retrieved_chunks)} chunks retrieved, "
                      f"{result.input_tokens}+{result.output_tokens} tokens, {cost}")
            except BudgetExceededError:
                raise
            except Exception as e:
                print(f"  ✗ ERROR: {e}")
                # Insert an empty result so evaluate.py doesn't mark it as missing
                results.append(PipelineResult(
                    qid=qid,
                    question=question,
                    query_type=query_type,
                    retrieved_chunks=[],
                    generated_answer=f"ERROR: {e}",
                    cited_sources=[],
                ))
            _save(output_path, generator, started, results, "running")
    except BudgetExceededError as e:
        status = "partial (budget)"
        print(f"\n[pipeline] STOPPED: {e}")
    except KeyboardInterrupt:
        status = "partial (interrupted)"
        print("\n[pipeline] Interrupted.")

    _save(output_path, generator, started, results, status)
    print(f"\n[pipeline] Saved {len(results)}/{total} results ({status}) → {output_path}")
    print(f"[pipeline] Total: {SESSION.summary()}")
    return results


def _extract_isin_list(question_entry: dict) -> Optional[list[str]]:
    """
    Derive the list of ISINs relevant to a question from its sources list.
    Used to route comparative queries correctly.
    """
    sources = question_entry.get("sources", [])
    hint    = question_entry.get("metadata_filter_hint") or {}

    # Collect unique ISINs from sources
    isins = list(dict.fromkeys(
        s["etf_isin"] for s in sources if "etf_isin" in s
    ))

    # Fall back to hint
    if not isins and "etf_isin" in hint:
        isins = [hint["etf_isin"]]

    return isins if isins else None


# ── CLI ────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="ETF RAG end-to-end pipeline")

    # Mode
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--query",    help="Single natural language question")
    mode.add_argument("--run_eval", action="store_true",
                      help="Run all ground truth questions and save output")

    # Single query options
    p.add_argument("--query_type",  type=int, default=1, choices=[1, 2, 3, 4])
    p.add_argument("--etf_isin",    default=None)
    p.add_argument("--isin_list",   nargs="+", default=None)
    p.add_argument("--doc_type",    default=None, choices=["factsheet", "kid"])
    p.add_argument("--year",        type=int, default=None)
    p.add_argument("--issuer",      default=None)
    p.add_argument("--k",           type=int, default=K_DEFAULT)

    # Eval batch options
    p.add_argument("--ground_truth", type=Path,
                   default=EVAL_DIR / "ground_truth.json")
    p.add_argument("--output",       type=Path, default=None,
                   help="Default: evaluation/runs/<timestamp>_<model>.json")
    p.add_argument("--qids",         nargs="+", default=None,
                   help="Only run these question ids (e.g. T1_001 T2_003)")
    p.add_argument("--limit",        type=int, default=None,
                   help="Only run the first N (selected) questions")

    # Shared
    p.add_argument("--provider", default=None, choices=PROVIDERS,
                   help="LLM provider (default: LLM_PROVIDER from .env)")
    p.add_argument("--model",    default=None,
                   help="Model id (default: LLM_MODEL from .env, else provider default)")
    p.add_argument("--db_path",  type=Path, default=DB_PATH)
    p.add_argument("--show_chunks", action="store_true",
                   help="Print retrieved chunk previews")
    return p


def main():
    args = build_parser().parse_args()

    retriever = Retriever(db_path=args.db_path)
    generator = Generator(provider=args.provider, model=args.model)

    if args.run_eval:
        output = args.output or default_output_path(generator)
        run_eval_batch(
            ground_truth_path=args.ground_truth,
            output_path=output,
            retriever=retriever,
            generator=generator,
            k=args.k,
            qids=args.qids,
            limit=args.limit,
        )
        subset = (f" --qids {' '.join(args.qids)}" if args.qids else "") + \
                 (f" --limit {args.limit}" if args.limit is not None else "")
        print(f"\nNext step → python evaluation/evaluate.py "
              f"--ground_truth {args.ground_truth} "
              f"--pipeline_output {output} "
              f"--report evaluation/report.json{subset}")
        return

    # ── Single query mode ──────────────────────────────────────────────────────
    filters: dict = {}
    if args.etf_isin: filters["etf_isin"] = args.etf_isin
    if args.doc_type: filters["doc_type"] = args.doc_type
    if args.year:     filters["year"]     = args.year
    if args.issuer:   filters["issuer"]   = args.issuer

    isin_list = args.isin_list or ([args.etf_isin] if args.etf_isin else None)

    result = run_query(
        question=args.query,
        retriever=retriever,
        generator=generator,
        query_type=args.query_type,
        isin_list=isin_list,
        metadata_filter_hint=filters,
        k=args.k,
    )

    if args.show_chunks:
        print(f"\n── RETRIEVED CHUNKS ({len(result.retrieved_chunks)}) ──────────────────")
        for i, c in enumerate(result.retrieved_chunks, 1):
            print(f"\n  [{i}] score={c['score']:.4f}  "
                  f"{c['etf_isin']} | {c['doc_type']} | {c['year']} | "
                  f"{c['section_heading'] or '—'} | {c['block_type']}")
            preview = c["text"][:200].replace("\n", " ")
            print(f"      {preview}{'…' if len(c['text']) > 200 else ''}")

    print(f"\n── ANSWER (type {result.query_type}) ──────────────────────────────────")
    print(result.generated_answer)

    print(f"\n── SOURCES CITED ({len(result.cited_sources)}) ──────────────────────────")
    for s in result.cited_sources:
        print(f"  {s['etf_isin']} | {s['issuer']} | {s['doc_type']} | {s['year']}")

    print(f"\n── COST  {SESSION.summary()} ──\n")


if __name__ == "__main__":
    main()
