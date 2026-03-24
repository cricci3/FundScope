"""
pipeline.py — End-to-end query handler
ETF RAG Project — Phase 4

Wires retrieve.py → generate.py into a single entry point.
Accepts a question, routes to the correct retrieval mode,
calls the LLM, and returns a structured response.

Output format is exactly what evaluate.py expects:
    {
      "qid":               "T1_001",
      "question":          "...",
      "query_type":        1,
      "retrieved_chunks":  [...],   ← list of chunk dicts with metadata
      "generated_answer":  "...",
      "cited_sources":     [...]    ← parsed from answer text
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

    # Run all ground truth questions and save output for evaluate.py
    python src/pipeline.py --run_eval \
                       --ground_truth evaluation/ground_truth.json \
                       --output      evaluation/pipeline_output.json

    # Then score:
    python evaluate.py --ground_truth evaluation/ground_truth.json \
                       --pipeline_output evaluation/pipeline_output.json \
                       --report evaluation/report.json
"""

import json
import argparse
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

from retrieve import Retriever, RetrievedChunk, route_query
from generate import Generator, GenerationResult


# ── Config defaults ────────────────────────────────────────────────────────────

DB_PATH    = Path("index/chroma_db/")
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
    input_tokens: int = 0
    output_tokens: int = 0


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
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


# ── Batch evaluation runner ────────────────────────────────────────────────────

def run_eval_batch(
    ground_truth_path: Path,
    output_path: Path,
    retriever: Retriever,
    generator: Generator,
    k: int = K_DEFAULT,
    skip_filled: bool = True,
) -> list[PipelineResult]:
    """
    Run the pipeline over every active question in ground_truth.json.
    Writes results to output_path in the format evaluate.py expects.

    skip_filled=True skips questions whose expected_answer still has <FILL>
    (they can't be auto-scored anyway, but you may still want answers for them).
    """
    gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    questions = gt["questions"]   # only active questions, not _parked_type3

    results: list[PipelineResult] = []
    total = len(questions)

    for i, q in enumerate(questions, 1):
        qid        = q["qid"]
        question   = q["question"]
        query_type = q["query_type"]
        hint       = q.get("metadata_filter_hint") or {}
        expected   = q.get("expected_answer", "")

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
            print(f"  ✓ {len(result.retrieved_chunks)} chunks retrieved, "
                  f"{result.input_tokens}+{result.output_tokens} tokens")
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

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [asdict(r) for r in results]
    output_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"\n[pipeline] Saved {len(results)} results → {output_path}")
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
                   default=Path("evaluation/ground_truth.json"))
    p.add_argument("--output",       type=Path,
                   default=Path("evaluation/pipeline_output.json"))

    # Shared
    p.add_argument("--db_path",  type=Path, default=DB_PATH)
    p.add_argument("--show_chunks", action="store_true",
                   help="Print retrieved chunk previews")
    return p


def main():
    args = build_parser().parse_args()

    retriever = Retriever(db_path=args.db_path)
    generator = Generator()

    if args.run_eval:
        run_eval_batch(
            ground_truth_path=args.ground_truth,
            output_path=args.output,
            retriever=retriever,
            generator=generator,
            k=args.k,
        )
        print(f"\nNext step → python evaluate.py "
              f"--ground_truth {args.ground_truth} "
              f"--pipeline_output {args.output} "
              f"--report evaluation/report.json")
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

    print(f"\n── TOKENS  in={result.input_tokens}  out={result.output_tokens} ──\n")


if __name__ == "__main__":
    main()
