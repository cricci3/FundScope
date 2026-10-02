"""
run_retrieval.py — Retrieval-only batch runner (no LLM, no API cost)

Runs only the retrieval step of the pipeline over the ground-truth questions
and writes an output that evaluate.py scores with --retrieval_only.

Usage:
    python evaluation/run_retrieval.py
    python evaluation/run_retrieval.py --limit 3 --output evaluation/pipeline_output.json

    python evaluation/evaluate.py --ground_truth evaluation/ground_truth.json \\
        --pipeline_output evaluation/runs/<timestamp>_retrieval-only.json --retrieval_only
"""

import sys
import json
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import INDEX_PATH, EVAL_DIR  # noqa: E402
from retrieve import Retriever  # noqa: E402
from pipeline import (  # noqa: E402
    K_DEFAULT, PipelineResult, _extract_isin_list, chunk_to_dict,
    retrieve_chunks, select_questions,
)


def run_retrieval(ground_truth_path: Path, retriever: Retriever, k: int = K_DEFAULT,
                  qids=None, limit=None) -> list[PipelineResult]:
    gt = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    questions = select_questions(gt["questions"], qids, limit)

    results = []
    for i, q in enumerate(questions, 1):
        chunks = retrieve_chunks(
            q["question"], retriever,
            query_type=q["query_type"],
            isin_list=_extract_isin_list(q),
            metadata_filter_hint=q.get("metadata_filter_hint") or {},
            k=k,
        )
        print(f"[{i}/{len(questions)}] {q['qid']}: {len(chunks)} chunks")
        results.append(PipelineResult(
            qid=q["qid"],
            question=q["question"],
            query_type=q["query_type"],
            retrieved_chunks=[chunk_to_dict(c) for c in chunks],
            generated_answer="",
            cited_sources=[],
            chunks_used=len(chunks),
        ))
    return results


def main():
    p = argparse.ArgumentParser(description="Retrieval-only run over the ground truth (no LLM)")
    p.add_argument("--ground_truth", type=Path, default=EVAL_DIR / "ground_truth.json")
    p.add_argument("--output",       type=Path, default=None,
                   help="Default: evaluation/runs/<timestamp>_retrieval-only.json")
    p.add_argument("--db_path",      type=Path, default=INDEX_PATH)
    p.add_argument("--k",            type=int, default=K_DEFAULT)
    p.add_argument("--qids",         nargs="+", default=None)
    p.add_argument("--limit",        type=int, default=None)
    args = p.parse_args()

    started = datetime.now(timezone.utc)
    output  = args.output or EVAL_DIR / "runs" / \
        f"{started.strftime('%Y%m%dT%H%M%SZ')}_retrieval-only.json"

    retriever = Retriever(db_path=args.db_path)
    results   = run_retrieval(args.ground_truth, retriever, args.k, args.qids, args.limit)

    payload = {
        "run": {
            "provider": None, "model": None, "mode": "retrieval_only",
            "started_at": started.isoformat(timespec="seconds"),
            "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "status": "complete", "n_questions": len(results),
            "totals": {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                       "cache_write_tokens": 0, "cost_usd": 0.0, "unpriced_questions": 0},
        },
        "results": [asdict(r) for r in results],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"\n[run_retrieval] Saved {len(results)} results → {output}")
    print(f"Next step → python evaluation/evaluate.py --ground_truth {args.ground_truth} "
          f"--pipeline_output {output} --retrieval_only")


if __name__ == "__main__":
    main()
