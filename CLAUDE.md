# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project roadmap — `TODO.md`

`TODO.md` (Italian) is the project roadmap: phases 0–9, each task with the files involved and a **"Done quando"** (done when) criterion. Work rules:

- **One phase at a time, in order.** Phases 0–1 are done. Phase 2 migrates generation to a cloud LLM (Ollama has been uninstalled and must not be reintroduced); phase 3 fixes the evaluation and sets the official baseline. Don't judge model quality before phase 3 is closed. Don't start phase N+1 until every task in phase N meets its "Done quando" criterion.
- Tick tasks `[x]` in `TODO.md` as they are completed.
- **At the end of each phase, launch the execution** to verify it end-to-end: rebuild the index if data/ingest/embed changed (`python setup.py --rebuild`), then run the pipeline + evaluation (commands below) and compare the report with the current reference (retrieval-only after phase 1; `evaluation/baseline_cloud.json` once task 3.7 is done). Report the metric deltas before moving on.
- Roadmap work happens on branch `refactor/cloud-llm` (task 0.1), not `main`.

Current reference: retrieval-only precision 0.988 · recall 1.000 (after phase 1). `evaluation/baseline_llama3.2-3b.json` is historical only (dirty index + buggy scorer) and is not comparable. The official generation baseline is set in task 3.7.

## Commands

No test suite, linter or packaging yet (planned in phase 8). Scripts in `src/` import each other as top-level modules (`from retrieve import Retriever`), so run them as `python src/<script>.py` — never as `python -m`.

Environment is managed by **uv** (`pyproject.toml` + `uv.lock`, Python 3.13): `uv sync` creates `.venv`; prefix the commands below with `uv run` (e.g. `uv run python setup.py`). Add deps with `uv add <pkg>`. `requirements.txt` is legacy (cleanup in TODO 8.1). Ollama is not installed and won't be used again: until phase 2 is done use `--skip_ollama`; after it, generation goes through a cloud provider configured in `.env` (see TODO phase 2).

```bash
uv sync

# Full setup: checks Ollama + model, ingests data/raw/ via src/metadata.json, builds the index
python setup.py [--rebuild] [--skip_ingest] [--skip_ollama] [--model llama3.2:3b]

# Individual stages
python src/ingest.py --batch data/raw/ --config src/metadata.json --output_dir data/processed/
python src/embed.py --input_dir data/processed/ --db_path index/chroma_db/ [--rebuild] [--dry_run]
python src/retrieve.py --query "..." --etf_isin IE00B4L5Y983 --doc_type factsheet [--mode comparative --isin_list A B] [--json]

# Entrypoints
python src/ask.py [--query "..."] [--show_chunks] [--model ...]     # keyword-routed RAG
python src/agent.py [--query "..."] [--show_calls] [--model ...]    # tool-calling agent (needs tool-capable model, e.g. mistral:7b)
python src/live_data.py --isin IE00B4L5Y983                         # yfinance market data

# Evaluation (17 ground-truth questions)
python src/pipeline.py --run_eval --ground_truth evaluation/ground_truth.json --output evaluation/pipeline_output.json
python evaluation/evaluate.py --ground_truth evaluation/ground_truth.json --pipeline_output evaluation/pipeline_output.json --report evaluation/report.json [--retrieval_only]
```

Single question through the batch pipeline: `python src/pipeline.py --query "..." --query_type 2 --isin_list IE00B4L5Y983 IE00BD4TXV59`.

`evaluation/run_retrieval.py` is referenced in the README but does not exist yet (TODO 3.5).

## Architecture

RAG over ETF factsheets and KIDs (PDF), all local: `pdfplumber` → chunks JSON → `all-MiniLM-L6-v2` embeddings → ChromaDB → Ollama via its OpenAI-compatible endpoint (`openai` client, `localhost:11434/v1`).

- **`src/metadata.json`** is the document registry: PDF filename → `isin`, `issuer`, `category`, `type` (factsheet|kid), `year`, … `ingest.py` attaches these to every chunk; output files in `data/processed/` are named by `build_output_stem` (e.g. `IE00B4L5Y983_2026_kid.json`).
- **Chunking** (`ingest.py`): per-doc-type sizes (factsheet 250 chars, KID 400); tables kept whole; a synthetic "key facts" chunk per document. Chunk ids come from `make_chunk_id`.
- **Retrieval** (`retrieve.py`): `Retriever` has three modes — `single` (metadata-filtered search), `comparative` (top-k per ISIN, then merged, so one ETF can't dominate), `cross_document` (top-k per doc_type for one ISIN). `route_query` dispatches between them. Filters are built by `_build_where`.
- **Query types** (used by `ground_truth.json`, `generate.py` prompts and `pipeline.py`): 1 factual, 2 comparative/cross-doc, 3 temporal (parked — `_parked_type3`, needs multi-year data), 4 synthetic reasoning.
- **Generation** (`generate.py`): per-type prompt templates, `build_context`, and citation parsing; answers cite sources as `[ISIN | issuer | doc_type | year]`.
- **Two entrypoints**: `ask.py` detects query type/ISINs with keyword rules and a hard-coded `KNOWN_ISINS`; `agent.py` lets the LLM choose tools (`search_etf_docs`, `get_live_data`, `list_available_etfs`). The roadmap (phase 4) makes the agent the only entrypoint.
- **Evaluation** (`evaluation/evaluate.py`): retrieval precision/recall against expected sources, heuristic faithfulness, attribution, unsupported-claims check, Type 4 rubric. Known to be unreliable until phase 3 is done.

### Paths, ids and dedup

- All paths and the collection/embedding-model names live in `src/config.py` (absolute, cwd-independent). `setup.py` adds `src/` to `sys.path` to import it. Don't hard-code `index/chroma_db` anywhere else.
- `embed.py` only indexes JSON files whose stem matches a `metadata.json` entry (`build_output_stem`); others are logged as `[skip]`. It also skips chunks whose `content_hash` is already indexed for the same `etf_isin` + `doc_type`.
- `chunk_id` = hash of `source_file | block_type | sha256(full text)`, independent of year; the key-facts chunk id ignores its `Year:`/`Reference date:` lines.
- Scripts print non-ASCII characters (`←`, `✓`); when stdout is piped on Windows set `PYTHONIOENCODING=utf-8` or they crash with `UnicodeEncodeError`.

### Known pitfalls

- ISIN/name/ticker maps are duplicated in `ask.py`, `agent.py` and `live_data.py`; adding an ETF means updating `metadata.json` plus these copies (until `src/registry.py`, TODO 4.3).
