# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project roadmap — `TODO.md`

`TODO.md` (Italian) is the project roadmap: phases 0–9, each task with the files involved and a **"Done quando"** (done when) criterion. Work rules:

- **One phase at a time, in order.** Phases 0–3 are done (phase 2 moved generation to Claude via the Anthropic API behind the provider-neutral `src/llm/` interface — Ollama has been uninstalled and must not be reintroduced; phase 3 fixed the evaluation, added the LLM judge and set the official baseline). Don't start phase N+1 until every task in phase N meets its "Done quando" criterion.
- Tick tasks `[x]` in `TODO.md` as they are completed.
- **At the end of each phase, launch the execution** to verify it end-to-end: rebuild the index if data/ingest/embed changed (`python setup.py --rebuild`), then run the pipeline + evaluation with the judge (`--judge llm --compare evaluation/baseline_cloud.json`) and report the metric deltas against the baseline before moving on.
- **Work only on `main` — never create branches.** Commit after each completed task, with a message that starts with the task number (e.g. `2.3: OpenAI-compatible backend`); push to `origin/main` at the end of each phase.
- **The LLM provider is Claude via the Anthropic API, with limited prepaid credit (~5 $).** Never hard-code a provider or model outside `src/llm/` and `config.py`. During development run the pipeline/evaluation on subsets (`--limit`, `--qids`); run the full 17-question evaluation only at the end of a phase, and always report the cost of each run.

Current reference: **`evaluation/baseline_cloud.json`** (task 3.7; Haiku 4.5 generator, Sonnet 5.5 judge, 17 questions): judge verdict pass 0.353 · judge correctness 0.467 · grounded 0.824 · abstained 0.353 · attribution 0.882 · retrieval precision 0.988 / recall 1.000 (document-level). Cost of a full run + judge ≈ 0.12 $. Most failures are abstentions: the right document is retrieved but not the chunk holding the value. `evaluation/baseline_llama3.2-3b.json` is historical only and not comparable.

## Commands

No test suite, linter or packaging yet (planned in phase 8). Scripts in `src/` import each other as top-level modules (`from retrieve import Retriever`), so run them as `python src/<script>.py` — never as `python -m`.

Environment is managed by **uv** (`pyproject.toml` + `uv.lock`, Python 3.13): `uv sync` creates `.venv`; prefix the commands below with `uv run` (e.g. `uv run python setup.py`). Add deps with `uv add <pkg>`. `requirements.txt` is legacy (cleanup in TODO 8.1).

LLM configuration lives in `.env` (template: `.env.example`; `.env` is git-ignored, never commit it): `LLM_PROVIDER` (default `anthropic`), `LLM_MODEL` (empty → `claude-haiku-4-5`), `JUDGE_PROVIDER`/`JUDGE_MODEL` (default `claude-sonnet-5-5`), per-provider keys `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY`, plus `LLM_MAX_COST_USD` (per-run budget, default 0.50) and `LLM_MIN_INTERVAL_S`. Precedence: `--provider`/`--model` flags > env/.env > defaults in `config.py`.

```bash
uv sync

# Full setup: checks deps, PDFs, LLM key + ping; ingests data/raw/ via src/metadata.json, builds the index
python setup.py [--rebuild] [--skip_ingest] [--skip_llm_check]

# LLM provider check (a few tokens; prints model, tokens, cost)
python src/llm/cli.py --ping [--provider groq --model ...] [--role judge]

# Individual stages
python src/ingest.py --batch data/raw/ --config src/metadata.json --output_dir data/processed/
python src/embed.py --input_dir data/processed/ --db_path index/chroma_db/ [--rebuild] [--dry_run]
python src/retrieve.py --query "..." --etf_isin IE00B4L5Y983 --doc_type factsheet [--mode comparative --isin_list A B] [--json]

# Entrypoints
python src/ask.py [--query "..."] [--show_chunks] [--show_cost] [--provider ...] [--model ...]   # keyword-routed RAG
python src/agent.py [--query "..."] [--show_calls] [--show_cost] [--provider ...] [--model ...]  # tool-calling agent
python src/live_data.py --isin IE00B4L5Y983                         # yfinance market data

# Evaluation (17 ground-truth questions). Output defaults to evaluation/runs/<UTC ts>_<model>.json
# During development use --limit N / --qids ...; run all 17 only at the end of a phase and report the cost.
python src/pipeline.py --run_eval [--limit 3] [--qids T1_001 T2_003] [--output ...]
python evaluation/evaluate.py --ground_truth evaluation/ground_truth.json --pipeline_output evaluation/runs/<run>.json --report evaluation/report.json [--judge heuristic|llm] [--compare evaluation/baseline_cloud.json] [--retrieval_only] [--limit N] [--qids ...]
python evaluation/run_retrieval.py [--limit N]          # retrieval only, no LLM; score with --retrieval_only
```

Single question through the batch pipeline: `python src/pipeline.py --query "..." --query_type 2 --isin_list IE00B4L5Y983 IE00BD4TXV59`.


## Architecture

RAG over ETF factsheets and KIDs (PDF): `pdfplumber` → chunks JSON → `all-MiniLM-L6-v2` embeddings → ChromaDB (all local) → cloud LLM through `src/llm/` (default Claude Haiku 4.5 via the Anthropic API).

- **`src/llm/`** is the only place that imports `anthropic` / `openai`. `chat()` also takes `response_schema` (structured JSON output) and `effort` (ignored where unsupported; Haiku 4.5 rejects it, Sonnet 5.5 rejects `temperature`). `base.py` has the neutral types (`Message`, `ToolSpec`, `ToolCall`, `Usage`, `LLMResponse`), the `LLMClient` interface (one method, `chat(messages, *, system, tools, temperature, max_tokens)`), and the process-wide `SESSION` cost tracker that enforces `LLM_MAX_COST_USD` (raises `BudgetExceededError`). `get_llm(role="generator"|"judge", provider, model)` in `__init__.py` builds the client. Backends: `anthropic_backend.py` (native SDK; SDK 1.x has no `temperature` kwarg, so it goes via `extra_body` for models that accept it; cache breakpoint on the system block — Haiku 4.5 only caches prefixes ≥ 4096 tokens) and `openai_compat.py` (one-line registry `OPENAI_COMPAT_PROVIDERS`, key from `<NAME>_API_KEY`). Prices in `pricing.py` (unknown model → cost `None`).
- In agent loops append `response.to_message()` (not a hand-built message): it carries the provider-native content (`raw`) that the backend replays verbatim.

- **`src/metadata.json`** is the document registry: PDF filename → `isin`, `issuer`, `category`, `type` (factsheet|kid), `year`, … `ingest.py` attaches these to every chunk; output files in `data/processed/` are named by `build_output_stem` (e.g. `IE00B4L5Y983_2026_kid.json`).
- **Chunking** (`ingest.py`): per-doc-type sizes from `config.CHUNK_SIZES` / `CHUNK_OVERLAPS` (factsheet 250 chars, KID 400); tables kept whole; a synthetic "key facts" chunk per document. Chunk ids come from `make_chunk_id`.
- **Retrieval** (`retrieve.py`): `Retriever` has three modes — `single` (metadata-filtered search), `comparative` (top-k per ISIN, then merged, so one ETF can't dominate), `cross_document` (top-k per doc_type for one ISIN). `route_query` dispatches between them. Filters are built by `_build_where`.
- **Query types** (used by `ground_truth.json`, `generate.py` prompts and `pipeline.py`): 1 factual, 2 comparative/cross-doc, 3 temporal (parked — `_parked_type3`, needs multi-year data), 4 synthetic reasoning.
- **Generation** (`generate.py`): per-type prompt templates, `build_context`, and citation parsing; answers cite sources as `[ISIN | issuer | doc_type | year]`.
- **Two entrypoints**: `ask.py` detects query type/ISINs with keyword rules and a hard-coded `KNOWN_ISINS`; `agent.py` lets the LLM choose tools (`search_etf_docs`, `get_live_data`, `list_available_etfs`). The roadmap (phase 4) makes the agent the only entrypoint.
- **Evaluation** (`evaluation/evaluate.py`): retrieval precision/recall (document-level: a chunk counts if its isin+doc_type+year match a source), attribution, and heuristics driven by `ground_truth.json` — `expected_values` (Type 1: list; Type 2: dict label → values, ISIN labels are checked in the part of the answer about that ETF; values are strings or lists of alternatives, numbers compared numerically), Type 4 rubric items `{item, keywords, min_match}`, unsupported numbers, abstention. Sentences that say information is missing are dropped before matching. `--judge llm` adds `evaluation/judge.py`: one call per question to `get_llm("judge")` (Sonnet 5.5, `JUDGE_EFFORT`), JSON enforced via `response_schema` and re-validated → correctness / grounded / rubric / verdict / reasoning in the report. The judge is the primary quality metric; heuristics are a free fallback. Reports carry `meta` (models, judge, chunk sizes, embedding model) and `--compare` prints deltas.

### Paths, ids and dedup

- All paths and the collection/embedding-model names live in `src/config.py` (absolute, cwd-independent). `setup.py` adds `src/` to `sys.path` to import it. Don't hard-code `index/chroma_db` anywhere else.
- `embed.py` only indexes JSON files whose stem matches a `metadata.json` entry (`build_output_stem`); others are logged as `[skip]`. It also skips chunks whose `content_hash` is already indexed for the same `etf_isin` + `doc_type`.
- `chunk_id` = hash of `source_file | block_type | sha256(full text)`, independent of year; the key-facts chunk id ignores its `Year:`/`Reference date:` lines.
- `pipeline.py --run_eval` writes `{"run": {...}, "results": [...]}` and rewrites the file after every question, so a budget stop or Ctrl+C keeps partial results; `evaluate.py` also accepts the old bare-list format.
- Scripts print non-ASCII characters (`←`, `✓`); when stdout is piped on Windows set `PYTHONIOENCODING=utf-8` or they crash with `UnicodeEncodeError`.

### Known pitfalls

- ISIN/name/ticker maps are duplicated in `ask.py`, `agent.py` and `live_data.py`; adding an ETF means updating `metadata.json` plus these copies (until `src/registry.py`, TODO 4.3).
