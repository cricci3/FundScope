# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project roadmap — `TODO.md`

`TODO.md` (Italian) is the project roadmap: phases 0–9, each task with the files involved and a **"Done quando"** (done when) criterion. Work rules:

- **One phase at a time, in order.** Phases 1–5 are done; phase 0 only lacks task 0.3 (delete the remote branch `origin/refactor/cloud-llm`, needs the user's confirmation). Phase 2 moved generation to Claude via the Anthropic API behind the provider-neutral `src/llm/` interface — Ollama has been uninstalled and must not be reintroduced; phase 3 fixed the evaluation, added the LLM judge and set the official baseline; phase 4 made the tool-calling agent the only entrypoint; phase 5 reworked retrieval (bigger chunks, multilingual embeddings, hybrid BM25 search, key facts) and added the full-context engine. Don't start phase N+1 until every task in phase N meets its "Done quando" criterion.
- Tick tasks `[x]` in `TODO.md` as they are completed.
- **At the end of each phase, launch the execution** to verify it end-to-end: rebuild the index if data/ingest/embed changed (`python setup.py --rebuild`), then run the pipeline + evaluation with the judge (`--judge llm --compare evaluation/baseline_cloud.json`) and report the metric deltas against the baseline before moving on.
- **Work only on `main` — never create branches.** Commit after each completed task, with a message that starts with the task number (e.g. `2.3: OpenAI-compatible backend`); push to `origin/main` at the end of each phase.
- **The LLM provider is Claude via the Anthropic API, with limited prepaid credit (~5 $).** Never hard-code a provider or model outside `src/llm/` and `config.py`. During development run the pipeline/evaluation on subsets (`--limit`, `--qids`); run the full 22-question evaluation only at the end of a phase, and always report the cost of each run.

Current reference: **`evaluation/baseline_cloud.json`** (task 3.7; Haiku 4.5 generator, Sonnet 5.5 judge, 17 questions): judge verdict pass 0.353 · judge correctness 0.467 · grounded 0.824 · abstained 0.353 · attribution 0.882 · retrieval precision 0.988 / recall 1.000 (document-level). Cost of a full run + judge ≈ 0.12 $. Most failures are abstentions: the right document is retrieved but not the chunk holding the value. `evaluation/baseline_llama3.2-3b.json` is historical only and not comparable.

Latest result (phase 5, agent engine — `evaluation/report_phase5_agent.json`, 22 questions): judge pass 0.682 · correctness 0.925 · grounded 0.727 · abstained 0 · attribution 0.909 · chunk value recall 0.947 · retrieval precision 0.777 / recall 0.955; on the 17 English questions pass 0.647 (phase 4: 0.706 — lost on groundedness: Haiku adds explanations the documents don't contain). The full-context engine (`report_phase5_full.json`) scores higher: pass 0.818. A full agent run + judge costs ≈ 0.35 $ (agent ≈ 0.007 $/question; full context ≈ 0.07 $ + judge 0.22 $), so keep development runs on `--qids` subsets. The baseline 3.7 table compares 17 questions: compare the same 17 when quoting deltas.

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
python src/retrieve.py --query "..." --etf_isin IE00B4L5Y983 --doc_type factsheet [--mode comparative --isin_list A B] [--json] [--vector_only]

# Entrypoints
python src/agent.py [--query "..."] [--show_calls] [--show_chunks] [--show_cost] [--provider ...] [--model ...]  # tool-calling agent (the entrypoint)
python src/ask.py ...                                                # alias of agent.py, same flags
python src/fullcontext.py --query "..." [--show_cost]                # no retrieval: every document in the cached prompt
python src/registry.py                                               # print the fund registry
python src/live_data.py --isin IE00B4L5Y983                         # yfinance market data

# Evaluation (22 ground-truth questions: 17 English + 5 Italian at the end). Output defaults to evaluation/runs/<UTC ts>_<model>[_agent|_full].json
# During development use --limit N / --qids ...; run all 22 only at the end of a phase and report the cost.
python src/pipeline.py --run_eval [--engine agent|rag|full] [--limit 3] [--qids T1_001 T2_003] [--output ...]
python evaluation/evaluate.py --ground_truth evaluation/ground_truth.json --pipeline_output evaluation/runs/<run>.json --report evaluation/report.json [--judge heuristic|llm] [--compare evaluation/baseline_cloud.json] [--retrieval_only] [--limit N] [--qids ...]
python evaluation/run_retrieval.py [--limit N]          # retrieval only, no LLM; score with --retrieval_only
```

Single question through the batch pipeline: `python src/pipeline.py --query "..."` (agent); the fixed RAG path takes routing hints: `python src/pipeline.py --engine rag --query "..." --query_type 2 --isin_list IE00B4L5Y983 IE00BD4TXV59`.


## Architecture

RAG over ETF factsheets and KIDs (PDF): `pdfplumber` → chunks JSON → `intfloat/multilingual-e5-base` embeddings → ChromaDB + BM25 (all local) → cloud LLM through `src/llm/` (default Claude Haiku 4.5 via the Anthropic API).

- **`src/llm/`** is the only place that imports `anthropic` / `openai`. `chat()` also takes `response_schema` (structured JSON output) and `effort` (ignored where unsupported; Haiku 4.5 rejects it, Sonnet 5.5 rejects `temperature`). `base.py` has the neutral types (`Message`, `ToolSpec`, `ToolCall`, `Usage`, `LLMResponse`), the `LLMClient` interface (one method, `chat(messages, *, system, tools, temperature, max_tokens)`), and the process-wide `SESSION` cost tracker that enforces `LLM_MAX_COST_USD` (raises `BudgetExceededError`). `get_llm(role="generator"|"judge", provider, model)` in `__init__.py` builds the client. Backends: `anthropic_backend.py` (native SDK; SDK 1.x has no `temperature` kwarg, so it goes via `extra_body` for models that accept it; cache breakpoint on the system block — Haiku 4.5 only caches prefixes ≥ 4096 tokens) and `openai_compat.py` (one-line registry `OPENAI_COMPAT_PROVIDERS`, key from `<NAME>_API_KEY`). Prices in `pricing.py` (unknown model → cost `None`).
- In agent loops append `response.to_message()` (not a hand-built message): it carries the provider-native content (`raw`) that the backend replays verbatim.

- **`src/metadata.json`** is the document registry: PDF filename → `isin`, `name`, `issuer`, `category`, `type` (factsheet|kid), `year`, `yahoo_ticker`, … (`name`/`yahoo_ticker` are fund-level, identical on every document of an ISIN). **`src/registry.py`** groups it by ISIN into `FUNDS` (name, issuer, ticker, doc types, years) — the only source of ISINs/names/tickers for the agent tools and `live_data.py`. `ingest.py` attaches the document fields to every chunk; output files in `data/processed/` are named by `build_output_stem` (e.g. `IE00B4L5Y983_2026_kid.json`).
- **Chunking** (`ingest.py`): per-doc-type sizes from `config.CHUNK_SIZES` / `CHUNK_OVERLAPS` (factsheet 1000 chars, KID 1200, overlap 15%, chosen by a chunk-value-recall sweep); tables kept whole (one-cell tables dropped); issuer glossary pages skipped (`is_glossary_page`); a synthetic "key facts" chunk per document built from case-sensitive label regexes (`_FACTSHEET_KF_PATTERNS` / `_KID_KF_PATTERNS`: TER, replication, methodology, SRI, KID costs…). Chunk ids come from `make_chunk_id`.
- **Retrieval** (`retrieve.py`): `Retriever` has three modes — `single` (metadata-filtered search), `comparative` (top-k per ISIN, then merged, so one ETF can't dominate), `cross_document` (top-k per doc_type for one ISIN). `route_query` dispatches between them. Filters are built by `_build_where`. Every mode goes through `_query`: vector top-20 + BM25 top-20 (same filters, applied in Python by `_matches`) fused with weighted RRF (`BM25_WEIGHT` 0.5; `config.HYBRID_SEARCH`); `score` stays the cosine similarity. The embedding model is stored in the collection metadata and `Retriever` refuses a mismatch; e5 needs the `query:`/`passage:` prefixes (`config.EMBEDDING_PREFIXES`). `apply_min_score` drops chunks below `config.MIN_SIMILARITY[model]` (key facts and chunks containing a query token with digits are kept).
- **Query types** (used by `ground_truth.json`, `generate.py` prompts and `pipeline.py`): 1 factual, 2 comparative/cross-doc, 3 temporal (parked — `_parked_type3`, needs multi-year data), 4 synthetic reasoning.
- **Citations** (`citations.py`): chunks shown to the model are numbered (`ChunkBook`, header `[Chunk N] ISIN | issuer | doc_type | year | section`); the model cites `[Chunk N]` and `resolve_citations` rewrites each tag as `[ISIN | issuer | doc_type | year]` from the chunk's real metadata (the format `evaluate.py`/judge read) and returns `cited_sources` + cited chunk numbers. Used by both engines.
- **Generation** (`generate.py`, `--engine rag` only): per-type prompt templates and `build_context`.
- **Agent** (`agent.py`, the entrypoint; `ask.py` is a 1-line alias): the LLM chooses tools — `search_etf_docs` (`isins`, `issuer`, `doc_type`, `year`, `mode` single|comparative|cross_doc; enums generated from the registry), `get_live_data`, `list_available_etfs`. Each search puts the key-facts chunk of every searched document first (`KEY_FACTS_FIRST`, up to `MAX_KEY_FACTS` documents), then drops low-similarity chunks and tells the model how many. `Agent.run()` returns an `AgentResult` (resolved answer, numbered chunks, cited sources, tool calls, usage/cost). Interactive mode keeps the last `HISTORY_TURNS` question/answer pairs (`reset` clears); `pipeline.py --engine agent` (default) uses `history_turns=0` and gives the agent only the question text, while `--engine rag` routes retrieval with the ground-truth hints and `--engine full` (`fullcontext.py`) puts the four documents in the cached system prompt (each document is one `[Chunk N]`).
- **Evaluation** (`evaluation/evaluate.py`): retrieval precision/recall (document-level: a chunk counts if its isin+doc_type+year match a source), chunk value recall (`chunk_value_recall`: share of expected values present in the retrieved chunks' text — the free chunk-level retrieval metric, also with `--retrieval_only`), attribution, and heuristics driven by `ground_truth.json` — `expected_values` (Type 1: list; Type 2: dict label → values, ISIN labels are checked in the part of the answer about that ETF; values are strings or lists of alternatives, numbers compared numerically), Type 4 rubric items `{item, keywords, min_match}`, unsupported numbers, abstention. Sentences that say information is missing are dropped before matching. `--judge llm` adds `evaluation/judge.py`: one call per question to `get_llm("judge")` (Sonnet 5.5, `JUDGE_EFFORT`), JSON enforced via `response_schema` and re-validated → correctness / grounded / rubric / verdict / reasoning in the report. The judge is the primary quality metric; heuristics are a free fallback. Reports carry `meta` (models, judge, chunk sizes, embedding model, hybrid, similarity threshold), group results by query type, difficulty and `language` (ground truth `language: "it"`), and `--compare` prints deltas. For `--engine full` runs the judge puts the shared documents in its cached system prompt.

### Paths, ids and dedup

- All paths and the collection/embedding-model names live in `src/config.py` (absolute, cwd-independent). `setup.py` adds `src/` to `sys.path` to import it. Don't hard-code `index/chroma_db` anywhere else.
- `embed.py` only indexes JSON files whose stem matches a `metadata.json` entry (`build_output_stem`); others are logged as `[skip]`. It also skips chunks whose `content_hash` is already indexed for the same `etf_isin` + `doc_type`.
- `chunk_id` = hash of `source_file | block_type | sha256(full text)`, independent of year; the key-facts chunk id ignores its `Year:`/`Reference date:` lines.
- `pipeline.py --run_eval` writes `{"run": {...}, "results": [...]}` and rewrites the file after every question, so a budget stop or Ctrl+C keeps partial results; `evaluate.py` also accepts the old bare-list format.
- Scripts print non-ASCII characters (`←`, `✓`); when stdout is piped on Windows set `PYTHONIOENCODING=utf-8` or they crash with `UnicodeEncodeError`.

### Known pitfalls

- On this machine Git Bash heredocs corrupt non-ASCII characters (`—`, `→`, `─` are common in this repo): write multi-line patches/scripts with the file Write/Edit tools, not `cat <<EOF`. This also applies to `python - <<'EOF'` patch scripts: search strings with non-ASCII characters or `\\` escapes silently fail to match.
- Changing `EMBEDDING_MODEL` (or its prefixes) needs `python setup.py --rebuild`: `Retriever` raises on an index built with another model. Startup takes ~40 s (imports + the 1.1 GB e5 model); `HF_HUB_OFFLINE=1` skips the Hugging Face network checks once the model is cached.
- Adding an ETF only needs `metadata.json` entries (with `name` and `yahoo_ticker`); `registry.py` raises if documents of the same ISIN disagree on a fund-level field.
