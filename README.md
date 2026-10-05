# FundScope — ETF Research Assistant (RAG)

A Retrieval-Augmented Generation (RAG) system that answers questions about ETF factsheets and Key Information Documents (KIDs) in plain English.

Built as a hands-on learning project to understand how RAG pipelines work end-to-end — from PDF ingestion to vector search to LLM-generated answers with source attribution.

---

## What it does

- Ingests ETF factsheets and KIDs (PDF) from issuers such as iShares and UBS
- Extracts and chunks text, preserving structured metadata (ISIN, issuer, doc type, section)
- Embeds chunks using `sentence-transformers` and stores them in a local ChromaDB vector store
- Answers questions by retrieving the most relevant chunks and passing them to a cloud LLM (Claude via the Anthropic API by default; other providers configurable in `.env`)
- Supports four query types: factual, comparative, cross-document, and synthetic reasoning
- Evaluates answer quality against a ground truth Q&A set

---

## Architecture

```
PDF files (factsheets + KIDs)
        │
        ▼
  src/ingest.py          Extract text + tables, clean, chunk, tag metadata
        │
        ▼
  data/processed/        JSON files — one array of chunks per document
        │
        ▼
  src/embed.py           Embed chunks with all-MiniLM-L6-v2, store in ChromaDB
        │
        ▼
  index/chroma_db/       Persisted vector store (local, no server needed)
        │
        ▼
  src/retrieve.py        Semantic search + metadata filtering
        │
        ▼
  src/generate.py        Prompt construction + LLM call (through src/llm/)
        │
        ▼
  src/ask.py             Interactive user interface
```

Embeddings and retrieval run locally; only answer generation calls the LLM provider.

---

## Project structure

```
FundScope/
│
├── src/
│   ├── ask.py                    # Interactive entrypoint — start here
│   ├── agent.py                  # Tool-calling agent (docs search + live market data)
│   ├── config.py                 # Paths + LLM settings (reads .env)
│   ├── metadata.json             # Document registry (ISIN, name, issuer, year, doc type, Yahoo ticker)
│   ├── registry.py               # Fund registry built from metadata.json (ISIN → name, issuer, ticker)
│   ├── ingest.py                 # PDF extraction, cleaning, chunking
│   ├── embed.py                  # Embedding + ChromaDB indexing
│   ├── retrieve.py               # Semantic search + metadata filtering
│   ├── generate.py               # Prompt templates + LLM call
│   ├── pipeline.py               # End-to-end batch runner
│   └── llm/                      # Provider-neutral LLM layer
│       ├── base.py               #   Message, ToolSpec, LLMResponse, LLMClient, cost tracker
│       ├── anthropic_backend.py  #   Native Anthropic backend (default)
│       ├── openai_compat.py      #   Groq / Gemini / OpenRouter (OpenAI-compatible APIs)
│       ├── pricing.py            #   $ per million tokens, for cost estimates
│       └── cli.py                #   --ping check
│
├── data/
│   ├── raw/                      # Original PDF files (not committed to git)
│   └── processed/                # Chunk JSON files produced by ingest.py
│
├── index/
│   └── chroma_db/                # Persisted ChromaDB vector store
│
├── evaluation/
│   ├── ground_truth.json         # 17 active Q&A pairs with expected answers
│   ├── evaluate.py               # Scoring: precision, recall, faithfulness, attribution
│   ├── runs/                     # One JSON per pipeline run (with provider, model, cost)
│   └── report.json               # Evaluation scores
│
├── .env.example                  # LLM provider configuration template
└── pyproject.toml                # Dependencies (managed with uv)
```

---

## Prerequisites

### Python environment

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

This creates `.venv` with every dependency. Prefix commands with `uv run` (e.g. `uv run python src/ask.py`), or activate `.venv`.

### LLM provider

FundScope generates answers with a cloud LLM. The default is **Claude Haiku 4.5** (`claude-haiku-4-5`) via the **Anthropic API**.

1. Get an API key at https://console.anthropic.com (Settings → API keys) and add some credit.
2. Create your `.env` from the template and paste the key:

```bash
cp .env.example .env
# then edit .env:  ANTHROPIC_API_KEY=sk-ant-...
```

3. Check that it works (a few tokens, a fraction of a cent):

```bash
python src/llm/cli.py --ping
# [ping] anthropic / claude-haiku-4-5-20251001 → 'pong'
# [ping] tokens in=15 out=5  cost=$0.000040
```

**Never commit `.env`** — it is git-ignored; only `.env.example` (without keys) is in the repo.

#### Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `LLM_PROVIDER` | `anthropic` | Provider for answers and the agent: `anthropic`, `groq`, `gemini`, `openrouter` |
| `LLM_MODEL` | provider default (`claude-haiku-4-5` for anthropic) | Model id. **Required** for providers other than anthropic |
| `JUDGE_PROVIDER` | `anthropic` | Provider for the LLM-as-judge used by the evaluation (phase 3) |
| `JUDGE_MODEL` | provider default (`claude-sonnet-5-5` for anthropic) | Judge model id |
| `JUDGE_EFFORT` | `medium` | Reasoning effort of the judge (`low` / `medium` / `high`; ignored by models without effort control) |
| `ANTHROPIC_API_KEY` | — | Key for `anthropic` |
| `GROQ_API_KEY` / `GEMINI_API_KEY` / `OPENROUTER_API_KEY` | — | Key for that provider; only the one in use is needed |
| `LLM_MAX_COST_USD` | `0.50` | Budget per run: a run stops (keeping its partial results) once the estimated cost reaches it |
| `LLM_MIN_INTERVAL_S` | `0` | Minimum seconds between two LLM calls (for free tiers with low request limits) |

Precedence: **CLI flags** (`--provider`, `--model`) > **environment / `.env`** > **defaults in `src/config.py`**.

#### Switching provider

Only `.env` changes — no code. Claude (default):

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=                     # empty → claude-haiku-4-5
ANTHROPIC_API_KEY=sk-ant-...
```

Groq:

```bash
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile   # any tool-capable model from the provider's model list
GROQ_API_KEY=gsk_...
LLM_MIN_INTERVAL_S=2                # stay under the free-tier rate limit
```

Gemini:

```bash
LLM_PROVIDER=gemini
LLM_MODEL=gemini-2.5-flash          # check Google's current model list
GEMINI_API_KEY=...
```

OpenRouter:

```bash
LLM_PROVIDER=openrouter
LLM_MODEL=anthropic/claude-haiku-4.5   # any model id listed on openrouter.ai/models
OPENROUTER_API_KEY=sk-or-...
```

(Model names for non-Anthropic providers are examples — check the provider's model list.) For a single run, override from the CLI instead:

```bash
python src/ask.py --provider groq --model llama-3.3-70b-versatile
python src/llm/cli.py --ping --provider gemini --model gemini-2.5-flash
```

`ask.py`, `agent.py`, `pipeline.py` and `generate.py` all accept `--provider` and `--model`.

#### Adding a new provider

- **OpenAI-compatible API** (most providers): add one line to `OPENAI_COMPAT_PROVIDERS` in `src/llm/openai_compat.py`, e.g. `"mistral": "https://api.mistral.ai/v1"`. The key is then read from `MISTRAL_API_KEY`, and `LLM_PROVIDER=mistral` works.
- **Anything else**: write a new backend in `src/llm/` that subclasses `LLMClient` and implements `_chat(messages, *, system, tools, temperature, max_tokens) -> LLMResponse` (converting the neutral `Message` / `ToolSpec` types to the provider's format), then hook it into `get_llm()` in `src/llm/__init__.py`.

Optionally add the model's prices to `src/llm/pricing.py`; models without a price still work, their cost is just reported as unknown.

#### Costs

Indicative prices (per million tokens, input / output): Claude Haiku 4.5 $1 / $5, Claude Sonnet 5.5 $2 / $10. A typical answer uses ~1,000 input and ~100–350 output tokens, i.e. **≈ $0.001–0.003 per question**; the full 17-question evaluation costs ≈ $0.03 with Haiku 4.5.

- `--show_cost` on `ask.py` / `agent.py` prints the tokens and estimated cost of every answer.
- `pipeline.py` and `evaluate.py` print the run total (pipeline + judge); each run file stores per-question and total tokens and cost.
- `LLM_MAX_COST_USD` caps what one run can spend.
- `pipeline.py --limit N` / `--qids T1_001 T2_003` run only a subset — use them while developing.

---

## Setup — running from zero

### Step 1 — Add your PDF documents

Place your ETF factsheets and KIDs in `data/raw/`. Then register each file in `src/metadata.json`:

```json
{
  "iShares_Core_MSCI_World_UCITS_ETF_USD_acc_factsheet.pdf": {
    "isin": "IE00B4L5Y983",
    "name": "iShares Core MSCI World UCITS ETF",
    "issuer": "ishares",
    "category": "MSCI World",
    "type": "factsheet",
    "currency": "USD",
    "share_class": "acc",
    "year": 2026,
    "quarter": "Q4",
    "yahoo_ticker": "EUNL.DE"
  }
}
```

`name` and `yahoo_ticker` are fund-level fields: repeat them identically on every document of the same ISIN (`src/registry.py` groups the documents by ISIN and rejects inconsistent entries; `python src/registry.py` prints the resulting fund list).

Supported issuers: `ishares`, `ubs`, `xtrackers`, `amundi`. Document types: `factsheet`, `kid`.

### Step 2 — Run setup

```bash
python setup.py
```

This single command checks the dependencies, the PDFs and the LLM provider (key present + a minimal `--ping`), ingests your PDFs, and builds the vector index.
Once it completes, `ask.py` is ready to use.

**Options:**

```bash
# Force full rebuild of the index (e.g. after changing chunk size or adding documents)
python setup.py --rebuild

# Skip ingestion if data/processed/ is already populated
python setup.py --skip_ingest

# Skip the LLM provider check (no API call)
python setup.py --skip_llm_check
```

If no provider key is configured (or the ping fails), setup still completes and builds the index.
You can use retrieval-only evaluation without a model — see the Evaluation section below.

---

## Asking questions

### Interactive mode (recommended)

```bash
python src/ask.py
```

This opens a prompt where you type questions in plain English. The system automatically detects the query type and retrieval mode.

```
════════════════════════════════════════════════════════════════
  FundScope — ETF Research Assistant
  Type a question, 'funds' to list ETFs, or 'exit' to quit.
════════════════════════════════════════════════════════════════

  Ask> What is the TER of the iShares MSCI World ETF?

  The TER of the iShares Core MSCI World UCITS ETF is 0.20%
  [IE00B4L5Y983 | ishares | factsheet | 2026]

  Sources:
    • iShares Core MSCI World (IE00B4L5Y983) | factsheet | 2026

  Ask> Compare the ongoing charges of both ETFs

  ...

  Ask> exit
```

Available commands inside the prompt:

| Command | Effect |
|---|---|
| Any question | Ask about the ETFs in the corpus |
| `funds` | List all ETFs available in the corpus |
| `chunks` | Toggle displaying retrieved source chunks |
| `help` | Show available commands |
| `exit` | Quit |

### Single question mode

```bash
python src/ask.py --query "What is the Summary Risk Indicator of IE00B4L5Y983?" --show_cost
```

### With source chunk inspection

```bash
python src/ask.py --query "Compare the replication methods of both ETFs" --show_chunks
```

This prints the retrieved chunks before the answer, showing exactly what the model was given — useful for understanding why an answer is correct or incorrect.

### Agent mode (tool calling)

```bash
python src/agent.py --query "How has the iShares Core MSCI World ETF performed this year?" --show_calls --show_cost
```

The model decides which tools to call: document search, live market data (yfinance), or the list of available ETFs.

### Specifying a different provider or model

```bash
python src/ask.py --provider groq --model llama-3.3-70b-versatile
```

---

## Example questions

**Factual (single ETF):**
- `What is the TER of the iShares Core MSCI World ETF?`
- `What is the ISIN of the UBS MSCI World ETF?`
- `What is the recommended holding period in the iShares KID?`
- `What is the Summary Risk Indicator of IE00B4L5Y983?`

**Comparative (two ETFs):**
- `Which ETF has a lower ongoing charge, iShares or UBS?`
- `Compare the replication methods of both MSCI World ETFs`
- `Do the two KIDs assign the same risk indicator?`
- `Compare the cost breakdown for both ETFs as reported in their KIDs`

**Cross-document (factsheet vs KID, same ETF):**
- `Does the iShares KID report the same ongoing cost as the factsheet?`
- `What replication method does the iShares factsheet report, and does the KID mention the same?`

**Synthetic (reasoning across documents):**
- `Which ETF would be more suitable for a cost-sensitive long-term investor?`
- `Summarise the key differences between the iShares and UBS MSCI World ETFs`

---

## Evaluation

To measure how well the system retrieves and answers questions against the ground truth set:

```bash
# Run the pipeline over all 17 active ground truth questions.
# Output: evaluation/runs/<UTC timestamp>_<model>.json (never overwritten)
python src/pipeline.py --run_eval

# While developing, run a subset to keep the cost low
python src/pipeline.py --run_eval --limit 3
python src/pipeline.py --run_eval --qids T1_001 T2_003

# Score a run (pass the same --limit / --qids if the run was a subset)
python evaluation/evaluate.py \
    --ground_truth evaluation/ground_truth.json \
    --pipeline_output evaluation/runs/<run>.json \
    --report evaluation/report.json
```

Each run file has the form `{"run": {provider, model, timestamps, status, totals}, "results": [...]}`, with tokens and estimated cost per question and in total.

### What is measured

- **Retrieval** precision/recall@k: is the right document (ISIN + doc type + year) retrieved?
- **Attribution**: are the right sources cited?
- **Heuristics** (free, no API calls), driven by `ground_truth.json`: expected key values found in the answer (`expected_values`, per ETF for comparisons), Type 4 rubric keywords, figures absent from the retrieved context, abstentions ("the documents do not contain…").
- **LLM judge** (`--judge llm`): a different model from the generator (default Claude Sonnet 5.5, `JUDGE_PROVIDER` / `JUDGE_MODEL` / `JUDGE_EFFORT` in `.env`) rates each answer's correctness against the reference, groundedness in the retrieved context and the rubric, and gives a pass / partial / fail verdict with a short motivation. This is the primary quality metric; it costs ≈ $0.005 per question.

```bash
# Score with the judge and compare with the official baseline
python evaluation/evaluate.py \
    --ground_truth evaluation/ground_truth.json \
    --pipeline_output evaluation/runs/<run>.json \
    --report evaluation/report.json \
    --judge llm --compare evaluation/baseline_cloud.json
```

The official baseline is `evaluation/baseline_cloud.json` (Haiku 4.5 answers judged by Sonnet 5.5): judge pass 35%, correctness 47%, grounded 82%, retrieval precision 0.99 / recall 1.00. Each report records the models, judge, chunk sizes and embedding model it was produced with, so reports stay comparable.

### Retrieval only (no LLM, no cost)

```bash
python evaluation/run_retrieval.py
python evaluation/evaluate.py \
    --ground_truth evaluation/ground_truth.json \
    --pipeline_output evaluation/runs/<timestamp>_retrieval-only.json \
    --retrieval_only
```

The report breaks down scores by query type (factual, comparative, synthetic) and difficulty (easy, medium, hard).

---

## Adding more ETFs

1. Place the new PDF files in `data/raw/`
2. Add entries to `src/metadata.json` (one per PDF, with the fund-level fields `name` and `yahoo_ticker` repeated identically on every document of the same ISIN). `src/registry.py` builds the fund list from this file, so the agent tools, `ask.py` and `live_data.py` pick up the new fund automatically
3. Re-run ingest, embed (without `--rebuild`), and you are ready to ask questions about the new fund

```bash
python src/ingest.py --batch data/raw/ --config src/metadata.json --output_dir data/processed/
python src/embed.py --input_dir data/processed/ --db_path index/chroma_db/
```

---

## Design notes

**Why ChromaDB?** It stores metadata alongside vectors and supports filtering before semantic search. This is essential for comparative queries — without metadata filtering, one ETF can dominate the top-k results and the other gets no representation.

**Why `all-MiniLM-L6-v2`?** It is 80 MB, runs on CPU in under a second per query, and produces 384-dimensional vectors. Good enough for a focused domain corpus of this size. The upgrade path is `all-mpnet-base-v2` (better quality, ~3x slower).

**Why a cloud LLM behind a neutral interface?** Small local models struggled with multi-document reasoning and tool calling. All code outside `src/llm/` talks only to `LLMClient` and its neutral types, so the provider is a configuration choice: Anthropic uses its native SDK (prompt caching, reliable tool use), every other provider goes through one generic OpenAI-compatible backend.

**Chunk size:** factsheets use 250-character chunks, KIDs use 400-character chunks. KID sections are short regulatory prose — cutting them at 250 characters loses the meaning of a section. Tables are kept whole and never split.

**Three retrieval modes:**
- `single` — one vector search with optional metadata filter (Type 1)
- `comparative` — one search per ETF independently, then merge (Type 2, multiple ETFs)
- `cross_doc` — one search per doc type for the same ETF, then merge (factsheet vs KID)

---

## Limitations

- Corpus is currently limited to two ETFs (iShares and UBS MSCI World). Adding more funds requires downloading documents and updating `metadata.json`.
- Answer generation needs network access and an API key with credit; without them only retrieval works.
- Temporal queries (comparing the same ETF across different years) require downloading documents from multiple years. The evaluation scaffolding for this is in place (`_parked_type3` in `ground_truth.json`) but inactive until multi-year data is available.
- Embedding runs on CPU. Embedding ~500 chunks takes approximately 10–20 seconds. No GPU is required.
