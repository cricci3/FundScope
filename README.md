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
  fundscope.ingest       Extract text + tables, clean, chunk, tag metadata, key-facts chunk
        │
        ▼
  data/processed/        JSON files — one array of chunks per document
        │
        ▼
  fundscope.embed        Embed chunks with multilingual-e5-base, store in ChromaDB
        │
        ▼
  index/chroma_db/       Persisted vector store (local, no server needed)
        │
        ▼
  fundscope.retrieve     Hybrid search (vectors + BM25, fused with RRF) + metadata filtering
        │
        ▼
  fundscope.agent        Tool-calling agent: the LLM (through fundscope.llm) decides what to
                         search; answers cite [Chunk N], resolved to real metadata
```

Two alternative engines are kept for comparison: `fundscope.generate` (the older fixed pipeline:
retrieval routed by hand + one LLM call, `--engine rag`) and `fundscope.fullcontext` (no retrieval:
every document in the prompt-cached system prompt, `--engine full`).

Embeddings and retrieval run locally; only answer generation calls the LLM provider.

---

## Project structure

```
FundScope/
│
├── src/fundscope/                # The Python package (installed in editable mode by `uv sync`)
│   ├── agent.py                  # Tool-calling agent: docs search + live data (`fundscope-cli` in the terminal)
│   ├── ask.py                    # Alias of agent.py
│   ├── bootstrap.py              # One-command setup (`fundscope-setup`): checks, ingest, index
│   ├── citations.py              # [Chunk N] citations → real source metadata
│   ├── config.py                 # Paths + settings (reads .env)
│   ├── log.py                    # Logging setup shared by the CLIs (--verbose)
│   ├── metadata.json             # Document registry (ISIN, name, issuer, year, doc type, Yahoo ticker)
│   ├── registry.py               # Fund registry built from metadata.json (ISIN → name, issuer, ticker)
│   ├── ingest.py                 # PDF extraction, cleaning, chunking
│   ├── embed.py                  # Embedding + ChromaDB indexing
│   ├── retrieve.py               # Hybrid search + metadata filtering
│   ├── live_data.py              # Market data from Yahoo Finance (disk cache)
│   ├── generate.py               # Fixed RAG pipeline: prompt templates + LLM call (--engine rag)
│   ├── fullcontext.py            # Full-context engine (--engine full)
│   ├── pipeline.py               # Batch runner over the ground truth (`fundscope-pipeline`)
│   ├── ui/                       # Web UI (Streamlit)
│   │   ├── launch.py             #   `fundscope`: starts the server and opens the browser
│   │   ├── app.py                #   the page: Ask, Live data, Evaluation, How it works
│   │   └── backend.py            #   formatting, evaluation reports, agent builder (tested without a browser)
│   └── llm/                      # Provider-neutral LLM layer
│       ├── base.py               #   Message, ToolSpec, LLMResponse, LLMClient, cost tracker
│       ├── anthropic_backend.py  #   Native Anthropic backend (default)
│       ├── openai_compat.py      #   Groq / Gemini / OpenRouter (OpenAI-compatible APIs)
│       ├── pricing.py            #   $ per million tokens, for cost estimates
│       └── cli.py                #   --ping check
│
├── tests/                        # pytest suite — no network, no API key (LLM and embeddings mocked)
│
├── data/
│   ├── raw/                      # Original PDF files (not committed to git)
│   └── processed/                # Chunk JSON files produced by ingest
│
├── index/
│   └── chroma_db/                # Persisted ChromaDB vector store
│
├── evaluation/
│   ├── ground_truth.json         # 22 Q&A pairs (17 English + 5 Italian) with expected values / rubrics
│   ├── evaluate.py               # Scoring: retrieval, heuristics, attribution, LLM judge
│   ├── judge.py                  # LLM-as-judge (one call per question)
│   ├── run_retrieval.py          # Retrieval-only run (no LLM, no cost)
│   ├── runs/                     # One JSON per pipeline run (with provider, model, cost)
│   └── baseline_cloud.json, report_phase*.json   # Reference reports
│
├── .github/workflows/ci.yml      # CI: ruff + pytest on every push
├── .env.example                  # LLM provider configuration template
└── pyproject.toml                # Package, console scripts, dependencies (managed with uv)
```

---

## Prerequisites

### Python environment

Python 3.13 and [uv](https://docs.astral.sh/uv/) are required.

```bash
uv sync
```

This creates `.venv` with every dependency and installs FundScope itself as the `fundscope` package (editable), with four commands: `fundscope` (the web UI), `fundscope-cli` (the agent in the terminal), `fundscope-setup` and `fundscope-pipeline`. Prefix commands with `uv run` (e.g. `uv run fundscope`), or activate `.venv`. The other modules run as `python -m fundscope.<module>` (e.g. `uv run python -m fundscope.retrieve --help`).

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
uv run python -m fundscope.llm.cli --ping
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
| `LIVE_DATA_TTL_S` | `3600` | Seconds Yahoo Finance answers stay cached in `data/cache/live_data/` (`0` disables) |

Precedence: **CLI flags** (`--provider`, `--model`) > **environment / `.env`** > **defaults in `src/fundscope/config.py`**.

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
uv run fundscope-cli --provider groq --model llama-3.3-70b-versatile
uv run python -m fundscope.llm.cli --ping --provider gemini --model gemini-2.5-flash
```

`fundscope`, `fundscope-pipeline`, `fundscope.generate` and `fundscope.fullcontext` all accept `--provider` and `--model`.

#### Adding a new provider

- **OpenAI-compatible API** (most providers): add one line to `OPENAI_COMPAT_PROVIDERS` in `src/fundscope/llm/openai_compat.py`, e.g. `"mistral": "https://api.mistral.ai/v1"`. The key is then read from `MISTRAL_API_KEY`, and `LLM_PROVIDER=mistral` works.
- **Anything else**: write a new backend in `src/fundscope/llm/` that subclasses `LLMClient` and implements `_chat(messages, *, system, tools, temperature, max_tokens) -> LLMResponse` (converting the neutral `Message` / `ToolSpec` types to the provider's format), then hook it into `get_llm()` in `src/fundscope/llm/__init__.py`.

Optionally add the model's prices to `src/fundscope/llm/pricing.py`; models without a price still work, their cost is just reported as unknown.

#### Costs

Indicative prices (per million tokens, input / output): Claude Haiku 4.5 $1 / $5, Claude Sonnet 5.5 $2 / $10. A typical answer uses ~1,000 input and ~100–350 output tokens, i.e. **≈ $0.001–0.003 per question**; a full 22-question agent run costs ≈ $0.15 and judging it ≈ $0.20 (see Results).

- `--show_cost` on `fundscope-cli` prints (the web UI always shows it) the tokens and estimated cost of every answer.
- `fundscope-pipeline` and `evaluate.py` print the run total (pipeline + judge); each run file stores per-question and total tokens and cost.
- `LLM_MAX_COST_USD` caps what one run can spend.
- `fundscope-pipeline --limit N` / `--qids T1_001 T2_003` run only a subset — use them while developing.

---

## Setup — running from zero

### Step 1 — Add your PDF documents

Place your ETF factsheets and KIDs in `data/raw/`. Then register each file in `src/fundscope/metadata.json`:

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

`name` and `yahoo_ticker` are fund-level fields: repeat them identically on every document of the same ISIN (`src/fundscope/registry.py` groups the documents by ISIN and rejects inconsistent entries; `uv run python -m fundscope.registry` prints the resulting fund list).

Supported issuers: `ishares`, `ubs`, `xtrackers`, `amundi`. Document types: `factsheet`, `kid`.

### Step 2 — Run setup

```bash
uv run fundscope-setup
```

This single command checks the dependencies, the PDFs and the LLM provider (key present + a minimal `--ping`), ingests your PDFs, and builds the vector index.
Once it completes, `uv run fundscope` opens the web UI (or use `fundscope-cli` in the terminal).

**Options:**

```bash
# Force full rebuild of the index (e.g. after changing chunk size or adding documents)
uv run fundscope-setup --rebuild

# Skip ingestion if data/processed/ is already populated
uv run fundscope-setup --skip_ingest

# Skip the LLM provider check (no API call)
uv run fundscope-setup --skip_llm_check
```

If no provider key is configured (or the ping fails), setup still completes and builds the index.
You can use retrieval-only evaluation without a model — see the Evaluation section below.

---

## Asking questions

### Web UI (recommended)

```bash
uv run fundscope                  # starts http://localhost:8501 and opens it in the browser
uv run fundscope --port 8600      # another port
uv run fundscope --no-browser     # server only
```

A local Streamlit app, built to demo the project:

- **Ask** — chat with the agent. Each answer shows its citations as coloured source badges, the live list of tool
  calls while it works, the tool calls and the retrieved chunks (cited ones marked) on demand, and the tokens and
  cost of the answer. Provider and model can be switched from the sidebar; *New conversation* clears the memory.
- **Live data** — price, returns, 52-week range, volume and a 12-month price chart from Yahoo Finance.
- **Evaluation** — the reference runs side by side (judge pass rate, correctness, groundedness, retrieval, cost).
- **How it works** — the architecture diagram and the main design choices.

The first question after a start waits ~10–30 s for the embedding model to load; the other tabs work immediately,
and also without an API key. Answers that compare funds or talk about suitability carry a "not financial advice"
note, which is also in the sidebar. Stop the server with Ctrl+C in the terminal.

### Terminal

```bash
uv run fundscope-cli
```

This opens a prompt where you type questions in plain English (`fundscope-cli` runs `fundscope.agent`; `python -m fundscope` and `fundscope.ask` are aliases). The LLM agent decides which funds, documents and retrieval mode to search, searches again with other wording if the first results miss the answer, and cites the chunks it used; the code turns each `[Chunk N]` citation into the real source metadata. Follow-up questions work: the last 5 questions and answers are remembered until you type `reset`.

```
════════════════════════════════════════════════════════════════
  FundScope Agent — ETF Research Assistant
  Powered by tool-calling. The LLM decides what to look up.
  Type a question or 'help' for commands.
════════════════════════════════════════════════════════════════

  Ask> What is the TER of the iShares MSCI World ETF?

  The TER of the iShares Core MSCI World UCITS ETF is 0.20%
  [IE00B4L5Y983 | ishares | factsheet | 2026]

  Sources:
    • iShares Core MSCI World UCITS ETF (IE00B4L5Y983) | factsheet | 2026

  Ask> And the UBS one?

  ...

  Ask> exit
```

Available commands inside the prompt:

| Command | Effect |
|---|---|
| Any question | Ask about the ETFs in the corpus |
| `funds` | List all ETFs available in the corpus |
| `chunks` | Toggle displaying retrieved source chunks (cited ones are marked) |
| `calls` | Toggle displaying the agent's tool calls |
| `reset` | Forget the conversation |
| `help` | Show available commands |
| `exit` | Quit |

### Single question mode

```bash
uv run fundscope-cli --query "What is the Summary Risk Indicator of IE00B4L5Y983?" --show_cost
```

### With source chunk inspection

```bash
uv run fundscope-cli --query "Compare the replication methods of both ETFs" --show_chunks
```

This prints the retrieved chunks before the answer, showing exactly what the model was given — useful for understanding why an answer is correct or incorrect.

### Tool calls and live data

```bash
uv run fundscope-cli --query "How has the iShares Core MSCI World ETF performed this year?" --show_calls --show_cost
```

The model decides which tools to call: document search (`search_etf_docs`, with optional ISINs, issuer, document type, year and mode — single, comparative or cross-document), live market data (yfinance), or the list of available ETFs. `--show_calls` prints each call.

### Specifying a different provider or model

```bash
uv run fundscope-cli --provider groq --model llama-3.3-70b-versatile
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
# Run the agent over all 22 active ground truth questions (17 in English, 5 in Italian;
# it sees only the question text).
# Output: evaluation/runs/<UTC timestamp>_<model>_agent.json (never overwritten)
uv run fundscope-pipeline --run_eval

# The older fixed pipeline, with retrieval routed by the ground-truth hints
# Output: evaluation/runs/<UTC timestamp>_<model>.json
uv run fundscope-pipeline --run_eval --engine rag

# No retrieval: every document in the (cached) prompt, for comparison
# Output: evaluation/runs/<UTC timestamp>_<model>_full.json
uv run fundscope-pipeline --run_eval --engine full

# While developing, run a subset to keep the cost low
uv run fundscope-pipeline --run_eval --limit 3
uv run fundscope-pipeline --run_eval --qids T1_001 T2_003

# Score a run (pass the same --limit / --qids if the run was a subset)
uv run python evaluation/evaluate.py \
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
uv run python evaluation/evaluate.py \
    --ground_truth evaluation/ground_truth.json \
    --pipeline_output evaluation/runs/<run>.json \
    --report evaluation/report.json \
    --judge llm --compare evaluation/baseline_cloud.json
```

The official baseline is `evaluation/baseline_cloud.json` (Haiku 4.5 answers judged by Sonnet 5.5): judge pass 35%, correctness 47%, grounded 82%, retrieval precision 0.99 / recall 1.00. Each report records the models, judge, chunk sizes and embedding model it was produced with, so reports stay comparable.

### Retrieval only (no LLM, no cost)

```bash
uv run python evaluation/run_retrieval.py
uv run python evaluation/evaluate.py \
    --ground_truth evaluation/ground_truth.json \
    --pipeline_output evaluation/runs/<timestamp>_retrieval-only.json \
    --retrieval_only
```

The report breaks down scores by query type (factual, comparative, synthetic), difficulty (easy, medium, hard) and language. Add `--min_recall 0.95` to exit with an error when the mean retrieval recall falls below it (used by CI).

---

## Adding more ETFs

1. Place the new PDF files in `data/raw/`
2. Add entries to `src/fundscope/metadata.json` (one per PDF, with the fund-level fields `name` and `yahoo_ticker` repeated identically on every document of the same ISIN). `src/fundscope/registry.py` builds the fund list from this file, so the agent tools and `live_data.py` pick up the new fund automatically
3. Re-run ingest, embed (without `--rebuild`), and you are ready to ask questions about the new fund

```bash
uv run python -m fundscope.ingest --batch data/raw/ --config src/fundscope/metadata.json --output_dir data/processed/
uv run python -m fundscope.embed --input_dir data/processed/ --db_path index/chroma_db/
```

---

## Results

All runs use Claude Haiku 4.5 as the generator and Claude Sonnet 5.5 as the judge (`--judge llm`).
"Pass" is the judge verdict, "correct" its correctness score (correct = 1, partial = ½),
"grounded" the share of answers whose claims are all supported by the retrieved context.

| Run | Engine | Questions | Pass | Correct | Grounded | Attribution | Retrieval P / R | Cost (run + judge) |
|---|---|---|---|---|---|---|---|---|
| `baseline_cloud.json` (phase 3) | fixed RAG | 17 EN | 35.3% | 46.7% | 82.4% | 0.88 | 0.99 / 1.00 | $0.03 + $0.09 |
| `report_phase4_agent.json` | agent | 17 EN | 70.6% | 83.3% | 76.5% | 0.82 | 0.85 / 0.94 | $0.21 + $0.13 |
| `report_phase5_agent.json` | agent | 22 (17 EN + 5 IT) | 68.2% | 92.5% | 72.7% | 0.91 | 0.78 / 0.96 | $0.15 + $0.20 |
| `report_phase5_full.json` | full context | 22 (17 EN + 5 IT) | 81.8% | 95.0% | 86.4% | 0.95 | 0.43 / 1.00 | $0.07 + $0.22 |

The agent fixed most of the baseline's failures (abstentions: the right document was retrieved
but not the chunk holding the value). Its remaining misses are mostly on groundedness: Haiku adds
explanations the documents do not contain. On this small corpus (~15k tokens) giving the model
every document is both better and cheaper; retrieval is expected to pay off as the corpus grows.
Retrieval precision is lower for the agent and full context by design: they look at more documents
than the minimum the question needs.

---

## Development

```bash
uv sync                     # also installs pytest and ruff (dev group)
uv run ruff check .         # lint
uv run pytest -q            # ~60 tests, ~2 min; no network, no API key
```

The tests never call an LLM or download a model: the embedding model is replaced by a hashing
embedder (a temporary ChromaDB index is built from a handful of chunks), the agent runs against a
scripted `LLMClient`, and both provider backends are checked against canned SDK responses (neutral
format ↔ Anthropic, neutral ↔ OpenAI), so switching provider stays verified without keys. Ingestion
is tested on a PDF generated by the test itself; when the corpus PDFs are present in `data/raw/`,
re-ingesting them must reproduce `data/processed/` chunk for chunk.

CI (`.github/workflows/ci.yml`) runs `uv sync --locked`, ruff and pytest on every push. The corpus
PDFs are not versioned, so the retrieval-only evaluation step (`evaluate.py --retrieval_only
--min_recall 0.95`) only runs where `data/raw/` holds them. Generation is evaluated by hand, since
it costs API credit.

Every command accepts `--verbose` (`-v`) to log progress and diagnostics to stderr; stdout carries
only the program output.

---

## Design notes

**Why ChromaDB?** It stores metadata alongside vectors and supports filtering before semantic search. This is essential for comparative queries — without metadata filtering, one ETF can dominate the top-k results and the other gets no representation.

**Why `intfloat/multilingual-e5-base`?** Questions may be in Italian while the documents are in English. In a retrieval-only comparison over the whole corpus (no metadata filters) it found the chunk holding the expected value earlier than `all-MiniLM-L6-v2`, `paraphrase-multilingual-MiniLM-L12-v2` and `multilingual-e5-small`, especially on the Italian questions (MRR 0.78 vs 0.43 for MiniLM). It runs on CPU (~1.1 GB, 768 dimensions, 512-token window) and needs the `query:` / `passage:` prefixes, configured in `config.EMBEDDING_PREFIXES`. The index records the model it was built with, and querying it with a different model is an error.

**Why hybrid search?** Embeddings rank exact tokens poorly: ISINs, acronyms (TER, OCF, SRI) and figures. Each search also ranks the same filtered chunks with BM25 and fuses the two rankings with Reciprocal Rank Fusion (BM25 weight 0.5). The agent then drops chunks below a per-model similarity threshold (`config.MIN_SIMILARITY`) and tells the model how many it discarded.

**Why a cloud LLM behind a neutral interface?** Small local models struggled with multi-document reasoning and tool calling. All code outside `src/fundscope/llm/` talks only to `LLMClient` and its neutral types, so the provider is a configuration choice: Anthropic uses its native SDK (prompt caching, reliable tool use), every other provider goes through one generic OpenAI-compatible backend.

**Chunk size:** factsheets use 1000-character chunks, KIDs 1200, with 15% overlap. Factsheets have a two-column layout that interleaves a label ("Product Structure :") with text from the other column; small chunks (250 characters) often separated a value from its label. A sweep measured with the chunk value recall of `evaluation/run_retrieval.py` went from 0.71 (250/400) to 1.00 (1000/1200). Tables are kept whole and never split, and each document gets a "key facts" chunk built from label patterns (TER, replication, SRI, costs).

**Full-context comparison:** the whole 2026 corpus is ~15k tokens, so `fundscope-pipeline --engine full` answers with every document in the prompt (prompt-cached: later questions read it at a tenth of the input price). It measures what retrieval adds compared with giving the model everything.

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
