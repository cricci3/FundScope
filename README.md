# FundScope 🔍

> A multi-document RAG system for querying and comparing ETF reports and prospectuses using natural language.

---

## Project Overview

FundScope is a Proof of Concept (POC) built to explore Retrieval-Augmented Generation (RAG) in a financial setting. The system ingests annual reports and prospectuses from ETFs and enables natural language querying — factual lookups, cross-fund comparisons, and temporal analysis — with full source attribution.

This project is a hands-on personal RAG learning exercise.

---

## Key Features

- 📄 **PDF ingestion** of ETF annual reports and prospectuses from issuers
- 🧩 **Metadata-aware chunking** — every chunk tagged with ETF ticker, issuer, category, year, and document type
- 🔍 **Semantic search** over a local FAISS vector store (CPU-native, no GPU required)
- 🏷️ **Metadata filtering** — scope retrieval to a specific fund, year, or asset category
- 🔄 **Comparative retrieval** — retrieve top-k chunks per ETF simultaneously for cross-fund queries
- 🤖 **LLM generation** via API (Claude / GPT-4o) with context-grounded answers
- 📌 **Source attribution** — every answer cites the ETF, year, and document section it drew from
- 📊 **Evaluation framework** — ground truth Q&A set with retrieval and faithfulness scoring

---

## Query Types Supported (in future)

**Type 1 — Factual** (single ETF, single fact)
```
"What is the total expense ratio of CSPX?"
```

**Type 2 — Comparative** (multiple ETFs, same dimension)
```
"Compare the expense ratios of CSPX, VUAA, and XSPX."
"Which of these bond ETFs describes more interest rate risk?"
```

**Type 3 — Temporal** (single ETF across years)
```
"How has IEMA's description of liquidity risk changed from 2021 to 2023?"
```

**Type 4 — Synthetic** (reasoning across multiple chunks)
```
"Which ETF in this corpus would be most suitable for a risk-averse investor and why?"
```

---

## Status

🚧 **In active development**