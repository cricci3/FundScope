# FundScope — TODO / Roadmap di miglioramento

Roadmap derivata dalla review del progetto (settembre 2026). Le fasi vanno eseguite **in ordine**:
le fasi 1–2 correggono bug che falsano le metriche, quindi vanno chiuse prima di giudicare il nuovo modello.

**Baseline attuale** (`evaluation/report.json`, modello `llama3.2:3b` locale):
retrieval recall 0.956 · precision 0.502 · faithfulness 26.7% · attribution 0.382 · unsupported claims 58.8%.

Convenzioni: ogni task ha i file coinvolti e un criterio **Done quando**. Spuntare `[x]` a lavoro finito.

---

## Fase 0 — Preparazione

- [x] **0.1 Branch di lavoro** — creare `refactor/cloud-llm` da `main`.
- [x] **0.2 Salvare la baseline** — copiare `evaluation/report.json` in `evaluation/baseline_llama3.2-3b.json` per confronto futuro.
  - Done quando: il file baseline esiste e non viene sovrascritto dalle run successive.

---

## Fase 1 — Bug di dati e percorsi (priorità massima)

- [x] **1.1 Config centralizzata dei percorsi**
  - Creare `src/config.py` con `PROJECT_ROOT = Path(__file__).resolve().parent.parent` e costanti assolute: `DATA_RAW`, `DATA_PROCESSED`, `INDEX_PATH = PROJECT_ROOT / "index" / "chroma_db"`, `METADATA_PATH`, `COLLECTION_NAME`, `EMBEDDING_MODEL`.
  - Sostituire i percorsi hard-coded in: `src/ask.py` (`DB_PATH`), `src/agent.py` (`DB_PATH`), `src/pipeline.py` (`DB_PATH`), `src/retrieve.py` (default CLI + costruttore), `src/embed.py` (default CLI), `setup.py` (`INDEX_PATH`, `METADATA`).
  - Oggi `ask.py`/`agent.py`/`setup.py` usano `src/index/chroma_db`, mentre `pipeline.py`/`retrieve.py` usano `index/chroma_db` relativo alla cwd → valutazione e app interrogano indici diversi.
  - Done quando: `grep -rn "chroma_db" src/ setup.py` mostra solo `config.py`; tutti gli script funzionano da qualsiasi cwd.

- [ ] **1.2 Eliminare indici e dati duplicati**
  - Cancellare `src/index/`, `src/src/`, `src/__pycache__/` e le cartelle UUID orfane in `index/chroma_db/`.
  - Cancellare da `data/processed/` i file obsoleti: `EUNL_*`, `UETW_*`, `*_2023_*` (sono gli stessi PDF re-ingeriti con metadati diversi; 302 chunk hanno `etf_isin` vuoto).
  - Done quando: dopo rebuild l'indice contiene ~300 chunk, tutti con `etf_isin` valorizzato e `year == 2026`.

- [x] **1.3 `embed.py` indicizza solo ciò che è registrato**
  - Caricare solo i JSON corrispondenti alle voci di `metadata.json` (via `build_output_stem` di `ingest.py`), non `*.json` alla cieca; loggare e ignorare i file non registrati.
  - Done quando: un JSON estraneo in `data/processed/` non finisce nell'indice.

- [x] **1.4 `chunk_id` stabile e anti-duplicato**
  - In `ingest.py` (`make_chunk_id`, `build_key_facts_chunk`) basare l'id su `source_file + block_type + hash(testo completo)`, non su `year + text[:80]`.
  - Aggiungere in `embed.py` un check di dedup su hash del testo per stessa `etf_isin` + `doc_type`.
  - Done quando: re-ingerire lo stesso PDF con anno diverso non produce chunk duplicati.

- [x] **1.5 `.gitignore`** — aggiungere `src/index/`, `.env`, `*.sqlite3`; rimuovere dal repo eventuali file già tracciati per errore (`git rm --cached`).

- [ ] **1.6 Rebuild pulito** — `python setup.py --rebuild` e verificare conteggi con `Retriever.collection_stats()`.

---

## Fase 2 — Valutazione affidabile

- [ ] **2.1 Rubric Type 4 corretta** (`evaluation/evaluate.py::_score_rubric`)
  - Oggi un item passa se compare *una qualsiasi parola* della frase (anche "for", "both") → T4 sempre 100%.
  - Sostituire con keyword esplicite per item nel `ground_truth.json` (es. `{"item": "...", "keywords": ["0.20%", "0.06%"], "min_match": 2}`) oppure con LLM-as-judge (2.4).
  - `must_not_contain` oggi cerca letteralmente la frase ("hallucinated figures") → sempre vero: sostituire con judge LLM.

- [ ] **2.2 Controllo "unsupported claims" meno rumoroso** (`_check_unsupported_claims`)
  - Rimuovere dal testo della risposta le citazioni `[ISIN | … | year]` prima di estrarre i numeri; ignorare anni (19xx/20xx) e frammenti di ISIN; normalizzare formati (`0,20 %` vs `0.20%`).

- [ ] **2.3 Match della risposta attesa meno rigido** (`_answer_contains_expected`)
  - Per Type 1: verificare la presenza dei *valori chiave* (aggiungere `expected_values: [...]` in `ground_truth.json`) invece della stringa intera.
  - Per Type 2: una lista di valori attesi per ogni ETF.

- [ ] **2.4 Attivare LLM-as-judge** — decommentare/completare `evaluate_faithfulness_llm` usando il client della Fase 3; flag `--judge llm|heuristic`. Usare un modello diverso/più grande di quello che genera, temperature 0.

- [ ] **2.5 Creare `evaluation/run_retrieval.py`** (citato nel README ma inesistente): esegue solo il retrieval sulla ground truth e scrive `pipeline_output.json` compatibile con `evaluate.py --retrieval_only`.

- [ ] **2.6 Report comparabile** — salvare in `report.json` anche `model`, `provider`, `chunk_size`, `embedding_model`, timestamp; aggiungere `--compare <baseline.json>` che stampa i delta.
  - Done quando (fase 2): rieseguendo la valutazione con `llama3.2:3b` sull'indice pulito si ottiene una nuova baseline credibile.

---

## Fase 3 — Passaggio a LLM in cloud (sostituisce Ollama come default)

- [ ] **3.1 Client LLM unico** — nuovo `src/llm.py`:
  - `PROVIDERS = {ollama, groq, gemini, openrouter}` → base_url compatibili OpenAI
    (`http://localhost:11434/v1`, `https://api.groq.com/openai/v1`, `https://generativelanguage.googleapis.com/v1beta/openai/`, `https://openrouter.ai/api/v1`).
  - Lettura da env: `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL` (nessun nome modello hard-coded: dipende dalla console del provider).
  - Funzione `get_client() -> (OpenAI, model_name)`; retry con backoff su 429/5xx (rate limit free tier).
  - Done quando: `generate.py` e `agent.py` non istanziano più `OpenAI(...)` direttamente.

- [ ] **3.2 `.env` + `python-dotenv`** — creare `.env.example` (senza chiavi reali) con le 3 variabili; caricarlo in `config.py`.

- [ ] **3.3 Aggiornare `generate.py` / `agent.py` / `ask.py`**
  - Rimuovere `DEFAULT_MODEL = "llama3.2:3b"`/`"mistral:7b"` e `_check_connection` specifico Ollama (fare check solo se provider == ollama).
  - Flag CLI `--provider` e `--model` che sovrascrivono l'env.

- [ ] **3.4 `setup.py`** — il check Ollama diventa opzionale: se `LLM_PROVIDER != ollama` verificare solo che `LLM_API_KEY` sia impostata (e fare una chiamata di prova).

- [ ] **3.5 Valutazione per provider** — eseguire la pipeline completa con Groq e con Gemini e salvare i report (`evaluation/report_<provider>_<model>.json`).
  - Done quando: esiste un confronto baseline 3B vs modelli cloud su tutte le metriche.

---

## Fase 4 — Agent come unico entrypoint

- [ ] **4.1 Robustezza loop in `agent.py`**
  - Decidere sui tool call con `if message.tool_calls:` (non `finish_reason == "tool_calls"`, inaffidabile tra provider).
  - Gestire `message.content is None` (niente `.strip()` su None).
  - Gestire JSON degli argomenti malformato con messaggio di errore restituito al modello.

- [ ] **4.2 Tool `search_etf_docs` più espressivo** — aggiungere parametri `isins: list[str]`, `mode: single|comparative|cross_doc`, `year`; enum di issuer/ISIN generati dal registro (4.4), non hard-coded.

- [ ] **4.3 Memoria conversazionale** — mantenere la history tra domande nella modalità interattiva (con comando `reset`), troncata a N turni.

- [ ] **4.4 Registro fondi unico** — `src/registry.py` che costruisce `KNOWN_ISINS` (nome, issuer, ticker Yahoo) da `metadata.json`; eliminare le copie in `ask.py`, `agent.py`, `live_data.py` (`ISIN_TO_NAME`, `ISIN_TO_TICKER`). Aggiungere il campo `yahoo_ticker` in `metadata.json`.

- [ ] **4.5 Deprecare il router a keyword di `ask.py`**
  - `ask.py` diventa un wrapper sottile sopra `Agent` (o viene rimosso), mantenendo i comandi `funds`, `chunks`, `help`, `exit`.
  - Se si vuole tenere un percorso non-agentico: router via LLM con output strutturato (JSON: isins, doc_types, mode, query_type) invece delle keyword (oggi `"less"` matcha `"unless"`, `cross_doc` non scatta mai perché `doc_type` è sempre impostato).
  - Adattare `pipeline.py` per poter valutare anche l'agent.

- [ ] **4.6 Citazioni strutturate** — far restituire al modello le fonti come riferimenti ai chunk (`[Chunk 3]`) e risolverle lato codice nei metadati reali, invece di far scrivere a mano `[ISIN | issuer | doc_type | year]` (riduce citazioni sbagliate).

---

## Fase 5 — Qualità del retrieval

- [ ] **5.1 Chunk più grandi** — con modelli cloud portare `CHUNK_SIZES` a ~800–1200 caratteri (factsheet) e ~1000–1500 (KID), overlap ~15%. Rieseguire la valutazione e scegliere i valori in base alle metriche.
- [ ] **5.2 Embedding multilingue** — provare `intfloat/multilingual-e5-small` (ricordare i prefissi `query:`/`passage:`) o `paraphrase-multilingual-MiniLM-L12-v2`; test con domande in italiano aggiunte alla ground truth. Rendere il modello configurabile in `config.py` e salvarlo nei metadati della collection (errore se query e indice usano modelli diversi).
- [ ] **5.3 Retrieval ibrido** — aggiungere BM25 (`rank_bm25`) e fondere con i risultati vettoriali (Reciprocal Rank Fusion): aiuta su ISIN, sigle (TER, OCF, SRI) e numeri.
- [ ] **5.4 Soglia di score** — scartare chunk sotto una similarità minima e dirlo al modello.
- [ ] **5.5 Modalità "full context" di confronto** — l'intero corpus 2026 è ~19k token: aggiungere una modalità che passa i documenti interi al modello, per misurare quanto il RAG aggiunge rispetto al contesto completo.

---

## Fase 6 — Dati di mercato (`src/live_data.py`)

- [ ] **6.1 Verificare il ticker UBS** — `0P0001FMRI.L` sembra un ID fondo, non il ticker di borsa dell'ETF; trovare il ticker Yahoo corretto su Xetra/Borsa Italiana e spostarlo in `metadata.json` (4.4).
- [ ] **6.2 Unità dei rendimenti** — verificare se `ytdReturn`, `threeYearAverageReturn`, `fiveYearAverageReturn` sono frazioni (0.12) e nel caso moltiplicare ×100 prima di `fmt_pct`.
- [ ] **6.3 Cache** — cache su disco con TTL (es. 1h) per evitare chiamate ripetute a Yahoo durante la valutazione.

---

## Fase 7 — Corpus

- [ ] **7.1 Aggiungere ETF** — almeno un altro MSCI World (Xtrackers, Amundi) e ETF su indici diversi (S&P 500, Emerging Markets, obbligazionario) per rendere significativi i confronti.
- [ ] **7.2 Documenti multi-anno** — scaricare factsheet/KID di anni diversi per attivare le domande Type 3 (`_parked_type3` in `ground_truth.json`) e il prompt `_prompt_type3`.
- [ ] **7.3 Estendere `ground_truth.json`** — domande sui nuovi ETF, domande in italiano, domande "senza risposta" (il modello deve dire che l'informazione manca).

---

## Fase 8 — Ingegneria del progetto

- [ ] **8.1 `requirements.txt`** — rimuovere `argparse`, `pathlib`, `typing` (stdlib); fissare le versioni; aggiungere `python-dotenv`, `yfinance`, `rank_bm25`, `pytest`. Valutare il passaggio a `pyproject.toml`.
- [ ] **8.2 Rinominare `setup.py`** in `bootstrap.py` (o `scripts/setup_index.py`): `setup.py` è il nome riservato al packaging setuptools.
- [ ] **8.3 Packaging `src/`** — trasformare `src/` in pacchetto (`fundscope/` con `__init__.py`) ed eliminare gli import che funzionano solo lanciando dalla cartella giusta (e il fallback `RetrievedChunk` duplicato in `generate.py`).
- [ ] **8.4 Logging** — sostituire i `print("[retrieve] …")` con `logging`, livello configurabile via `--verbose`.
- [ ] **8.5 Test (`tests/`)** con pytest:
  - `_build_where`, `parse_citations`, `split_text`, `make_chunk_id` (unit);
  - ingest di un PDF di esempio → numero e metadati dei chunk;
  - retrieval end-to-end su indice temporaneo (senza LLM);
  - agent con client LLM mockato (sequenza tool call → risposta).
- [ ] **8.6 CI GitHub Actions** — su push: lint (`ruff`), test, valutazione retrieval-only; fallire se la recall scende sotto una soglia.
- [ ] **8.7 README** — riallineare comandi (`python src/ask.py`, percorso `src/metadata.json`, campo `isin` invece di `ticker` nell'esempio), sezione provider cloud + `.env`, tabella risultati della valutazione per modello, rimuovere riferimenti a file inesistenti.

---

## Fase 9 — Interfaccia e pubblicazione (opzionale, per portfolio)

- [ ] **9.1 UI web** — app Streamlit o Gradio: chat, visualizzazione fonti/chunk usati, pannello dati live, selettore provider/modello.
- [ ] **9.2 Deploy gratuito** — Hugging Face Spaces o Streamlit Community Cloud, con chiave API come secret; indice costruito al primo avvio o committato come artefatto.
- [ ] **9.3 Disclaimer** — nota "non è consulenza finanziaria" nella UI e nelle risposte che confrontano l'idoneità dei fondi.

---

## Riferimenti rapidi provider (free tier, settembre 2026 — verificare nella console)

| Provider | Base URL OpenAI-compatibile | Limiti indicativi |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | ~1.000 req/giorno, ~30 RPM |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | 20–1.500 req/giorno secondo il modello |
| OpenRouter | `https://openrouter.ai/api/v1` | 50 req/giorno sui modelli `:free` |
| Ollama (locale) | `http://localhost:11434/v1` | limitato dall'hardware |
