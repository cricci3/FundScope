# FundScope — TODO / Roadmap di miglioramento

Roadmap derivata dalla review del progetto (settembre 2026, riordinata il 01/10/2026). Le fasi vanno eseguite **in ordine**.

**Decisione (01/10/2026): Ollama è stato disinstallato e non verrà più usato.** La generazione passa a un LLM in cloud
(fase 2) *prima* della revisione della valutazione (fase 3): il judge LLM della fase 3 richiede un modello in cloud,
e senza modello locale non è possibile rimisurare la generazione del 3B.

**Riferimenti di metrica**
- `evaluation/baseline_llama3.2-3b.json` — solo **storico**: misurato su indice sporco (chunk triplicati) e con scorer
  difettoso, quindi **non confrontabile** con le run future.
- Retrieval-only dopo la fase 1: precision 0.988 · recall 1.000 (questa è la baseline valida del retrieval).
- La **baseline di riferimento della generazione** viene fissata a fine fase 3 (task 3.7), con modello cloud e scorer corretto.
  Le metriche di generazione misurate durante la fase 2 servono solo come smoke test: non trarne conclusioni.

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

- [x] **1.1b Ambiente uv** — `pyproject.toml` + `uv.lock` + `.python-version` (3.13); `.venv` gestito da `uv sync`, script lanciati con `uv run python ...`.

- [x] **1.2 Eliminare indici e dati duplicati**
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

- [x] **1.6 Rebuild pulito** — `python setup.py --rebuild` e verificare conteggi con `Retriever.collection_stats()`.
  - Esito (30/09/2026): 299 chunk (factsheet 198, kid 101; IE00B4L5Y983 115, IE00BD4TXV59 184), tutti con `etf_isin` e `year == 2026`. Retrieval-only: precision 0.502 → 0.988, recall 0.956 → 1.000. Metriche di generazione non rimisurate (Ollama non installato).

---

## Fase 2 — Migrazione a LLM in cloud (rimozione di Ollama)

- [ ] **2.1 Client LLM unico** — nuovo `src/llm.py`:
  - `PROVIDERS = {groq, gemini, openrouter}` → base_url compatibili OpenAI
    (`https://api.groq.com/openai/v1`, `https://generativelanguage.googleapis.com/v1beta/openai/`, `https://openrouter.ai/api/v1`).
    Struttura a dizionario: aggiungere un provider (anche a pagamento) = una riga.
  - Lettura da env (via `config.py`): `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`. Nessun nome di modello hard-coded nel codice: dipende dalla console del provider.
  - `get_client() -> (OpenAI, model_name)`; errore chiaro se la chiave manca.
  - Retry con backoff esponenziale su 429/5xx e throttling configurabile (`LLM_MIN_INTERVAL_S`): i free tier hanno limiti per minuto (es. Groq ~30 RPM / pochi k token/min).
  - Done quando: `src/llm.py` esiste e una chiamata di prova (`uv run python src/llm.py --ping`) risponde.

- [ ] **2.2 `.env` + configurazione**
  - `.env.example` committato (senza chiavi reali) con `LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`, `LLM_MIN_INTERVAL_S`; `.env` già in `.gitignore` (verificare).
  - `config.py` carica `.env` con `python-dotenv` (già tra le dipendenze in `pyproject.toml`).
  - Done quando: copiando `.env.example` in `.env` e inserendo la chiave, tutto funziona senza altri passaggi.

- [ ] **2.3 Rimuovere Ollama da `generate.py` e `agent.py`**
  - Usare `get_client()` al posto di `OpenAI(base_url=OLLAMA_BASE_URL, …)`; eliminare `OLLAMA_BASE_URL`, `DEFAULT_MODEL = "llama3.2:3b"` / `"mistral:7b"`, `_check_connection` / `_check_model` specifici di Ollama.
  - Flag CLI `--provider` e `--model` che sovrascrivono l'env (anche in `ask.py` e `pipeline.py`).
  - Done quando: `grep -rni "ollama\|11434\|llama3.2\|mistral:7b" src/` non restituisce nulla.

- [ ] **2.4 Robustezza minima del loop in `agent.py`** (necessaria per provider diversi da Ollama)
  - Decidere sui tool call con `if message.tool_calls:` invece di `finish_reason == "tool_calls"` (inaffidabile tra provider).
  - Gestire `message.content is None` (niente `.strip()` su None).
  - Argomenti JSON malformati → messaggio d'errore restituito al modello come risultato del tool, non eccezione.
  - Done quando: `uv run python src/agent.py --query "..." --show_calls` funziona con almeno due provider.

- [ ] **2.5 Aggiornare `setup.py`**
  - Rimuovere `check_ollama`, `--skip_ollama`, `--model` e tutti i messaggi su `ollama pull`.
  - Nuovo check: `LLM_API_KEY` impostata + chiamata di prova al provider; se fallisce, il setup completa comunque l'indice e lo segnala (modalità retrieval-only).
  - Done quando: `uv run python setup.py` gira da zero senza alcun riferimento a Ollama.

- [ ] **2.6 Metadati della run** — `pipeline.py` salva in `pipeline_output.json` `provider`, `model`, timestamp e token usati (oggi `model` è `None`).

- [ ] **2.7 Smoke test end-to-end**
  - `uv run python src/ask.py --query "What is the TER of the iShares Core MSCI World ETF?"` e una domanda comparativa.
  - Pipeline completa sulle 17 domande con il provider principale; salvare l'output in `evaluation/runs/` (non sovrascrivere).
  - Done quando: tutte le 17 domande producono una risposta senza errori di rete/rate limit. Le metriche **non** sono ancora affidabili (fase 3).

- [ ] **2.8 Documentazione minima** — aggiornare `CLAUDE.md` (sezioni Commands/Architecture: niente più Ollama né `--skip_ollama`) e la sezione "Prerequisites" del `README.md` (provider cloud + `.env`). Il README completo resta nel task 8.7.

---

## Fase 3 — Valutazione affidabile

Le correzioni allo scorer (3.1–3.3) sono solo codice e si possono verificare sull'output della fase 2.7; il judge (3.4) richiede il client cloud.

- [ ] **3.1 Rubric Type 4 corretta** (`evaluation/evaluate.py::_score_rubric`)
  - Oggi un item passa se compare *una qualsiasi parola* della frase (anche "for", "both") → T4 sempre 100%.
  - Keyword esplicite per item nel `ground_truth.json` (es. `{"item": "...", "keywords": ["0.20%", "0.06%"], "min_match": 2}`) come scorer euristico di riserva.
  - `must_not_contain` oggi cerca letteralmente la frase ("hallucinated figures") → sempre vero: delegarlo al judge (3.4).

- [ ] **3.2 Controllo "unsupported claims" meno rumoroso** (`_check_unsupported_claims`)
  - Rimuovere dalla risposta le citazioni `[ISIN | … | year]` prima di estrarre i numeri; ignorare anni (19xx/20xx) e frammenti di ISIN; normalizzare i formati (`0,20 %` vs `0.20%`).

- [ ] **3.3 Match della risposta attesa meno rigido** (`_answer_contains_expected`)
  - Type 1: verificare i *valori chiave* (nuovo campo `expected_values: [...]` in `ground_truth.json`) invece dell'intera stringa.
  - Type 2: lista di valori attesi per ogni ETF.

- [ ] **3.4 LLM-as-judge** — completare `evaluate_faithfulness_llm` usando `src/llm.py`; flag `--judge llm|heuristic`.
  - Judge configurabile separatamente (`JUDGE_PROVIDER`, `JUDGE_MODEL`, `JUDGE_API_KEY`): idealmente **un modello diverso da quello che genera** (es. genera con Groq, giudica con Gemini), temperature 0, output JSON validato.
  - Done quando: il judge restituisce per ogni domanda verdetto + motivazione salvati nel report.

- [ ] **3.5 Creare `evaluation/run_retrieval.py`** (citato nel README ma inesistente): solo retrieval sulla ground truth, output compatibile con `evaluate.py --retrieval_only`. Nessun LLM richiesto.

- [ ] **3.6 Report comparabili** — salvare nel report `provider`, `model`, `judge_model`, `chunk_size`, `embedding_model`, timestamp; salvare ogni run in `evaluation/runs/<timestamp>_<model>.json`; flag `--compare <report.json>` che stampa i delta.

- [ ] **3.7 Baseline ufficiale**
  - Rieseguire pipeline + valutazione (judge attivo) con il provider principale e almeno un secondo modello.
  - Salvare come `evaluation/baseline_cloud.json` e riportare le metriche in testa a questo file e in `CLAUDE.md`.
  - Done quando: esiste una baseline di generazione affidabile; tutte le fasi successive si confrontano con questa.

---

## Fase 4 — Agent come unico entrypoint

- [ ] **4.1 Tool `search_etf_docs` più espressivo** — parametri `isins: list[str]`, `mode: single|comparative|cross_doc`, `year`; enum di issuer/ISIN generati dal registro (4.3), non hard-coded.

- [ ] **4.2 Memoria conversazionale** — history mantenuta tra domande in modalità interattiva (comando `reset`), troncata a N turni.

- [ ] **4.3 Registro fondi unico** — `src/registry.py` che costruisce `KNOWN_ISINS` (nome, issuer, ticker Yahoo) da `metadata.json`; eliminare le copie in `ask.py`, `agent.py`, `live_data.py` (`ISIN_TO_NAME`, `ISIN_TO_TICKER`). Aggiungere il campo `yahoo_ticker` in `metadata.json`.

- [ ] **4.4 Deprecare il router a keyword di `ask.py`**
  - `ask.py` diventa un wrapper sottile sopra `Agent` (o viene rimosso), mantenendo i comandi `funds`, `chunks`, `help`, `exit`.
  - In alternativa, per un percorso non agentico: router via LLM con output strutturato (JSON: isins, doc_types, mode, query_type) al posto delle keyword (oggi `"less"` matcha `"unless"` e `cross_doc` non scatta mai perché `doc_type` è sempre impostato).
  - `pipeline.py` deve poter valutare anche l'agent.

- [ ] **4.5 Citazioni strutturate** — il modello cita i chunk (`[Chunk 3]`) e il codice li risolve nei metadati reali, invece di fargli scrivere a mano `[ISIN | issuer | doc_type | year]`.
  - Done quando (fase 4): metriche ≥ baseline 3.7, attribution in miglioramento.

---

## Fase 5 — Qualità del retrieval

- [ ] **5.1 Chunk più grandi** — con modelli cloud portare `CHUNK_SIZES` a ~800–1200 caratteri (factsheet) e ~1000–1500 (KID), overlap ~15%; scegliere in base alle metriche.
- [ ] **5.2 Embedding multilingue** — provare `intfloat/multilingual-e5-small` (prefissi `query:`/`passage:`) o `paraphrase-multilingual-MiniLM-L12-v2`; aggiungere domande in italiano alla ground truth. Salvare il modello di embedding nei metadati della collection (errore se query e indice usano modelli diversi).
- [ ] **5.3 Retrieval ibrido** — BM25 (`rank_bm25`) + vettoriale fusi con Reciprocal Rank Fusion: aiuta su ISIN, sigle (TER, OCF, SRI) e numeri.
- [ ] **5.4 Soglia di score** — scartare chunk sotto una similarità minima e segnalarlo al modello.
- [ ] **5.5 Modalità "full context" di confronto** — l'intero corpus 2026 è ~19k token: modalità che passa i documenti interi al modello, per misurare quanto il RAG aggiunge rispetto al contesto completo (attenzione ai limiti token/min dei free tier).

---

## Fase 6 — Dati di mercato (`src/live_data.py`)

- [ ] **6.1 Verificare il ticker UBS** — `0P0001FMRI.L` sembra un ID fondo, non il ticker di borsa dell'ETF; trovare il ticker Yahoo corretto (Xetra/Borsa Italiana) e spostarlo in `metadata.json` (4.3).
- [ ] **6.2 Unità dei rendimenti** — verificare se `ytdReturn`, `threeYearAverageReturn`, `fiveYearAverageReturn` sono frazioni (0.12) e in tal caso moltiplicare ×100 prima di `fmt_pct`.
- [ ] **6.3 Cache** — cache su disco con TTL (es. 1h) per evitare chiamate ripetute a Yahoo durante la valutazione.

---

## Fase 7 — Corpus

- [ ] **7.1 Aggiungere ETF** — almeno un altro MSCI World (Xtrackers, Amundi) e ETF su indici diversi (S&P 500, Emerging Markets, obbligazionario).
- [ ] **7.2 Documenti multi-anno** — factsheet/KID di anni diversi per attivare le domande Type 3 (`_parked_type3` in `ground_truth.json`) e `_prompt_type3`.
- [ ] **7.3 Estendere `ground_truth.json`** — domande sui nuovi ETF, domande in italiano, domande "senza risposta" (il modello deve dire che l'informazione manca).

---

## Fase 8 — Ingegneria del progetto

- [ ] **8.1 Dipendenze** — eliminare `requirements.txt` (sostituito da `pyproject.toml` + `uv.lock`); aggiungere come dev-dependency `pytest` e `ruff` (`uv add --dev`); `rank_bm25` quando serve (5.3).
- [ ] **8.2 Rinominare `setup.py`** in `bootstrap.py` (o `scripts/setup_index.py`): con un `pyproject.toml` presente, `setup.py` è il nome riservato al packaging setuptools e può essere eseguito per errore da `pip install .`.
- [ ] **8.3 Packaging `src/`** — trasformare `src/` in pacchetto (`fundscope/` con `__init__.py`), eliminare gli import che dipendono dalla cwd, il `sys.path` hack in `setup.py` e il fallback `RetrievedChunk` duplicato in `generate.py`.
- [ ] **8.4 Logging** — sostituire i `print("[retrieve] …")` con `logging`, livello via `--verbose`; risolve anche il problema `UnicodeEncodeError` su Windows con stdout in pipe.
- [ ] **8.5 Test (`tests/`)** con pytest:
  - unit: `_build_where`, `parse_citations`, `split_text`, `make_chunk_id`, scorer di `evaluate.py`;
  - ingest di un PDF di esempio → numero e metadati dei chunk;
  - retrieval end-to-end su indice temporaneo (senza LLM);
  - agent con client LLM **mockato** (sequenza tool call → risposta): nessuna chiamata di rete nei test.
- [ ] **8.6 CI GitHub Actions** — su push: `uv sync`, `ruff`, test, valutazione retrieval-only; fallire se la recall scende sotto soglia. Nessuna chiave API in CI (la valutazione della generazione resta manuale).
- [ ] **8.7 README** — riallineare comandi (`uv run python src/...`, `src/metadata.json`, campo `isin` nell'esempio), sezione provider cloud + `.env`, tabella risultati per modello, rimuovere riferimenti a Ollama e a file inesistenti.

---

## Fase 9 — Interfaccia e pubblicazione (opzionale, per portfolio)

- [ ] **9.1 UI web** — Streamlit o Gradio: chat, fonti/chunk usati, pannello dati live, selettore provider/modello.
- [ ] **9.2 Deploy gratuito** — Hugging Face Spaces o Streamlit Community Cloud, chiave API come secret; indice costruito al primo avvio o committato come artefatto.
- [ ] **9.3 Disclaimer** — "non è consulenza finanziaria" nella UI e nelle risposte che confrontano l'idoneità dei fondi.

---

## Riferimenti rapidi provider (free tier, settembre 2026 — verificare nella console)

| Provider | Base URL OpenAI-compatibile | Limiti indicativi |
|---|---|---|
| Groq | `https://api.groq.com/openai/v1` | ~1.000 req/giorno, ~30 RPM |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | 20–1.500 req/giorno secondo il modello |
| OpenRouter | `https://openrouter.ai/api/v1` | 50 req/giorno sui modelli `:free` |
