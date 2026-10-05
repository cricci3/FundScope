# FundScope — TODO / Roadmap di miglioramento

Roadmap derivata dalla review del progetto (settembre 2026, riordinata il 01/10/2026). Le fasi vanno eseguite **in ordine**.

**Decisione (01/10/2026): Ollama è stato disinstallato e non verrà più usato.** La generazione passa a un LLM in cloud
(fase 2) *prima* della revisione della valutazione (fase 3): il judge LLM della fase 3 richiede un modello in cloud,
e senza modello locale non è possibile rimisurare la generazione del 3B.

**Decisione (02/10/2026): provider principale = Claude via API Anthropic** (chiave già disponibile, credito prepagato 5 $),
con supporto al cambio di provider tramite `.env`. Il credito è limitato: durante lo sviluppo usare sottoinsiemi
di domande (`--limit`, `--qids`) e lanciare le valutazioni complete solo a fine fase.

**Workflow git: si lavora solo su `main`, nessun branch.** Un commit per task completato (messaggio che cita il numero
del task, es. `2.3: backend OpenAI-compatibile`), push su `origin/main` a fine fase.

**Riferimenti di metrica**
- `evaluation/baseline_llama3.2-3b.json` — solo **storico**: misurato su indice sporco (chunk triplicati) e con scorer
  difettoso, quindi **non confrontabile** con le run future.
- Retrieval-only dopo la fase 1: precision 0.988 · recall 1.000 (questa è la baseline valida del retrieval).
- **Baseline ufficiale della generazione (02/10/2026, task 3.7): `evaluation/baseline_cloud.json`** — tutte le fasi
  successive si confrontano con questa (`evaluate.py ... --judge llm --compare evaluation/baseline_cloud.json`).
  Generatore Claude Haiku 4.5, judge Claude Sonnet 5.5 (effort medium), 17 domande, run
  `evaluation/runs/20261002T134517Z_claude-haiku-4-5.json`. Costo: pipeline 0,028 $ + judge 0,093 $ = 0,121 $.

  | Metrica | Overall | Type 1 (7) | Type 2 (8) | Type 4 (2) |
  |---|---|---|---|---|
  | Judge verdict pass | **35.3%** | 71.4% | 12.5% | 0.0% |
  | Judge correctness (correct=1, partial=½) | 46.7% | | | |
  | Judge grounded | 82.4% | | | |
  | Judge abstained ("i documenti non contengono…") | 35.3% | | | |
  | Retrieval precision / recall (a livello di documento) | 0.988 / 1.000 | | | |
  | Attribution | 88.2% | 100% | 87.5% | 50.0% |
  | Euristica: expected values trovati (T1–2) | 42.9% | | | |

  Lettura: il retrieval trova sempre il *documento* giusto (P/R ≈ 1 sono misurate per documento), ma spesso non il
  *chunk* con il dato → il modello si astiene (35%). Il collo di bottiglia è il retrieval a livello di chunk, non la generazione.
- Le metriche di generazione della fase 2 (smoke test) non sono confrontabili con la baseline (scorer diverso).

Convenzioni: ogni task ha i file coinvolti e un criterio **Done quando**. Spuntare `[x]` a lavoro finito.

---

## Stato attuale (aggiornato 2026-10-05)

- **Branch:** solo `main`, allineato e pushato su `origin/main` (`d66e573`, fase 4 chiusa). Branch locale `refactor/cloud-llm` cancellato; resta `origin/refactor/cloud-llm` su GitHub → task 0.3 aperto (cancellazione remota da confermare con l'utente).
- **Test/lint:** nessuna suite né ruff (`uv run pytest` / `uv run ruff` → "program not found"; previsti in fase 8). Le verifiche della fase 4 (risoluzione `[Chunk N]`, memoria con LLM finto, argomenti di `search_etf_docs`, rimozione del preambolo) sono state fatte con script temporanei non salvati.
- **Fatto nell'ultima sessione:** fase 4 completa e verificata con run completa + judge (esito e tabella sotto la fase 4): judge pass 35% → 71%, astensioni 35% → 6%. Credito Anthropic speso finora ≈ 1,16 $.
- **Problemi aperti / scoperti:**
  - Retrieval: nel factsheet iShares "Product Structure : Physical" è spezzato dal layout a colonne e non viene trovato → T1_006 e T2_007 falliscono anche dopo 4–6 ricerche (fase 5 e/o ingest).
  - Ticker UBS `0P0001FMRI.L` → 404 su Yahoo (task 6.1); `EUNL.DE` funziona con dati reali.
  - Agent ≈ 0,012 $/domanda (3,2 chiamate in media, prompt che cresce a ogni ricerca, nessun cache hit perché Haiku 4.5 richiede prefissi ≥ 4096 token): da valutare un breakpoint di cache sull'ultimo messaggio o meno chunk per ricerca.
  - Haiku a volte inventa esempi numerici (T4_001, "€100.000 in 20 anni") nonostante la regola nel system prompt. Judge non deterministico (±1 domanda tra run).
- **Decisioni recenti:** `pipeline.py` valuta di default l'agent, che riceve solo la domanda (la pipeline fissa resta come `--engine rag`, per confronto); il judge riceve il registro fondi dato all'agent (`reference_data`), perché fa parte del contesto dell'agent; `baseline_cloud.json` resta la baseline ufficiale (la promozione di `report_phase4_agent.json` non è stata decisa).
- **Proposta (da approvare, non in roadmap):** metrica di retrieval a livello di chunk (il chunk recuperato/citato contiene il valore atteso?); l'output della pipeline salva già `cited_chunk_ids`.
- **Prossimo passo:** 1) fase 5: ispezionare i chunk del factsheet iShares (`data/processed/IE00B4L5Y983_2026_Q4_factsheet.json`) e provare 5.1 (chunk più grandi) misurando con `evaluation/run_retrieval.py` (gratis) e `--qids T1_006 T2_007`; 2) chiedere all'utente se cancellare `origin/refactor/cloud-llm` (chiude 0.3).

---

## Fase 0 — Preparazione

- [x] **0.1 Branch di lavoro** — ~~creare `refactor/cloud-llm` da `main`~~. Superato: il lavoro della fase 1 è già stato integrato in `main`; da ora si lavora solo su `main` (vedi 0.3).
- [x] **0.2 Salvare la baseline** — copiare `evaluation/report.json` in `evaluation/baseline_llama3.2-3b.json` per confronto futuro.
  - Done quando: il file baseline esiste e non viene sovrascritto dalle run successive.
- [ ] **0.3 Eliminare il branch `refactor/cloud-llm`**
  - Verificare che sia interamente contenuto in `main` (`git branch --merged main` deve elencarlo), poi `git branch -d refactor/cloud-llm`; se esiste anche su GitHub, `git push origin --delete refactor/cloud-llm`.
  - Done quando: `git branch -a` mostra solo `main` (e `origin/main`).

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

## Fase 2 — Migrazione a LLM in cloud: Claude (Anthropic) + provider intercambiabili

Obiettivo: **Claude via API Anthropic come provider principale** (chiave già disponibile, credito prepagato 5 $),
con un'architettura che permetta di cambiare provider (Groq, Gemini, OpenRouter, …) **solo modificando `.env`**, senza toccare il codice.

Scelte di progetto:
- Backend **nativo** per Anthropic (SDK `anthropic`, non il layer compatibile OpenAI, che la documentazione Anthropic sconsiglia fuori dai test e che non supporta prompt caching, citazioni native e schema garantito dei tool).
- Backend **OpenAI-compatibile** generico per tutti gli altri provider (client `openai` già presente).
- Il resto del codice (`generate.py`, `agent.py`, `evaluate.py`) usa **solo un'interfaccia neutra** e non sa quale provider c'è sotto.
- Modelli di default: generazione/agent `claude-haiku-4-5-20251001` (1 $ / 5 $ per M token in/out); judge della fase 3 `claude-sonnet-5-5` (2 $ / 10 $). Verificare gli identificativi nella documentazione Anthropic prima di fissarli in `.env.example`.

- [x] **2.1 Interfaccia LLM neutra** — nuovo pacchetto/modulo `src/llm/` (o `src/llm.py` se resta piccolo):
  - Tipi neutri: `Message(role, content, tool_calls?, tool_call_id?)`, `ToolSpec(name, description, parameters_json_schema)`, `ToolCall(id, name, arguments: dict)`, `LLMResponse(text, tool_calls, usage: Usage(input_tokens, output_tokens, cache_read_tokens, cache_write_tokens), model, provider, stop_reason)`.
  - Interfaccia `LLMClient` con un solo metodo: `chat(messages, *, system=None, tools=None, temperature=0.0, max_tokens=1024) -> LLMResponse`.
  - Factory `get_llm(role="generator"|"judge") -> LLMClient` che legge la configurazione (2.4).
  - Done quando: `generate.py` e `agent.py` possono essere scritti senza importare né `openai` né `anthropic`.

- [x] **2.2 Backend Anthropic (nativo, default)** — `AnthropicClient(LLMClient)`:
  - `uv add anthropic`; `anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)` → `client.messages.create(...)`.
  - Conversione formati: `system` come parametro separato (non come messaggio); tool definiti con `input_schema`; risposta con blocchi `text` / `tool_use` → `LLMResponse`; risultati dei tool inviati come blocchi `tool_result` in un messaggio `user`.
  - Gestione errori: retry con backoff su 429/529/5xx (`anthropic.RateLimitError`, `APIStatusError`); errore chiaro su credito esaurito / chiave non valida (non ritentare).
  - Prompt caching sul blocco `system` + definizioni dei tool (si ripetono a ogni iterazione dell'agent): `cache_control` sull'ultimo blocco stabile. Verificare nella documentazione la soglia minima di token cacheabili per Haiku; se il prompt è sotto soglia, lasciare il codice pronto ma senza effetto.
  - Done quando: `uv run python -m ... --ping` (o `uv run python src/llm/cli.py --ping`) risponde con Haiku e stampa token e costo.

- [x] **2.3 Backend OpenAI-compatibile (provider alternativi)** — `OpenAICompatClient(LLMClient)`:
  - Registro provider: `{"groq": "https://api.groq.com/openai/v1", "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/", "openrouter": "https://openrouter.ai/api/v1"}`; aggiungere un provider = una riga.
  - Conversione tool/messaggi da/verso il formato OpenAI (`tools=[{"type":"function",...}]`, `message.tool_calls`, ruolo `tool`).
  - Decidere sui tool call con `if message.tool_calls:` (non `finish_reason`), gestire `content is None`.
  - Done quando: lo stesso `--ping` funziona con `LLM_PROVIDER=groq` (se si dispone di una chiave; altrimenti coperto da test mockati in 8.5).
  - *Esito (02/10/2026): nessuna chiave Groq disponibile; conversione messaggi/tool, argomenti malformati, `content=None` e usage verificati con client mockato. Test permanenti in 8.5.*

- [x] **2.4 Configurazione e `.env`**
  - Variabili: `LLM_PROVIDER` (default `anthropic`), `LLM_MODEL`, `JUDGE_PROVIDER`, `JUDGE_MODEL`, chiavi **per provider** `ANTHROPIC_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`, `OPENROUTER_API_KEY` (così si cambia provider cambiando solo `LLM_PROVIDER`/`LLM_MODEL`), `LLM_MIN_INTERVAL_S`, `LLM_MAX_COST_USD`.
  - `config.py` carica `.env` con `python-dotenv`; `.env.example` committato senza chiavi; verificare che `.env` sia in `.gitignore` e **mai** committato.
  - Precedenza: flag CLI `--provider` / `--model` > variabili d'ambiente > default in `config.py`.
  - Done quando: passare da Claude a un altro provider richiede solo di modificare `.env` (o un flag CLI).

- [x] **2.5 Tracciamento costi e protezione del credito** (credito disponibile: 5 $)
  - Tabella prezzi per modello in `config.py` (o `src/llm/pricing.py`), facilmente aggiornabile; prezzo sconosciuto → costo `null` + warning, non errore.
  - Ogni `LLMResponse` accumula token e costo stimato in un contatore di sessione; `ask.py`/`agent.py` mostrano il costo della domanda con `--show_cost`; `pipeline.py` e `evaluate.py` stampano il totale a fine run.
  - Budget per run: se il costo stimato supera `LLM_MAX_COST_USD` (default 0.50) la run si interrompe salvando i risultati parziali.
  - `pipeline.py` / `evaluate.py`: opzioni `--limit N` e `--qids T1_001 T2_003 …` per lavorare su sottoinsiemi durante lo sviluppo.
  - Done quando: ogni run riporta token e costo; una run oltre budget si ferma senza perdere i risultati già ottenuti.

- [x] **2.6 Rimuovere Ollama e usare l'interfaccia in `generate.py`, `agent.py`, `ask.py`, `pipeline.py`**
  - Eliminare `OLLAMA_BASE_URL`, `DEFAULT_MODEL = "llama3.2:3b"` / `"mistral:7b"`, `_check_connection` / `_check_model`, ogni `OpenAI(...)` diretto.
  - `agent.py`: `TOOLS` riscritti come `ToolSpec` neutri; il loop usa `LLMResponse.tool_calls`; argomenti malformati → errore restituito al modello come risultato del tool; history dei messaggi in formato neutro.
  - Done quando: `grep -rniE "ollama|11434|llama3\.2|mistral:7b|from openai|import anthropic" src/` trova solo i file dei backend in `src/llm/`.

- [x] **2.7 Aggiornare `setup.py`**
  - Rimuovere `check_ollama`, `--skip_ollama`, `--model` e i messaggi su `ollama pull`; aggiungere `anthropic` alla lista dei pacchetti verificati.
  - Nuovo check: chiave del provider configurato presente + chiamata `--ping` minima (pochi token); se fallisce, l'indice viene comunque costruito e il setup segnala la modalità retrieval-only.
  - Done quando: `uv run python setup.py` gira da zero senza alcun riferimento a Ollama.

- [x] **2.8 Metadati della run** — `pipeline_output.json` salva `provider`, `model`, timestamp, token (input/output/cache) e costo stimato per domanda e totale (oggi `model` è `None`).

- [x] **2.9 Smoke test end-to-end (economico)**
  - `uv run python src/ask.py --query "What is the TER of the iShares Core MSCI World ETF?" --show_cost` + una domanda comparativa + `agent.py --show_calls` su una domanda che usa `get_live_data`.
  - Pipeline su `--limit 3`, poi sulle 17 domande una sola volta; output in `evaluation/runs/` (non sovrascrivere).
  - Done quando: tutte le 17 domande producono una risposta senza errori; costo totale riportato. Le metriche **non** sono ancora affidabili (fase 3).
  - *Esito (02/10/2026): 17/17 risposte senza errori con Haiku 4.5, 14 623 token in + 2 606 out, costo stimato 0,028 $ → `evaluation/runs/20261002T132512Z_claude-haiku-4-5.json`. Retrieval invariato (P 0.988 · R 1.000). Agent: tool loop OK; Yahoo Finance non raggiungibile dalla sandbox, errore gestito.*

- [x] **2.10 Documentazione: `README.md`** — nuova sezione **"LLM provider"** (sostituisce "Ollama" e aggiorna "Prerequisites"):
  - provider di default (Claude Haiku 4.5 via API Anthropic) e come ottenere/inserire la chiave (`cp .env.example .env`);
  - tabella delle variabili d'ambiente (2.4) con esempi;
  - **come cambiare provider**: esempio completo di `.env` per Anthropic e per Groq/Gemini/OpenRouter, e override da CLI (`--provider`, `--model`);
  - **come aggiungere un nuovo provider** (una riga nel registro se OpenAI-compatibile, altrimenti un nuovo backend che implementa `LLMClient`);
  - costi indicativi, `--show_cost`, `LLM_MAX_COST_USD`, `--limit`;
  - nota: la chiave non va mai committata.
  - Aggiornare anche `CLAUDE.md` (Commands/Architecture: niente Ollama, nuova architettura `src/llm/`, variabili d'ambiente).
  - Done quando: una persona che clona il repo configura e cambia provider seguendo solo il README.

---

## Fase 3 — Valutazione affidabile

Le correzioni allo scorer (3.1–3.3) sono solo codice e si possono verificare sull'output della fase 2.9 senza nuove chiamate API; il judge (3.4) usa l'interfaccia `get_llm(role="judge")`.

- [x] **3.1 Rubric Type 4 corretta** (`evaluation/evaluate.py::_score_rubric`)
  - Oggi un item passa se compare *una qualsiasi parola* della frase (anche "for", "both") → T4 sempre 100%.
  - Keyword esplicite per item nel `ground_truth.json` (es. `{"item": "...", "keywords": ["0.20%", "0.06%"], "min_match": 2}`) come scorer euristico di riserva.
  - `must_not_contain` oggi cerca letteralmente la frase ("hallucinated figures") → sempre vero: delegarlo al judge (3.4).

- [x] **3.2 Controllo "unsupported claims" meno rumoroso** (`_check_unsupported_claims`)
  - Rimuovere dalla risposta le citazioni `[ISIN | … | year]` prima di estrarre i numeri; ignorare anni (19xx/20xx) e frammenti di ISIN; normalizzare i formati (`0,20 %` vs `0.20%`).

- [x] **3.3 Match della risposta attesa meno rigido** (`_answer_contains_expected`)
  - Type 1: verificare i *valori chiave* (nuovo campo `expected_values: [...]` in `ground_truth.json`) invece dell'intera stringa.
  - Type 2: lista di valori attesi per ogni ETF.

- [x] **3.4 LLM-as-judge** — completare `evaluate_faithfulness_llm` usando `get_llm(role="judge")`; flag `--judge llm|heuristic`.
  - Judge configurabile separatamente (`JUDGE_PROVIDER`, `JUDGE_MODEL`, 2.4): **un modello diverso da quello che genera** — default Claude Sonnet 5.5 che giudica le risposte di Haiku 4.5. Temperature 0, output JSON validato (con il backend Anthropic usare gli output strutturati per avere lo schema garantito).
  - Il costo del judge rientra nel conteggio e nel budget della run (2.5).
  - Done quando: il judge restituisce per ogni domanda verdetto + motivazione salvati nel report.
  - *Esito (02/10/2026): `evaluation/judge.py`, output JSON con schema garantito (`output_config.format`) e ri-validato.
    Sonnet 5.5 non accetta `temperature` (400): si usa il default del modello, con `effort` (`JUDGE_EFFORT`, default medium).
    Prompt caching attivo sul system prompt del judge (~0,005 $/domanda).*

- [x] **3.5 Creare `evaluation/run_retrieval.py`** (citato nel README ma inesistente): solo retrieval sulla ground truth, output compatibile con `evaluate.py --retrieval_only`. Nessun LLM richiesto.

- [x] **3.6 Report comparabili** — salvare nel report `provider`, `model`, `judge_model`, `chunk_size`, `embedding_model`, timestamp; salvare ogni run in `evaluation/runs/<timestamp>_<model>.json`; flag `--compare <report.json>` che stampa i delta.

- [x] **3.7 Baseline ufficiale**
  - Rieseguire pipeline + valutazione (judge attivo) con il provider principale (Claude Haiku 4.5). Opzionale, se il credito lo consente o con una chiave gratuita: un secondo modello/provider per confronto.
  - Facoltativo per risparmiare: usare la Batch API di Anthropic (−50 %) per le run di valutazione, che non richiedono risposte immediate.
  - Salvare come `evaluation/baseline_cloud.json` e riportare le metriche in testa a questo file e in `CLAUDE.md`.
  - Done quando: esiste una baseline di generazione affidabile; tutte le fasi successive si confrontano con questa.
  - *Esito (02/10/2026): baseline in testa a questo file. Indice ricostruito prima della run (299 chunk, invariato).
    Batch API non usata: costo totale già 0,12 $. Nessun secondo provider (nessuna chiave gratuita configurata).*

---

## Fase 4 — Agent come unico entrypoint

- [x] **4.1 Tool `search_etf_docs` più espressivo** — parametri `isins: list[str]`, `mode: single|comparative|cross_doc`, `year`; enum di issuer/ISIN generati dal registro (4.3), non hard-coded.

- [x] **4.2 Memoria conversazionale** — history mantenuta tra domande in modalità interattiva (comando `reset`), troncata a N turni.

- [x] **4.3 Registro fondi unico** — `src/registry.py` che costruisce `KNOWN_ISINS` (nome, issuer, ticker Yahoo) da `metadata.json`; eliminare le copie in `ask.py`, `agent.py`, `live_data.py` (`ISIN_TO_NAME`, `ISIN_TO_TICKER`). Aggiungere il campo `yahoo_ticker` in `metadata.json`.

- [x] **4.4 Deprecare il router a keyword di `ask.py`**
  - `ask.py` diventa un wrapper sottile sopra `Agent` (o viene rimosso), mantenendo i comandi `funds`, `chunks`, `help`, `exit`.
  - In alternativa, per un percorso non agentico: router via LLM con output strutturato (JSON: isins, doc_types, mode, query_type) al posto delle keyword (oggi `"less"` matcha `"unless"` e `cross_doc` non scatta mai perché `doc_type` è sempre impostato).
  - `pipeline.py` deve poter valutare anche l'agent.

- [x] **4.5 Citazioni strutturate** — il modello cita i chunk (`[Chunk 3]`) e il codice li risolve nei metadati reali, invece di fargli scrivere a mano `[ISIN | issuer | doc_type | year]`.
  - Alternativa da valutare con il backend Anthropic: le **citazioni native** (chunk passati come documenti; la risposta contiene i passaggi citati). Va esposta tramite l'interfaccia neutra in modo opzionale (gli altri provider ricadono sul formato `[Chunk N]`).
  - Done quando (fase 4): metriche ≥ baseline 3.7, attribution in miglioramento.
  - *Citazioni native Anthropic non implementate: il formato `[Chunk N]` risolto in `src/citations.py` vale per tutti i provider.*

**Esito fase 4 (05/10/2026)** — run `evaluation/runs/20261005T130107Z_claude-haiku-4-5_agent.json`, report
`evaluation/report_phase4_agent.json` (agent, Haiku 4.5, judge Sonnet 5.5, 17 domande; indice ricostruito, 299 chunk).
L'agent vede solo il testo della domanda (nessun hint dalla ground truth). Costo: pipeline 0,210 $ + judge 0,129 $ = 0,339 $
(≈ 0,012 $/domanda, 3,2 chiamate LLM/domanda; l'agent costa ~7× la pipeline fissa).

| Metrica | Baseline 3.7 | Fase 4 (agent) | Delta |
|---|---|---|---|
| Judge verdict pass | 35.3% | **70.6%** | +35.3 |
| Judge correctness | 46.7% | 83.3% | +36.6 |
| Judge grounded | 82.4% | 76.5% | −5.9 (1 domanda) |
| Judge abstained | 35.3% | 5.9% | −29.4 |
| Attribution | 88.2% | 82.4% | −5.9 (1 domanda) |
| Retrieval precision / recall (documento) | 0.988 / 1.000 | 0.851 / 0.941 | −0.137 / −0.059 |

Lettura: il criterio "metriche ≥ baseline" è rispettato sulla metrica primaria (pass ×2, astensioni da 35% a 6%) ma
non alla lettera su grounded/attribution/retrieval:
- T1_003 ("qual è l'ISIN di …") è risolta dal registro fondi nel system prompt senza cercare: 0 chunk e nessuna
  citazione → da sola vale −0.059 su recall e attribution. Il judge ora riceve il registro come "reference data"
  (`reference_data` nell'output della pipeline) e la considera grounded.
- La precision scende perché l'agent fa più ricerche (fino a 10) e recupera anche documenti non richiesti dalla ground truth.
- Grounded −1 domanda è nell'ordine del rumore del judge; i non-grounded rimasti sono T4_001 (esempio numerico
  inventato, nonostante la regola nel prompt), T4_002, T2_007, T2_008.
- Fallimenti residui = retrieval: "Product Structure: Physical" del factsheet iShares è spezzato dal layout a colonne
  e non viene trovato (T1_006, T2_007) → fase 5 / ingest.

---

## Fase 5 — Qualità del retrieval

- [ ] **5.1 Chunk più grandi** — con modelli cloud portare `CHUNK_SIZES` a ~800–1200 caratteri (factsheet) e ~1000–1500 (KID), overlap ~15%; scegliere in base alle metriche.
- [ ] **5.2 Embedding multilingue** — provare `intfloat/multilingual-e5-small` (prefissi `query:`/`passage:`) o `paraphrase-multilingual-MiniLM-L12-v2`; aggiungere domande in italiano alla ground truth. Salvare il modello di embedding nei metadati della collection (errore se query e indice usano modelli diversi).
- [ ] **5.3 Retrieval ibrido** — BM25 (`rank_bm25`) + vettoriale fusi con Reciprocal Rank Fusion: aiuta su ISIN, sigle (TER, OCF, SRI) e numeri.
- [ ] **5.4 Soglia di score** — scartare chunk sotto una similarità minima e segnalarlo al modello.
- [ ] **5.5 Modalità "full context" di confronto** — l'intero corpus 2026 è ~19k token: modalità che passa i documenti interi al modello, per misurare quanto il RAG aggiunge rispetto al contesto completo (attenzione ai limiti token/min dei free tier). Con Claude usare il **prompt caching** sul corpus: le letture dalla cache costano un decimo dell'input.

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

- [ ] **8.1 Dipendenze** — eliminare `requirements.txt` (sostituito da `pyproject.toml` + `uv.lock`); `anthropic` aggiunto in 2.2; dev-dependency `pytest` e `ruff` (`uv add --dev`); `rank_bm25` quando serve (5.3).
- [ ] **8.2 Rinominare `setup.py`** in `bootstrap.py` (o `scripts/setup_index.py`): con un `pyproject.toml` presente, `setup.py` è il nome riservato al packaging setuptools e può essere eseguito per errore da `pip install .`.
- [ ] **8.3 Packaging `src/`** — trasformare `src/` in pacchetto (`fundscope/` con `__init__.py`), eliminare gli import che dipendono dalla cwd, il `sys.path` hack in `setup.py` e il fallback `RetrievedChunk` duplicato in `generate.py`.
- [ ] **8.4 Logging** — sostituire i `print("[retrieve] …")` con `logging`, livello via `--verbose`; risolve anche il problema `UnicodeEncodeError` su Windows con stdout in pipe.
- [ ] **8.5 Test (`tests/`)** con pytest:
  - unit: `_build_where`, `parse_citations`, `split_text`, `make_chunk_id`, scorer di `evaluate.py`;
  - ingest di un PDF di esempio → numero e metadati dei chunk;
  - retrieval end-to-end su indice temporaneo (senza LLM);
  - agent con client LLM **mockato** (sequenza tool call → risposta): nessuna chiamata di rete nei test;
  - conversione dei formati per **entrambi** i backend (neutro ↔ Anthropic, neutro ↔ OpenAI) con risposte registrate/finte, così il cambio di provider resta verificato anche senza chiavi;
  - calcolo dei costi e interruzione per budget (2.5).
- [ ] **8.6 CI GitHub Actions** — su push: `uv sync`, `ruff`, test, valutazione retrieval-only; fallire se la recall scende sotto soglia. Nessuna chiave API in CI (la valutazione della generazione resta manuale).
- [ ] **8.7 README** — riallineare il resto del README (la sezione "LLM provider" è già in 2.10): comandi (`uv run python src/...`, `src/metadata.json`, campo `isin` nell'esempio), architettura aggiornata, tabella risultati per modello, eventuali residui di Ollama e riferimenti a file inesistenti.

---

## Fase 9 — Interfaccia e pubblicazione (opzionale, per portfolio)

- [ ] **9.1 UI web** — Streamlit o Gradio: chat, fonti/chunk usati, pannello dati live, selettore provider/modello.
- [ ] **9.2 Deploy gratuito** — Hugging Face Spaces o Streamlit Community Cloud, chiave API come secret; indice costruito al primo avvio o committato come artefatto.
- [ ] **9.3 Disclaimer** — "non è consulenza finanziaria" nella UI e nelle risposte che confrontano l'idoneità dei fondi.

---

## Riferimenti rapidi provider (ottobre 2026 — verificare nella console)

| Provider | Accesso | Limiti / costi indicativi |
|---|---|---|
| **Anthropic (default)** | SDK nativo `anthropic` | a consumo: Haiku 4.5 1 $/5 $, Sonnet 5.5 2 $/10 $ per M token (in/out); Batch −50 %, cache read 0,1× |
| Groq | `https://api.groq.com/openai/v1` | ~1.000 req/giorno, ~30 RPM |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta/openai/` | 20–1.500 req/giorno secondo il modello |
| OpenRouter | `https://openrouter.ai/api/v1` | 50 req/giorno sui modelli `:free` |
