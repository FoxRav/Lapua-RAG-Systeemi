# Lapua-RAG

**Paikallinen SOTA-RAG-alusta suomenkielisille asiakirjamassoille.** Pohjalla
PaddleOCR PP-StructureV3 + PaddleOCR-VL, päällä asiakkaan omaan
asiakirjakieleen hienosäädetty Qwen2.5 + LoRA
([`CCG-FAKTUM/lapua-llm-v2`](https://huggingface.co/CCG-FAKTUM/lapua-llm-v2)).

## Ydinkäsite: **Systeemi**

*Systeemi* on se kuratoitu datakokonaisuus, jota järjestelmä käsittelee
ja josta **SLM-tekoäly** (Small Language Model: Qwen2.5-1.5B +
LoRA `lapua-llm-v2`) **ainoana** hakee tietonsa. Kaikki vastaukset
perustellaan Systeemin sisällä olevilla katkelmilla, lähde (§, sivu,
pvm) näkyviin. Jos Systeemi ei sisällä relevanttia tietoa, palvelu
pidättäytyy vastaamasta – malli ei saa vuotaa esikoulutuksen muistia
tuotteen kautta. Tämä **closed-book over Systeemi** -takuu on kovakoodattu
`AnswerService`:een, ei mallin promptiin.

Lapuan kaupungin data toimii ensimmäisenä malliesimerkkinä ja
myyntireferenssinä; arkkitehtuuri on multi-tenant ja sama pino palvelee
tilitoimistoja, juridisia tiimejä ja teollisuuden ERP-dokumentaatiota
omalla Systeemillä + LoRA-adapterilla per asiakas.

---

## 1. Tilanneraportti (v0.2 · 2026-04-17)

### Valmista

- **PaddleOCR-ympäristö pystyssä** (ks. §6). PP-StructureV3 ajettu onnistuneesti
  45-sivuiselle *Talouden toteutumaraportti 31.3.25* -osavuosikatsaukselle,
  keskimääräinen OCR-luottamus > 0.9, GPU NVIDIA RTX 4050 Laptop.
- **Lapua-RAG skeleton-koodi kirjoitettu** – 18 moduulia, ~1400 riviä Python 3.10.
  Kaikki rajapinnat paikallaan, toteutus kevyt muttei placeholderia.
- **Systeemi-alijärjestelmä (`src/lapua_rag/systeemi/`)**: tilastot, versionti
  (deterministinen SHA-256 indeksoitujen dokumenttien yli), kattavuusraportti.
  API-endpointit `/v1/system/stats|version|coverage` ja CLI-aliryhmä
  `lapua-rag system {stats,version,coverage}` – tyhjällä DB:llä smoke-testattu.
- **Closed-book-vahtikoira `AnswerService`:ssa**: LLM:ää ei edes kutsuta jos
  Systeemistä ei löydy riittävän luotettavaa kontekstia. `RagAnswer`
  kantaa `abstained` + `abstain_reason` -kenttiä – hallusinaatiota ei voi
  sattua huolimattomuudesta.
- **34 yksikkötestiä vihreinä** (mojibake-korjaus, pykäläpohjainen chunkkaus,
  doctype-heuristiikka, SHA-256-dedup, RRF-fuusio, suomen stemmer,
  storage-layout, OCR-fallback-sääntö, Systeemin tilastot & versiointi,
  closed-book-guard). `ruff` puhdas, `pytest` 2.7 s.
- **Asennetut komponentit**: `peft`, `sentence-transformers`, `qdrant-client`,
  `sqlmodel`, `structlog`, `watchdog`, `lm-format-enforcer`, `snowballstemmer`,
  `rank-bm25`, `typer[all]` + dev-pakki (`ruff`, `mypy`, `pytest(-asyncio,-cov)`,
  `black`, `isort`).
- **`lapua-rag`-CLI asennettu editable-tilassa**, toimii:
  `init / ingest / ingest-dir / query / serve / system {stats,version,coverage}`.
- **Deploy-konfigurointi** (`deploy/docker-compose.yml`) Qdrantille ja
  Meilisearchille; vLLM-palvelu kommentoitu valmiiksi Linux/WSL2-siirtoa
  varten.

### Osittain (rajapinnat paikallaan, ei vielä end-to-end-ajoa)

- **OCR → postprocess → embed → extract -pipeline** (`src/lapua_rag/pipeline.py`)
  on koodattu, mutta ajamatta kokonaisuutena yhdellekään dokumentille.
- **Qdrant + BM25 + reranker -hybridihaku** – luokat valmiita, kokonaisen
  retrieve-ketjun smoke-test tekemättä.
- **Qwen + LoRA -constrained extract** – `LocalLlmClient` (CPU) ja
  `RemoteVllmClient` (vLLM) olemassa; ensimmäistä ei ole ajettu livenä.

### Ei vielä aloitettu

- Multi-tenant-todennus (tenant-kenttä on rakenteessa, ACL-kerros puuttuu).
- Prometheus-metriikka, auditloki, UI (Streamlit / Next.js).
- LLM-as-judge eval-skripti ja Lapuan kultajoukko-dataset.

### v0.3 · 2026-04-18 – vLLM-synteesi tuotantonopeudella

- **109 dokumenttia, 7 922 chunkia, 1.13 M tokenia** indeksoitu Qdrantiin
  (`scripts/batch_ingest.py`, 108 PDF:ää DATA_päättävät_elimet_20251202 → 11 h 15 min
  OCR+embed, extraction skipattu).
- **vLLM WSL2-distrossa pystyyn** (`lapua-vllm` / F:\wsl\lapua-vllm, Ubuntu 22.04).
  Qwen2.5-1.5B bf16 + lapua-llm-v2 LoRA rinnakkain `--lora-modules lapua-v2=…`,
  OpenAI-endpoint `http://localhost:8000/v1`. Ks. §10.
- **`lapua-rag query` vLLM-backendilla end-to-end:** vastaus ~15 s (LLM osuus
  1–3 s, loput embedding/reranker mallilatauksia CLI-prosessissa; FastAPI
  `_answer_service()` on `lru_cache`d → toisen kyselyn latenssi pelkkä LLM-aika).
- **Closed-book-vahtikoira validoitu elävällä mallilla** – LoRA palauttaa
  "En löydä Systeemistä vastausta" kysymyksille joihin kontekstin
  max_source_score < kynnys.

### v0.5 · 2026-04-18 – Systeemi-frontend (Next.js + shadcn/ui)

Tähän asti Systeemiä on ajettu CLI:stä ja FastAPI:n raakana
JSON-rajapintana — toimivaa, mutta ei sellaisesta kiitellä korpuksen
tutkimisesta. v0.5:ssä saatiin täysi web-UI tuotantolaadulla:

- **`frontend/`** — Next.js 16 + React 19 + Tailwind 4 + shadcn/ui
  (base-nova preset, Base UI primitiivit). Yhden käyttäjän käyttöön
  optimoitu, mutta koodirakenne kestää myös asiakasdemoa.
- **Tyyppiturva backendin → frontendin yli**: `frontend/scripts/gen-types.mjs`
  generoi TypeScript-tyypit `lib/api/openapi.d.ts`:ään suoraan FastAPI:n
  OpenAPI-spekin pohjalta (`openapi-typescript`). Skripti yrittää ensin
  livesessiota (`/openapi.json`), fallbackaa offlineen `tmp/openapi.json`:iin.
  CLI-tuki: `lapua-rag openapi-dump` kirjoittaa specin levylle ilman
  serveriä.
- **Komponenttirakenne** (`frontend/components/`):
  - `chat-panel.tsx` — chat-näkymä, Ctrl+Enter lähettää, optimistinen
    pending-tila skeletoneilla, virhekäsittely toastilla.
  - `answer-card.tsx` — johtopäätös + perustelut markdown-renderöitynä,
    abstain-bannerit per syy (`no_context`, `below_threshold`,
    `model_refused`), reranker-score-mittari.
  - `source-card.tsx` — doc_id (klikkaa → kopioi), sivu, section,
    score-badge, snippet → "Näytä koko chunk" lazy-fetchaa
    `GET /v1/chunks/{chunk_id}`.
  - `pdf-viewer.tsx` — Dialog-modaali, iframe selaimen sisäänrakennettuun
    PDF-katsojaan `#page=N&zoom=page-fit`-ankkurilla.
  - `system-sidebar.tsx` — korpus-stats, kattavuus per doc_type, versio,
    **mode-toggle** (synth ↔ retrieve, persistoidaan LocalStorage:iin).
  - `query-history.tsx` — viimeiset 20 kyselyä LocalStorage:sta, klikkaa
    → injektoi inputtiin.
  - `theme-toggle.tsx` — light / dark / system, `next-themes` SSR-safe.
- **Backend-laajennukset**:
  - `QueryRequest.mode: AnswerMode | None` — per-pyyntö override
    `Settings.answer_mode`:lle. `_answer_service` cachetetaan moodikohtaisesti
    (`lru_cache(maxsize=2)`), joten embedder/reranker eivät uudelleenladata.
  - `GET /v1/documents/{doc_id}/source` — striimaa PDF:n
    `data/storage/<tenant>/YYYY/MM/<doc_id>/source.pdf`-polusta inline-PDF:nä,
    selain renderöi sen sisäänrakennetulla katsojallaan.
  - `GET /v1/chunks/{chunk_id}` — koko chunkin teksti + metadata,
    backend "näytä koko chunk" -ekspanderille.
  - `RagSource.chunk_id` — uusi kenttä jotta UI tietää mitä chunkia
    pyytää koko tekstiksi (oletus optional, taaksepäin yhteensopiva).
- **Tiedonkulku UI ↔ taustapalvelut**: TanStack Query cachettaa
  `system/stats|coverage|version` ja `chunks/{id}` -hauista 30 s, kyselyt
  itse menevät `useMutation`-kautta ilman cachea (jokainen kysely tuore).
- **CLI: `lapua-rag ui`** — käynnistää Next.js dev-serverin
  (`frontend/`) subprocessina; `--install` ajaa `npm install`:n
  ensimmäisellä kerralla.
- **5 uutta backend-yksikkötestiä** (`test_api_documents.py`,
  `test_api_query.py`): PDF-streaming + 404-haarat, chunk-endpointti,
  `mode`-overriden + `Literal`-validoinnin, kaikki yhteensä **46 testiä
  vihreänä**. `next build` + `tsc --noEmit` puhtaita.

Suosittu kysely-flow: open `http://localhost:3000` → kirjoita kysymys →
pieni *retrieve*-badge top-oikealla näyttää aktiivisen tilan → 5 lähde-
korttia, joista jokainen aukeaa sekä koko-chunk-näkymäksi että PDF:n
oikealla sivulla.

### v0.4 · 2026-04-18 – AnswerService kestäväksi: retrieve-mode + diagnoosit

Diagnostinen ajo (`tmp/debug_query.py`) paljasti että `lapua-llm-v2` LoRA on
ylikoulutettu kieltäytymään: malli palautti *"En löydä Systeemistä vastausta"*
vaikka rerankerin top-1 score oli 0.99 ja chunkki sisälsi vastauksen sanasta
sanaan (ks. §11). Pelkkä prompti- tai parametriviritys ei riittänyt — even
few-shot-esimerkki ei muuttanut käytöstä. Korjattu rakenteellisesti:

- **`AnswerService` jaettu kahteen moodiin** (`Settings.answer_mode`):
  - `synth` = retrieve → rerank → vLLM extracts JSON (alkuperäinen polku)
  - `retrieve` = retrieve → rerank → palauta top-N siteeratut katkelmat,
    LLM ohitetaan kokonaan. **Tällä hetkellä default `.env`:ssä**, koska
    tuottaa luotettavasti relevantteja vastauksia ilman synteesimallin
    virhepäätöksiä.
- **AnswerService-parametrit Settings:iin** (`LAPUA_ANSWER_*`): mode,
  min_score, max_context_chunks (3 → 5, vastaa rerankerin top_k_final:ia),
  max_chars_per_chunk (800 → 1500), retrieve_snippet_chars (600).
- **Abstain-recovery**: jos LoRA palauttaa abstain-tekstin mutta unohtaa
  flagin, post-processor flippaa `abstained=true`.
- **Diagnostinen `rag.llm_call`-loki**: kirjaa n_chunks, context_chars,
  top_score → silent truncationit ja chunk-cap-mismatchit nyt näkyvissä.
- **Few-shot esimerkki promptissa** (synth-modea varten, kestävä koodi
  lapua-llm-v3:n tullessa).
- **40 yksikkötestiä vihreinä** (ml. 4 uutta retrieve-mode + abstain-recovery
  -testiä; testit ajavat ilman Qwen-latausta `_RecordingLlm`-mockilla).
- **Smoke-testi onnistunut**: kyselyllä *"Kuka valittiin Jytyn pääluottamus-
  mieheksi Lapualla?"* retrieve-mode palauttaa 5 lähdettä joista top-1
  (score 0.99) sisältää suoran vastauksen *"Samuli Taivalmaan kaudelle
  2025-2028"*.

---

## 2. Miten ohjelma toimii

### 2.1 Tiedon elinkaari (ingest)

```
PDF (inbox)
    │  SHA-256 → doc_id = sha256[:16]      (idempotentti, deduplikoi)
    ▼
data/storage/<tenant>/YYYY/MM/<doc_id>/source.pdf
    │
    ▼   OcrPipeline (PP-StructureV3, GPU)
        pages/NNN.md, NNN.res.json, NNN.png, tables/NNN_MM.html
    │
    ▼   postprocess.consolidate_markdown   (mojibake-korjaus .res.json:sta)
        document.md
    │
    ▼   postprocess.detect_doc_type        (poytakirja / osavuosi / tilinpäätös / …)
    ▼   postprocess.chunk_document         (§ N -rajoilla pöytäkirjoille,
                                            heading-rajoilla muille)
    │
    ├─▶ Embedder (E5/BGE-M3)   → Qdrant.upsert     (dense)
    ├─▶ stem_finnish + FTS5    → BM25Index.upsert  (sparse)
    └─▶ ExtractionPipeline     → structured.json
        (Qwen + lapua-llm-v2, JSON-schema-constrained)
    │
    ▼
  documents.status = INDEXED
```

**Kaikki vaiheet ovat idempotentteja.** Sama PDF samalla sisällöllä on no-op;
deterministiset `chunk_id`:t = vektori-upsert ei duplikoi.

### 2.2 Kysely (query) – closed-book over Systeemi

```
user query
    │  Embedder.embed_query    → Qdrant top-30 (dense)    over Systeemi
    │  stem_finnish + FTS5     → BM25   top-30 (sparse)   over Systeemi
    ▼
  rrf_fuse(k=60)               → fused top-30
    ▼
  Reranker (BGE cross-encoder) → top-5 RetrievalResult
    ▼                                                     ┌── abstain: no_context
  AnswerService guard (min_score-kynnys)  ────────────────┤   abstain: below_threshold
    ▼                                                     └── ei LLM-kutsua
  SLM (Qwen2.5-1.5B + LoRA, JSON-schema-constrained)
    ▼
  RagAnswer { johtopaatos, perustelut, lahteet[],
              abstained, abstain_reason, max_source_score }
```

Vastausformaatti on **sama kuin LoRA:an on opetettu**: *Johtopäätös →
Perustelut → Lähteet (§, kokous, pvm)*. `lm-format-enforcer` takaa että
ulostulo parsetuu joka kerta suoraan Pydanticiksi ilman regex-korjauksia.

**Vahtikoira on deterministinen:** `AnswerService.answer` palauttaa
abstention-vastauksen kutsumatta LLM:ää lainkaan, kun Systeemi ei anna
relevanttia kontekstia (`no_context`) tai rerankerin paras pisteytys jää
konfiguroidun kynnyksen alle (`below_threshold`). Kolmas syy
`model_refused` = LLM itse ilmoitti JSON-outputissa ettei kontekstissa ollut
riittävää tietoa. Kolmannen syyn malli on opetettu käyttämään LoRA:n kautta.

### 2.3 Moduulikartta

```
src/lapua_rag/
├── config.py              # pydantic-settings (LAPUA_*)
├── models/                # Pydantic-domainmallit + StrEnum-tyypit
├── storage/layout.py      # deterministiset polut per dokumentti
├── observability/         # structlog JSON + correlation_id = doc_id
├── db/                    # SQLModel schema + session_scope
├── ingest/                # SHA-256 dedup, watchdog inbox, SQLite queue
├── ocr/                   # PP-StructureV3 wrapper + VL fallback -sääntö
├── postprocess/           # mojibake-fix, doctype-heuristic, § chunking, tables
├── extract/               # Qwen+LoRA: LocalLlmClient & RemoteVllmClient (vLLM)
├── embed/                 # sentence-transformers, E5/BGE-prefiksit
├── index/                 # Qdrant, SQLite FTS5 + suomen stemmer, RRF-fuusio
├── rerank/                # BGE reranker v2 m3 cross-encoder
├── retrieve/              # hybrid → rerank -ketju
├── rag/                   # RagAnswer-synteesi + closed-book-vahtikoira
├── systeemi/              # stats, version (SHA-256), coverage-raportit
├── api/                   # FastAPI: /ingest, /query, /documents, /system/*
├── mcp/                   # FastMCP-työkalut Cursorille / Claude Desktopille
├── pipeline.py            # end-to-end orkestrointi (idempotentti)
└── cli.py                 # Typer CLI (ml. `system`-aliryhmä)
```

Funktionaalinen ydin (`postprocess`, `models`, `rag`, `index.hybrid`) on puhdasta
koodia ilman I/O-sivuvaikutuksia. Raskaat ulkoiset clientit
(Paddle, Qwen, Qdrant, SQLModel, tiedostojärjestelmä) ovat omissa moduuleissaan.

Arkkitehtuuridetalji: [`docs/architecture.md`](docs/architecture.md)
Tuotenäkökulma: [`docs/sales_overview.md`](docs/sales_overview.md)
Operointi: [`docs/operations.md`](docs/operations.md)
Täysi suunnitelma + roadmap: [`tmp/rag_system_design.md`](tmp/rag_system_design.md)

---

## 3. Pikakäynnistys

```powershell
cd F:\-DEV-\76.PaddleOCR
.\.venv\Scripts\Activate.ps1

# 1. Käynnistä Qdrant (Docker Desktop vaatii)
docker compose -f deploy\docker-compose.yml up -d qdrant

# 2. Kopio env-pohja
copy .env.example .env

# 3. Luo DB-skeema ja Qdrant-kokoelma
lapua-rag init

# 4. Ingestoi yksi PDF
lapua-rag ingest "F:\...\Talouden toteutumaraportti.pdf"

# 5. Kysele
lapua-rag query "Mitä Q1 osavuosikatsaus kertoo investoinneista?"

# 6. Tarkista Systeemin tila
lapua-rag system stats       # dokumentti-/chunk-/token-määrät per tenant
lapua-rag system version     # Systeemin deterministinen sisältöhash
lapua-rag system coverage    # indeksoitu / kesken / epäonnistunut per doc_type

# 7. Nosta HTTP-API
lapua-rag serve    # → http://127.0.0.1:8080/docs

# 8. Nosta web-UI (uudessa terminaalissa, vaatii Node 20+)
lapua-rag ui --install   # ekan kerran (asentaa frontend/node_modules:n)
lapua-rag ui             # jatkossa  → http://localhost:3000
```

> Vaihtoehtoisesti `cd frontend && npm install && npm run dev`.

### HTTP-API (FastAPI)

| Endpoint               | Metodi | Tehtävä                                              |
|------------------------|--------|------------------------------------------------------|
| `/v1/ingest`           | POST   | Lataa PDF → ajaa koko pipelinen                      |
| `/v1/query`            | POST   | `{query, tenant?}` → `RagAnswer` (closed-book)       |
| `/v1/documents`        | GET    | Listaa indeksoidut dokumentit (suodattimet)          |
| `/v1/documents/{id}`   | GET    | Yhden dokumentin metadata                            |
| `/v1/system/stats`     | GET    | Systeemin määrät per tenant                          |
| `/v1/system/version`   | GET    | Deterministinen Systeemi-hash (cache-avain)          |
| `/v1/system/coverage`  | GET    | Kattavuus per `doc_type` + jumissa olevat ingest-työt |
| `/healthz`             | GET    | Elossa-tarkistus                                     |

### Testit

```powershell
pytest tests/unit -q --no-cov     # 34 testiä, ~2.7 s
ruff check src tests              # lint
mypy src/lapua_rag                # strict type-check
```

---

## 4. Konfigurointi (.env)

Kaikki asetukset `LAPUA_`-prefiksillä, täydellinen pohja `.env.example`:ssä.
Tärkeimmät:

| Muuttuja                         | Oletus                               | Tehtävä                          |
|----------------------------------|--------------------------------------|----------------------------------|
| `LAPUA_TENANT`                   | `lapua`                              | Asiakkaan tunniste datarivillä   |
| `LAPUA_STORAGE_ROOT`             | `data/storage`                       | PDF:t + per-doc-artefaktit       |
| `LAPUA_QDRANT_URL`               | `http://localhost:6333`              | Dense-vector-DB                  |
| `LAPUA_EMBEDDING_MODEL`          | `intfloat/multilingual-e5-large`     | Vaihda → `BAAI/bge-m3` halutessasi |
| `LAPUA_LLM_BASE`                 | `Qwen/Qwen2.5-1.5B-Instruct`         | Pohjamalli                       |
| `LAPUA_LLM_LORA`                 | `CCG-FAKTUM/lapua-llm-v2`            | **Per-asiakas LoRA-adapteri**    |
| `LAPUA_LLM_DEVICE`               | `cpu`                                | Windows-MVP; Linuxissa `cuda`    |
| `LAPUA_LLM_VLLM_URL`             | tyhjä                                | Aseta → transparent remote vLLM  |
| `LAPUA_LLM_VLLM_MODEL`           | `lapua-v2`                           | LoRA-moduulin nimi (vastaa `--lora-modules <nimi>=<path>`) |
| `LAPUA_OCR_DEVICE`               | `gpu:0`                              | Paddle-laitevalinta              |
| `LAPUA_ANSWER_MODE`              | `synth`                              | `synth`=LLM-synteesi, `retrieve`=top-N siteeratut katkelmat (suositus kunnes lapua-llm-v3 valmis) |
| `LAPUA_ANSWER_MIN_SCORE`         | `0.0`                                | Rerank-kynnys closed-book-guardille |
| `LAPUA_ANSWER_MAX_CONTEXT_CHUNKS`| `5`                                  | LLM:lle/käyttäjälle näkyvien chunkien määrä |
| `LAPUA_ANSWER_MAX_CHARS_PER_CHUNK`| `1500`                              | Per-chunk merkkibudjetti synth-modessa |
| `LAPUA_ANSWER_RETRIEVE_SNIPPET_CHARS`| `600`                            | Per-source snippet retrieve-modessa |

Multi-tenant: uusi asiakas = uusi `tenant`-arvo + **oma Systeemi**
(eristetty `storage/<tenant>/` + per-tenant Qdrant-filtteri + per-tenant
BM25-filtteri) + oma LoRA-adapteri (74 MB / asiakas). vLLM tukee
`--lora-modules` -moodia, jossa useita adaptereita ajetaan rinnakkain
samalla pohjamallilla.

---

## 5. Seuraavat konkreettiset tehtävät

### Viikko 1 – **Systeemin ensimmäinen dokumentti + SLM live**

1. **Ajoa koko pipeline läpi** jo prosessoidulle *Talouden
   toteutumaraportti*-PDF:lle: `lapua-rag ingest "<pdf>"`, jonka jälkeen
   `lapua-rag system stats` näyttää kyseisen dokumentin Systeemissä.
   Odotettavia ongelmia: Qwen-mallin alkulataus (~3 GB disk), CPU-inferenssin
   hitaus (10–30 s / chunk), `lm-format-enforcer`-tokenizer-yhteensopivuus.
2. **Ajaa SLM livenä** `lapua-rag query` -kautta: varmista että
   `lapua-llm-v2` lataa Qwen2.5:n päälle, `RagAnswer` parsetuu ja
   closed-book-guard palauttaa abstain-vastauksen kysymykselle jota
   Systeemistä ei löydy.
3. **Kirjoittaa smoke-testi `tests/integration/test_pipeline_e2e.py`** joka
   ajaa koko ketjun pienellä kultadokumentilla ja varmistaa että
   `structured.json` sisältää odotetut pykälät + Systeemin tila muuttuu
   (uusi versiohash, +1 indexed_count).
4. **Verifioida mojibake-korjaus** ajamalla `consolidate_markdown` tuolle
   PDF:lle ja diff-vertaamalla `rec_texts`-lähtöiseen tekstiin.
5. **Luoda kultajoukko** 3–5 dokumentista eri tyypeistä (pöytäkirja,
   osavuosi, tilinpäätös) → `tests/fixtures/gold/`.

### Viikko 2 – **laadun mittarointi**

5. **Ajaa 10 todellista Lapua-PDF:ää** eräajona
   (`lapua-rag ingest-dir data/inbox`).
   Mittari: OCR-luottamus per dokumentti, ekstraktion pykäläkattavuus,
   hybridihaun recall@5.
6. **Virittää chunkkaussäännöt tilinpäätökselle** (taulukkorakenteinen, ei
   pykäliä) – lisätä `postprocess/chunking.py`:hin taulukko-tietoiset
   sliding-windowit.
7. **Kirjoittaa `scripts/eval_judge.py`** – LLM-as-judge automaattiarviointi
   omalla LoRA:lla + referenssinä GPT-4.

### Viikko 3 – **tuotantoketju**

8. **Pystyttää vLLM Linux/WSL2-koneeseen** `--enable-lora`-moodissa;
   aseta `LAPUA_LLM_VLLM_URL=http://wsl:8000/v1` ja mittaa ekstraktion
   nopeuskasvu (tavoite: 0.2–1 s / chunk vs. 10–30 s CPU:lla).
9. **Indeksoida 100 dokumenttia** eräajona; ensimmäinen realistinen
   volyymimittaus.
10. **Prometheus-metriikka** (`/metrics`-endpoint): `ingest_total`,
    `ingest_duration_seconds`, `retrieve_hit_rate_at_5`,
    `llm_generate_duration_seconds`.

### Viikko 4 – **demo + myynti**

11. **Streamlit- tai Next.js-UI**: kysymyspalkki, tulosluettelo sitaateilla,
    klikattavat bbox-highlightit sivu-PNG:iin.
12. **Asiakasdemo-skenaariot**: "Näytä kaikki kaupunginhallituksen päätökset
    2024 joita koskevat ympäristönsuojeluluvat", "Mitkä investoinnit ylittyivät
    budjetissa Q1 aikana?".
13. **Air-gap-paketointi**: `docker compose` -stack asiakkaiden koneisiin
    ilman ulkoista verkkoa.

### Hyväksymiskriteerit v1.0

- [ ] 200 Lapua-PDF:ää **Systeemissä** (indeksoitu, haettavissa, rakenteellisena JSON:na)
- [ ] Jokaisella vastauksella ≥ 1 lähde (§ + sivu) **Systeemistä**
- [ ] Closed-book-guard: 0 hallusinaatiota kultajoukon "off-topic"-kysymyksille
  (SLM abstaineeraa tai palvelu abstaineeraa ennen SLM-kutsua)
- [ ] LLM-as-judge > 80 % vastauksista "oikeellinen"
- [ ] OCR-luottamus > 0.9 keskimäärin; alle 0.6 päätynyt VL-fallbackiin
- [ ] `data/storage/` ainoa source-of-truth; indeksit uudelleenrakennettavissa < 1 h
- [ ] Multi-tenant: sama stack ≥ 2 Systeemiä (asiakasta) ilman koodimuutoksia
- [ ] `lapua-rag system version` stabiili: sama Systeemi ⇒ sama hash
- [ ] CI: ruff + mypy --strict + pytest vihreänä
- [ ] Auditloki: kuka kysyi mitä, milloin, mihin dokumentteihin päätyi

---

## 6. Alla oleva PaddleOCR-ympäristö

Kaikki mitä Lapua-RAG tekee OCR-rintamalla perustuu alla kuvattuun täysin
paikalliseen PaddleOCR-asennukseen. Jos jokin toiminto ei toimi, aloita
diagnoosi tältä tasolta.

### 6.1 Layout

```text
F:\-DEV-\76.PaddleOCR\
├── .venv\                           # Python 3.10 venv (do not commit)
│   └── Lib\site-packages\
│       ├── _paddleocr_torch_dll_fix.pth  # loads the preload hook at startup
│       └── _paddleocr_preload.py         # pre-loads torch DLLs (see below)
├── PaddleOCR\                       # Upstream clone (main)
│   ├── paddleocr\                   # editable `paddleocr` package
│   ├── langchain-paddleocr\         # editable `langchain_paddleocr`
│   └── mcp_server\                  # editable `paddleocr_mcp`
├── src\lapua_rag\                   # Lapua-RAG product (this repo's own code)
├── tests\                           # unit + integration tests
├── scripts\                         # utility scripts (cleanup, eval, …)
├── docs\                            # architecture, operations, sales
├── deploy\docker-compose.yml        # Qdrant + (optional) Meilisearch + vLLM
├── data\                            # runtime: inbox / storage / index (gitignored)
├── tmp\
│   ├── rag_system_design.md         # full SOTA product plan
│   ├── verify_env.py                # full stack verification
│   ├── diag_torch.py                # torch DLL diagnostics
│   └── extras-requirements.txt      # optional extras installed on top of [all]
├── pyproject.toml                   # lapua-rag package + tooling config
├── .env.example
├── .gitignore
└── README.md                        # this file
```

### 6.2 Installed stack

| Component                            | Version       | Notes                                                   |
|--------------------------------------|---------------|---------------------------------------------------------|
| Python                               | 3.10.11       | global interpreter consumed by `.venv`                  |
| `paddlepaddle-gpu`                   | 3.3.1         | CUDA 11.8 + cuDNN bundled (`run_check()` passes on GPU) |
| `paddleocr` (editable)               | 3.4.1 dev     | from `PaddleOCR\` with `[all]` extras                   |
| `paddlex`                            | 3.4.3         | `[ocr, genai-client, ie, trans, serving]`               |
| `paddleformers`                      | 1.1.1         | VL / LLM inference support                              |
| `langchain_paddleocr` (editable)     | 0.1.1         | from `PaddleOCR\langchain-paddleocr\`                   |
| `paddleocr_mcp` (editable)           | 0.5.0         | from `PaddleOCR\mcp_server\` with `[local]`             |
| `mcp` / `fastmcp`                    | 1.27 / 3.2.4  | Model Context Protocol server stack                     |
| `onnx` / `onnxruntime-gpu`           | 1.17 / 1.23.2 | GPU inference via ONNX Runtime                          |
| `fastapi` / `uvicorn[standard]`      | 0.136 / 0.44  | `paddlex[serving]` HPS gateway + Lapua-RAG API          |
| `transformers` / `tokenizers`        | 4.57 / 0.22   | HF stack for VL / LLM                                   |
| `torch` / `torchvision`              | 2.5.1+cpu     | required by `albumentations`, `accelerate`, Qwen, embeddings |
| `peft`                               | 0.19          | LoRA adapter loading for Qwen                           |
| `sentence-transformers`              | 5.4           | E5 / BGE-M3 embeddings + BGE reranker                   |
| `qdrant-client`                      | 1.17          | Vector DB client                                        |
| `sqlmodel`                           | 0.0.38        | Metadata DB (SQLite / PostgreSQL)                       |
| `lm-format-enforcer`                 | 0.11          | JSON-schema-constrained LLM decoding                    |
| `snowballstemmer`                    | 3.0           | Finnish BM25 stemming                                   |
| `structlog` / `watchdog`             | 25 / 6        | JSON logs + inbox file-watch                            |
| `typer[all]`                         | 0.24          | CLI                                                     |
| `pytest` / `ruff` / `mypy` / `black` | latest        | dev tooling                                             |
| GPU                                  |               | NVIDIA RTX 4050 Laptop, 6 GB VRAM (SM 8.9)              |

`paddle.utils.run_check()` passes: **"PaddlePaddle works well on 1 GPU."**
`python tmp\verify_env.py` must report **every** component as `OK` except
`paddle2onnx` (see *Known issues*).

### 6.3 Aktivointi

PowerShell:

```powershell
cd F:\-DEV-\76.PaddleOCR
.\.venv\Scripts\Activate.ps1
```

Cmd:

```bat
cd /d F:\-DEV-\76.PaddleOCR
.venv\Scripts\activate.bat
```

Deaktivointi: `deactivate`.

### 6.4 Pelkkä PaddleOCR CLI (ilman Lapua-RAG-kerrosta)

Jos haluat suoran PaddleOCR-ajon ohi Lapua-RAG-pipelinen, esim. testauksen
tai debug-tarkoitukseen:

| Tehtävä                                  | Komento                                                                |
|------------------------------------------|------------------------------------------------------------------------|
| PP-OCRv5 – tekstin tunnistus             | `paddleocr ocr --input img.png`                                        |
| PP-StructureV3 – PDF → Markdown+JSON     | `paddleocr pp_structurev3 --input doc.pdf --save_path output\`         |
| PaddleOCR-VL – vaikeat skannaukset       | `paddleocr doc_parser --input doc.pdf --save_path output\`             |
| PP-DocTranslation                        | `paddleocr pp_doctranslation --input doc.pdf --save_path output\`      |
| PP-ChatOCRv4 – KIE                       | `paddleocr pp_chatocrv4_doc --input doc.pdf --save_path output\`       |
| Täysi alikomentolista                    | `paddleocr --help`                                                     |

Python-API suoraan:

```python
from paddleocr import PaddleOCR

ocr = PaddleOCR(lang="en", use_gpu=True)
result = ocr.predict("your_document.pdf")
for page in result:
    page.print()
    page.save_to_markdown("output/")
    page.save_to_json("output/")
```

**Huom.** Lapua-RAG ei käytä tätä reittiä vaan `PPStructureV3`-luokkaa
kääreenä (`src/lapua_rag/ocr/pipeline.py`) koska malli on tarpeen ladata
**kerran per prosessi**, ei per ajo.

---

## 7. Torch + Paddle DLL coexistence fix

Both `paddlepaddle-gpu` and `torch` ship their own copy of `libiomp5md.dll`
(Intel OpenMP) plus MSVC/cuDNN shims under `.../lib/`. The library that
loads *first* wins the process-wide DLL slots; the one that loads *later*
crashes with `OSError: [WinError 127] … shm.dll`.

The venv bundles a pre-load hook so this ordering is handled transparently:

- `.venv\Lib\site-packages\_paddleocr_torch_dll_fix.pth` – picked up by
  CPython at interpreter start-up and executes `import _paddleocr_preload`.
- `.venv\Lib\site-packages\_paddleocr_preload.py` – registers `torch/lib`
  via `os.add_dll_directory`, keeps the cookie alive on `builtins`, prepends
  the directory to `PATH`, and pre-loads `torch_python.dll` with
  `ctypes.WinDLL` so torch's OpenMP wins the slots before PaddlePaddle is
  even touched.

Without this hook `import albumentations`, `import accelerate`,
`import timm`, and every `paddleocr` pipeline that reaches for
`albumentations` fail when Paddle has been imported first. **Keep both
files in `site-packages`** if you duplicate this venv elsewhere.

---

## 8. Known issues

1. **`import paddle2onnx` fails with `DLL load failed`** on this Windows +
   Paddle 3.3.1 combination. Converting models to ONNX still works via the
   upstream CLI entry point; runtime inference via `onnxruntime-gpu` works
   normally. Install the VC++ 2015-2022 redistributable and retry
   `pip install --force-reinstall paddle2onnx` if needed.
2. **HPI / GenAI server wheels are Linux-only today.** `ultra-infer-*` and
   `fastdeploy-gpu`'s `tool_helpers` have no Windows distributions. Use the
   pure Paddle / ONNX Runtime inference paths on Windows, or move serving
   to Linux / WSL2 when you need TensorRT-level throughput. **Lapua-RAG's
   vLLM-backed extraction requires exactly this Linux step** – see
   `docs/operations.md` § *Production topology*.
3. **`paddlenlp==2.5.2`** pin from `ppstructure/kie/requirements.txt` is
   incompatible with modern PaddlePaddle and is intentionally **not**
   installed. Needed only for *training* KIE models; inference works
   without it.
4. **6 GB VRAM is shared, not concurrent.** PP-StructureV3 / PaddleOCR-VL
   consume ~4 GB during ingest. vLLM serving Qwen2.5-1.5B bf16 + LoRA in
   the `lapua-vllm` WSL2 distro reserves ~4.5 GB (2.89 GB weights + 0.3 GB
   KV cache @ 4096 tokens + 1.4 GB peak activations + 0.21 GB non-torch).
   **You cannot run ingest (OCR) and query (vLLM) at the same time** – the
   second allocator will OOM. Run them sequentially, or stop vLLM before
   ingesting by killing the `lapua-vllm` distro’s `vllm serve` process.
5. **`torch==2.5.1+cpu`** is the pinned build. torch 2.11 ships a broken
   `shm.dll` on this box even with the pre-load hook; do not upgrade
   without re-verifying.

---

## 9. Reproduce from scratch

```powershell
# 1. clone upstream (shallow)
git clone --depth 1 https://github.com/PaddlePaddle/PaddleOCR.git PaddleOCR

# 2. venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel

# 3. GPU Paddle (CUDA 11.8)
python -m pip install --upgrade paddlepaddle-gpu `
    -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# 4. PaddleOCR editable with all extras
python -m pip install -e .\PaddleOCR[all]

# 5. extra optional components (serving, ONNX, doc parser, KIE infra)
python -m pip install -r tmp\extras-requirements.txt

# 6. paddleformers + transformers stack for VL local inference
python -m pip install paddleformers transformers accelerate timm einops pynvml

# 7. stable torch CPU wheel (see DLL coexistence fix above)
python -m pip install --upgrade --force-reinstall `
    torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu

# 8. repo subpackages (editable)
python -m pip install -e .\PaddleOCR\langchain-paddleocr
python -m pip install -e .\PaddleOCR\mcp_server[local]

# 9. torch DLL preload hook – copy both files into the venv site-packages:
#      _paddleocr_torch_dll_fix.pth   (single line: `import _paddleocr_preload`)
#      _paddleocr_preload.py          (see section above)

# 10. Lapua-RAG product dependencies + editable install
python -m pip install peft sentence-transformers qdrant-client sqlmodel `
    structlog watchdog lm-format-enforcer snowballstemmer rank-bm25 `
    typer[all] pytest pytest-asyncio pytest-cov ruff black isort mypy types-requests
python -m pip install -e . --no-deps

# 11. verify
python tmp\verify_env.py
pytest tests/unit -q --no-cov
ruff check src tests
lapua-rag --help
```

---

## 10. vLLM WSL2 -pystytys (production answer path)

Tämä reitti on tarkoitettu yhden kysyjän työasemalle, jossa Windows-puolella on
Lapua-RAG + PaddleOCR ja **CUDA on saatavilla ainoastaan WSL2:n läpi**. vLLM ei
toimi natiivisti Windowsissa, joten Qwen-synteesi siirretään erilliseen
WSL2-distroon, joka puhuu OpenAI-yhteensopivaa REST-API:a `localhost:8000`:sta.
Windows-Lapua-RAG näkee sen `LAPUA_LLM_VLLM_URL`:n kautta.

**Esivaatimukset:** WSL 2.4+ (testattu 2.5.9), NVIDIA-ajuri ≥ 555 Windowsissa
(testattu 581.04), ~15 GB vapaata levyä kohdeasemalla (malli + venv), GPU
jossa ≥ 5 GB vapaata VRAM:ia kun OCR ei aja.

### 10.1 Ubuntu-distron asennus valitulle levyasemalle

```powershell
# F-asema koska C: oli ahdas; vaihda sijainti tarpeen mukaan
New-Item -ItemType Directory -Force -Path F:\wsl\lapua-vllm
wsl --install Ubuntu-22.04 --location F:\wsl\lapua-vllm --name lapua-vllm --no-launch
wsl -d lapua-vllm -- bash -c "nvidia-smi --query-gpu=name,memory.free --format=csv,noheader"
# pitäisi näkyä "NVIDIA GeForce RTX 4050 Laptop GPU, 5036 MiB"
```

### 10.2 Python + vLLM

```bash
# sisällä lapua-vllm-distrossa, root-tilillä
apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-venv python3.10-dev build-essential git curl ca-certificates
python3.10 -m venv /root/vllm-venv
/root/vllm-venv/bin/pip install --upgrade pip wheel
/root/vllm-venv/bin/pip install 'vllm==0.6.6.post1'
# vllm 0.6.6 odottaa transformers 4.x:ää; uusin 5.x rikkoo tokenizerin
/root/vllm-venv/bin/pip install 'transformers<5.0,>=4.45'
```

**Kolme varoitusta:**

- Triton (vllm:n riippuvuus) kääntää runtime-kernelit gcc:llä → tarvitsee
  `python3.10-dev`-paketin (Python.h). Ilman sitä `profile_run` kaatuu.
- `transformers>=5` poistaa `Qwen2Tokenizer.all_special_tokens_extended`:n
  → vllm 0.6.6 virheilmoittaa tokenizerin alustuksessa.
- WSL2:n ZMQ-stack ei tue `zmq_poll(timeout=-1)`:tä → käytä
  `--disable-frontend-multiprocessing` (ajaa enginen samassa prosessissa).

### 10.3 Mallien esilataus HF-cacheen

```bash
# prefetch ennen ensimmäistä startia jotta käynnistys ei odota latausta
python /mnt/f/-DEV-/76.PaddleOCR/tmp/hf_prefetch.py
# → /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/...
# → /root/.cache/huggingface/hub/models--CCG-FAKTUM--lapua-llm-v2/...
```

### 10.4 vllm serve -käynnistys

Skripti `tmp/start_vllm.sh` (committed) vie nämä kaikki oikein:

```bash
/root/vllm-venv/bin/vllm serve Qwen/Qwen2.5-1.5B-Instruct \
    --enable-lora --max-loras 1 --max-lora-rank 16 \
    --lora-modules "lapua-v2=/root/.cache/huggingface/hub/models--CCG-FAKTUM--lapua-llm-v2/snapshots/<sha>" \
    --max-model-len 4096 --gpu-memory-utilization 0.80 --dtype bfloat16 \
    --enforce-eager --disable-frontend-multiprocessing \
    --guided-decoding-backend outlines \
    --host 0.0.0.0 --port 8000 --disable-log-requests
```

Keskeiset valinnat pienelle 6 GB VRAM:ille:

- `--enforce-eager` ohittaa CUDA-graph-cachen (säästää ~0.5 GB).
- `--max-model-len 4096` pitää KV-cachen pienenä (~0.3 GB); AnswerService
  lähettää alle 1 k tokenia per kutsu.
- `--max-loras 1 --max-lora-rank 16` vastaa `lapua-llm-v2`:n `adapter_config.json`:ia.
- `--guided-decoding-backend outlines` – xgrammar 0.1.33 rikkoo `TokenizerInfo`-API:n.

### 10.5 Windowsista käyttö

```powershell
# .env
LAPUA_LLM_VLLM_URL=http://localhost:8000/v1
LAPUA_LLM_VLLM_MODEL=lapua-v2
```

WSL 2.4+ välittää `localhost:8000` Windowsin puolelta automaattisesti distroon
(ei tarvita `wsl hostname -I`:tä eikä Windows Defender -sääntöjä).
Testi:

```powershell
curl http://localhost:8000/v1/models                 # listaa Qwen + lapua-v2
lapua-rag query "Mitä päätettiin § 123 kokouksessa?" # end-to-end
```

### 10.6 GPU-konflikti ingestin kanssa

OCR-pipeline ja vLLM molemmat varaavat ~4 GB VRAM:ia. 6 GB kortilla ne eivät
mahdu yhtäaikaa. Käytännön työjärjestys:

1. `wsl -d lapua-vllm -- pkill -f "vllm serve"` – vapauta GPU
2. `lapua-rag ingest-dir <dir>` tai `python scripts/batch_ingest.py ...`
3. `wsl -d lapua-vllm -- bash /root/start_vllm.sh` – käynnistä vLLM takaisin
4. `lapua-rag query ...` / `lapua-rag serve` (FastAPI)

---

## 11. Avoin tiketti: lapua-llm-v3 retrain (ylikoulutettu abstain)

### Juurisyy

`lapua-llm-v2` LoRA-painot palauttavat **deterministisesti**
*"En löydä Systeemistä vastausta tähän kysymykseen"* myös silloin kun
retrieval-kontekstissa on vastauksen exact-match. Diagnoosi (`tmp/debug_query.py`,
2026-04-18):

| Mittari | Havainto |
|---|---|
| Reranker top-1 score | 0.99 (BGE reranker-v2-m3) |
| Top-1 chunk sisältö | *"Jytyn vaalitoimikunta on päättänyt valita Lapuan kaupungin uudeksi pääluottamusmieheksi Samuli Taivalmaan kaudelle 2025-2028."* |
| lapua-v2 vastaus    | Abstain-template (model_refused) |
| Base Qwen vastaus   | Rikkinäinen JSON (`"johtopaatos": "johtopaatos"` literal placeholder) |
| Few-shot kokeilu    | Ei vaikutusta lapua-v2:n käytökseen |

→ Ongelma on **mallin painoissa**, ei promptissa eikä parametreissa.
Koulutusdata oli ilmeisesti voimakkaasti vinoutunut kieltäytymisesimerkkeihin.

### Korjaus (lapua-llm-v3)

1. **Tasapainota koulutusdata 50/50** abstain vs. extract -esimerkkien välillä.
   Nykyinen suhde on tuntematon mutta ilmeisesti > 80/20 abstain-painotteinen.
2. **Lisää positiivisia extract-esimerkkejä** suomenkielisistä päätös-
   teksteistä: kuka valittiin / mitä päätettiin / milloin astuu voimaan -tyyppiä.
   Tavoite: ≥ 200 (chunk → JSON) -paria joissa vastaus on chunkissa.
3. **Säilytä JSON-skema-koulutus**: nykyinen v2 outputtaa syntaktisesti oikeaa
   JSON:ia — tämä on ainoa asia jonka v2 osaa paremmin kuin base.
4. **Validointi koulutuksen jälkeen**: 10 kontrollikyselyä joista 5:llä on
   suora vastaus indeksissä, 5:llä ei. Hyväksymiskriteeri: ≥ 4/5 oikeaa
   extract:ia, ≥ 4/5 oikeaa abstainia.
5. **Deploy**: julkaise `CCG-FAKTUM/lapua-llm-v3`, päivitä
   `LAPUA_LLM_LORA=CCG-FAKTUM/lapua-llm-v3` ja vaihda `LAPUA_ANSWER_MODE=synth`.

### Sillaikaa

`LAPUA_ANSWER_MODE=retrieve` (tämän commitin default `.env`:ssä) tuottaa
luotettavasti relevantteja vastauksia — käyttäjä lukee top-N siteeratut
katkelmat suoraan. Closed-book-guarantee säilyy, hallusinaatioriski = 0,
relevanttiusarvio reranker-scoreissa näkyvissä.

### Aikataulu

1–2 päivää: datan revisio + LoRA-koulutus (RTX 4050, 1.5B base, r=16, ~2 h
per epoch × 3-5 epochia), validointi, julkaisu.

---

## 12. Frontend (Systeemi-UI, `frontend/`)

### 12.1 Stack

| Komponentti       | Versio | Miksi                                                              |
|-------------------|--------|--------------------------------------------------------------------|
| Next.js (App)     | 16     | Stabiili App Router + Turbopack dev. SSR ei pakollinen, mutta saadaan ilmaiseksi. |
| React             | 19     | Vakaa peer Next 16:lle.                                            |
| Tailwind CSS      | 4      | CSS-muuttujat = vaivaton dark mode.                                |
| shadcn/ui         | base-nova preset | Base UI -primitiivit (`@base-ui/react`) — tyylittelee shadcn:n logiikkaa, ei lukitse Radixiin. |
| TanStack Query    | 5      | Stats/version/coverage cache; mutaatiot kysely-flowiin.           |
| react-markdown    | latest | Johtopäätös + perustelut markdown-renderöitynä (GFM-taulukot).    |
| openapi-typescript | 7     | Generoi TypeScript-tyypit FastAPI:n OpenAPI-spekistä.              |
| next-themes       | latest | Light/dark/system, SSR-safe (`suppressHydrationWarning`).          |
| sonner            | latest | Toastit (verkko/abstain).                                          |

### 12.2 Dev-flow

```powershell
# Terminal 1 – backend
cd F:\-DEV-\76.PaddleOCR
.\.venv\Scripts\Activate.ps1
lapua-rag serve                # → http://localhost:8080

# Terminal 2 – WSL2 vLLM (vain jos LAPUA_ANSWER_MODE=synth)
wsl -d lapua-vllm -- bash /root/start_vllm.sh    # → http://localhost:8000

# Terminal 3 – frontend
lapua-rag ui --install         # ekan kerran
lapua-rag ui                   # → http://localhost:3000
```

### 12.3 OpenAPI → TypeScript

Tyypit ovat **autogeneroituja** — backendin `RagAnswer` ja kaverit
muotoutuvat suoraan `frontend/lib/api/openapi.d.ts`:ksi. Kun muutat backendin
Pydantic-skeemoja, päivitä tyypit:

```powershell
# vaihtoehto A — backend pyörii
cd frontend
npm run gen-types

# vaihtoehto B — backend ei pyöri (CI / offline)
lapua-rag openapi-dump          # tmp/openapi.json
cd frontend
npm run gen-types               # fallbackaa offlineen
```

Generoitu `openapi.d.ts` on **gitignoroitu** koska se on aina
johdannainen FastAPI-skeemasta — git-historian saastuttaminen ei ole
arvonsa väärti.

### 12.4 Komponenttikartta (`frontend/`)

```
frontend/
├── app/
│   ├── layout.tsx          # html lang=fi, suppressHydrationWarning, Providers
│   ├── page.tsx            # main shell (chat | sidebar grid)
│   └── globals.css         # Tailwind 4 + shadcn theme tokens
├── components/
│   ├── providers.tsx       # ThemeProvider + QueryClientProvider + TooltipProvider + Toaster
│   ├── header.tsx          # logo + theme-toggle
│   ├── theme-toggle.tsx    # light/dark/system kierto
│   ├── chat-panel.tsx      # textarea + viestihistoria, Ctrl+Enter
│   ├── answer-card.tsx     # markdown-render, abstain-banner, score-meter
│   ├── source-card.tsx     # doc_id+sivu+score, snippet → koko chunk lazy
│   ├── pdf-viewer.tsx      # Dialog + iframe browser PDF viewerille
│   ├── system-sidebar.tsx  # stats/coverage/version + mode-toggle
│   ├── query-history.tsx   # LocalStorage history (max 20)
│   └── ui/                 # shadcn-generoidut primitiivit
├── lib/
│   ├── api/
│   │   ├── client.ts       # typed fetch wrapper, ApiError
│   │   └── openapi.d.ts    # ⚠ autogen, gitignored
│   ├── history.ts          # LocalStorage-helperit
│   └── utils.ts            # shadcn cn()
└── scripts/
    └── gen-types.mjs       # OpenAPI → TS, live-fallback offlineen
```

### 12.5 Tuotanto-build (myöhemmin)

Toistaiseksi UI ajetaan dev-tilassa (`npm run dev`). Tuotantokäyttöön
kaksi vaihtoehtoa kun aika tulee:

1. **Itsenäinen Node-prosessi**: `npm run build && npm run start`
   (port 3000), backend FastAPI port 8080 — selkeä erottelu, helppo
   skaalata erikseen.
2. **Static export FastAPI:n alle**: `next export` → `out/`-kansio,
   palvellaan FastAPI:n `/static/`-mountista yhdellä portilla (8080).
   Yhden binäärin/Docker-imagen deploy. Vaatii että UI ei käytä
   server-rendered route-kohtaista tilaa (tällä hetkellä ei käytä).

Kumpaakin ei tarvita ennen kuin Systeemiä on tarkoitus jakaa toiselle
käyttäjälle — yksilökäytössä `lapua-rag ui` riittää.
