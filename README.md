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
- vLLM-palvelun pystytys Linuxille.

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
```

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
| `LAPUA_OCR_DEVICE`               | `gpu:0`                              | Paddle-laitevalinta              |
| `LAPUA_ANSWER_MIN_SCORE`*        | `0.0`                                | Rerank-kynnys closed-book-guardille |

*) suunniteltu; tällä hetkellä kynnys asetetaan `AnswerService(min_score=…)`
konstruktorissa. README päivittyy kun settings-kentät ovat käytössä.

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
4. **6 GB VRAM**: PaddleOCR-VL-0.9B and PP-StructureV3 run comfortably.
   Qwen2.5-1.5B in bf16 requires ~4 GB; running OCR *and* Qwen on the same
   GPU simultaneously will OOM. Use CPU for Qwen on Windows, or offload
   Qwen to a separate Linux host via `LAPUA_LLM_VLLM_URL`.
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
