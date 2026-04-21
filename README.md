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

## 1. Tilanneraportti (v0.8.0 · 2026-04-21)

Repo julkaistu GitHubiin: <https://github.com/FoxRav/Lapua-RAG-Systeemi>.

### v0.8.0 · 2026-04-21 – Observability-loop + auditloki + aggregate-UI

Ohjeen `INSTRUCTIONS FOLDER/CURSOR_v0.8_OHJE.md` tehtävät A–E toteutettu
ennen `lapua-llm-v3`-julkaisua:

- **Prometheus call-site-instrumentointi** (`src/lapua_rag/rag/answer.py`,
  `src/lapua_rag/pipeline.py`): `AnswerService.answer()` kirjaa jokaiselle
  kyselylle `query_total`/`query_duration_seconds`/`retrieve_top_score` +
  `query_abstained_total`-laskurin. Ingestissä `ingest_total`/
  `ingest_duration_seconds` kirjautuu myös skip-polulla, ja onnistuneen
  indeksoinnin jälkeen `corpus_documents_total`/`corpus_chunks_total` päivittyvät
  `gather_stats()`-kautta. Metriikoiden epäonnistuminen ei koskaan kaada
  vastauspolkua (`try/except` defensiivisesti).
- **Auditloki** (`src/lapua_rag/audit/service.py`, `/v1/audit`-endpoint,
  uusi `AuditLog`-taulu): `/v1/query` ja `/v1/aggregate` kirjaavat taustalla
  (FastAPI `BackgroundTasks`) kyselyn, tenantin, moodin, abstain-lipun,
  top-scoren, source-doc-id:t, latenssin, client-IP:n ja User-Agent:n
  SQLite-tauluun **sekä** `structlog`-JSON-eventtinä. Read-only API:
  `GET /v1/audit?tenant=…&limit=…` palauttaa uusimmat kirjaukset. Kaikki
  epäonnistumiset logitetaan, mutta ne eivät koskaan riko pääflowta.
- **Frontend aggregate-reititys**: uusi `frontend/lib/classify-query.ts`
  tunnistaa "kuinka monta / paljonko rahaa / summa"-kuviot ja reitittää
  kyselyn automaattisesti `/v1/aggregate`-endpointiin. Tulos renderöidään
  uudessa `AggregateCard`-komponentissa (lokaliisoitu numeroformaatti,
  ikonit, "not_supported"-fallback). Chat-paneelin `ChatTurn` on nyt
  erottava union (`kind: "rag" | "aggregate"`).
- **`scripts/train_v3_lora.py`** — Unsloth-pohjainen LoRA-koulutusskripti
  `lapua-llm-v3`:lle (WSL2). Raskaat ML-importit tehdään `run()`:n sisällä,
  joten `--help` ja yksikkötestit toimivat ilman GPU-pinoa. `TrainConfig`
  + `parse_args()` + `load_records()` ovat puhtaita funktioita; Unit-testit
  kattavat argumenttiparsinnan ja JSONL-lukujen.
- **`scripts/expand_gold_set.py`** — laajentaa
  `tests/fixtures/gold/lapua_gold_v1.jsonl`:n 30 riviin: olemassa olevat
  rivit säilytetään, off-topic-kysymykset (abstain) lisätään, loput
  täytetään Qdrantista vedetyistä päätös-chunkeista generoiduilla
  extract-kysymyksillä. `section_id` + lyhyt `doc_id` toimivat
  dedup-disambiguaattorina, jotta toistuva "## Päätös"-otsikko ei romahduta
  rivejä yhdeksi.
- **42 uutta yksikkötestiä** (yhteensä **157 vihreänä**, ~11 s): Prometheus
  call-site (AnswerService + pipeline), auditpalvelun happy/failure-polku,
  audit-endpoint, classify-query-funktio, train_v3_lora-parseri,
  expand_gold_set-generaattori. `ruff check` puhdas, mypy-baseline
  ennallaan (uusissa tiedostoissa ei uusia virheitä).

### v0.7.0 · 2026-04-21 – Eval-pipeline, aggregate-endpoint, Prometheus

Jatko-ohjeen (`INSTRUCTIONS FOLDER/CURSOR_JATKO-OHJE.md`) tehtävät 1–4 ja 6
toteutettu ennen `lapua-llm-v3`-retrainia:

- **`scripts/audit_training_data.py`** — luokittelee ChatML-JSONL:n
  abstain/extract-pareihin, raportoi suhteen ja nostaa varoituksen kun
  abstain-osuus > 60 % tai < 40 %.
- **`scripts/build_v3_dataset.py`** — rakentaa balansoidun
  `data/training/v3_dataset.jsonl`-datasetin Qdrantista. Pure-function
  osat (`extract_question_from_chunk`, `make_extract_example`,
  `make_abstain_example`, `build_dataset`) yksikkötestattuja ilman
  Qdrant-kutsuja.
- **`tests/fixtures/gold/lapua_gold_v1.jsonl` + `scripts/eval_rag.py`** —
  10-rivinen seed-kultajoukko + eval-skripti joka ajaa FastAPI:n läpi ja
  raportoi abstain-tarkkuuden, extract-tarkkuuden, keskilatenssin sekä
  v1.0-hyväksymisrajat.
- **`/v1/aggregate`-endpoint** (`src/lapua_rag/api/routes/aggregate.py`):
  COUNT/SUM-kysymykset reititetään olemassaolevalle `DecisionRow`-taululle
  SQL:llä. Deterministinen luokittelija + etunimi-sukunimi-heuristiikka
  entity-ekstraktioon. Taustalla UI-tunnistus tulossa frontendiin
  erikseen.
- **Prometheus-metriikka** (`src/lapua_rag/observability/metrics.py`):
  `/metrics`-endpoint tuottaa tekstimuotoisen exposition seitsemälle
  instrumentille (`lapua_rag_ingest_total`, `..._ingest_duration_seconds`,
  `..._query_total`, `..._query_duration_seconds`,
  `..._retrieve_top_score`, `..._corpus_documents_total`,
  `..._corpus_chunks_total`). Oma `CollectorRegistry` jotta MCP-sidecar
  ja testit eivät kolaroi globaaliin rekisteriin.
- **61 uutta yksikkötestiä** (yhteensä **130 vihreinä**, ~5 s); ruff
  puhdas, mypy puhdas uusissa tiedostoissa (aggregate-endpoint käyttää
  `sqlmodel.col()`-apuria strict-tarkistuksen toteuttamiseksi).
- **Suunnitelma** `tmp/jatko_plan_2026-04-21.md` kuvaa mitä tehtiin ja
  mitkä ohjeen kohdat (v3-koulutus, batch-ingest-lisäerät, frontendin
  aggregate-reititin) jäivät seuraavaan iteraatioon.

### Valmista (end-to-end ajossa)

- **PaddleOCR-ympäristö pystyssä** (ks. §6). PP-StructureV3 ajettu eräajona
  108 PDF:lle (DATA_päättävät_elimet_20251202), keskimääräinen OCR-luottamus
  > 0.9, GPU NVIDIA RTX 4050 Laptop.
- **Korpus indeksoitu**: **109 dokumenttia, 7 922 chunkia, 1.13 M tokenia**
  Qdrantissa + SQLite FTS5:ssa (`scripts/batch_ingest.py`, 11 h 15 min
  OCR+embed). Pipeline (`src/lapua_rag/pipeline.py`) ajettu kokonaisuutena.
- **Hybridihaku tuotantokäytössä**: Qdrant dense + BM25 sparse + RRF-fuusio
  + BGE reranker-v2-m3 cross-encoder, top_k_final=8 + chunk-type boost
  (+0.20 `## Päätös`+verbi, -0.15 osallistujalistat).
- **vLLM WSL2-distrossa live** (`lapua-vllm` / F:\wsl\lapua-vllm, Ubuntu 22.04).
  Qwen2.5-1.5B bf16 + LoRA `lapua-llm-v2` rinnakkain `--lora-modules
  lapua-v2=…`, OpenAI-endpoint `http://localhost:8000/v1`. Ks. §10.
- **Vastaustila `extract` default** (v0.6 silta, v0.6.2 viritetty): LoRA
  lainaa 1–3 virkettä → Python renderöi yhden siteeratun vastauksen.
  Cross-chunk fallback ja `answer_min_score=0.10` siisti abstain
  COUNT/off-topic-kysymyksille. Ks. §11.
- **Closed-book-vahtikoira `AnswerService`:ssa**: LLM:ää ei kutsuta jos
  Systeemistä ei löydy riittävän luotettavaa kontekstia. `RagAnswer`
  kantaa `abstained` + `abstain_reason` (`no_context` / `below_threshold`
  / `model_refused`) – hallusinaatioriski = 0 by construction.
- **Systeemi-alijärjestelmä (`src/lapua_rag/systeemi/`)**: tilastot, versionti
  (deterministinen SHA-256 indeksoitujen dokumenttien yli), kattavuusraportti.
  API-endpointit `/v1/system/stats|version|coverage` ja CLI-aliryhmä
  `lapua-rag system {stats,version,coverage}` ajettu live-korpuksella.
- **Frontend tuotantolaadulla** (`frontend/`, ks. §12): Next.js 16 +
  React 19 + Tailwind 4 + shadcn/ui (base-nova). Chat-paneeli, lähdekortit,
  PDF-katsoja, 3-tilan moodikytkin (`extract / retrieve / synth`),
  kyselyhistoria, light/dark/system theme. Tyypit autogeneroitu
  FastAPI:n OpenAPI-spekistä.
- **69 yksikkötestiä vihreinä, ~7 s** (mojibake-korjaus, pykäläpohjainen
  chunkkaus, doctype-heuristiikka, SHA-256-dedup, RRF-fuusio, suomen
  stemmer, storage-layout, OCR-fallback, Systeemin tilastot & versiointi,
  closed-book-guard, retrieve/extract-tilojen flow, chunk-type boost,
  cross-chunk fallback, lenient JSON-parser, PDF-streaming, mode-override).
  `ruff` puhdas, `next build` + `tsc --noEmit` + ESLint OK.
- **`lapua-rag`-CLI asennettu editable-tilassa**: `init / ingest / ingest-dir /
  query / serve / ui / openapi-dump / system {stats,version,coverage}`.
- **Deploy-konfigurointi** (`deploy/docker-compose.yml`) Qdrantille ja
  Meilisearchille; vLLM-palvelu kommentoitu valmiiksi Linux/WSL2-siirtoa
  varten.

### Ei vielä aloitettu

- Multi-tenant-todennus (tenant-kenttä on rakenteessa, ACL-kerros puuttuu).
- LLM-as-judge eval-automaatio (v0.7:n `scripts/eval_rag.py` käyttää
  substring-match + abstain-lippua; judge-LLM tulee v1.0 acceptance-kuopan
  osana, §5).
- `lapua-llm-v3` retrainin varsinainen ajo (ylikoulutettu abstain, ks. §11):
  `scripts/build_v3_dataset.py` rakentaa datasetin, `scripts/train_v3_lora.py`
  (v0.8) ajaa Unsloth-LoRA-koulutuksen WSL2:ssa. Palautuu v1.0:aan kun uusi
  LoRA on julkaistu, jolloin `LAPUA_ANSWER_MODE=synth` voidaan ottaa
  käyttöön ja `extract`-silta jää backupiksi.

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

### v0.6.2 · 2026-04-18 – Reranker-boost + cross-chunk fallback + tiukempi gate

Vahvistus: live smoke-testi paljasti että v0.6:n `extract`-tila *toimii*
(LoRA tuottaa quoteja eikä abstainaa kapeassa tehtävässä), mutta kolme
erillistä vikaa ohjasi vastauksen pieleen. Kaikki kolme on nyt korjattu.

| # | Ongelma v0.6.0:ssa | Korjaus v0.6.2:ssa |
|---|--------------------|---------------------|
| 1 | "Saapuvillaolleet jäsenet"-chunkit voittivat reranker-pisteissä (~0.99) myös spesifeihin "kuka on" -kysymyksiin | `_chunk_type_boost`: +0.20 chunkeille joissa `## Päätös` + valinta-verbi; -0.15 osallistujalistoille |
| 2 | Reranker top-k_final=5 → oikea chunk saattoi jäädä pooliin pääsemättä | `top_k_final = 8` (riittävä headroom myös laajemmalle korpukselle) |
| 3 | Python-fallback käytti aina top-1 chunkkia → kun reranker oli väärässä, fallback toisti virheen | `_python_fallback_across_chunks`: hakee parhaan token-overlap-virkkeen *kaikista* top-N chunkista |
| 4 | `min_score=0.0` salli aggregointi-/skooppitappio-kysymysten edetä LLM:lle (Q3 top-score 0.030 antoi "Sami Kuula." -tason vastauksen) | `answer_min_score = 0.10` (BGE-sigmoid: relevant >0.5, marginal 0.10-0.50, irrelevant <0.05) |

**Smoke-testi v0.6.2:lla** (samat 3 kysymystä jotka aiemmin tuottivat
ohilaukauksia):

| Kysymys | top_score | abstained | Tulos |
|---------|-----------|-----------|-------|
| "Kuka on Lapuan kaupunginhallituksen puheenjohtaja?" | 1.189 (boost) | False | LoRA suoralla sitaatilla "Lapuan kaupunginhallituksen puheenjohtaja Kai Pöntinen" |
| "Kuka on Lapuan kaupunginjohtaja?" | 1.144 (boost) | False | Cross-chunk Python-fallback löysi "Satu Kankare … Kaupunginjohtaja" oikealta sivulta |
| "Kuinka monessa päätöksessä Sami Kuula on ollut mukana?" | 0.030 | **True** | Siisti abstention: "Parhaan vastaavuuden pisteet (0.030) jäivät kynnyksen (0.100) alle" |

**13 uutta yksikkötestiä** kattaa boost-funktion (5), end-to-end-flippauksen
stub-rerankerilla (2), cross-chunk-fallbackin kahdessa tilassa (2), lenient
JSON-parserin (2), markdown-fence-toipumisen (1) ja garbage-syötteen (1).
**Yhteensä 69 testiä vihreinä**, ruff puhdas.

### v0.6.1 · 2026-04-18 – vLLM-stabiliteetti + plain-text + lenient JSON

Live smoke-testi v0.6.0:lla paljasti että **vLLM AsyncLLMEngine kuoli
60 s -timeoutiin** ensimmäiseen heavy-promptiin: 5169-merkin konteksti +
outlines `guided_json` RTX 4050 Laptopilla (6 GB) ylitti
`ENGINE_ITERATION_TIMEOUT_S`-rajan, ja kuoltuaan engine ei toipunut →
kaikki seuraavat kutsut 500 / ConnectError. Korjaukset:

- **`generate_text`** rinnakkainen polku `LlmClient`-protokollaan: plain
  chat completion ilman `guided_json`:ia. Käytetään extract-tilassa, jossa
  skeema on niin pieni (3 kenttää) että outlines-overhead ei ole
  perusteltu. Lenient JSON-parseri (`_parse_extract_response`) toipuu
  myös markdown-koodiaitojen sisään käärityistä vastauksista, joita
  lapua-llm-v2 empiirisesti emittoi.
- **Kevyempi extract-promptin context**: `extract_max_chunks=3`
  (vs. 5) × `extract_max_chars_per_chunk=800` (vs. 1500). Konteksti
  tippui 5169 → ~2400 merkkiä (≥50 % vähennys). Synth-tila käyttää
  edelleen 5×1500 vakiokokoa.
- **vLLM-startup-skriptin kovetus**: `VLLM_ENGINE_ITERATION_TIMEOUT_S=180`,
  `--max-num-seqs 1` (single-user laptop), `--enforce-eager`.

### v0.6 · 2026-04-18 – `extract`-tila: yksi yhtenäinen vastaus ilman v3:a

Käyttäjäpalaute v0.5:n `retrieve`-tilasta: *"vastaukset eivät ole järkeviä,
tarvitaan yksi yhtenäinen vastaus eikä järjetöntä määrää ohilaukauksia"*.
`retrieve`-tila on suunniteltu chunk-dumpiksi (käyttäjä lukee evidenssin
itse) ja `synth`-tila on rikki kunnes `lapua-llm-v3` on koulutettu (ks. §11).
Tarvittiin kolmas tila joka antaa yhden siteeratun vastauksen heti.

- **Uusi `extract`-vastaustila** (oletus v0.6:sta lähtien):
  - **Stage A – LoRA-extract**: sama `lapua-llm-v2`, mutta kapealla
    quote-only-tehtävällä (`_ExtractResponse {quote, chunk_index, no_match}`).
    Kapeampi tehtävä ohittaa abstain-bias:n koska prompt ei pyydä mallia
    *vastaamaan* — vain *lainaamaan sanatarkasti*.
  - **Stage B – Python-render**: rakentaa `RagAnswer`:n templateksi:
    johtopäätös = suora lainaus, perustelut = lähde-header + reranker-score,
    lähteet = lainattu lähde ensin + tukevat lähteet seuraavina.
  - **Python-fallback**: jos LoRA palauttaa `no_match=true`, tyhjän quote:n,
    tai epäonnistuu (httpx/JSON/pydantic), heuristiikka valitsee top-1
    chunkista virkkeen jolla on suurin token-overlap kysymyksen sisältö-
    sanojen kanssa. Käyttäjä saa **aina** yhden siteeratun vastauksen.
  - Suomi-tietoinen virkkeenjako (`_split_sentences`) joka strippaa
    HTML-taulukot ja markdown-headerit OCR:n tuottamasta tekstistä.
- **Closed-book-takuut säilyvät**: `no_context` ja `below_threshold` -portit
  laukeavat ennen LLM-kutsua myös `extract`-tilassa. Hallusinaatioriski = 0
  koska Stage B ei koskaan keksi sanoja kontekstin ulkopuolelta.
- **Backend-laajennukset**:
  - `AnswerMode` Literal: `"synth" | "retrieve" | "extract"`.
  - `Settings.answer_mode` default `"extract"`.
  - `_answer_service` `lru_cache(maxsize=3)` kattaa kaikki kolme tilaa.
- **Frontend-laajennukset**:
  - `SystemSidebar`: Switch korvattu 3-tilan `Tabs`:lla, näyttää
    moodikohtaisen kuvauksen.
  - `app/page.tsx`: `DEFAULT_MODE = "extract"`, LocalStorage tunnistaa
    kaikki kolme arvoa.
  - `AnswerCard`: badge-tyylit (`extract`+`synth` highlighted, `retrieve`
    secondary) viestii että ekstraktio on aktiivinen synthesointi-luokka.
- **10 uutta yksikkötestiä** (`test_extract_mode.py`): LoRA-quote happy path,
  Python-fallback `no_match`/empty/exception-poluilla, chunk_index-clampaus,
  no_context/below_threshold-portit, virkkeenjakajan robustius
  HTML/markdown-noiselle. **Yhteensä 56 testiä vihreinä**, `answer.py`
  coverage 98%. Ruff puhdas, `next build` + `tsc --noEmit` + ESLint OK.

Vaikutus käyttäjän näkemään: aiempi 5-chunkin dump muuttuu yhdeksi
selkeäksi *"Tarkastuslautakunta on kutsunut kaupunginjohtaja Satu Kankareen
kertomaan kaupungin ajankohtaisista asioista."*-tyyppiseksi vastaukseksi
joka linkittyy oikeaan PDF:ään ja säilyttää tukimateriaalit alapuolella
tarkistettavaksi.

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
    **mode-toggle** (v0.5: synth ↔ retrieve Switch; v0.6: extract / retrieve /
    synth Tabs, persistoidaan LocalStorage:iin).
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
# Kloonaa repo (uusi GitHub-koti)
git clone https://github.com/FoxRav/Lapua-RAG-Systeemi.git F:\-DEV-\76.PaddleOCR
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
| `/v1/aggregate`        | POST   | COUNT/SUM-kysymykset SQL:llä `decisions`-taulusta (ei RAG) |
| `/v1/audit`            | GET    | Viimeisten kyselyjen auditloki (read-only)           |
| `/metrics`             | GET    | Prometheus-exposition                                |
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
| `LAPUA_ANSWER_MODE`              | `extract`                            | `extract`=LoRA-quote + Python-render (oletus, v0.6 silta), `retrieve`=top-N siteeratut katkelmat, `synth`=LLM-syntheesi (vaatii toimivan LoRAn, vrt. §11) |
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

### Tehty (viikot 1–4 tähän mennessä)

- [x] Pipeline ajettu kokonaisuutena → 109 dokumenttia indeksoitu (v0.3)
- [x] vLLM WSL2-distrossa live `--enable-lora`-moodissa (v0.3, §10)
- [x] SLM ajossa `lapua-rag query` -kautta (extract-tila default v0.6 alkaen)
- [x] Hybridihaku Qdrant + BM25 + reranker validoitu live-korpuksella
- [x] Closed-book-guard validoitu elävällä mallilla (no_context /
      below_threshold / model_refused, ks. §11)
- [x] Frontend (Next.js + shadcn/ui) tuotantolaadulla (v0.5, ks. §12)
- [x] 69 yksikkötestiä vihreänä, ruff puhdas

### Viikko 1 jäljellä – **integraatiotestaus + kultajoukko**

1. **Kirjoittaa smoke-testi `tests/integration/test_pipeline_e2e.py`** joka
   ajaa koko ketjun pienellä kultadokumentilla ja varmistaa että
   `structured.json` sisältää odotetut pykälät + Systeemin tila muuttuu
   (uusi versiohash, +1 indexed_count).
2. **Verifioida mojibake-korjaus** ajamalla `consolidate_markdown` yhdelle
   PDF:lle ja diff-vertaamalla `rec_texts`-lähtöiseen tekstiin.
3. **Luoda kultajoukko** 3–5 dokumentista eri tyypeistä (pöytäkirja,
   osavuosi, tilinpäätös) → `tests/fixtures/gold/`.

### Viikko 2 – **laadun mittarointi**

4. **Mitata indeksoidun korpuksen laatu**: OCR-luottamus per dokumentti,
   ekstraktion pykäläkattavuus, hybridihaun recall@5 109-dokumentin
   korpuksella.
5. **Virittää chunkkaussäännöt tilinpäätökselle** (taulukkorakenteinen, ei
   pykäliä) – lisätä `postprocess/chunking.py`:hin taulukko-tietoiset
   sliding-windowit.
6. **Kirjoittaa `scripts/eval_judge.py`** – LLM-as-judge automaattiarviointi
   omalla LoRA:lla + referenssinä GPT-4.

### Viikko 3 – **tuotantoketju + reranker-laajennus**

7. **Reranker-parannus** (§11.4): rerank-pool top-K 8 → 20, query-rewrite
   3 reformulointia, max-pool-pisteytys. Tavoite: ratkaista loputkin
   hallitus/valtuusto-sekoitukset jotka v0.6.2:n boost ei korjannut.
8. **Aggregointi-endpoint** `POST /v1/aggregate` (§11.4): COUNT-kysymykset
   reititetään SQL-pohjaiselle haulle `decisions`-taulusta. UI tunnistaa
   "kuinka monta/montako" -kuviot ja siirtää kutsun automaattisesti.
9. **Prometheus-metriikka** (`/metrics`-endpoint): `ingest_total`,
   `ingest_duration_seconds`, `retrieve_hit_rate_at_5`,
   `llm_generate_duration_seconds`.

### Viikko 4 – **demo + myynti**

10. **lapua-llm-v3 retrain** (§11): 50/50 abstain/extract data, ≥200
    positiivista esimerkkiä, validointi 10 kontrollikyselyllä, julkaisu.
    Sen jälkeen `LAPUA_ANSWER_MODE=synth` voidaan ottaa käyttöön; extract
    jää backupiksi.
11. **Asiakasdemo-skenaariot**: "Näytä kaikki kaupunginhallituksen päätökset
    2024 joita koskevat ympäristönsuojeluluvat", "Mitkä investoinnit
    ylittyivät budjetissa Q1 aikana?".
12. **PDF-katsojan bbox-highlightit**: sivu-PNG + reranker-osumakohdan
    bounding box klikattavana — täydentää nykyistä `pdf-viewer.tsx`-modaalia.
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
- [x] Auditloki: kuka kysyi mitä, milloin, mihin dokumentteihin päätyi
      (v0.8: SQLite `audit_log` + `structlog` + `GET /v1/audit`)

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
# 0. clone Lapua-RAG-Systeemi itsensä
git clone https://github.com/FoxRav/Lapua-RAG-Systeemi.git F:\-DEV-\76.PaddleOCR
cd F:\-DEV-\76.PaddleOCR

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

### Sillaikaa: `extract`-tila (v0.6 silta)

Käyttäjäpalaute v0.5:n `retrieve`-tilasta: *"vastaukset eivät ole järkeviä,
tarvitaan yksi yhtenäinen vastaus eikä järjetöntä määrää ohilaukauksia"*.
Vastaus tähän on **kolmas vastaustila** `extract` (default `.env.example`:ssa
v0.6:sta lähtien), joka antaa yhden siteeratun vastauksen ilman v3-koulutuksen
odottelua.

**Toimintaperiaate**:

1. **Stage A – LoRA-extract**: Sama `lapua-llm-v2` LoRA, mutta kapealla
   tehtävällä: *"lainaa kontekstista 1–3 virkettä jotka sisältävät vastauksen,
   palauta `chunk_index` ja `no_match`"*. Kapeampi tehtävä ohittaa abstain-bias:n
   koska prompt ei pyydä mallia *vastaamaan* — vain *lainaamaan sanatarkasti*.
2. **Stage B – Python-render**: Vastaus rakennetaan templateksi: johtopäätös =
   suora lainaus, perustelut = lähde-header + reranker-score, lähteet = lainattu
   lähde ensin + tukevat lähteet seuraavina.
3. **Fallback**: Jos LoRA palauttaa `no_match=true`, tyhjän quote:n, tai
   epäonnistuu (httpx/JSON/pydantic), Python-heuristiikka valitsee top-1
   chunkista virkkeen jolla on suurin token-overlap kysymyksen sisältösanojen
   kanssa. Käyttäjä saa aina yhden siteeratun vastauksen — ei koskaan dump:ia.

**Mitä `extract` EI ratkaise**:

- **Reranker-virheet** (osittain korjattu v0.6.2:ssa): Jos top-1 chunk on
  väärä, extract lainaa väärästä lähteestä yhtä varmasti kuin synth
  syntetisoisi väärin. v0.6.2:n `_chunk_type_boost` (`## Päätös`+verbi
  +0.20, osallistujalistat -0.15) ratkaisi smoke-testin Q1/Q2-virheet,
  mutta yleisempi reranker-laajennus (query-rewrite, MMR-diversifikaatio,
  isompi rerank-pool 8 → 20) jää roadmapille §11.4.
- **Aggregointikysymykset**: *"Kuinka monessa päätöksessä Sami Kuula on
  ollut mukana?"* on COUNT-kysymys joka ei kuulu RAG:iin lainkaan. v0.6.2:n
  `answer_min_score=0.10` abstainaa nämä siististi sen sijaan että ne
  tuottaisivat järjettömiä vastauksia, mutta lopullinen ratkaisu on
  oma SQL-pohjainen `/v1/aggregate`-endpoint (ks. roadmap §11.5).

**Konfigurointi**: `LAPUA_ANSWER_MODE=extract` (default), tai per-pyyntö
`POST /v1/query {"query":"...", "mode":"extract"}`. UI:ssa 3-tilan
välilehtivalinta `Extract / Retrieve / Synth` `SystemSidebar`:ssä.

### Aikataulu

1–2 päivää: datan revisio + LoRA-koulutus (RTX 4050, 1.5B base, r=16, ~2 h
per epoch × 3-5 epochia), validointi, julkaisu.

### 11.4 Roadmap-tikettien jatkot

- **Reranker-parannus** (Q1-tyyppisille hallitus/valtuusto-sekoituksille):
  rerank-input top-K 5 → 20, query-rewrite step joka tuottaa 3 reformulointia,
  max-pool-pisteytys.
- **Aggregointi-endpoint** (Q3-tyyppisille COUNT-kysymyksille):
  `POST /v1/aggregate` joka käyttää LoRA:a vain entity-extractioniin
  (henkilö/päätös/päivämäärä) ja suorittaa SQL-haun
  `decisions`-taulusta. UI tunnistaa "kuinka monta/montako" -kysymykset
  ja reitittää ne automaattisesti.

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
