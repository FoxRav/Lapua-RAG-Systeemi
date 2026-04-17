# Lapua-RAG – SOTA-tuotteen suunnitelma

**Tuotteen idea:** paikallinen, yksityisyyden säilyttävä RAG-alusta joka muuttaa
organisaation PDF-asiakirjamassat (pöytäkirjat, tilinpäätökset, kirjanpito,
sopimukset, raportit) **haettavaksi rakenteelliseksi dataksi**, ja vastaa
luonnollisen kielen kysymyksiin **asiakaskohtaisesti hienosäädetyllä**
Qwen2.5 + LoRA -mallilla. Lapuan kaupungin data on ensimmäinen malliesimerkki,
joka toimii myös myyntireferenssinä.

Tavoite: satoja dokumentteja, kymmeniä tuhansia sivuja, inkrementaalinen
ingest, tuotteistettavissa eri asiakkaille omilla LoRA-adaptereilla.

---

## 0. Tuotekehitys lyhyesti

| Vaihe           | Sisältö                                                   | Aikataulu |
|-----------------|-----------------------------------------------------------|-----------|
| v0.1 (tehty)    | Skeleton-koodi, OCR-ajo todistettu, moduulit + testit     | ✅ nyt     |
| v0.2 (1–2 vk)   | Ensimmäinen koko pipeline end-to-end yhdelle dokumentille | seuraava  |
| v0.3 (2–3 vk)   | Eräajoa, Qdrant + BM25, hybridi + reranker, FastAPI-rajap.| –         |
| v0.4 (3–4 vk)   | vLLM Linuxille, LoRA tuotantomoodi, metrics, logi-exportti| –         |
| v1.0 (4–6 vk)   | Multi-tenant, asiakasdemot, SLA-ominaisuudet, auditlokit   | –         |

---

## 1. LoRA-mallin käyttö: `CCG-FAKTUM/lapua-llm-v2`

Mallikortista (huggingface.co/CCG-FAKTUM/lapua-llm-v2) keskeistä:

- Pohjamalli: `Qwen/Qwen2.5-1.5B-Instruct`
- Adapteri: ~74 MB, r=16, α=32, dropout=0.05
- Target-moduulit: `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- Vastausformaatti on **treenattu sisään**: *Johtopäätös → Perustelut →
  Lähteet (§, kokous, pvm)*. Pipeline kunnioittaa tätä formaattia
  (`src/lapua_rag/rag/answer.py`).
- Käyttöohje: käytä **aina RAG-kontekstin kanssa**; älä muistivarastona.

### Kaksi rinnakkaista polkua

```
Extraction (rakenne-irrotus):        Answering (käyttäjän kysymys):

chunk text                           user query
   │                                    │
   ▼  Qwen+LoRA, JSON-schema           ▼  Qwen+LoRA, RagAnswer-schema
DecisionItem { pykala, paatos, … }    RagAnswer { johtopaatos, perustelut, lahteet }
```

Molemmat käyttävät samaa adapteria. **Constrained decoding** (`lm-format-enforcer`
+ JSON-schema) takaa että vastaus parsetuu suoraan Pydanticiksi ilman regex-
korjauksia.

### Missä malli ajautuu

| Ympäristö           | Paikka                | Latenssi/chunk |
|---------------------|-----------------------|----------------|
| MVP Windows CPU     | transformers + PEFT   | 10–30 s        |
| Tuotanto Linux GPU  | **vLLM + --enable-lora** | 0.2–1 s      |
| vaikutus            | `LAPUA_LLM_VLLM_URL`  | transparent    |

Pipeline vaihtaa toteutuksen automaattisesti env-muuttujan perusteella
(`src/lapua_rag/extract/llm.py` – `default_client()`).

---

## 2. Tuotearkkitehtuuri (toteutettu)

Repo sisältää nyt:

```
lapua_rag
├── config.py              # pydantic-settings (LAPUA_*)
├── models/                # Pydantic-domainmallit
│   ├── document.py         # Document, Page, Chunk, DocumentType, DocumentStatus
│   ├── decisions.py        # DecisionItem, DocumentStructure (LoRA-formaatti)
│   └── manifest.py         # ProcessingManifest + stage-records
├── storage/layout.py       # deterministiset polut per doc
├── observability/logging.py# structlog JSON + correlation_id
├── db/                    # SQLModel schema + session_scope
├── ingest/                # SHA-256 dedup, watchdog inbox, SQLite queue
├── ocr/pipeline.py        # PP-StructureV3 wrapper + confidence-fallback
├── postprocess/           # mojibake fix, doctype heuristic, § chunking, tables
├── extract/               # LocalLlmClient (transformers+PEFT) & RemoteVllmClient
├── embed/embedder.py      # E5/BGE-M3 sentence-transformers, E5-prefiksit
├── index/                 # Qdrant, SQLite FTS5 + Finnish stemmer, RRF fusion
├── rerank/reranker.py     # BGE cross-encoder
├── retrieve/search.py     # hybrid → rerank → text-load
├── rag/answer.py          # Johtopäätös → Perustelut → Lähteet synth
├── api/                   # FastAPI: /ingest, /query, /documents
├── mcp/server.py          # FastMCP tools stub
├── pipeline.py            # end-to-end orchestrator (idempotent)
└── cli.py                 # Typer: init, ingest, ingest-dir, query, serve
```

Testit: `tests/unit/*` kattaa puhtaat funktiot (mojibake, chunking, dedup, RRF,
Finnish-stemmer, layout, doctype, fallback-sääntö).

Deploy: `deploy/docker-compose.yml` nostaa Qdrantin ja valinnaisesti
Meilisearchin. vLLM-service on valmiina kommentoituna (Linux/WSL2).

Dokumentaatio: `docs/architecture.md`, `docs/operations.md`,
`docs/sales_overview.md`.

---

## 3. Tiedon elinkaari

```
PDF → SHA-256 → doc_id (16 hex)
     │
     ▼
  data/storage/<tenant>/YYYY/MM/<doc_id>/
     ├── source.pdf
     ├── pages/NNN.md, NNN.res.json, NNN.png
     ├── document.md                 (yhdistetty, mojibake-korjattu)
     ├── tables/NNN_MM.html + .parquet
     ├── structured.json             (Qwen+LoRA ekstraktio)
     └── manifest.json               (versiot + stage-records)
     │
     ▼
  SQL (metadata)
     ├── documents (id, tenant, doc_type, status, indexed_at, …)
     ├── pages
     ├── chunks (section_id, text, vector_id)
     ├── tables
     └── decisions (flatattu structured.json pääelementti)
     │
     ▼
  Qdrant (dense vectors)       BM25 (SQLite FTS5, suomen stemmer)
     │                            │
     └─────────── hybrid ─────────┘
                    │
                    ▼
                 Reranker (BGE v2 m3)
                    │
                    ▼
                 Qwen + LoRA  →  RagAnswer JSON
```

**Idempotentti joka vaiheessa.** Sama PDF samalla sisällöllä = no-op.
Sama chunkkausalgoritmi = deterministiset `chunk_id`:t = vektori-upsert ei
duplikoi.

---

## 4. Mitä PaddleOCR-tulosteesta on säilytettävä

Aiempi PP-StructureV3-ajo synnytti 557 tiedostoa yhden PDF:n ympärille.
Siivousskripti `scripts/cleanup_paddleocr_output.py` pudottaa levytilaa
~80 %.

Pidetään (per sivu):
- `*.md` – markdown teksti
- `*_res.json` – koko rakenne, luottamuspisteet, OCR-tekstit, taulukot
- `*_table_*.html` – taulukot strukturoituna (→ Parquet postprocessissa)
- koko sivun PNG-render (sitaattinäytölle)
- alkuperäinen PDF

Pudotetaan:
- `*.docx`, `*.tex`, leike-PNG:t (`*_img_*.png`) – redundantti tai debug.

`document.md` rakennetaan ensisijaisesti **`_res.json`:n `rec_texts`:stä**,
ei `*.md`-pinkasta, koska PaddleOCR:n markdown-writer tuottaa suomessa
ajoittain cp1252→utf-8 mojibakea (`k�sittely`). Tämä on toteutettu
funktiossa `postprocess.consolidate._page_text_from_json`.

---

## 5. Ekstraktio-skeema (tiukka, Pydantic)

```python
class DecisionItem(BaseModel):
    pykala: str                 # "§ 42"
    otsikko: str
    paatos: str
    perustelut: str
    esittelija: str | None
    aanestys: str | None
    euro_summa: float | None
    paivamaara: date | None
    sivu: int
    lahteet: list[SourceRef]

class DocumentStructure(BaseModel):
    doc_id: str
    tenant: str
    doc_type: Literal[...]
    elin: str | None
    paivamaara: date | None
    otsikko: str | None
    paatokset: list[DecisionItem]
    taulukot: list[TableRef]
    schema_version: int = 1
```

`schema_version` on **tuotteen kontrahti asiakkaalle**. Kun muuttuu, vain
`extract`-vaihe ajetaan uudelleen – ei kallis OCR.

---

## 6. Hybridihaku + rerank

- **Dense top-30**: multilingual E5 large (oletusasetus; vaihda BGE-M3:en
  `.env`:n `LAPUA_EMBEDDING_MODEL`:llä). E5-prefiksit (`query: `/`passage: `)
  hoidetaan automaattisesti `Embedder`-luokassa.
- **Sparse top-30**: SQLite FTS5 + `snowballstemmer` suomen kielelle.
  Tokenisaattorina `unicode61 remove_diacritics 2` – kestää ä, ö, å.
- **RRF-fuusio (k=60)**: `rrf_fuse` on puhdas funktio (ks. testit).
- **Rerank top-5**: `BAAI/bge-reranker-v2-m3` cross-encoder. Tarkkuushyöty
  top-5 vastauksissa on dokumentoitu ~+15–20 % MRR.

---

## 7. Tuotteistaminen: multi-tenant & myyntinäkökulma

- **`tenant` jokaisessa rivissä ja vektorissa**: yksi binääri, monta
  asiakasta, kova data-separaatio.
- **Yksi LoRA per asiakas**: pohja Qwen2.5-1.5B-Instruct jaetaan; LoRA-adapteri
  on 74 MB tiedosto. `LAPUA_LLM_LORA`-env vaihtaa mallin per tenant.
  vLLM:ssä `--lora-modules` tukee useiden adapterien rinnakkaisajoa
  samalla pohjalla.
- **Schema-versio** kulkee `manifest.json`:issa; uusi asiakas = uusi schema
  tarvittaessa, mutta yhteiset ekstraktiotyypit (päätös, raportti, tilinpäätös)
  riittävät useimmille kuntiin/tilitoimistoihin.
- **Sales overview** on valmiina (`docs/sales_overview.md`): arvolupaus,
  vertailutaulukko PDF-chatteihin, demo-reitti, hinnoitteluaihio.

---

## 8. Skaalautuvuus numerolla

RTX 4050 (6 GB) + Windows + CPU-Qwen = MVP.
Ammatillisessa käytössä:

| Komponentti      | Windowsilla (MVP) | Linux/WSL2 + GPU (tuotanto) |
|------------------|-------------------|------------------------------|
| OCR              | ~1.1 s/sivu (GPU) | 0.6 s/sivu (A10G)            |
| Embeddings (E5)  | ~0.05 s/chunk     | ~0.01 s/chunk                |
| Qwen-ekstraktio  | 10–30 s/chunk     | 0.2–1 s/chunk (vLLM-batched) |
| Rerank top-30    | ~0.5 s            | ~0.1 s                        |

Siis 5 000 sivun indeksointi:
- MVP: OCR ~2 h, ekstraktio 40–80 h → **tuntuva pullonkaula on LLM CPU:lla**.
- Tuotanto: OCR ~1 h, ekstraktio 2–4 h. Alle työpäivän.

**Asynkroninen kaksireittinen ajo**: `/v1/ingest` palauttaa heti kun OCR +
indeksointi valmiit; ekstraktio jatkuu taustalla. Dokumentti on hakukelpoinen
minuuteissa, rakenteellinen JSON täydentyy tuntien sisällä.

---

## 9. Mitä seuraavaksi (konkreettinen roadmap)

### Viikko 1 (end-to-end yhdelle PDF:lle)

1. `docker compose -f deploy/docker-compose.yml up -d qdrant`
2. `pip install -e .[dev]`
3. `lapua-rag init` – luo metadata-DB + Qdrant-kokoelma.
4. `lapua-rag ingest "<aiempi testi-PDF>"` – koko pipeline läpi.
   Korjaa esiin nousevat ongelmat (mallin lataus, ecxceptionit, polut).
5. Luo 3–5 dokumentin kultajoukko: käsin tarkastetut odotetut pykälät ja
   taulukot → `tests/fixtures/gold/`.

### Viikko 2 (kattavuus + laatu)

6. Aja 10 todellista Lapua-PDF:ää eri tyypeistä (pöytäkirja, osavuosi,
   tilinpäätös). Mittaa OCR-luottamus, ekstraktion F1, vastauslaatu.
7. Viritä chunkkaussäännöt tilinpäätökselle (rakenteelliset taulukot, ei
   pykäliä).
8. Lisää `LLM-as-judge` -skripti `scripts/eval_judge.py` – automatisoi
   RAG-vastausten pisteyttäminen oman LoRA:n + GPT-4:n kanssa.

### Viikko 3 (tuotantoketju)

9. Pystytä vLLM Linux-koneeseen / WSL2:en; aseta `LAPUA_LLM_VLLM_URL`.
10. Aja 100 todellista dokumenttia eräajona.
11. Lisää Prometheus-metriikka (`ingest_total`, `ingest_duration_seconds`,
    `retrieve_hit_rate`).

### Viikko 4 (demo + myynti)

12. Streamlit- tai Next.js-UI: kysymyspalkki, tulosluettelo sitaateilla,
    klikattavat bbox-highlightit sivu-PNG:iin.
13. Asiakasdemo-skenaario: "Näytä kaikki kaupunginhallituksen päätökset
    vuodelta 2024 jotka koskevat ympäristönsuojelulupia."
14. Paketoi `docker compose` -stack asiakkaiden koneisiin (air-gap-kelpoinen).

---

## 10. Hyväksymiskriteerit v1.0

- [ ] 200 Lapua-PDF:ää indeksoitu, haettavissa, rakenteellisina JSON:eina.
- [ ] Jokaisella vastauksella vähintään 1 lähde (§ + sivu).
- [ ] LLM-as-judge: > 80 % vastauksista "oikeellinen".
- [ ] OCR-luottamus > 0.9 keskimäärin; alle 0.6 päätynyt VL-fallbackiin.
- [ ] `data/storage/` on ainoa source-of-truth; indeksit uudelleenrakennetaan
      alle 1 h:ssa 200 docille.
- [ ] Multi-tenant: sama stack palvelee ≥2 asiakasta ilman koodimuutoksia.
- [ ] CI: ruff + mypy --strict + pytest vihreänä.
- [ ] Auditlokit: kuka kysyi mitä, milloin, mihin dokumentteihin päätyi.

Tämä on myyntipakettisi **minimumi ominaisuusjoukko**; kaikki yllä oleva on
arkkitehtuurissa valmiina, toteutusta vaille.
