# Operating Lapua-RAG

## Local development (Windows, GPU for OCR)

```powershell
cd F:\-DEV-\76.PaddleOCR
.\.venv\Scripts\Activate.ps1
pip install -e .[dev]

# 1. start Qdrant
docker compose -f deploy\docker-compose.yml up -d qdrant

# 2. copy env template
copy .env.example .env

# 3. bootstrap
lapua-rag init

# 4. ingest one PDF
lapua-rag ingest "F:/path/to/file.pdf"

# 5. query
lapua-rag query "Mitä kaupunginhallitus päätti § 12?"

# 6. run HTTP API
lapua-rag serve
```

## Production topology (recommended)

* **Workstation or single GPU host (Windows or Linux)** runs PaddleOCR
  (PP-StructureV3, GPU) and the FastAPI service.
* **Linux/WSL2 host with NVIDIA GPU** runs `vllm/vllm-openai` with
  `--enable-lora --lora-modules lapua-llm-v2=CCG-FAKTUM/lapua-llm-v2`.
  Point the FastAPI service at it via `LAPUA_LLM_VLLM_URL`.
* **Managed Qdrant** (or local container) for vectors.
* **PostgreSQL** replaces SQLite for `LAPUA_DATABASE_URL`.
* **Meilisearch** (optional) replaces SQLite FTS5 if you need distributed BM25.

## Reindex scenarios

| Change                                | Rerun                                              |
|---------------------------------------|----------------------------------------------------|
| New LoRA adapter                      | `extract` stage only (`pipeline._extract(...)`)    |
| New chunking strategy                 | `postprocess` + `embed` + `index`                  |
| New embedding model                   | Drop Qdrant collection, re-embed all chunks        |
| New source PDF                        | `lapua-rag ingest <pdf>`; deduplicates by SHA-256  |
| Updated structured schema_version     | bump `schema_version`; rerun `extract` selectively |

## Observability

All logs are structured JSON with `correlation_id = doc_id`. Common
fields:

```
{"event": "pipeline.ocr_start", "doc_id": "abc123…", "level": "info"}
{"event": "pipeline.embed_start", "doc_id": "…", "chunks": 42}
{"event": "pipeline.indexed", "doc_id": "…", "chunks": 42}
```

Ship via `fluentbit` / `vector` to Loki, Elastic, or Datadog.

## Backups

* `data/storage/` – source PDFs + per-doc artefacts (authoritative).
* `data/index/metadata.sqlite` – metadata DB (rebuildable from storage).
* Qdrant volume (`qdrant_data`) – rebuildable from storage via
  `lapua-rag ingest-dir`.

The source-of-truth is **`data/storage/`**. Everything else can be rebuilt.
