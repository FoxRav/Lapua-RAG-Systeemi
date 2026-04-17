# Lapua-RAG architecture

## 1. Guiding principles

* **Functional core, imperative shell.** Pure functions in `postprocess`,
  `models`, `rag` (easy to test); external clients (Paddle, Qwen, Qdrant,
  SQLModel, FS) isolated in `ocr`, `extract`, `embed`, `index`, `db`, `api`.
* **Idempotent pipeline.** Every stage writes deterministic IDs so reruns
  replace rather than duplicate. Reingesting the same PDF costs O(1).
* **Versioned models.** Every ingested document records the exact model
  revisions used (Paddle, E5, reranker, Qwen base, LoRA, schema_version).
  Re-running only the stages that changed is a one-liner.
* **Tenant-first.** Every row and every vector carries `tenant`. The same
  deployment can serve multiple customers (Lapua today, other municipalities
  and companies tomorrow) with hard data separation.
* **Local-first.** No data leaves the host unless explicitly configured.
  All embeddings, the LLM, and the vector DB run on-prem.

## 2. Module map

```
lapua_rag
├── config.py              # Settings via pydantic-settings (LAPUA_* env)
├── models/                # Pydantic + StrEnum domain models
├── storage/               # Deterministic on-disk layout per doc
├── observability/         # structlog JSON + correlation_id
├── db/                    # SQLModel schema + session_scope
├── ingest/                # SHA-256 dedup, watchdog inbox, SQLite queue
├── ocr/                   # PP-StructureV3 wrapper + VL fallback rule
├── postprocess/           # mojibake fix, doctype rules, § chunking, tables
├── extract/               # Qwen+LoRA local & vLLM clients; JSON-constrained
├── embed/                 # sentence-transformers (E5/BGE-M3) with prefixing
├── index/                 # Qdrant client, SQLite FTS5 BM25, RRF fusion
├── rerank/                # BGE reranker v2 m3
├── retrieve/              # hybrid → rerank composition
├── rag/                   # answer service (Johtopäätös → Perustelut → Lähteet)
├── api/                   # FastAPI routes: /ingest, /query, /documents
├── mcp/                   # FastMCP tools for Cursor / Claude Desktop
├── pipeline.py            # end-to-end orchestrator (idempotent)
└── cli.py                 # Typer CLI
```

## 3. Ingest data flow

```
pdf → SHA-256 → doc_id
      │
      ▼
  storage/<tenant>/<YYYY>/<MM>/<doc_id>/source.pdf
      │
      ▼  ocr.OcrPipeline                           (GPU, stateful)
  pages/NNN.md  + pages/NNN.res.json  + pages/NNN.png
      │
      ▼  postprocess.consolidate_markdown          (pure)
  document.md  (mojibake-safe, from rec_texts when possible)
      │
      ▼  postprocess.detect_doc_type               (pure)
      ▼  postprocess.chunk_document                (pure)
  RawChunk[]   (section_id = '§ N' for poytakirja, heading otherwise)
      │
      ├──▶ embed.Embedder.embed_passages → Qdrant.upsert
      ├──▶ postprocess.stem_finnish      → BM25Index.upsert  (SQLite FTS5)
      └──▶ extract.ExtractionPipeline    → structured.json
                                           (Qwen + lapua-llm-v2 via
                                            lm-format-enforcer)
      │
      ▼
  DB: documents.status = indexed
```

## 4. Query data flow

```
user query
  │
  ├─▶ Embedder.embed_query         → dense top-30
  ├─▶ BM25.search (Finnish stemmer) → sparse top-30
  │
  ▼ rrf_fuse (k=60)
  fused top-30
  │
  ▼ Reranker.rerank (BGE cross-encoder)
  top-5 RetrievalResult
  │
  ▼ AnswerService
  RagAnswer = {johtopaatos, perustelut, lahteet[]}
  (Qwen + lapua-llm-v2, lm-format-enforcer JSON-constrained)
```

## 5. State machine

`DocumentStatus` transitions (`models/document.py`):

```
QUEUED → OCR → POSTPROC → EMBEDDED → EXTRACT → INDEXED
                                              ↘ FAILED
```

Pipeline writes the current status before each stage; a crashed worker resumes
from the stage that never reached `SUCCESS` in `manifest.json`.

## 6. Scaling path

1. **MVP (Windows workstation, GPU for OCR, CPU for Qwen).** This repo today.
2. **Split processes.** OCR worker on Windows GPU; Qwen extraction + RAG
   answering on a Linux host running vLLM with the LoRA adapter loaded
   (`--enable-lora`). `LAPUA_LLM_VLLM_URL` triggers the remote client.
3. **Horizontal workers.** Replace `session_scope` SQLite with PostgreSQL;
   replace `BM25Index` SQLite with Meilisearch or Tantivy; keep the same
   interfaces. No business-logic changes required.
4. **Multi-tenant.** Every module already parameterizes on `tenant`.
   Deploy a single binary, onboard new customers by creating their tenant
   row and serving them with their own Qwen+LoRA (one LoRA per customer).
