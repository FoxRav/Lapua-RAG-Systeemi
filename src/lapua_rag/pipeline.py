"""End-to-end ingest pipeline orchestrator.

Chains the individual stages (ingest → OCR → post-process → embed → extract)
while keeping every stage idempotent. Re-running on an already-indexed
document is a no-op.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

from lapua_rag.config import get_settings
from lapua_rag.db.schema import ChunkRow, DocumentRow, PageRow
from lapua_rag.db.session import create_all, session_scope
from lapua_rag.embed.embedder import Embedder
from lapua_rag.extract.llm import default_client
from lapua_rag.extract.pipeline import ExtractionPipeline
from lapua_rag.index.bm25 import BM25Index
from lapua_rag.index.qdrant import QdrantIndex
from lapua_rag.ingest.dedup import compute_sha256, doc_id_from_sha256
from lapua_rag.models.document import DocumentStatus, DocumentType
from lapua_rag.observability import bind_correlation_id, get_logger
from lapua_rag.observability.metrics import get_instruments
from lapua_rag.ocr.pipeline import OcrPipeline
from lapua_rag.postprocess.chunking import chunk_document, chunk_id
from lapua_rag.postprocess.consolidate import consolidate_markdown
from lapua_rag.postprocess.doctype import detect_doc_type
from lapua_rag.storage.layout import DocumentLayout
from lapua_rag.systeemi.stats import gather_stats

_log = get_logger(__name__)


@dataclass(slots=True)
class IngestResult:
    doc_id: str
    status: DocumentStatus
    page_count: int
    chunk_count: int


@dataclass(slots=True)
class LapuaPipeline:
    """Single-process orchestrator, safe to instantiate once per worker."""

    ocr: OcrPipeline
    embedder: Embedder
    qdrant: QdrantIndex
    bm25: BM25Index
    extractor: ExtractionPipeline

    def ingest(
        self,
        *,
        pdf_path: Path,
        tenant: str | None = None,
        skip_extract: bool = False,
    ) -> IngestResult:
        t_start = time.perf_counter()
        settings = get_settings()
        tenant = tenant or settings.tenant
        create_all()

        sha = compute_sha256(pdf_path)
        doc_id = doc_id_from_sha256(sha)
        bind_correlation_id(doc_id)

        existing = _lookup(doc_id)
        if existing is not None and existing.status == DocumentStatus.INDEXED.value:
            _log.info("pipeline.skip_already_indexed", doc_id=doc_id)
            _record_ingest_metrics(
                tenant=tenant,
                doc_type=existing.doc_type,
                status="skipped",
                duration_s=time.perf_counter() - t_start,
            )
            return IngestResult(
                doc_id=doc_id,
                status=DocumentStatus.INDEXED,
                page_count=existing.page_count,
                chunk_count=0,
            )

        layout = DocumentLayout.for_document(
            storage_root=settings.storage_root,
            tenant=tenant,
            doc_id=doc_id,
            bucket=date.today(),
        )
        if not layout.source_pdf.exists():
            shutil.copy2(pdf_path, layout.source_pdf)
        _upsert_document(doc_id=doc_id, tenant=tenant, sha=sha, layout=layout)

        _log.info("pipeline.ocr_start", doc_id=doc_id)
        _set_status(doc_id, DocumentStatus.OCR)
        ocr_result = self.ocr.run(doc_id=doc_id, pdf_path=layout.source_pdf, layout=layout)
        _persist_pages(doc_id=doc_id, layout=layout, page_count=ocr_result.page_count)

        _log.info("pipeline.postprocess_start", doc_id=doc_id)
        _set_status(doc_id, DocumentStatus.POSTPROC)
        consolidate_markdown(
            pages_dir=layout.pages_dir,
            out_path=layout.document_md,
            page_count=ocr_result.page_count,
        )
        document_text = layout.document_md.read_text(encoding="utf-8")
        doc_type = detect_doc_type(document_text)
        _update_doctype(doc_id=doc_id, doc_type=doc_type, page_count=ocr_result.page_count)

        raw_chunks = chunk_document(text=document_text, doc_type=doc_type)
        chunk_rows = _store_chunks(
            doc_id=doc_id,
            tenant=tenant,
            doc_type=doc_type,
            raw_chunks=raw_chunks,
        )

        _log.info("pipeline.embed_start", doc_id=doc_id, chunks=len(chunk_rows))
        _set_status(doc_id, DocumentStatus.EMBEDDED)
        vectors = self.embedder.embed_passages([row.text for row in chunk_rows])
        self.qdrant.ensure_collection(dim=self.embedder.dimension())
        self.qdrant.upsert(
            ids=[row.chunk_id for row in chunk_rows],
            vectors=vectors,
            payloads=[_chunk_payload(row) for row in chunk_rows],
        )
        self.bm25.upsert(
            rows=[
                {
                    "chunk_id": row.chunk_id,
                    "doc_id": row.doc_id,
                    "tenant": row.tenant,
                    "page_no": row.page_no,
                    "text": row.text,
                }
                for row in chunk_rows
            ],
        )

        if skip_extract:
            _log.info("pipeline.extract_skipped", doc_id=doc_id)
        else:
            _log.info("pipeline.extract_start", doc_id=doc_id)
            _set_status(doc_id, DocumentStatus.EXTRACT)
            structured = self.extractor.extract_document(
                doc_id=doc_id,
                tenant=tenant,
                doc_type=doc_type,
                chunks=raw_chunks,
            )
            layout.structured_json.write_text(
                structured.model_dump_json(indent=2), encoding="utf-8",
            )

        _set_status(doc_id, DocumentStatus.INDEXED)
        _log.info("pipeline.indexed", doc_id=doc_id, chunks=len(chunk_rows))
        _record_ingest_metrics(
            tenant=tenant,
            doc_type=doc_type.value,
            status="indexed",
            duration_s=time.perf_counter() - t_start,
        )
        _update_corpus_gauges(tenant=tenant)
        return IngestResult(
            doc_id=doc_id,
            status=DocumentStatus.INDEXED,
            page_count=ocr_result.page_count,
            chunk_count=len(chunk_rows),
        )


def _lookup(doc_id: str) -> DocumentRow | None:
    with session_scope() as db:
        return db.get(DocumentRow, doc_id)


def _upsert_document(
    *,
    doc_id: str,
    tenant: str,
    sha: str,
    layout: DocumentLayout,
) -> None:
    with session_scope() as db:
        row = db.get(DocumentRow, doc_id)
        if row is None:
            row = DocumentRow(
                doc_id=doc_id,
                tenant=tenant,
                source_path=str(layout.source_pdf),
                sha256=sha,
                doc_type=DocumentType.MUU.value,
                status=DocumentStatus.QUEUED.value,
            )
        db.add(row)


def _set_status(doc_id: str, status: DocumentStatus) -> None:
    with session_scope() as db:
        row = db.get(DocumentRow, doc_id)
        if row is None:
            return
        row.status = status.value
        if status is DocumentStatus.INDEXED:
            row.indexed_at = datetime.utcnow()
        db.add(row)


def _update_doctype(*, doc_id: str, doc_type: DocumentType, page_count: int) -> None:
    with session_scope() as db:
        row = db.get(DocumentRow, doc_id)
        if row is None:
            return
        row.doc_type = doc_type.value
        row.page_count = page_count
        db.add(row)


def _persist_pages(*, doc_id: str, layout: DocumentLayout, page_count: int) -> None:
    with session_scope() as db:
        for page_no in range(page_count):
            md = layout.page_md(page_no)
            js = layout.page_res_json(page_no)
            if not md.exists() or not js.exists():
                continue
            row = PageRow(
                doc_id=doc_id,
                page_no=page_no,
                md_path=str(md),
                json_path=str(js),
                ocr_confidence_avg=_avg_confidence(js),
            )
            db.add(row)


def _avg_confidence(res_json: Path) -> float:
    try:
        data = json.loads(res_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    res = data.get("res", {}) if isinstance(data, dict) else {}
    scores = res.get("overall_ocr_res", {}).get("rec_scores") if isinstance(res, dict) else None
    if not isinstance(scores, list) or not scores:
        return 0.0
    try:
        return float(sum(scores)) / len(scores)
    except (TypeError, ValueError):
        return 0.0


def _store_chunks(
    *,
    doc_id: str,
    tenant: str,
    doc_type: DocumentType,
    raw_chunks: list,
) -> list[ChunkRow]:
    rows: list[ChunkRow] = []
    with session_scope() as db:
        for index, chunk in enumerate(raw_chunks):
            cid = chunk_id(doc_id=doc_id, section_id=chunk.section_id, index=index)
            row = ChunkRow(
                chunk_id=cid,
                doc_id=doc_id,
                tenant=tenant,
                page_no=chunk.page_start,
                section_id=chunk.section_id,
                section_title=chunk.section_title,
                doc_type=doc_type.value,
                text=chunk.text,
                token_count=len(chunk.text) // 4,
                vector_id=cid,
            )
            db.merge(row)
            rows.append(row)
    return rows


def _chunk_payload(row: ChunkRow) -> dict[str, object]:
    return {
        "chunk_id": row.chunk_id,
        "doc_id": row.doc_id,
        "tenant": row.tenant,
        "page_no": row.page_no,
        "section_id": row.section_id,
        "section_title": row.section_title,
        "doc_type": row.doc_type,
    }


def _record_ingest_metrics(
    *,
    tenant: str,
    doc_type: str,
    status: str,
    duration_s: float,
) -> None:
    """Update Prometheus counters for a single ingest outcome.

    Defensive: metric failures (e.g. in tests where the process-wide
    registry was torn down) must never surface as an ingest error.
    """
    try:
        get_instruments().record_ingest(
            tenant=tenant,
            doc_type=doc_type,
            status=status,
            duration_s=duration_s,
        )
    except Exception:  # pragma: no cover - defensive
        _log.warning("pipeline.metrics_record_failed", tenant=tenant, doc_type=doc_type)


def _update_corpus_gauges(*, tenant: str) -> None:
    """Refresh the corpus gauges after a successful indexing run."""
    try:
        stats = gather_stats(tenant=tenant)
        get_instruments().set_corpus_size(
            tenant=tenant,
            documents=stats.document_count,
            chunks=stats.chunk_count,
        )
    except Exception:  # pragma: no cover - defensive
        _log.warning("pipeline.corpus_gauge_failed", tenant=tenant)


def build_default() -> LapuaPipeline:
    settings = get_settings()
    return LapuaPipeline(
        ocr=OcrPipeline(device=settings.ocr_device),
        embedder=Embedder(
            model_name=settings.embedding_model,
            device=settings.embedding_device,
            batch_size=settings.embedding_batch_size,
        ),
        qdrant=QdrantIndex(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            api_key=settings.qdrant_api_key,
        ),
        bm25=BM25Index(path=settings.index_dir / "bm25.sqlite"),
        extractor=ExtractionPipeline(client=default_client()),
    )
