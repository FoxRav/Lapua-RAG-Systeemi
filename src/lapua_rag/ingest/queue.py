"""SQLite-backed ingest queue.

Intentionally minimal; no Redis or Celery. Each row represents one attempted
ingest of a PDF and moves through the documents.status state machine.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from sqlmodel import select

from lapua_rag.db.schema import DocumentRow
from lapua_rag.db.session import session_scope
from lapua_rag.models.document import DocumentStatus


@dataclass(frozen=True, slots=True)
class QueueItem:
    doc_id: str
    source_path: Path
    status: DocumentStatus


class IngestQueue:
    """Thin wrapper over the `documents` table providing queue semantics."""

    def enqueue(
        self,
        *,
        doc_id: str,
        tenant: str,
        source_path: Path,
        sha256: str,
        doc_type: str = "muu",
    ) -> QueueItem:
        with session_scope() as db:
            row = db.get(DocumentRow, doc_id)
            if row is None:
                row = DocumentRow(
                    doc_id=doc_id,
                    tenant=tenant,
                    source_path=str(source_path),
                    sha256=sha256,
                    doc_type=doc_type,
                    status=DocumentStatus.QUEUED.value,
                )
                db.add(row)
            return QueueItem(
                doc_id=doc_id,
                source_path=Path(row.source_path),
                status=DocumentStatus(row.status),
            )

    def mark(self, doc_id: str, status: DocumentStatus) -> None:
        with session_scope() as db:
            row = db.get(DocumentRow, doc_id)
            if row is None:
                msg = f"Unknown doc_id={doc_id}"
                raise KeyError(msg)
            row.status = status.value
            if status is DocumentStatus.INDEXED:
                row.indexed_at = datetime.utcnow()
            db.add(row)

    def pending(self, limit: int = 32) -> Iterator[QueueItem]:
        terminal = {DocumentStatus.INDEXED.value, DocumentStatus.FAILED.value}
        with session_scope() as db:
            stmt = (
                select(DocumentRow)
                .where(DocumentRow.status.not_in(terminal))  # type: ignore[attr-defined]
                .order_by(DocumentRow.created_at)
                .limit(limit)
            )
            for row in db.exec(stmt):
                yield QueueItem(
                    doc_id=row.doc_id,
                    source_path=Path(row.source_path),
                    status=DocumentStatus(row.status),
                )
