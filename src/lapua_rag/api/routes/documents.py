"""Document metadata endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from sqlmodel import select

from lapua_rag.db.schema import DocumentRow
from lapua_rag.db.session import session_scope

router = APIRouter()


@router.get("/documents")
def list_documents(
    tenant: str | None = None,
    doc_type: str | None = None,
    limit: int = 50,
) -> list[dict[str, object]]:
    with session_scope() as db:
        stmt = select(DocumentRow)
        if tenant:
            stmt = stmt.where(DocumentRow.tenant == tenant)
        if doc_type:
            stmt = stmt.where(DocumentRow.doc_type == doc_type)
        stmt = stmt.order_by(DocumentRow.created_at.desc()).limit(limit)  # type: ignore[attr-defined]
        return [row.model_dump() for row in db.exec(stmt)]


@router.get("/documents/{doc_id}")
def get_document(doc_id: str) -> dict[str, object]:
    with session_scope() as db:
        row = db.get(DocumentRow, doc_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"doc_id {doc_id} not found")
        return row.model_dump()
