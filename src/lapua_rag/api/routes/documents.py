"""Document + chunk metadata endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict
from sqlmodel import select

from lapua_rag.db.schema import ChunkRow, DocumentRow
from lapua_rag.db.session import session_scope

router = APIRouter()


class ChunkDetail(BaseModel):
    """Full chunk payload for the UI's "expand snippet" action."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    doc_id: str
    tenant: str
    page_no: int
    section_id: str | None
    section_title: str | None
    doc_type: str
    text: str
    token_count: int


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


@router.get("/documents/{doc_id}/source")
def get_document_source(doc_id: str) -> FileResponse:
    """Stream the original PDF from disk so the UI can deep-link to a page.

    The frontend's PDF modal points an iframe at this URL with a `#page=N`
    fragment which the browser's built-in viewer honours (Chrome/Edge/Firefox).
    """
    with session_scope() as db:
        row = db.get(DocumentRow, doc_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"doc_id {doc_id} not found")
        source_path = Path(row.source_path)

    if not source_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                f"source.pdf for doc_id {doc_id} not found on disk "
                f"({source_path}); index may be out of sync with storage"
            ),
        )
    return FileResponse(
        path=source_path,
        media_type="application/pdf",
        filename=f"{doc_id}.pdf",
        # inline disposition is required so the browser previews the PDF
        # rather than triggering a download.
        headers={"Content-Disposition": f'inline; filename="{doc_id}.pdf"'},
    )


@router.get("/chunks/{chunk_id}", response_model=ChunkDetail)
def get_chunk(chunk_id: str) -> ChunkDetail:
    """Return the full chunk text + metadata.

    The /v1/query endpoint already returns truncated snippets per source;
    this endpoint backs the "show full chunk" expander in the UI without
    re-running retrieval.
    """
    with session_scope() as db:
        row = db.get(ChunkRow, chunk_id)
        if row is None:
            raise HTTPException(status_code=404, detail=f"chunk_id {chunk_id} not found")
        return ChunkDetail(
            chunk_id=row.chunk_id,
            doc_id=row.doc_id,
            tenant=row.tenant,
            page_no=row.page_no,
            section_id=row.section_id,
            section_title=row.section_title,
            doc_type=row.doc_type,
            text=row.text,
            token_count=row.token_count,
        )
