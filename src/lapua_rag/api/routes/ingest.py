"""POST /v1/ingest endpoint."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile
from pydantic import BaseModel

from lapua_rag.api.auth import require_api_key
from lapua_rag.config import get_settings
from lapua_rag.pipeline import build_default

router = APIRouter()


class IngestAccepted(BaseModel):
    doc_id: str
    status: str
    page_count: int
    chunk_count: int


@router.post("/ingest", response_model=IngestAccepted)
async def ingest_pdf(
    file: UploadFile,
    background: BackgroundTasks,
    tenant: Annotated[str, Depends(require_api_key)],
) -> IngestAccepted:
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files accepted")

    settings = get_settings()
    inbox_path = _save_to_inbox(file, inbox_dir=settings.inbox_dir)

    pipeline = build_default()
    # When auth is enabled the key-bound tenant wins; otherwise fall back
    # to the configured single-tenant default (same as pre-v0.9).
    effective_tenant = tenant if settings.auth_enabled else None
    result = pipeline.ingest(pdf_path=inbox_path, tenant=effective_tenant)
    return IngestAccepted(
        doc_id=result.doc_id,
        status=result.status.value,
        page_count=result.page_count,
        chunk_count=result.chunk_count,
    )


def _save_to_inbox(file: UploadFile, *, inbox_dir: Path) -> Path:
    inbox_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=".pdf",
        dir=inbox_dir,
        delete=False,
    ) as tmp:
        shutil.copyfileobj(file.file, tmp)
        return Path(tmp.name)
