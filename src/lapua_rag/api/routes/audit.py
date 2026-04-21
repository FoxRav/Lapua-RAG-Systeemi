"""GET /v1/audit — read-only access to the audit log.

Thin reader over ``audit_log``. Mutations happen only via
:func:`lapua_rag.audit.log_entry` (fired as BackgroundTasks from the
query / aggregate routes) — this endpoint is deliberately read-only so
operators can't accidentally edit history from the HTTP surface.

Note: ``AuditLog`` is a SQLModel *table* class; using it directly as
FastAPI's ``response_model`` yields empty payloads because table classes
don't expose fields to Pydantic's serializer. We declare a dedicated
``AuditLogRead`` Pydantic model mirroring the same fields so the
response schema is stable for clients and for the OpenAPI generator.
"""

from __future__ import annotations

from datetime import datetime
from typing import Final

from fastapi import APIRouter, Query
from pydantic import BaseModel, ConfigDict
from sqlmodel import col, desc, select

from lapua_rag.db.schema import AuditLog
from lapua_rag.db.session import session_scope

router = APIRouter()

_MAX_LIMIT: Final[int] = 500
_DEFAULT_LIMIT: Final[int] = 50


class AuditLogRead(BaseModel):
    """Response-model view of :class:`AuditLog`."""

    model_config = ConfigDict(from_attributes=True)

    id: int | None = None
    ts: datetime
    tenant: str
    endpoint: str
    query_text: str
    mode: str | None = None
    abstained: bool = False
    abstain_reason: str | None = None
    max_source_score: float | None = None
    source_doc_ids: str | None = None
    latency_ms: int | None = None
    client_ip: str | None = None
    user_agent: str | None = None


@router.get("/audit", response_model=list[AuditLogRead])
def get_audit_log(
    tenant: str | None = None,
    limit: int = Query(default=_DEFAULT_LIMIT, ge=1, le=_MAX_LIMIT),
) -> list[AuditLogRead]:
    """Return the most recent ``limit`` audit rows, newest first."""
    # Materialise into AuditLogRead *inside* the session so SQLAlchemy's
    # lazy-loading proxies don't trip DetachedInstanceError after the
    # session closes.
    with session_scope() as session:
        stmt = select(AuditLog).order_by(desc(col(AuditLog.ts))).limit(limit)
        if tenant is not None:
            stmt = stmt.where(col(AuditLog.tenant) == tenant)
        return [AuditLogRead.model_validate(row) for row in session.exec(stmt).all()]
