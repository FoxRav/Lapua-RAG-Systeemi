"""Audit-log writer: SQLite + structlog JSON.

Both write paths are fire-and-forget (designed to be invoked via a
FastAPI ``BackgroundTask``): a failure here must never surface to the
caller or corrupt the response. All exceptions are caught and logged
as ``audit.write_failed`` so operators can spot a degraded audit
pipeline without tripping the request path.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import TYPE_CHECKING

from lapua_rag.db.schema import AuditLog
from lapua_rag.db.session import session_scope
from lapua_rag.observability import get_logger

if TYPE_CHECKING:
    from lapua_rag.rag.answer import RagAnswer

_log = get_logger(__name__)

# Truncate the query preview written to the structured log — avoids
# accidentally echoing long paste-bins into shared log drains.
_QUERY_LOG_PREVIEW_CHARS: int = 120


def log_entry(
    *,
    tenant: str,
    endpoint: str,
    query_text: str,
    mode: str | None = None,
    abstained: bool = False,
    abstain_reason: str | None = None,
    max_source_score: float | None = None,
    source_doc_ids: list[str] | None = None,
    latency_ms: int | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Write one audit record to SQLite and emit a structlog event.

    Never raises. Designed for use as a FastAPI ``BackgroundTask``.
    """
    try:
        ids_json = (
            json.dumps(source_doc_ids, ensure_ascii=False)
            if source_doc_ids is not None
            else None
        )
        row = AuditLog(
            ts=datetime.utcnow(),
            tenant=tenant,
            endpoint=endpoint,
            query_text=query_text,
            mode=mode,
            abstained=abstained,
            abstain_reason=abstain_reason,
            max_source_score=max_source_score,
            source_doc_ids=ids_json,
            latency_ms=latency_ms,
            client_ip=client_ip,
            user_agent=user_agent,
        )
        with session_scope() as session:
            session.add(row)
        _log.info(
            "audit.query",
            tenant=tenant,
            endpoint=endpoint,
            mode=mode,
            query=query_text[:_QUERY_LOG_PREVIEW_CHARS],
            abstained=abstained,
            abstain_reason=abstain_reason,
            score=max_source_score,
            latency_ms=latency_ms,
        )
    except Exception:
        # Narrow except is impractical: DB outage, disk full, SQLModel
        # schema drift all need the same "degraded audit" handling.
        _log.exception("audit.write_failed", query=query_text[:60])


def log_query(
    *,
    tenant: str,
    endpoint: str,
    query_text: str,
    answer: RagAnswer,
    mode: str | None = None,
    latency_ms: int | None = None,
    client_ip: str | None = None,
    user_agent: str | None = None,
) -> None:
    """Convenience wrapper extracting audit fields from a :class:`RagAnswer`."""
    source_ids = [s.doc_id for s in (answer.lahteet or [])]
    log_entry(
        tenant=tenant,
        endpoint=endpoint,
        query_text=query_text,
        mode=mode,
        abstained=answer.abstained,
        abstain_reason=answer.abstain_reason,
        max_source_score=answer.max_source_score,
        source_doc_ids=source_ids,
        latency_ms=latency_ms,
        client_ip=client_ip,
        user_agent=user_agent,
    )
