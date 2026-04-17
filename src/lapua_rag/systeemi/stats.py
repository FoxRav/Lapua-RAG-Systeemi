"""Runtime statistics of Systeemi.

Read-only snapshot over the metadata DB; does not hit Qdrant / BM25 to keep
it cheap enough for a health-check endpoint.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import func, select

from lapua_rag.db.schema import ChunkRow, DocumentRow
from lapua_rag.db.session import session_scope
from lapua_rag.models.document import DocumentStatus


@dataclass(slots=True, frozen=True)
class _DocFacts:
    """Materialised subset of :class:`DocumentRow` that survives session close."""

    tenant: str
    doc_type: str
    status: str
    page_count: int
    indexed_at: datetime | None


class DocTypeCount(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_type: str
    count: int


class StatusCount(BaseModel):
    model_config = ConfigDict(frozen=True)

    status: str
    count: int


class TenantStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    tenant: str
    document_count: int
    indexed_count: int
    chunk_count: int
    token_count: int
    page_count: int
    first_indexed: datetime | None = None
    last_indexed: datetime | None = None
    doc_types: list[DocTypeCount] = Field(default_factory=list)
    statuses: list[StatusCount] = Field(default_factory=list)


class SystemStats(BaseModel):
    """Full Systeemi snapshot across all tenants."""

    model_config = ConfigDict(frozen=True)

    generated_at: datetime = Field(default_factory=datetime.utcnow)
    tenant_count: int
    document_count: int
    indexed_count: int
    failed_count: int
    chunk_count: int
    token_count: int
    page_count: int
    per_tenant: list[TenantStats] = Field(default_factory=list)


def gather_stats(*, tenant: str | None = None) -> SystemStats:
    """Return a snapshot of Systeemi. Filter by ``tenant`` when given."""
    with session_scope() as db:
        doc_stmt = select(
            DocumentRow.tenant,
            DocumentRow.doc_type,
            DocumentRow.status,
            DocumentRow.page_count,
            DocumentRow.indexed_at,
        )
        if tenant is not None:
            doc_stmt = doc_stmt.where(DocumentRow.tenant == tenant)
        doc_rows = [
            _DocFacts(
                tenant=r[0],
                doc_type=r[1],
                status=r[2],
                page_count=r[3],
                indexed_at=r[4],
            )
            for r in db.exec(doc_stmt)
        ]

        chunk_stmt = select(ChunkRow.tenant, func.count(), func.sum(ChunkRow.token_count))
        if tenant is not None:
            chunk_stmt = chunk_stmt.where(ChunkRow.tenant == tenant)
        chunk_stmt = chunk_stmt.group_by(ChunkRow.tenant)
        chunk_agg: dict[str, tuple[int, int]] = {
            row[0]: (int(row[1] or 0), int(row[2] or 0)) for row in db.exec(chunk_stmt)
        }

    if not doc_rows:
        return SystemStats(
            tenant_count=0,
            document_count=0,
            indexed_count=0,
            failed_count=0,
            chunk_count=0,
            token_count=0,
            page_count=0,
        )

    by_tenant: dict[str, list[_DocFacts]] = {}
    for row in doc_rows:
        by_tenant.setdefault(row.tenant, []).append(row)

    per_tenant = [
        _tenant_stats(name=name, rows=rows, chunk_agg=chunk_agg.get(name, (0, 0)))
        for name, rows in sorted(by_tenant.items())
    ]

    return SystemStats(
        tenant_count=len(by_tenant),
        document_count=sum(t.document_count for t in per_tenant),
        indexed_count=sum(t.indexed_count for t in per_tenant),
        failed_count=sum(
            sum(sc.count for sc in t.statuses if sc.status == DocumentStatus.FAILED.value)
            for t in per_tenant
        ),
        chunk_count=sum(t.chunk_count for t in per_tenant),
        token_count=sum(t.token_count for t in per_tenant),
        page_count=sum(t.page_count for t in per_tenant),
        per_tenant=per_tenant,
    )


def _tenant_stats(
    *,
    name: str,
    rows: list[_DocFacts],
    chunk_agg: tuple[int, int],
) -> TenantStats:
    statuses = Counter(row.status for row in rows)
    doc_types = Counter(row.doc_type for row in rows)
    indexed_times = [row.indexed_at for row in rows if row.indexed_at is not None]
    indexed_count = statuses.get(DocumentStatus.INDEXED.value, 0)

    return TenantStats(
        tenant=name,
        document_count=len(rows),
        indexed_count=indexed_count,
        chunk_count=chunk_agg[0],
        token_count=chunk_agg[1],
        page_count=sum(row.page_count for row in rows),
        first_indexed=min(indexed_times) if indexed_times else None,
        last_indexed=max(indexed_times) if indexed_times else None,
        doc_types=[
            DocTypeCount(doc_type=dt, count=count) for dt, count in doc_types.most_common()
        ],
        statuses=[
            StatusCount(status=status, count=count) for status, count in statuses.most_common()
        ],
    )
