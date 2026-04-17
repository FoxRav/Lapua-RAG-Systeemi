"""Coverage report: what is (and is not) in Systeemi right now."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import select

from lapua_rag.db.schema import DocumentRow
from lapua_rag.db.session import session_scope
from lapua_rag.models.document import DocumentStatus


@dataclass(slots=True, frozen=True)
class _DocFacts:
    doc_id: str
    doc_type: str
    status: str
    paivamaara: date | None
    created_at: datetime


class DoctypeCoverage(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_type: str
    indexed: int
    failed: int
    in_progress: int
    earliest_pvm: date | None = None
    latest_pvm: date | None = None


class CoverageReport(BaseModel):
    """Per-tenant per-doc_type view of Systeemi completeness."""

    model_config = ConfigDict(frozen=True)

    tenant: str | None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    by_doctype: list[DoctypeCoverage] = Field(default_factory=list)
    missing_recent: list[str] = Field(
        default_factory=list,
        description="doc_ids in-progress > 24 h: likely a stuck pipeline.",
    )


_TERMINAL = {DocumentStatus.INDEXED.value, DocumentStatus.FAILED.value}


def compute_coverage(*, tenant: str | None = None) -> CoverageReport:
    """Aggregate Systeemi coverage metrics."""
    with session_scope() as db:
        stmt = select(
            DocumentRow.doc_id,
            DocumentRow.doc_type,
            DocumentRow.status,
            DocumentRow.paivamaara,
            DocumentRow.created_at,
        )
        if tenant is not None:
            stmt = stmt.where(DocumentRow.tenant == tenant)
        rows = [
            _DocFacts(
                doc_id=r[0],
                doc_type=r[1],
                status=r[2],
                paivamaara=r[3],
                created_at=r[4],
            )
            for r in db.exec(stmt)
        ]

    by_type: dict[str, list[_DocFacts]] = {}
    for row in rows:
        by_type.setdefault(row.doc_type, []).append(row)

    coverage = [
        _doctype_coverage(doc_type=dt, rows=subset)
        for dt, subset in sorted(by_type.items())
    ]

    stuck_cutoff = datetime.utcnow()
    missing = [
        row.doc_id
        for row in rows
        if row.status not in _TERMINAL
        and (stuck_cutoff - row.created_at).total_seconds() > 86_400
    ]

    return CoverageReport(tenant=tenant, by_doctype=coverage, missing_recent=missing)


def _doctype_coverage(*, doc_type: str, rows: list[_DocFacts]) -> DoctypeCoverage:
    indexed = sum(1 for r in rows if r.status == DocumentStatus.INDEXED.value)
    failed = sum(1 for r in rows if r.status == DocumentStatus.FAILED.value)
    in_progress = len(rows) - indexed - failed
    pvms = [r.paivamaara for r in rows if r.paivamaara is not None]
    return DoctypeCoverage(
        doc_type=doc_type,
        indexed=indexed,
        failed=failed,
        in_progress=in_progress,
        earliest_pvm=min(pvms) if pvms else None,
        latest_pvm=max(pvms) if pvms else None,
    )
