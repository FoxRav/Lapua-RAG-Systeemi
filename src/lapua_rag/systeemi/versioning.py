"""Deterministic version hash of the entire Systeemi.

Two documents in Systeemi = one version hash. If **any** document is added,
removed, re-extracted, or re-embedded, the hash changes. Consumers can cache
answers keyed on ``(tenant, query, system_version)`` and invalidate
automatically.
"""

from __future__ import annotations

import hashlib
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field
from sqlmodel import select

from lapua_rag.db.schema import DocumentRow
from lapua_rag.db.session import session_scope
from lapua_rag.models.document import DocumentStatus


class ModelFingerprint(BaseModel):
    """The pinned models whose outputs are baked into Systeemi."""

    model_config = ConfigDict(frozen=True)

    embedder: str
    reranker: str
    llm_base: str
    llm_lora: str
    schema_version: int = 1


class SystemVersion(BaseModel):
    """A content-hash over the current Systeemi state."""

    model_config = ConfigDict(frozen=True)

    tenant: str | None
    generated_at: datetime = Field(default_factory=datetime.utcnow)
    document_count: int
    content_hash: str = Field(description="sha256 over sorted (doc_id, sha256) pairs")
    models: ModelFingerprint


def _content_hash(pairs: list[tuple[str, str]]) -> str:
    hasher = hashlib.sha256()
    for doc_id, sha in sorted(pairs):
        hasher.update(doc_id.encode("utf-8"))
        hasher.update(b"\x00")
        hasher.update(sha.encode("utf-8"))
        hasher.update(b"\n")
    return hasher.hexdigest()


def compute_system_version(
    *,
    tenant: str | None = None,
    models: ModelFingerprint,
) -> SystemVersion:
    """Compute a stable hash of all INDEXED documents."""
    with session_scope() as db:
        stmt = select(DocumentRow.doc_id, DocumentRow.sha256).where(
            DocumentRow.status == DocumentStatus.INDEXED.value,
        )
        if tenant is not None:
            stmt = stmt.where(DocumentRow.tenant == tenant)
        pairs = [(row[0], row[1]) for row in db.exec(stmt)]

    return SystemVersion(
        tenant=tenant,
        document_count=len(pairs),
        content_hash=_content_hash(pairs),
        models=models,
    )
