"""Retrieval service: hybrid fetch → text loading → rerank."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlmodel import select

from lapua_rag.db.schema import ChunkRow
from lapua_rag.db.session import session_scope
from lapua_rag.index.hybrid import HybridRetriever
from lapua_rag.rerank.reranker import Reranker


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk_id: str
    doc_id: str
    page_no: int
    section_id: str | None
    score: float
    text: str


@dataclass(slots=True)
class SearchService:
    retriever: HybridRetriever
    reranker: Reranker
    top_k_final: int = 5

    def search(
        self,
        *,
        query: str,
        tenant: str,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        fused = self.retriever.retrieve(query=query, tenant=tenant, filters=filters)
        if not fused:
            return []

        chunk_ids = [cid for cid, _ in fused]
        texts = _load_chunk_texts(chunk_ids)
        candidates = [(cid, texts[cid]) for cid in chunk_ids if cid in texts]
        reranked = self.reranker.rerank(
            query=query,
            candidates=candidates,
            top_k=self.top_k_final,
        )

        rows = _load_chunk_facts([cid for cid, _ in reranked])
        results: list[RetrievalResult] = []
        for chunk_id, score in reranked:
            row = rows.get(chunk_id)
            if row is None:
                continue
            results.append(
                RetrievalResult(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    page_no=row["page_no"],
                    section_id=row["section_id"],
                    score=score,
                    text=row["text"],
                )
            )
        return results


def _load_chunk_texts(chunk_ids: list[str]) -> dict[str, str]:
    if not chunk_ids:
        return {}
    with session_scope() as db:
        stmt = select(ChunkRow.chunk_id, ChunkRow.text).where(ChunkRow.chunk_id.in_(chunk_ids))  # type: ignore[attr-defined]
        return {row[0]: row[1] for row in db.exec(stmt)}


def _load_chunk_facts(chunk_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Materialise chunk columns as plain dicts to survive session close."""
    if not chunk_ids:
        return {}
    with session_scope() as db:
        stmt = select(
            ChunkRow.chunk_id,
            ChunkRow.doc_id,
            ChunkRow.page_no,
            ChunkRow.section_id,
            ChunkRow.text,
        ).where(ChunkRow.chunk_id.in_(chunk_ids))  # type: ignore[attr-defined]
        return {
            row[0]: {
                "chunk_id": row[0],
                "doc_id": row[1],
                "page_no": row[2],
                "section_id": row[3],
                "text": row[4],
            }
            for row in db.exec(stmt)
        }
