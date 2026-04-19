"""Retrieval service: hybrid fetch → text loading → rerank → chunk-type boost."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from sqlmodel import select

from lapua_rag.db.schema import ChunkRow
from lapua_rag.db.session import session_scope
from lapua_rag.index.hybrid import HybridRetriever
from lapua_rag.observability import get_logger
from lapua_rag.rerank.reranker import Reranker

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    chunk_id: str
    doc_id: str
    page_no: int
    section_id: str | None
    score: float
    text: str


# Decision-bearing chunks contain a "## Päätös" header and a selection verb.
# These actually answer "kuka on / valittiin / nimettiin"-style questions
# and should outrank attendance-list chunks that merely enumerate names.
# Patterns are anchored on chunk-start to avoid matching incidental mentions
# deep in a long minutes block.
_DECISION_HEADER_RE = re.compile(
    r"##\s*P[äa]?[äa]t[öo]s",
    flags=re.IGNORECASE,
)
_SELECTION_VERB_RE = re.compile(
    r"\b(valits[ie]|valittiin|nime(?:si|ttiin|tt[äa]v[äa])|"
    r"esitt[äa][äa]|m[äa][äa]r[äa]si|hyv[äa]ksyi)\b",
    flags=re.IGNORECASE,
)
# Attendance-list chunks repeat "## Saapuvillaolleet jäsenet" / "## Osallistujat"
# verbatim across every meeting; they score high on any "puheenjohtaja"-flavoured
# query because the role names appear, but they almost never answer the actual
# question (who currently holds the role). Penalise them at the boost stage.
_ATTENDANCE_HEADER_RE = re.compile(
    r"^\s*##\s*(Saapuvilla\s*olleet|Osallistujat)",
    flags=re.IGNORECASE | re.MULTILINE,
)

_DECISION_BOOST: float = 0.20
_ATTENDANCE_PENALTY: float = -0.15


def _chunk_type_boost(text: str) -> float:
    """Return an additive score adjustment for chunk ``text``.

    Pure function: deterministic, side-effect free, easily unit-testable.
    The exact magnitudes are tuned against the v0.6.1 smoke set (Q1 attendance
    chunk scored 0.991 vs decision chunk ~0.85; +0.20 / -0.15 closes the
    gap without inverting other rankings).
    """
    delta = 0.0
    if _DECISION_HEADER_RE.search(text) and _SELECTION_VERB_RE.search(text):
        delta += _DECISION_BOOST
    if _ATTENDANCE_HEADER_RE.search(text):
        delta += _ATTENDANCE_PENALTY
    return delta


@dataclass(slots=True)
class SearchService:
    retriever: HybridRetriever
    reranker: Reranker
    # v0.6.2: bumped from 5 → 8 so decision chunks that BGE rerankers occasionally
    # rank below attendance-lists still enter the post-rerank boost pool.
    top_k_final: int = 8
    # Toggle the chunk-type boost (mainly so unit tests can pin behaviour).
    apply_chunk_type_boost: bool = True

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
            adjusted = score
            if self.apply_chunk_type_boost:
                delta = _chunk_type_boost(row["text"])
                if delta != 0.0:
                    _log.info(
                        "rerank.chunk_type_boost",
                        chunk_id=chunk_id,
                        base_score=score,
                        delta=delta,
                    )
                    adjusted = score + delta
            results.append(
                RetrievalResult(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    page_no=row["page_no"],
                    section_id=row["section_id"],
                    score=adjusted,
                    text=row["text"],
                )
            )
        # Re-sort after boost so chunk #1 is genuinely the new top-1.
        results.sort(key=lambda r: r.score, reverse=True)
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
