"""Hybrid retrieval: RRF fusion of dense + sparse results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lapua_rag.embed.embedder import Embedder
from lapua_rag.index.bm25 import BM25Index
from lapua_rag.index.qdrant import QdrantIndex


def rrf_fuse(
    *,
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """Reciprocal Rank Fusion – simple, robust, parameter-light.

    Pure function.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (chunk_id, _score) in enumerate(ranking):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda pair: pair[1], reverse=True)


@dataclass(slots=True)
class HybridRetriever:
    embedder: Embedder
    qdrant: QdrantIndex
    bm25: BM25Index
    top_k_dense: int = 30
    top_k_sparse: int = 30
    top_k_fused: int = 30

    def retrieve(
        self,
        *,
        query: str,
        tenant: str,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        vector = self.embedder.embed_query(query)
        dense = self.qdrant.search(
            vector=vector,
            top_k=self.top_k_dense,
            tenant=tenant,
            filters=filters,
        )
        dense_ranked = [(cid, score) for cid, score, _ in dense]
        sparse = self.bm25.search(query=query, tenant=tenant, top_k=self.top_k_sparse)
        fused = rrf_fuse(rankings=[dense_ranked, sparse])
        return fused[: self.top_k_fused]
