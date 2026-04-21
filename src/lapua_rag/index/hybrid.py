"""Hybrid retrieval: RRF fusion of dense + sparse results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from lapua_rag.embed.embedder import Embedder
from lapua_rag.index.bm25 import BM25Index
from lapua_rag.index.qdrant import QdrantIndex

# ``lapua_rag.retrieve.query_rewrite`` is imported lazily inside
# ``HybridRetriever.retrieve`` because ``retrieve/__init__`` re-exports
# ``SearchService``, which in turn imports ``HybridRetriever`` — a top-
# level import here would make the module graph circular.


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
    # When True the retriever expands the query via
    # ``retrieve.query_rewrite.rewrite_query``, runs each variant through
    # the hybrid pipeline, and max-pools the resulting rankings. Off by
    # default in tests so the single-query fixtures still work unchanged.
    enable_query_rewrite: bool = True

    def retrieve(
        self,
        *,
        query: str,
        tenant: str,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float]]:
        from lapua_rag.retrieve.query_rewrite import rewrite_query  # noqa: PLC0415

        queries = rewrite_query(query) if self.enable_query_rewrite else [query]

        rankings: list[list[tuple[str, float]]] = []
        for variant in queries:
            vector = self.embedder.embed_query(variant)
            dense = self.qdrant.search(
                vector=vector,
                top_k=self.top_k_dense,
                tenant=tenant,
                filters=filters,
            )
            dense_ranked = [(cid, score) for cid, score, _ in dense]
            sparse = self.bm25.search(
                query=variant,
                tenant=tenant,
                top_k=self.top_k_sparse,
            )
            rankings.append(dense_ranked)
            rankings.append(sparse)

        fused = rrf_fuse(rankings=rankings)
        return fused[: self.top_k_fused]
