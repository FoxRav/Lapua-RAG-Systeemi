"""Qdrant client wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

from lapua_rag.observability import get_logger

_log = get_logger(__name__)


@dataclass(slots=True)
class QdrantIndex:
    url: str
    collection: str
    api_key: str | None = None
    _client: QdrantClient | None = field(default=None, repr=False, compare=False)

    @property
    def client(self) -> QdrantClient:
        if self._client is None:
            self._client = QdrantClient(url=self.url, api_key=self.api_key)
        return self._client

    def ensure_collection(self, *, dim: int) -> None:
        collections = {c.name for c in self.client.get_collections().collections}
        if self.collection in collections:
            return
        _log.info("qdrant.create_collection", collection=self.collection, dim=dim)
        self.client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
        for payload_field in ("tenant", "doc_id", "doc_type", "page_no", "section_id"):
            self.client.create_payload_index(
                collection_name=self.collection,
                field_name=payload_field,
                field_schema=qm.PayloadSchemaType.KEYWORD
                if payload_field != "page_no"
                else qm.PayloadSchemaType.INTEGER,
            )

    def upsert(
        self,
        *,
        ids: list[str],
        vectors: list[list[float]],
        payloads: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        self.client.upsert(
            collection_name=self.collection,
            points=[
                qm.PointStruct(id=_uuid_for(i), vector=v, payload=p)
                for i, v, p in zip(ids, vectors, payloads, strict=True)
            ],
        )

    def search(
        self,
        *,
        vector: list[float],
        top_k: int,
        tenant: str,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[str, float, dict[str, Any]]]:
        must: list[qm.FieldCondition] = [
            qm.FieldCondition(key="tenant", match=qm.MatchValue(value=tenant)),
        ]
        for key, value in (filters or {}).items():
            must.append(qm.FieldCondition(key=key, match=qm.MatchValue(value=value)))

        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            query_filter=qm.Filter(must=must),
            with_payload=True,
        )
        return [
            (str(h.payload.get("chunk_id", h.id)), float(h.score), dict(h.payload or {}))
            for h in response.points
        ]


def _uuid_for(chunk_id: str) -> str:
    """Qdrant requires UUID-shaped IDs; derive deterministically from chunk_id."""
    import uuid

    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"lapua-rag:chunk:{chunk_id}"))
