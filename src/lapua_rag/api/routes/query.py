"""POST /v1/query endpoint."""

from __future__ import annotations

import time
from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field

from lapua_rag.api.auth import require_api_key
from lapua_rag.audit.service import log_query
from lapua_rag.config import get_settings
from lapua_rag.embed.embedder import Embedder
from lapua_rag.extract.llm import default_client
from lapua_rag.index.bm25 import BM25Index
from lapua_rag.index.hybrid import HybridRetriever
from lapua_rag.index.qdrant import QdrantIndex
from lapua_rag.rag.answer import AnswerMode, AnswerService, RagAnswer
from lapua_rag.rerank.reranker import Reranker
from lapua_rag.retrieve.search import SearchService

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1024)
    tenant: str | None = None
    # Per-request override of Settings.answer_mode. None → fall back to
    # the configured default. Lets the UI flip synth↔retrieve without a
    # server restart, which is essential while lapua-llm-v3 is in training.
    mode: AnswerMode | None = None


@router.post("/query", response_model=RagAnswer)
def query(
    request: QueryRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    auth_tenant: Annotated[str, Depends(require_api_key)],
) -> RagAnswer:
    settings = get_settings()
    mode: AnswerMode = request.mode or settings.answer_mode
    # When auth is enabled the bound tenant wins; the request-body
    # field is only honoured in dev mode for backwards-compat scripts.
    tenant = auth_tenant if settings.auth_enabled else (request.tenant or auth_tenant)
    svc = _answer_service(mode)

    started = time.perf_counter()
    answer = svc.answer(query=request.query, tenant=tenant)
    latency_ms = int((time.perf_counter() - started) * 1000)

    client = http_request.client
    background_tasks.add_task(
        log_query,
        tenant=tenant,
        endpoint="/v1/query",
        query_text=request.query,
        answer=answer,
        mode=mode,
        latency_ms=latency_ms,
        client_ip=client.host if client is not None else None,
        user_agent=http_request.headers.get("user-agent"),
    )
    return answer


@lru_cache(maxsize=3)
def _answer_service(mode: AnswerMode) -> AnswerService:
    """Build an AnswerService for a given mode. Cached separately per mode
    so we share the heavy embedder/reranker instances across requests but
    still respect per-request mode overrides without rebuilding everything.
    Cache size = 3 to cover synth/retrieve/extract.
    """
    settings = get_settings()
    embedder = Embedder(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
    )
    qdrant = QdrantIndex(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
    )
    bm25 = BM25Index(path=settings.index_dir / "bm25.sqlite")
    retriever = HybridRetriever(embedder=embedder, qdrant=qdrant, bm25=bm25)
    reranker = Reranker(model_name=settings.reranker_model, device=settings.reranker_device)
    search = SearchService(retriever=retriever, reranker=reranker)
    return AnswerService(
        search=search,
        llm=default_client(),
        min_score=settings.answer_min_score,
        max_context_chunks=settings.answer_max_context_chunks,
        max_chars_per_chunk=settings.answer_max_chars_per_chunk,
        mode=mode,
        retrieve_snippet_chars=settings.answer_retrieve_snippet_chars,
    )
