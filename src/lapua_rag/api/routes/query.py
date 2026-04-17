"""POST /v1/query endpoint."""

from __future__ import annotations

from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel, Field

from lapua_rag.config import get_settings
from lapua_rag.embed.embedder import Embedder
from lapua_rag.extract.llm import default_client
from lapua_rag.index.bm25 import BM25Index
from lapua_rag.index.hybrid import HybridRetriever
from lapua_rag.index.qdrant import QdrantIndex
from lapua_rag.rag.answer import AnswerService, RagAnswer
from lapua_rag.rerank.reranker import Reranker
from lapua_rag.retrieve.search import SearchService

router = APIRouter()


class QueryRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1024)
    tenant: str | None = None


@router.post("/query", response_model=RagAnswer)
def query(request: QueryRequest) -> RagAnswer:
    settings = get_settings()
    svc = _answer_service()
    return svc.answer(query=request.query, tenant=request.tenant or settings.tenant)


@lru_cache(maxsize=1)
def _answer_service() -> AnswerService:
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
    return AnswerService(search=search, llm=default_client())
