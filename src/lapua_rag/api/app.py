"""FastAPI app factory."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from lapua_rag import __version__
from lapua_rag.api.routes import aggregate, audit, documents, ingest, query, system
from lapua_rag.config import get_settings
from lapua_rag.db.session import create_all
from lapua_rag.observability import configure_logging
from lapua_rag.observability.metrics import router as metrics_router


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)
    create_all()

    app = FastAPI(
        title="Lapua-RAG",
        version=__version__,
        description="Local RAG platform for Finnish municipal documents.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(ingest.router, prefix="/v1", tags=["ingest"])
    app.include_router(query.router, prefix="/v1", tags=["query"])
    app.include_router(documents.router, prefix="/v1", tags=["documents"])
    app.include_router(system.router, prefix="/v1", tags=["system"])
    app.include_router(aggregate.router, prefix="/v1", tags=["aggregate"])
    app.include_router(audit.router, prefix="/v1", tags=["audit"])
    app.include_router(metrics_router)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok", "tenant": settings.tenant}

    return app
