"""Prometheus metrics for the Lapua-RAG service.

Surface:

* ``/metrics`` — Prometheus text-format exposition (APIRouter below).
* Counters / histograms / gauges — module-level instruments the service
  code imports and updates in-place.

Design notes:

* Instruments live on a dedicated :class:`CollectorRegistry` so tests
  can tear them down between runs without globally corrupting the
  default registry (prometheus_client raises ``Duplicated`` otherwise
  if tests re-import).
* The API router is assembled in ``build_metrics_router`` so callers
  can choose to mount metrics under a registry they own (e.g. test
  fixtures). :data:`router` is the default-registry instance that
  ``create_app`` mounts.
"""

from __future__ import annotations

from typing import Final

from fastapi import APIRouter
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

# Bucket edges for retrieve top-1 reranker score. BGE reranker-v2-m3
# emits sigmoid-ish scores with the practical 0..1.5 range observed in
# production. Buckets follow the closed-book threshold (0.10) so we can
# visualise abstain vs. answered in Grafana.
_RETRIEVE_SCORE_BUCKETS: Final[tuple[float, ...]] = (
    0.0,
    0.05,
    0.1,
    0.3,
    0.5,
    0.7,
    0.9,
    1.0,
    1.2,
    1.5,
)

# Ingest stage latency varies wildly (OCR minutes, embed seconds); use a
# log-ish bucket set from 0.1s to 30 minutes.
_DURATION_BUCKETS: Final[tuple[float, ...]] = (
    0.1,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    1800.0,
)


def build_registry() -> CollectorRegistry:
    """Return a fresh empty registry. Primarily for tests."""
    return CollectorRegistry()


def build_instruments(registry: CollectorRegistry) -> MetricInstruments:
    """Register all instruments against ``registry`` and return them."""
    return MetricInstruments(
        ingest_total=Counter(
            "lapua_rag_ingest_total",
            "Indeksoitujen dokumenttien kokonaismäärä",
            labelnames=("tenant", "doc_type", "status"),
            registry=registry,
        ),
        ingest_duration=Histogram(
            "lapua_rag_ingest_duration_seconds",
            "Ingestoinnin kesto sekunteina",
            labelnames=("tenant", "stage"),
            buckets=_DURATION_BUCKETS,
            registry=registry,
        ),
        query_total=Counter(
            "lapua_rag_query_total",
            "Kyselyiden kokonaismäärä",
            labelnames=("tenant", "mode", "abstained"),
            registry=registry,
        ),
        query_duration=Histogram(
            "lapua_rag_query_duration_seconds",
            "Kyselyn kesto sekunteina",
            labelnames=("tenant", "mode"),
            buckets=_DURATION_BUCKETS,
            registry=registry,
        ),
        retrieve_score=Histogram(
            "lapua_rag_retrieve_top_score",
            "Rerankerin top-1 pisteytys per kysely",
            buckets=_RETRIEVE_SCORE_BUCKETS,
            registry=registry,
        ),
        corpus_documents=Gauge(
            "lapua_rag_corpus_documents_total",
            "Indeksoitujen dokumenttien määrä",
            labelnames=("tenant",),
            registry=registry,
        ),
        corpus_chunks=Gauge(
            "lapua_rag_corpus_chunks_total",
            "Indeksoitujen chunkkien määrä",
            labelnames=("tenant",),
            registry=registry,
        ),
    )


class MetricInstruments:
    """Typed holder for the set of instruments we expose.

    Instances are created via :func:`build_instruments` and passed to
    recording helpers. Exposed as a class (not a NamedTuple) so FastAPI
    dependency injection can carry it when we later move off module-
    level globals.
    """

    __slots__ = (
        "corpus_chunks",
        "corpus_documents",
        "ingest_duration",
        "ingest_total",
        "query_duration",
        "query_total",
        "retrieve_score",
    )

    def __init__(
        self,
        *,
        ingest_total: Counter,
        ingest_duration: Histogram,
        query_total: Counter,
        query_duration: Histogram,
        retrieve_score: Histogram,
        corpus_documents: Gauge,
        corpus_chunks: Gauge,
    ) -> None:
        self.ingest_total = ingest_total
        self.ingest_duration = ingest_duration
        self.query_total = query_total
        self.query_duration = query_duration
        self.retrieve_score = retrieve_score
        self.corpus_documents = corpus_documents
        self.corpus_chunks = corpus_chunks

    def record_ingest(
        self,
        *,
        tenant: str,
        doc_type: str,
        status: str,
        duration_s: float,
        stage: str = "pipeline",
    ) -> None:
        """Count one ingest outcome and record its duration."""
        self.ingest_total.labels(tenant=tenant, doc_type=doc_type, status=status).inc()
        self.ingest_duration.labels(tenant=tenant, stage=stage).observe(duration_s)

    def record_query(
        self,
        *,
        tenant: str,
        mode: str,
        abstained: bool,
        duration_s: float,
        top_score: float | None,
    ) -> None:
        """Count one query outcome and record duration / reranker score."""
        self.query_total.labels(
            tenant=tenant, mode=mode, abstained=str(abstained).lower()
        ).inc()
        self.query_duration.labels(tenant=tenant, mode=mode).observe(duration_s)
        if top_score is not None:
            self.retrieve_score.observe(top_score)

    def set_corpus_size(self, *, tenant: str, documents: int, chunks: int) -> None:
        """Update corpus gauges; call this from systeemi.stats producers."""
        self.corpus_documents.labels(tenant=tenant).set(documents)
        self.corpus_chunks.labels(tenant=tenant).set(chunks)


# Process-wide default registry + instrument set. We deliberately use
# a dedicated registry (not ``REGISTRY``) so importing this module
# doesn't pollute the global default — that matters when tests or the
# MCP sidecar also instantiate metrics.
_DEFAULT_REGISTRY: Final[CollectorRegistry] = build_registry()
_instruments: MetricInstruments = build_instruments(_DEFAULT_REGISTRY)


def get_instruments() -> MetricInstruments:
    """Return the process-wide instrument set."""
    return _instruments


def build_metrics_router(registry: CollectorRegistry) -> APIRouter:
    """Return an APIRouter that exposes ``GET /metrics`` for ``registry``."""
    local_router = APIRouter(tags=["observability"])

    @local_router.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    return local_router


# Default router used by ``app.py``; tests can mount their own via
# :func:`build_metrics_router` against a throwaway registry.
router: Final[APIRouter] = build_metrics_router(_DEFAULT_REGISTRY)
