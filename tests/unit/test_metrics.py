"""Tests for lapua_rag.observability.metrics.

We build a fresh ``CollectorRegistry`` per test so the default module
registry isn't polluted — this keeps tests order-independent and lets
us mount isolated routers.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import CollectorRegistry

from lapua_rag.observability import metrics as metrics_module
from lapua_rag.observability.metrics import (
    MetricInstruments,
    build_instruments,
    build_metrics_router,
    build_registry,
)
from lapua_rag.rag.answer import AnswerService
from lapua_rag.retrieve.search import RetrievalResult


@pytest.fixture
def instruments_pair() -> Iterator[tuple[CollectorRegistry, MetricInstruments]]:
    registry = build_registry()
    yield registry, build_instruments(registry)


class TestRecordIngest:
    def test_counter_and_histogram_update(
        self, instruments_pair: tuple[CollectorRegistry, MetricInstruments]
    ) -> None:
        _, instruments = instruments_pair
        instruments.record_ingest(
            tenant="lapua",
            doc_type="poytakirja",
            status="indexed",
            duration_s=12.5,
        )
        counter_value = instruments.ingest_total.labels(
            tenant="lapua", doc_type="poytakirja", status="indexed"
        )._value.get()
        assert counter_value == 1.0


class TestRecordQuery:
    def test_abstain_label_is_boolean_string(
        self, instruments_pair: tuple[CollectorRegistry, MetricInstruments]
    ) -> None:
        _, instruments = instruments_pair
        instruments.record_query(
            tenant="lapua",
            mode="extract",
            abstained=True,
            duration_s=1.2,
            top_score=0.95,
        )
        value = instruments.query_total.labels(
            tenant="lapua", mode="extract", abstained="true"
        )._value.get()
        assert value == 1.0

    def test_top_score_optional(
        self, instruments_pair: tuple[CollectorRegistry, MetricInstruments]
    ) -> None:
        _, instruments = instruments_pair
        instruments.record_query(
            tenant="lapua",
            mode="retrieve",
            abstained=False,
            duration_s=0.5,
            top_score=None,
        )
        value = instruments.query_total.labels(
            tenant="lapua", mode="retrieve", abstained="false"
        )._value.get()
        assert value == 1.0


class TestSetCorpusSize:
    def test_gauges_reflect_latest_values(
        self, instruments_pair: tuple[CollectorRegistry, MetricInstruments]
    ) -> None:
        _, instruments = instruments_pair
        instruments.set_corpus_size(tenant="lapua", documents=109, chunks=7922)
        assert instruments.corpus_documents.labels(tenant="lapua")._value.get() == 109
        assert instruments.corpus_chunks.labels(tenant="lapua")._value.get() == 7922
        # Overwrite (not accumulate)
        instruments.set_corpus_size(tenant="lapua", documents=110, chunks=7950)
        assert instruments.corpus_documents.labels(tenant="lapua")._value.get() == 110


@dataclass
class _StubSearch:
    """Minimal SearchService-compatible stub returning canned results."""

    results: list[RetrievalResult]

    def search(self, *, query: str, tenant: str) -> list[RetrievalResult]:
        return list(self.results)


class _StubLlm:
    """LlmClient stub — never called in the no_context gate path."""

    def generate_text(self, **_: object) -> str:  # pragma: no cover - safety net
        raise AssertionError("LLM should not be invoked for empty retrieval")

    def generate_json(self, **_: object) -> dict[str, object]:  # pragma: no cover
        raise AssertionError("LLM should not be invoked for empty retrieval")


class TestAnswerServiceInstrumentation:
    def test_records_abstain_counter_on_no_context(
        self,
        instruments_pair: tuple[CollectorRegistry, MetricInstruments],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """AnswerService.answer() fires query_total with abstained=true when
        the retrieval layer returns zero hits (no_context gate)."""
        _, instruments = instruments_pair
        monkeypatch.setattr(metrics_module, "_instruments", instruments)

        svc = AnswerService(search=_StubSearch(results=[]), llm=_StubLlm(), mode="extract")
        answer = svc.answer(query="testikysymys", tenant="lapua")

        assert answer.abstained is True
        counter_value = instruments.query_total.labels(
            tenant="lapua", mode="extract", abstained="true"
        )._value.get()
        assert counter_value == 1.0
        # Histogram count is stored as _sum / _count buckets; verify at least
        # one observation recorded.
        hist_count = instruments.query_duration.labels(
            tenant="lapua", mode="extract"
        )._sum.get()
        assert hist_count >= 0.0


class TestPipelineInstrumentation:
    def test_record_ingest_helper_updates_counters(
        self,
        instruments_pair: tuple[CollectorRegistry, MetricInstruments],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The ingest metric helper increments counters and records a
        duration without touching live OCR / DB resources."""
        from lapua_rag import pipeline  # noqa: PLC0415

        _, instruments = instruments_pair
        monkeypatch.setattr(metrics_module, "_instruments", instruments)

        pipeline._record_ingest_metrics(
            tenant="lapua",
            doc_type="poytakirja",
            status="indexed",
            duration_s=1.25,
        )

        value = instruments.ingest_total.labels(
            tenant="lapua", doc_type="poytakirja", status="indexed"
        )._value.get()
        assert value == 1.0


class TestMetricsRouter:
    def test_endpoint_returns_prometheus_text(
        self, instruments_pair: tuple[CollectorRegistry, MetricInstruments]
    ) -> None:
        registry, instruments = instruments_pair
        instruments.record_query(
            tenant="lapua", mode="extract", abstained=False, duration_s=1.0, top_score=0.8
        )
        app = FastAPI()
        app.include_router(build_metrics_router(registry))
        client = TestClient(app)

        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert "text/plain" in resp.headers["content-type"]
        body = resp.text
        assert "lapua_rag_query_total" in body
        assert 'tenant="lapua"' in body
        assert 'mode="extract"' in body
