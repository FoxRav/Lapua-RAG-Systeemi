"""Unit tests for lapua_rag.audit.service.

The service must:
 * Persist one row per call (via session_scope).
 * Emit a structlog event with the expected fields.
 * Never raise — DB failures are logged and swallowed (fire-and-forget).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import cast
from unittest.mock import MagicMock

import pytest

from lapua_rag.audit import service as audit_service
from lapua_rag.db.schema import AuditLog
from lapua_rag.rag.answer import RagAnswer, RagSource


def _dummy_answer(*, abstained: bool = False) -> RagAnswer:
    sources = (
        []
        if abstained
        else [
            RagSource(
                chunk_id="c1",
                doc_id="doc-A",
                page_no=3,
                section_id="§12",
                snippet="Esimerkkilainaus.",
            )
        ]
    )
    return RagAnswer(
        johtopaatos="Testi",
        perustelut="Testi",
        lahteet=sources,
        abstained=abstained,
        abstain_reason="no_context" if abstained else None,
        max_source_score=0.9,
    )


class _FakeSession:
    """Captures adds without touching a real DB engine."""

    def __init__(self) -> None:
        self.added: list[AuditLog] = []

    def add(self, row: AuditLog) -> None:
        self.added.append(row)


@pytest.fixture
def fake_session(monkeypatch: pytest.MonkeyPatch) -> Iterator[_FakeSession]:
    session = _FakeSession()

    @contextmanager
    def _fake_scope() -> Iterator[_FakeSession]:
        yield session

    monkeypatch.setattr(audit_service, "session_scope", _fake_scope)
    yield session


class TestLogEntry:
    def test_success_persists_one_row(self, fake_session: _FakeSession) -> None:
        audit_service.log_entry(
            tenant="lapua",
            endpoint="/v1/query",
            query_text="Kuka on kaupunginjohtaja?",
            mode="extract",
            abstained=False,
            abstain_reason=None,
            max_source_score=0.87,
            source_doc_ids=["doc-A", "doc-B"],
            latency_ms=250,
            client_ip="127.0.0.1",
            user_agent="pytest",
        )
        assert len(fake_session.added) == 1
        row = fake_session.added[0]
        assert row.tenant == "lapua"
        assert row.endpoint == "/v1/query"
        assert row.abstained is False
        assert row.latency_ms == 250
        assert row.source_doc_ids is not None
        assert json.loads(row.source_doc_ids) == ["doc-A", "doc-B"]

    def test_abstain_fields_recorded(self, fake_session: _FakeSession) -> None:
        audit_service.log_entry(
            tenant="lapua",
            endpoint="/v1/query",
            query_text="Off-topic?",
            mode="extract",
            abstained=True,
            abstain_reason="no_context",
            max_source_score=None,
            source_doc_ids=[],
            latency_ms=10,
        )
        assert len(fake_session.added) == 1
        row = fake_session.added[0]
        assert row.abstained is True
        assert row.abstain_reason == "no_context"
        assert row.source_doc_ids == "[]"

    def test_db_failure_is_swallowed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A broken session_scope must not propagate — the caller is the
        HTTP response path and cannot be blocked by an audit hiccup."""

        @contextmanager
        def _broken_scope() -> Iterator[object]:
            raise RuntimeError("db down")
            yield  # pragma: no cover - unreachable

        monkeypatch.setattr(audit_service, "session_scope", _broken_scope)
        audit_service.log_entry(
            tenant="lapua",
            endpoint="/v1/query",
            query_text="Jotain",
            latency_ms=5,
        )


class TestLogQuery:
    def test_delegates_with_extracted_fields(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """log_query is a thin adapter over log_entry — verify it maps
        RagAnswer fields onto the primitive log_entry signature."""
        captured: dict[str, object] = {}

        def _fake_entry(**kwargs: object) -> None:
            captured.update(kwargs)

        monkeypatch.setattr(audit_service, "log_entry", _fake_entry)
        answer = _dummy_answer()
        audit_service.log_query(
            tenant="lapua",
            endpoint="/v1/query",
            query_text="Kuka on?",
            answer=answer,
            mode="extract",
            latency_ms=120,
        )
        assert captured["tenant"] == "lapua"
        assert captured["endpoint"] == "/v1/query"
        assert captured["abstained"] is False
        assert captured["source_doc_ids"] == ["doc-A"]
        assert captured["max_source_score"] == pytest.approx(0.9)
        assert captured["mode"] == "extract"

    def test_abstained_answer_has_empty_source_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, object] = {}
        monkeypatch.setattr(
            audit_service,
            "log_entry",
            lambda **kw: captured.update(kw),
        )
        audit_service.log_query(
            tenant="lapua",
            endpoint="/v1/query",
            query_text="Off-topic?",
            answer=_dummy_answer(abstained=True),
        )
        assert captured["abstained"] is True
        ids = cast(list[str], captured["source_doc_ids"])
        assert ids == []


class TestSessionInteraction:
    def test_add_called_with_correct_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression: the mock-session contract that higher-level
        integration code relies on stays stable."""
        session = MagicMock()

        @contextmanager
        def _scope() -> Iterator[MagicMock]:
            yield session

        monkeypatch.setattr(audit_service, "session_scope", _scope)
        audit_service.log_entry(
            tenant="lapua",
            endpoint="/v1/aggregate",
            query_text="Kuinka monta?",
            mode="count",
        )
        session.add.assert_called_once()
        row = session.add.call_args.args[0]
        assert isinstance(row, AuditLog)
        assert row.mode == "count"
