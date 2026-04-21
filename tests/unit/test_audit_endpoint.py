"""Integration test for GET /v1/audit.

Uses an in-memory SQLite DB via monkey-patched ``get_engine`` so the
test is hermetic and fast — no live database setup required.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine

from lapua_rag.api.routes import audit as audit_routes
from lapua_rag.db import session as db_session
from lapua_rag.db.schema import AuditLog


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    # In-memory SQLite is per-connection; StaticPool makes the single
    # shared connection survive across session_scope calls in the same
    # process. Without this the route's session cannot see the rows the
    # fixture inserted.
    engine: Engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)

    def _fake_engine() -> Engine:
        return engine

    monkeypatch.setattr(db_session, "get_engine", _fake_engine)
    # get_engine is cached via lru_cache elsewhere; make the patched
    # one visible to callers that imported it directly.
    monkeypatch.setattr("lapua_rag.api.routes.audit.session_scope",
                        db_session.session_scope)

    now = datetime.utcnow()
    with Session(engine) as s:
        s.add(AuditLog(
            ts=now - timedelta(minutes=5),
            tenant="lapua",
            endpoint="/v1/query",
            query_text="Kuka on?",
            mode="extract",
            abstained=False,
            latency_ms=120,
        ))
        s.add(AuditLog(
            ts=now,
            tenant="lapua",
            endpoint="/v1/aggregate",
            query_text="Kuinka monta?",
            mode="count",
            abstained=False,
            latency_ms=30,
        ))
        s.add(AuditLog(
            ts=now - timedelta(minutes=1),
            tenant="toinen",
            endpoint="/v1/query",
            query_text="Ei tätä.",
            mode="extract",
            abstained=True,
            abstain_reason="no_context",
            latency_ms=50,
        ))
        s.commit()

    app = FastAPI()
    app.include_router(audit_routes.router, prefix="/v1")
    yield TestClient(app)


def test_returns_rows_newest_first(client: TestClient) -> None:
    resp = client.get("/v1/audit?limit=10")
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 3
    # Newest row = /v1/aggregate (ts=now).
    assert rows[0]["endpoint"] == "/v1/aggregate"
    assert rows[-1]["endpoint"] == "/v1/query"


def test_tenant_filter(client: TestClient) -> None:
    resp = client.get("/v1/audit", params={"tenant": "lapua"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 2
    assert all(r["tenant"] == "lapua" for r in rows)


def test_limit_clamps(client: TestClient) -> None:
    resp = client.get("/v1/audit", params={"limit": 1})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1


def test_limit_upper_bound_enforced(client: TestClient) -> None:
    # le=500 in the route signature → 422 for values over the ceiling.
    resp = client.get("/v1/audit", params={"limit": 10_000})
    assert resp.status_code == 422
