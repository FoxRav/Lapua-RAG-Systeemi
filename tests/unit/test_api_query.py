"""Test that POST /v1/query honours per-request `mode` overrides.

We don't construct a real AnswerService here — that would pull in
embedder + reranker + LLM. Instead we monkeypatch _answer_service to
record which mode the route asked for.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from lapua_rag.api.routes import query as query_routes
from lapua_rag.rag.answer import RagAnswer


class _FakeAnswerService:
    def __init__(self, mode: str) -> None:
        self.mode = mode

    def answer(self, *, query: str, tenant: str) -> RagAnswer:
        return RagAnswer(
            johtopaatos=f"mode={self.mode} query={query} tenant={tenant}",
            perustelut="fake",
            lahteet=[],
            abstained=False,
        )


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    requested_modes: list[str] = []

    def _fake_factory(mode: str) -> _FakeAnswerService:
        requested_modes.append(mode)
        return _FakeAnswerService(mode=mode)

    monkeypatch.setattr(query_routes, "_answer_service", _fake_factory)
    app = FastAPI()
    app.include_router(query_routes.router, prefix="/v1")
    client = TestClient(app)
    client.requested_modes = requested_modes  # type: ignore[attr-defined]
    yield client


class _StubSettings:
    """Stand-in for get_settings() used by the query route."""

    answer_mode = "synth"
    tenant = "lapua"
    # v0.9: the route inspects auth_enabled to decide whether the
    # request-body tenant is honoured. Disable it here so the legacy
    # behaviour (body.tenant or settings.tenant) still applies.
    auth_enabled = False


def _stub_settings() -> _StubSettings:
    return _StubSettings()


def test_query_uses_settings_default_when_mode_omitted(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without `mode` in the body the route falls back to Settings.answer_mode."""
    monkeypatch.setattr(query_routes, "get_settings", _stub_settings)

    resp = client.post("/v1/query", json={"query": "Mikä on Lapuassa?"})

    assert resp.status_code == 200
    assert "mode=synth" in resp.json()["johtopaatos"]
    assert client.requested_modes == ["synth"]  # type: ignore[attr-defined]


def test_query_per_request_mode_overrides_default(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Body's `mode` field beats Settings.answer_mode on a single request."""
    monkeypatch.setattr(query_routes, "get_settings", _stub_settings)

    resp = client.post(
        "/v1/query",
        json={"query": "Mikä on Lapuassa?", "mode": "retrieve"},
    )

    assert resp.status_code == 200
    assert "mode=retrieve" in resp.json()["johtopaatos"]
    assert client.requested_modes == ["retrieve"]  # type: ignore[attr-defined]


def test_query_rejects_invalid_mode(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(query_routes, "get_settings", _stub_settings)

    resp = client.post("/v1/query", json={"query": "abc", "mode": "bogus"})

    assert resp.status_code == 422  # pydantic Literal validation
