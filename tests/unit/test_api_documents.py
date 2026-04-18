"""Tests for /v1/documents/{doc_id}/source and /v1/chunks/{chunk_id}.

We exercise the routes through a FastAPI app wired against an in-memory
SQLite database with a single fixture document + chunk + source PDF on
disk. No embedder/LLM is constructed — the heavy AnswerService factory
is not touched here.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

from lapua_rag.api.routes import documents as documents_routes
from lapua_rag.db import session as session_module
from lapua_rag.db.schema import ChunkRow, DocumentRow

_FAKE_PDF_BYTES = b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n%test\n%%EOF\n"


@pytest.fixture
def app_with_fixture_doc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """Spin up a FastAPI app with only the documents router mounted and an
    isolated in-memory DB containing one document + one chunk + a real PDF
    on disk. We swap ``get_engine()`` so the route's real ``session_scope``
    talks to our throwaway engine without needing to mock the context
    manager itself."""
    # File-backed SQLite (not :memory:) so multiple Session instances from
    # the route share the same tables — the in-memory variant gives each
    # connection its own private DB and the route's session can't see the
    # rows the fixture inserted.
    db_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)

    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(_FAKE_PDF_BYTES)

    with Session(engine) as db:
        db.add(
            DocumentRow(
                doc_id="abcd1234",
                tenant="lapua",
                source_path=str(pdf_path),
                sha256="0" * 64,
                doc_type="poytakirja",
                page_count=1,
                status="indexed",
            )
        )
        db.add(
            ChunkRow(
                chunk_id="chunk_xyz",
                doc_id="abcd1234",
                tenant="lapua",
                page_no=3,
                section_id="§ 12",
                section_title="Talousarviomuutos",
                doc_type="poytakirja",
                text="Kaupunginhallitus päätti hyväksyä talousarvion muutokset." * 5,
                token_count=42,
            )
        )
        db.commit()

    def _override_get_engine() -> Engine:
        return engine

    monkeypatch.setattr(session_module, "get_engine", _override_get_engine)

    app = FastAPI()
    app.include_router(documents_routes.router, prefix="/v1")
    yield TestClient(app)


def test_get_document_source_streams_pdf(app_with_fixture_doc: TestClient) -> None:
    resp = app_with_fixture_doc.get("/v1/documents/abcd1234/source")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == "application/pdf"
    assert "inline" in resp.headers["content-disposition"]
    assert resp.content == _FAKE_PDF_BYTES


def test_get_document_source_404_when_doc_id_unknown(
    app_with_fixture_doc: TestClient,
) -> None:
    resp = app_with_fixture_doc.get("/v1/documents/does-not-exist/source")
    assert resp.status_code == 404


def test_get_document_source_404_when_pdf_missing_on_disk(
    app_with_fixture_doc: TestClient,
    tmp_path: Path,
) -> None:
    """Index-disk drift: row points at a path that no longer exists."""
    (tmp_path / "source.pdf").unlink()
    resp = app_with_fixture_doc.get("/v1/documents/abcd1234/source")
    assert resp.status_code == 404
    assert "source.pdf" in resp.json()["detail"]


def test_get_chunk_returns_full_text(app_with_fixture_doc: TestClient) -> None:
    resp = app_with_fixture_doc.get("/v1/chunks/chunk_xyz")

    assert resp.status_code == 200
    body = resp.json()
    assert body["chunk_id"] == "chunk_xyz"
    assert body["doc_id"] == "abcd1234"
    assert body["page_no"] == 3
    assert body["section_id"] == "§ 12"
    assert body["section_title"] == "Talousarviomuutos"
    assert body["text"].startswith("Kaupunginhallitus päätti")
    assert body["token_count"] == 42


def test_get_chunk_404_when_chunk_id_unknown(app_with_fixture_doc: TestClient) -> None:
    resp = app_with_fixture_doc.get("/v1/chunks/missing")
    assert resp.status_code == 404
