from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from lapua_rag.config import get_settings
from lapua_rag.db import session as session_mod
from lapua_rag.db.schema import ChunkRow, DocumentRow
from lapua_rag.db.session import create_all, session_scope
from lapua_rag.systeemi.coverage import compute_coverage
from lapua_rag.systeemi.stats import gather_stats
from lapua_rag.systeemi.versioning import ModelFingerprint, compute_system_version


@pytest.fixture
def isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the whole stack at a throwaway SQLite in tmp_path."""
    db_file = tmp_path / "metadata.sqlite"
    monkeypatch.setenv("LAPUA_DATABASE_URL", f"sqlite:///{db_file.as_posix()}")
    monkeypatch.setenv("LAPUA_STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setenv("LAPUA_INDEX_DIR", str(tmp_path / "index"))
    monkeypatch.setenv("LAPUA_INBOX_DIR", str(tmp_path / "inbox"))
    get_settings.cache_clear()  # type: ignore[attr-defined]
    session_mod.get_engine.cache_clear()  # type: ignore[attr-defined]
    create_all()


def _seed(
    docs: list[dict[str, object]],
    chunks: list[dict[str, object]] | None = None,
) -> None:
    with session_scope() as db:
        for d in docs:
            db.add(DocumentRow(**d))  # type: ignore[arg-type]
        for c in chunks or []:
            db.add(ChunkRow(**c))  # type: ignore[arg-type]


def test_gather_stats_empty(isolated_db: None) -> None:
    stats = gather_stats()
    assert stats.tenant_count == 0
    assert stats.document_count == 0
    assert stats.per_tenant == []


def test_gather_stats_aggregates_per_tenant(isolated_db: None) -> None:
    _seed(
        docs=[
            {
                "doc_id": "a1" * 8,
                "tenant": "lapua",
                "source_path": "/x.pdf",
                "sha256": "a" * 64,
                "doc_type": "poytakirja",
                "status": "indexed",
                "page_count": 10,
                "indexed_at": datetime(2026, 4, 10),
            },
            {
                "doc_id": "b2" * 8,
                "tenant": "lapua",
                "source_path": "/y.pdf",
                "sha256": "b" * 64,
                "doc_type": "osavuosikatsaus",
                "status": "indexed",
                "page_count": 45,
                "indexed_at": datetime(2026, 4, 17),
            },
            {
                "doc_id": "c3" * 8,
                "tenant": "demo",
                "source_path": "/z.pdf",
                "sha256": "c" * 64,
                "doc_type": "poytakirja",
                "status": "failed",
                "page_count": 0,
            },
        ],
        chunks=[
            {
                "chunk_id": "ch1",
                "doc_id": "a1" * 8,
                "tenant": "lapua",
                "page_no": 0,
                "doc_type": "poytakirja",
                "text": "teksti",
                "token_count": 100,
            },
            {
                "chunk_id": "ch2",
                "doc_id": "b2" * 8,
                "tenant": "lapua",
                "page_no": 0,
                "doc_type": "osavuosikatsaus",
                "text": "teksti2",
                "token_count": 250,
            },
        ],
    )

    stats = gather_stats()
    assert stats.tenant_count == 2
    assert stats.document_count == 3
    assert stats.indexed_count == 2
    assert stats.failed_count == 1
    assert stats.chunk_count == 2
    assert stats.token_count == 350

    lapua = next(t for t in stats.per_tenant if t.tenant == "lapua")
    assert lapua.document_count == 2
    assert lapua.page_count == 55
    assert lapua.first_indexed == datetime(2026, 4, 10)
    assert lapua.last_indexed == datetime(2026, 4, 17)
    assert {d.doc_type for d in lapua.doc_types} == {"poytakirja", "osavuosikatsaus"}


def test_system_version_is_deterministic(isolated_db: None) -> None:
    _seed(
        docs=[
            {
                "doc_id": "a1" * 8,
                "tenant": "lapua",
                "source_path": "/x.pdf",
                "sha256": "a" * 64,
                "doc_type": "poytakirja",
                "status": "indexed",
            },
            {
                "doc_id": "b2" * 8,
                "tenant": "lapua",
                "source_path": "/y.pdf",
                "sha256": "b" * 64,
                "doc_type": "osavuosikatsaus",
                "status": "indexed",
            },
        ],
    )

    fp = ModelFingerprint(
        embedder="e5",
        reranker="bge",
        llm_base="qwen",
        llm_lora="lapua-v2",
    )
    v1 = compute_system_version(tenant="lapua", models=fp)
    v2 = compute_system_version(tenant="lapua", models=fp)
    assert v1.content_hash == v2.content_hash
    assert v1.document_count == 2
    assert len(v1.content_hash) == 64


def test_system_version_changes_when_doc_added(isolated_db: None) -> None:
    fp = ModelFingerprint(embedder="e", reranker="r", llm_base="b", llm_lora="l")
    _seed(
        docs=[
            {
                "doc_id": "a1" * 8,
                "tenant": "lapua",
                "source_path": "/x.pdf",
                "sha256": "a" * 64,
                "doc_type": "poytakirja",
                "status": "indexed",
            }
        ],
    )
    before = compute_system_version(models=fp)
    _seed(
        docs=[
            {
                "doc_id": "b2" * 8,
                "tenant": "lapua",
                "source_path": "/y.pdf",
                "sha256": "b" * 64,
                "doc_type": "poytakirja",
                "status": "indexed",
            }
        ],
    )
    after = compute_system_version(models=fp)
    assert before.content_hash != after.content_hash
    assert after.document_count == before.document_count + 1


def test_coverage_splits_by_doc_type_and_status(isolated_db: None) -> None:
    _seed(
        docs=[
            {
                "doc_id": "a" * 16,
                "tenant": "lapua",
                "source_path": "/a.pdf",
                "sha256": "a" * 64,
                "doc_type": "poytakirja",
                "status": "indexed",
            },
            {
                "doc_id": "b" * 16,
                "tenant": "lapua",
                "source_path": "/b.pdf",
                "sha256": "b" * 64,
                "doc_type": "poytakirja",
                "status": "failed",
            },
            {
                "doc_id": "c" * 16,
                "tenant": "lapua",
                "source_path": "/c.pdf",
                "sha256": "c" * 64,
                "doc_type": "tilinpaatos",
                "status": "ocr",
            },
        ],
    )
    coverage = compute_coverage(tenant="lapua")
    by_type = {row.doc_type: row for row in coverage.by_doctype}
    assert by_type["poytakirja"].indexed == 1
    assert by_type["poytakirja"].failed == 1
    assert by_type["tilinpaatos"].in_progress == 1
