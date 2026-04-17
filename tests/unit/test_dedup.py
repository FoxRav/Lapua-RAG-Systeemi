from __future__ import annotations

from pathlib import Path

import pytest

from lapua_rag.ingest.dedup import compute_sha256, doc_id_from_sha256


def test_sha256_stable(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello lapua")
    sha = compute_sha256(p)
    assert len(sha) == 64
    assert sha == compute_sha256(p)


def test_doc_id_from_sha256_truncates() -> None:
    assert len(doc_id_from_sha256("0" * 64)) == 16
    assert doc_id_from_sha256("0" * 64, length=8) == "00000000"


def test_doc_id_rejects_bad_length() -> None:
    with pytest.raises(ValueError):
        doc_id_from_sha256("0" * 64, length=4)
