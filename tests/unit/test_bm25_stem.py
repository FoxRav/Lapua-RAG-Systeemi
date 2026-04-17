from __future__ import annotations

from lapua_rag.index.bm25 import stem_finnish


def test_stems_finnish_words() -> None:
    stemmed = stem_finnish("kaupunginhallituksen päätökset")
    assert "kaupunginhallitu" in stemmed
    assert "päätök" in stemmed


def test_empty() -> None:
    assert stem_finnish("") == ""
