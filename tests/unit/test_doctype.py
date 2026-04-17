from __future__ import annotations

from lapua_rag.models.document import DocumentType
from lapua_rag.postprocess.doctype import detect_doc_type


def test_detects_osavuosikatsaus() -> None:
    text = "Talouden toteutumaraportti 1.1.-31.3.2025 osavuosikatsaus"
    assert detect_doc_type(text) == DocumentType.OSAVUOSIKATSAUS


def test_detects_poytakirja() -> None:
    text = "Kaupunginhallituksen pöytäkirja 3/2024\nPuheenjohtaja: ...\nLäsnäolijat: ..."
    assert detect_doc_type(text) == DocumentType.POYTAKIRJA


def test_detects_tilinpaatos() -> None:
    text = "Tilinpäätös 2024\nTuloslaskelma\nTase"
    assert detect_doc_type(text) == DocumentType.TILINPAATOS


def test_unknown_defaults_to_muu() -> None:
    assert detect_doc_type("random text") == DocumentType.MUU
