from __future__ import annotations

from lapua_rag.ocr.fallback import should_fallback_to_vl


def test_fallback_when_low_average_confidence() -> None:
    assert should_fallback_to_vl(
        ocr_confidence_avg=0.4,
        low_confidence_pages=(),
        page_count=10,
    ) is True


def test_fallback_when_too_many_low_conf_pages() -> None:
    assert should_fallback_to_vl(
        ocr_confidence_avg=0.9,
        low_confidence_pages=(0, 1, 2, 3),
        page_count=10,
        low_ratio=0.3,
    ) is True


def test_no_fallback_for_healthy_doc() -> None:
    assert should_fallback_to_vl(
        ocr_confidence_avg=0.95,
        low_confidence_pages=(),
        page_count=10,
    ) is False


def test_empty_document_does_not_fallback() -> None:
    assert should_fallback_to_vl(
        ocr_confidence_avg=0.0,
        low_confidence_pages=(),
        page_count=0,
    ) is False
