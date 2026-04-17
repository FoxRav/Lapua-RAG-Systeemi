"""PaddleOCR-VL fallback rules."""

from __future__ import annotations


def should_fallback_to_vl(
    *,
    ocr_confidence_avg: float,
    low_confidence_pages: tuple[int, ...],
    page_count: int,
    threshold: float = 0.6,
    low_ratio: float = 0.2,
) -> bool:
    """Return True when a document warrants re-processing with PaddleOCR-VL.

    Pure function – trivially testable.
    """
    if page_count == 0:
        return False
    if ocr_confidence_avg < threshold:
        return True
    return len(low_confidence_pages) / page_count > low_ratio
