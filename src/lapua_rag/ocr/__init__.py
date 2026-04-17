"""OCR + layout extraction via PaddleOCR PP-StructureV3."""

from __future__ import annotations

from lapua_rag.ocr.fallback import should_fallback_to_vl
from lapua_rag.ocr.pipeline import OcrPipeline, OcrResult

__all__ = ["OcrPipeline", "OcrResult", "should_fallback_to_vl"]
