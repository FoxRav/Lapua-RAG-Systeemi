"""Thin wrapper around PaddleOCR PP-StructureV3.

The pipeline is kept as a stateful class so that model weights are loaded once
per process. Pure parsing of the raw PaddleX output into our domain models
lives in :mod:`lapua_rag.ocr.parse`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from lapua_rag.observability import get_logger
from lapua_rag.storage.layout import DocumentLayout

if TYPE_CHECKING:
    pass  # heavy paddle imports deferred to __init__

_log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class OcrResult:
    doc_id: str
    page_count: int
    layout: DocumentLayout
    ocr_confidence_avg: float
    low_confidence_pages: tuple[int, ...]


class OcrPipeline:
    """Adapter around PP-StructureV3.

    Keep one instance per process. Safe to share across documents; not safe to
    share across threads (paddle's GIL + GPU state).
    """

    def __init__(self, *, device: str = "gpu:0") -> None:
        self._device = device
        self._pipeline: object | None = None

    def _ensure_loaded(self) -> object:
        if self._pipeline is not None:
            return self._pipeline
        from paddleocr import PPStructureV3  # imported lazily; heavy

        _log.info("ocr.model_load", device=self._device)
        self._pipeline = PPStructureV3(device=self._device)
        return self._pipeline

    def run(
        self,
        *,
        doc_id: str,
        pdf_path: Path,
        layout: DocumentLayout,
    ) -> OcrResult:
        """Run PP-StructureV3 on a single PDF and persist per-page artefacts."""
        pipeline = self._ensure_loaded()

        confidences: list[float] = []
        low_conf: list[int] = []
        page_count = 0

        pdf_stem = pdf_path.stem  # filename paddle uses as output prefix
        for page_index, page_result in enumerate(pipeline.predict(str(pdf_path))):  # type: ignore[attr-defined]
            page_result.save_to_markdown(str(layout.pages_dir))
            page_result.save_to_json(str(layout.pages_dir))
            _canonicalise_page_files(
                pages_dir=layout.pages_dir,
                pdf_stem=pdf_stem,
                page_index=page_index,
                layout=layout,
            )
            avg_conf = _extract_avg_confidence(page_result)
            if avg_conf == 0.0:
                avg_conf = _avg_confidence_from_file(layout.page_res_json(page_index))
            confidences.append(avg_conf)
            if avg_conf < 0.6:
                low_conf.append(page_index)
            page_count += 1

        avg = sum(confidences) / len(confidences) if confidences else 0.0
        _log.info(
            "ocr.completed",
            doc_id=doc_id,
            page_count=page_count,
            ocr_confidence_avg=avg,
            low_confidence_pages=low_conf,
        )
        return OcrResult(
            doc_id=doc_id,
            page_count=page_count,
            layout=layout,
            ocr_confidence_avg=avg,
            low_confidence_pages=tuple(low_conf),
        )


def _extract_avg_confidence(page_result: object) -> float:
    """Pull the average recognition score from a PP-StructureV3 page result."""
    res = getattr(page_result, "res", None) or getattr(page_result, "_res", None)
    if not isinstance(res, dict):
        return 0.0
    scores = res.get("overall_ocr_res", {}).get("rec_scores")
    if scores is None:
        return 0.0
    try:
        total = float(sum(scores))
        return total / len(scores) if len(scores) else 0.0
    except (TypeError, ValueError):
        return 0.0


def _canonicalise_page_files(
    *,
    pages_dir: Path,
    pdf_stem: str,
    page_index: int,
    layout: DocumentLayout,
) -> None:
    """Rename PaddleOCR's ``<stem>_<i>.md`` / ``<stem>_<i>_res.json`` outputs to
    our canonical ``NNN.md`` / ``NNN.res.json`` layout.

    PaddleOCR derives filenames from the input PDF's stem, which makes the
    filesystem layout non-deterministic across different source filenames.
    Rewriting here is idempotent: the second invocation of a rerun finds the
    canonical names already in place and is a no-op.
    """
    legacy_md = pages_dir / f"{pdf_stem}_{page_index}.md"
    legacy_json = pages_dir / f"{pdf_stem}_{page_index}_res.json"
    canonical_md = layout.page_md(page_index)
    canonical_json = layout.page_res_json(page_index)
    if legacy_md.exists() and not canonical_md.exists():
        legacy_md.replace(canonical_md)
    if legacy_json.exists() and not canonical_json.exists():
        legacy_json.replace(canonical_json)


def _avg_confidence_from_file(res_json: Path) -> float:
    """Fallback: read per-page ``rec_scores`` from the saved ``_res.json``.

    Used when the live ``page_result`` object does not expose ``res`` as a
    simple dict (PaddleOCR API drift across versions).
    """
    try:
        data = json.loads(res_json.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return 0.0
    if not isinstance(data, dict):
        return 0.0
    scores: object = None
    # The current PP-StructureV3 schema nests under "res" or "overall_ocr_res"
    # depending on version; we try both paths.
    for path in (("res", "overall_ocr_res", "rec_scores"), ("overall_ocr_res", "rec_scores")):
        cursor: object = data
        ok = True
        for key in path:
            if isinstance(cursor, dict) and key in cursor:
                cursor = cursor[key]
            else:
                ok = False
                break
        if ok and isinstance(cursor, list) and cursor:
            scores = cursor
            break
    if not isinstance(scores, list) or not scores:
        return 0.0
    try:
        return float(sum(scores)) / len(scores)
    except (TypeError, ValueError):
        return 0.0
