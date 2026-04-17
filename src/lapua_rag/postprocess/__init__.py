"""Pure post-processing of PP-StructureV3 output."""

from __future__ import annotations

from lapua_rag.postprocess.chunking import chunk_document, split_by_pykala
from lapua_rag.postprocess.consolidate import consolidate_markdown
from lapua_rag.postprocess.doctype import detect_doc_type
from lapua_rag.postprocess.encoding import fix_finnish_mojibake
from lapua_rag.postprocess.tables import parse_table_html

__all__ = [
    "chunk_document",
    "consolidate_markdown",
    "detect_doc_type",
    "fix_finnish_mojibake",
    "parse_table_html",
    "split_by_pykala",
]
