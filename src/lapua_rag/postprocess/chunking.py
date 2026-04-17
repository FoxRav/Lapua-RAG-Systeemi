"""Chunking strategies for Lapua documents.

The primary strategy for meeting minutes is splitting by ``§ N`` section
markers, falling back to heading-level splits for financial/osavuosi-style
reports, with a sliding-window safety net for outlier chunks.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass

from lapua_rag.models.document import DocumentType

_PYKALA_RE = re.compile(r"(?m)^\s*(?:§\s*(\d+[a-z]?)|(\d+[a-z]?)\s*§)\s*(.*)$")
_HEADING_RE = re.compile(r"(?m)^(#{1,3})\s+(.+)$")
_APPROX_CHARS_PER_TOKEN = 4


@dataclass(frozen=True, slots=True)
class RawChunk:
    section_id: str | None
    section_title: str | None
    text: str
    page_start: int
    page_end: int


def split_by_pykala(text: str) -> list[RawChunk]:
    """Split meeting minutes by ``§ N`` markers."""
    markers = list(_PYKALA_RE.finditer(text))
    if not markers:
        return []
    chunks: list[RawChunk] = []
    for idx, match in enumerate(markers):
        start = match.start()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        section = match.group(1) or match.group(2) or ""
        title = (match.group(3) or "").strip() or None
        body = text[start:end].strip()
        if not body:
            continue
        pages = _pages_in_range(text, start, end)
        chunks.append(
            RawChunk(
                section_id=f"§ {section}" if section else None,
                section_title=title,
                text=body,
                page_start=pages[0],
                page_end=pages[1],
            )
        )
    return chunks


def _split_by_heading(text: str) -> list[RawChunk]:
    markers = list(_HEADING_RE.finditer(text))
    if not markers:
        return []
    chunks: list[RawChunk] = []
    for idx, match in enumerate(markers):
        start = match.start()
        end = markers[idx + 1].start() if idx + 1 < len(markers) else len(text)
        body = text[start:end].strip()
        if not body:
            continue
        pages = _pages_in_range(text, start, end)
        chunks.append(
            RawChunk(
                section_id=None,
                section_title=match.group(2).strip(),
                text=body,
                page_start=pages[0],
                page_end=pages[1],
            )
        )
    return chunks


_PAGE_MARKER_RE = re.compile(r"<!--\s*page:\s*(\d+)\s*-->")


def _pages_in_range(text: str, start: int, end: int) -> tuple[int, int]:
    prefix_matches = list(_PAGE_MARKER_RE.finditer(text, 0, start + 1))
    current_page = int(prefix_matches[-1].group(1)) if prefix_matches else 0
    inside = [int(m.group(1)) for m in _PAGE_MARKER_RE.finditer(text, start, end)]
    if inside:
        return current_page, max(inside)
    return current_page, current_page


def _sliding_window(
    chunk: RawChunk,
    *,
    max_chars: int,
    overlap_chars: int,
) -> list[RawChunk]:
    if len(chunk.text) <= max_chars:
        return [chunk]
    parts: list[RawChunk] = []
    step = max_chars - overlap_chars
    for start in range(0, len(chunk.text), step):
        body = chunk.text[start : start + max_chars]
        if not body.strip():
            continue
        parts.append(
            RawChunk(
                section_id=chunk.section_id,
                section_title=chunk.section_title,
                text=body,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            )
        )
    return parts


def chunk_document(
    *,
    text: str,
    doc_type: DocumentType,
    max_tokens: int = 1500,
    overlap_tokens: int = 150,
) -> list[RawChunk]:
    """Return a list of chunks ordered by section appearance.

    Pure function.
    """
    primary: list[RawChunk] = (
        split_by_pykala(text)
        if doc_type in {DocumentType.POYTAKIRJA, DocumentType.ESITYSLISTA, DocumentType.LAUTAKUNTA}
        else []
    )
    if not primary:
        primary = _split_by_heading(text) or [
            RawChunk(section_id=None, section_title=None, text=text, page_start=0, page_end=0)
        ]

    max_chars = max_tokens * _APPROX_CHARS_PER_TOKEN
    overlap_chars = overlap_tokens * _APPROX_CHARS_PER_TOKEN
    return [
        part
        for chunk in primary
        for part in _sliding_window(chunk, max_chars=max_chars, overlap_chars=overlap_chars)
    ]


def chunk_id(*, doc_id: str, section_id: str | None, index: int) -> str:
    """Deterministic chunk id for idempotent vector upsert."""
    raw = f"{doc_id}:{section_id or ''}:{index}".encode()
    return hashlib.sha1(raw).hexdigest()[:32]


def iter_raw_chunks(chunks: list[RawChunk]) -> Iterator[tuple[int, RawChunk]]:
    return enumerate(chunks)
