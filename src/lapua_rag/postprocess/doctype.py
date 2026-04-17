"""Lightweight document-type detection by keyword voting.

Deliberately rule-based first; a LLM-based classifier can be plugged in later
through the same function signature.
"""

from __future__ import annotations

from collections import Counter

from lapua_rag.models.document import DocumentType

_KEYWORDS: dict[DocumentType, tuple[str, ...]] = {
    DocumentType.POYTAKIRJA: (
        "pöytäkirja",
        "kokous",
        "puheenjohtaja",
        "läsnäolijat",
        "käsiteltävät asiat",
    ),
    DocumentType.ESITYSLISTA: ("esityslista",),
    DocumentType.OSAVUOSIKATSAUS: (
        "osavuosikatsaus",
        "talouden toteutumaraportti",
        "toteumaraportti",
    ),
    DocumentType.TILINPAATOS: (
        "tilinpäätös",
        "tuloslaskelma",
        "tase",
        "rahoituslaskelma",
    ),
    DocumentType.LAUTAKUNTA: ("lautakunta", "lautakunnan päätös"),
    DocumentType.LIITE: ("liite",),
}


def detect_doc_type(text: str, *, sample_chars: int = 4000) -> DocumentType:
    """Return the best-matching ``DocumentType`` for the given text."""
    if not text:
        return DocumentType.MUU
    sample = text[:sample_chars].lower()
    scores: Counter[DocumentType] = Counter()
    for doctype, kws in _KEYWORDS.items():
        scores[doctype] = sum(sample.count(kw) for kw in kws)
    best, best_score = scores.most_common(1)[0]
    return best if best_score > 0 else DocumentType.MUU
