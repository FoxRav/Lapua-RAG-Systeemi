"""Finnish mojibake repair.

PP-StructureV3's markdown writer has occasionally emitted cp1252-decoded UTF-8
bytes (``k�sittely`` instead of ``käsittely``). We recover by (a) re-encoding
suspicious strings round-trip and (b) repairing a small set of Finnish-specific
heuristics.
"""

from __future__ import annotations

_REPLACEMENT_CHAR = "\ufffd"


def looks_like_mojibake(text: str) -> bool:
    """Quick heuristic: replacement chars or cp1252→utf-8 patterns."""
    if _REPLACEMENT_CHAR in text:
        return True
    return any(marker in text for marker in ("Ã¤", "Ã¶", "Ã¥", "Â§"))


def fix_finnish_mojibake(text: str) -> str:
    """Best-effort repair of common Finnish encoding issues.

    Pure function: never mutates input.
    """
    if not text or not looks_like_mojibake(text):
        return text

    try:
        recovered = text.encode("cp1252", errors="strict").decode("utf-8", errors="strict")
    except (UnicodeEncodeError, UnicodeDecodeError):
        recovered = text

    return (
        recovered.replace(_REPLACEMENT_CHAR, "")
        .replace("–", "-")
        .replace("—", "-")
    )
