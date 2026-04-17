"""Content-addressable deduplication via SHA-256."""

from __future__ import annotations

import hashlib
from pathlib import Path

_CHUNK_SIZE = 1 << 20  # 1 MiB


def compute_sha256(path: Path) -> str:
    """Stream-hash a file to avoid loading large PDFs fully in memory."""
    hasher = hashlib.sha256()
    with path.open("rb") as fp:
        while chunk := fp.read(_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def doc_id_from_sha256(sha256: str, *, length: int = 16) -> str:
    """Derive a short deterministic doc_id from the file hash."""
    if not 8 <= length <= 64:
        msg = f"doc_id length must be between 8 and 64, got {length}"
        raise ValueError(msg)
    return sha256[:length]
