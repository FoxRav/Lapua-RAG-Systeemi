"""Sparse lexical index backed by SQLite FTS5 with Finnish stemming."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from snowballstemmer import FinnishStemmer

_STEMMER = FinnishStemmer()


def stem_finnish(text: str) -> str:
    """Pure tokenizer+stemmer for Finnish text."""
    tokens = (t.strip().lower() for t in text.split() if t.strip())
    return " ".join(_STEMMER.stemWord(t) for t in tokens)


def _fts5_match_expr(stemmed: str) -> str:
    """Escape a stemmed query into a safe FTS5 MATCH expression.

    FTS5 treats bare reserved words (AND, OR, NOT, NEAR) and punctuation as
    operators; wrapping each token in double quotes forces literal matching.
    Embedded double quotes are doubled per SQLite's FTS5 string escape rules.
    """
    parts = [f'"{tok.replace(chr(34), chr(34) * 2)}"' for tok in stemmed.split() if tok]
    return " ".join(parts)


@dataclass(slots=True)
class BM25Index:
    """Append-only FTS5 index.

    We pre-stem on the Python side and store both the raw text (for display)
    and the stemmed text (for FTS search).
    """

    path: Path

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
            USING fts5(
                chunk_id UNINDEXED,
                doc_id UNINDEXED,
                tenant UNINDEXED,
                page_no UNINDEXED,
                stemmed,
                text,
                tokenize='unicode61 remove_diacritics 2'
            )
            """
        )
        return conn

    def upsert(self, *, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        with self._connect() as conn:
            conn.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id = ?",
                [(r["chunk_id"],) for r in rows],
            )
            conn.executemany(
                """
                INSERT INTO chunks_fts(chunk_id, doc_id, tenant, page_no, stemmed, text)
                VALUES (:chunk_id, :doc_id, :tenant, :page_no, :stemmed, :text)
                """,
                [
                    {
                        "chunk_id": r["chunk_id"],
                        "doc_id": r["doc_id"],
                        "tenant": r["tenant"],
                        "page_no": r["page_no"],
                        "stemmed": stem_finnish(r["text"]),
                        "text": r["text"],
                    }
                    for r in rows
                ],
            )

    def search(self, *, query: str, tenant: str, top_k: int) -> list[tuple[str, float]]:
        match = _fts5_match_expr(stem_finnish(query))
        if not match:
            return []
        with self._connect() as conn:
            cur = conn.execute(
                """
                SELECT chunk_id, bm25(chunks_fts) AS score
                FROM chunks_fts
                WHERE chunks_fts MATCH ? AND tenant = ?
                ORDER BY score LIMIT ?
                """,
                (match, tenant, top_k),
            )
            return [(row[0], float(row[1])) for row in cur.fetchall()]
