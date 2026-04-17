"""Lapua-RAG: local RAG platform for Finnish municipal documents.

The package is structured as a functional core (pure functions in
`postprocess`, `models`, `rag`) surrounded by an imperative shell (external
clients in `ocr`, `extract`, `embed`, `index`, `db`, `api`).
"""

from __future__ import annotations

__version__ = "0.1.0"
