"""Audit logging for Lapua-RAG query endpoints."""

from __future__ import annotations

from lapua_rag.audit.service import log_entry, log_query

__all__ = ["log_entry", "log_query"]
