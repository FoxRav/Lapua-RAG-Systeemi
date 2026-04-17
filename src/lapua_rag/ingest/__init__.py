"""Ingestion: file watcher, deduplication, queue."""

from __future__ import annotations

from lapua_rag.ingest.dedup import compute_sha256, doc_id_from_sha256
from lapua_rag.ingest.queue import IngestQueue, QueueItem
from lapua_rag.ingest.watcher import InboxWatcher

__all__ = [
    "InboxWatcher",
    "IngestQueue",
    "QueueItem",
    "compute_sha256",
    "doc_id_from_sha256",
]
