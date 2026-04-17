"""Inbox folder watcher (watchdog)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from lapua_rag.observability import get_logger

_log = get_logger(__name__)


class _PdfHandler(FileSystemEventHandler):
    def __init__(self, on_pdf: Callable[[Path], None]) -> None:
        self._on_pdf = on_pdf

    def on_created(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return
        path = Path(str(event.src_path))
        if path.suffix.lower() != ".pdf":
            return
        _log.info("inbox.pdf_detected", path=str(path))
        self._on_pdf(path)


class InboxWatcher:
    """Watch ``inbox_dir`` for new PDFs and invoke a callback per file.

    The callback must be non-blocking; push work onto the queue, don't process
    synchronously here.
    """

    def __init__(self, inbox_dir: Path, on_pdf: Callable[[Path], None]) -> None:
        self._inbox_dir = inbox_dir
        self._on_pdf = on_pdf
        self._observer: Observer | None = None

    def start(self) -> None:
        self._inbox_dir.mkdir(parents=True, exist_ok=True)
        observer = Observer()
        observer.schedule(_PdfHandler(self._on_pdf), str(self._inbox_dir), recursive=False)
        observer.start()
        self._observer = observer
        _log.info("inbox.watcher_started", path=str(self._inbox_dir))

    def stop(self) -> None:
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=5.0)
            self._observer = None
            _log.info("inbox.watcher_stopped")
