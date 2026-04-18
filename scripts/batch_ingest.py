"""Batch-ingest a folder tree into Systeemi (one-shot, in-process).

Unlike the CLI's ``ingest-dir`` this walks a directory tree recursively,
skips configured sub-folders, and uses a single :class:`LapuaPipeline`
instance so PP-StructureV3, embedder, BM25 and Qdrant clients are loaded
exactly once for the whole batch.

Usage::

    python scripts/batch_ingest.py <root> [--tenant lapua] [--skip-extract]
                                          [--exclude rag_output,tmp]
                                          [--log data/batch_ingest.log]

Design notes:
* Pure orchestration; all ingest logic lives in
  :class:`lapua_rag.pipeline.LapuaPipeline`.
* Continues the batch on per-document failure (logs traceback, counts
  failed).
* Prints a per-document line with doc_id, status, pages, chunks, elapsed
  seconds -- easy to eyeball progress in a terminal.
* Writes a summary JSONL log for later analysis.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from lapua_rag.observability import configure_logging
from lapua_rag.pipeline import LapuaPipeline, build_default


@dataclass(slots=True)
class BatchItemResult:
    """Outcome of ingesting a single PDF in the batch."""

    path: str
    doc_id: str | None
    status: str
    pages: int
    chunks: int
    elapsed_s: float
    error: str | None = None


@dataclass(slots=True)
class BatchSummary:
    """Aggregate stats across the batch."""

    total: int = 0
    indexed: int = 0
    skipped: int = 0
    failed: int = 0
    elapsed_s: float = 0.0
    items: list[BatchItemResult] = field(default_factory=list)


def _iter_pdfs(root: Path, excludes: set[str]) -> Iterable[Path]:
    """Yield every ``*.pdf`` under root whose relative path does not
    traverse an excluded directory name."""
    for candidate in sorted(root.rglob("*.pdf")):
        if any(part in excludes for part in candidate.relative_to(root).parts):
            continue
        yield candidate


def _ingest_one(
    *,
    pipeline: LapuaPipeline,
    pdf: Path,
    tenant: str,
    skip_extract: bool,
) -> BatchItemResult:
    """Run one PDF through the pipeline; never raises."""
    start = time.perf_counter()
    try:
        result = pipeline.ingest(pdf_path=pdf, tenant=tenant, skip_extract=skip_extract)
        elapsed = time.perf_counter() - start
        return BatchItemResult(
            path=str(pdf),
            doc_id=result.doc_id,
            status=result.status.value,
            pages=result.page_count,
            chunks=result.chunk_count,
            elapsed_s=round(elapsed, 2),
        )
    except Exception as exc:
        elapsed = time.perf_counter() - start
        return BatchItemResult(
            path=str(pdf),
            doc_id=None,
            status="failed",
            pages=0,
            chunks=0,
            elapsed_s=round(elapsed, 2),
            error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=3)}",
        )


def run_batch(
    *,
    root: Path,
    tenant: str,
    skip_extract: bool,
    excludes: set[str],
    log_path: Path,
) -> BatchSummary:
    """Ingest every PDF under ``root`` that is not under an excluded folder."""
    pdfs = list(_iter_pdfs(root, excludes))
    summary = BatchSummary(total=len(pdfs))
    if not pdfs:
        print(f"[batch] no PDFs under {root}", flush=True)
        return summary

    print(f"[batch] {len(pdfs)} PDFs under {root} (excludes={sorted(excludes)})", flush=True)
    print(f"[batch] building pipeline (skip_extract={skip_extract}, tenant={tenant})", flush=True)
    pipeline = build_default()

    log_path.parent.mkdir(parents=True, exist_ok=True)
    batch_start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log:
        for i, pdf in enumerate(pdfs, start=1):
            item = _ingest_one(
                pipeline=pipeline,
                pdf=pdf,
                tenant=tenant,
                skip_extract=skip_extract,
            )
            summary.items.append(item)
            if item.status == "indexed":
                summary.indexed += 1
                tag = "OK "
            elif item.status == "failed":
                summary.failed += 1
                tag = "ERR"
            else:
                summary.skipped += 1
                tag = "SKP"

            rel = pdf.relative_to(root)
            print(
                f"[{i:03d}/{len(pdfs):03d}] {tag} {item.elapsed_s:6.1f}s "
                f"pages={item.pages:3d} chunks={item.chunks:3d} "
                f"doc={item.doc_id or '-':16s} {rel}",
                flush=True,
            )
            if item.error:
                print(f"    !! {item.error.splitlines()[0]}", flush=True)

            log.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")
            log.flush()

    summary.elapsed_s = round(time.perf_counter() - batch_start, 2)
    return summary


def _parse_excludes(raw: str) -> set[str]:
    return {part.strip() for part in raw.split(",") if part.strip()}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="Folder tree to walk recursively")
    parser.add_argument("--tenant", default="lapua", help="Tenant tag for stored chunks")
    parser.add_argument(
        "--skip-extract",
        action="store_true",
        help="Skip LLM structured extraction (required on CPU-only hosts).",
    )
    parser.add_argument(
        "--exclude",
        default="rag_output,tmp,.git",
        help="Comma-separated folder names to skip anywhere in the tree.",
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("data/batch_ingest.jsonl"),
        help="Path for per-document JSONL result log.",
    )
    args = parser.parse_args(argv)

    if not args.root.exists():
        print(f"[batch] root does not exist: {args.root}", file=sys.stderr)
        return 2

    configure_logging(level="INFO", fmt="json")
    summary = run_batch(
        root=args.root.resolve(),
        tenant=args.tenant,
        skip_extract=args.skip_extract,
        excludes=_parse_excludes(args.exclude),
        log_path=args.log,
    )

    print(
        f"[batch] done: total={summary.total} indexed={summary.indexed} "
        f"skipped={summary.skipped} failed={summary.failed} "
        f"elapsed={summary.elapsed_s:.1f}s",
        flush=True,
    )
    return 0 if summary.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
