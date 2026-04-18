"""Typer CLI entry point.

Usage::

    lapua-rag init                           # create DB schema + Qdrant collection
    lapua-rag ingest path/to/file.pdf        # ingest one PDF
    lapua-rag ingest-dir path/to/folder      # ingest a folder
    lapua-rag query "Mitä § 78 koskee?"
    lapua-rag serve                          # run FastAPI on port 8080
    lapua-rag ui                             # run Next.js frontend dev server
    lapua-rag openapi-dump                   # write OpenAPI JSON for gen-types
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import typer
import uvicorn
from rich import print as rprint
from rich.table import Table

from lapua_rag.api.app import create_app
from lapua_rag.api.routes.query import _answer_service
from lapua_rag.config import get_settings
from lapua_rag.db.session import create_all
from lapua_rag.embed.embedder import Embedder
from lapua_rag.index.qdrant import QdrantIndex
from lapua_rag.observability import configure_logging
from lapua_rag.pipeline import build_default
from lapua_rag.systeemi import compute_coverage, compute_system_version, gather_stats
from lapua_rag.systeemi.versioning import ModelFingerprint

app = typer.Typer(help="Lapua-RAG command line interface")
system_app = typer.Typer(help="Inspect Systeemi: stats, version, coverage")
app.add_typer(system_app, name="system")


@app.callback()
def _bootstrap() -> None:
    settings = get_settings()
    configure_logging(level=settings.log_level, fmt=settings.log_format)


@app.command()
def init() -> None:
    """Create DB schema and ensure Qdrant collection exists."""
    settings = get_settings()
    create_all()
    embedder = Embedder(model_name=settings.embedding_model, device=settings.embedding_device)
    qdrant = QdrantIndex(
        url=settings.qdrant_url,
        collection=settings.qdrant_collection,
        api_key=settings.qdrant_api_key,
    )
    qdrant.ensure_collection(dim=embedder.dimension())
    rprint("[green]Initialised DB and Qdrant collection.[/green]")


@app.command()
def ingest(
    pdf: Path,
    tenant: str | None = None,
    skip_extract: bool = typer.Option(
        False,
        "--skip-extract/--extract",
        help="Skip LLM structured extraction (useful on CPU-only runs).",
    ),
) -> None:
    """Ingest a single PDF into Systeemi."""
    pipeline = build_default()
    result = pipeline.ingest(pdf_path=pdf, tenant=tenant, skip_extract=skip_extract)
    table = Table(title="Ingest result")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("doc_id", result.doc_id)
    table.add_row("status", result.status.value)
    table.add_row("pages", str(result.page_count))
    table.add_row("chunks", str(result.chunk_count))
    rprint(table)


@app.command("ingest-dir")
def ingest_dir(
    folder: Path,
    tenant: str | None = None,
    skip_extract: bool = typer.Option(
        False,
        "--skip-extract/--extract",
        help="Skip LLM structured extraction.",
    ),
) -> None:
    """Ingest every ``*.pdf`` in a folder (non-recursive)."""
    pipeline = build_default()
    for pdf in sorted(folder.glob("*.pdf")):
        rprint(f"[bold]Ingesting[/bold] {pdf.name}")
        pipeline.ingest(pdf_path=pdf, tenant=tenant, skip_extract=skip_extract)


@app.command()
def query(question: str, tenant: str | None = None) -> None:
    """Ask a question against the indexed corpus."""
    settings = get_settings()
    svc = _answer_service(settings.answer_mode)
    answer = svc.answer(query=question, tenant=tenant or settings.tenant)
    rprint(answer.model_dump_json(indent=2))


@app.command()
def serve(host: str | None = None, port: int | None = None) -> None:
    """Run the FastAPI server."""
    settings = get_settings()
    uvicorn.run(
        "lapua_rag.api.app:create_app",
        factory=True,
        host=host or settings.api_host,
        port=port or settings.api_port,
        reload=False,
    )


@app.command()
def ui(
    port: int = typer.Option(3000, help="Port for the Next.js dev server."),
    install: bool = typer.Option(
        False,
        "--install/--no-install",
        help="Run `npm install` in frontend/ before starting (first-run setup).",
    ),
) -> None:
    """Run the Next.js frontend dev server (frontend/).

    The backend (`lapua-rag serve`) must be running separately on the URL
    pointed at by `frontend/.env.local`'s NEXT_PUBLIC_API_URL.
    """
    repo_root = Path(__file__).resolve().parents[2]
    frontend_dir = repo_root / "frontend"
    if not (frontend_dir / "package.json").is_file():
        raise typer.BadParameter(
            f"frontend/ not found at {frontend_dir}; did the repo move?",
        )

    npm = shutil.which("npm")
    if npm is None:
        raise typer.BadParameter(
            "npm not found on PATH. Install Node.js 20+ from https://nodejs.org/.",
        )

    if install:
        rprint("[bold]Running `npm install`...[/bold]")
        subprocess.run([npm, "install", "--no-fund", "--no-audit"], cwd=frontend_dir, check=True)

    env = {**os.environ, "PORT": str(port)}
    rprint(f"[bold green]Starting Next.js dev server on http://localhost:{port}[/bold green]")
    # Use exec-style spawn so Ctrl+C cleanly terminates the npm child.
    subprocess.run([npm, "run", "dev"], cwd=frontend_dir, env=env, check=False)


# Module-level singleton for the openapi-dump default. Required to avoid
# Ruff B008 (typer.Option in defaults) while keeping the CLI ergonomic.
_OPENAPI_DEFAULT_OUT: Path = Path("tmp/openapi.json")


@app.command("openapi-dump")
def openapi_dump(
    out: Path = typer.Option(  # noqa: B008
        _OPENAPI_DEFAULT_OUT,
        "--out",
        "-o",
        help="Where to write the OpenAPI JSON.",
    ),
) -> None:
    """Dump the OpenAPI schema to disk for offline TypeScript generation.

    Used by `frontend/scripts/gen-types.mjs` when the backend isn't running.
    """
    spec = create_app().openapi()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    rprint(f"[green]Wrote OpenAPI spec to {out} ({out.stat().st_size} bytes)[/green]")
    rprint(
        "Now in frontend/: [bold]npm run gen-types[/bold] to regenerate TypeScript types.",
    )


@system_app.command("stats")
def system_stats_cmd(tenant: str | None = None) -> None:
    """Print Systeemi document + chunk counts per tenant."""
    stats = gather_stats(tenant=tenant)
    table = Table(title="Systeemi stats")
    table.add_column("tenant")
    table.add_column("docs", justify="right")
    table.add_column("indexed", justify="right")
    table.add_column("pages", justify="right")
    table.add_column("chunks", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("last indexed")
    for t in stats.per_tenant:
        table.add_row(
            t.tenant,
            str(t.document_count),
            str(t.indexed_count),
            str(t.page_count),
            str(t.chunk_count),
            f"{t.token_count:,}",
            str(t.last_indexed) if t.last_indexed else "-",
        )
    rprint(table)
    rprint(
        f"[bold]Totals:[/bold] tenants={stats.tenant_count} "
        f"docs={stats.document_count} indexed={stats.indexed_count} "
        f"failed={stats.failed_count} chunks={stats.chunk_count}"
    )


@system_app.command("version")
def system_version_cmd(tenant: str | None = None) -> None:
    """Print the deterministic Systeemi content hash."""
    settings = get_settings()
    fingerprint = ModelFingerprint(
        embedder=settings.embedding_model,
        reranker=settings.reranker_model,
        llm_base=settings.llm_base,
        llm_lora=settings.llm_lora,
    )
    version = compute_system_version(tenant=tenant, models=fingerprint)
    rprint(version.model_dump_json(indent=2))


@system_app.command("coverage")
def system_coverage_cmd(tenant: str | None = None) -> None:
    """Print per-doc_type coverage for Systeemi."""
    report = compute_coverage(tenant=tenant)
    table = Table(title=f"Coverage (tenant={tenant or 'all'})")
    table.add_column("doc_type")
    table.add_column("indexed", justify="right")
    table.add_column("in progress", justify="right")
    table.add_column("failed", justify="right")
    table.add_column("earliest")
    table.add_column("latest")
    for row in report.by_doctype:
        table.add_row(
            row.doc_type,
            str(row.indexed),
            str(row.in_progress),
            str(row.failed),
            str(row.earliest_pvm) if row.earliest_pvm else "-",
            str(row.latest_pvm) if row.latest_pvm else "-",
        )
    rprint(table)
    if report.missing_recent:
        rprint(f"[yellow]Stuck > 24h:[/yellow] {', '.join(report.missing_recent)}")


if __name__ == "__main__":  # pragma: no cover
    app()
