"""Deterministic filesystem layout for per-document artefacts.

    <storage_root>/
        <tenant>/
            <YYYY>/<MM>/
                <doc_id>/
                    source.pdf
                    document.md
                    structured.json
                    manifest.json
                    pages/
                        <NN>.md
                        <NN>.res.json
                        <NN>.png
                    tables/
                        <NN>_<M>.html
                        <NN>_<M>.parquet
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path


@dataclass(frozen=True, slots=True)
class DocumentLayout:
    """Resolved paths for a single document within the storage root."""

    root: Path

    @classmethod
    def for_document(
        cls,
        *,
        storage_root: Path,
        tenant: str,
        doc_id: str,
        bucket: date,
    ) -> DocumentLayout:
        root = storage_root / tenant / f"{bucket.year:04d}" / f"{bucket.month:02d}" / doc_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "pages").mkdir(parents=True, exist_ok=True)
        (root / "tables").mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    @property
    def source_pdf(self) -> Path:
        return self.root / "source.pdf"

    @property
    def document_md(self) -> Path:
        return self.root / "document.md"

    @property
    def structured_json(self) -> Path:
        return self.root / "structured.json"

    @property
    def manifest_json(self) -> Path:
        return self.root / "manifest.json"

    @property
    def pages_dir(self) -> Path:
        return self.root / "pages"

    @property
    def tables_dir(self) -> Path:
        return self.root / "tables"

    def page_md(self, page_no: int) -> Path:
        return self.pages_dir / f"{page_no:03d}.md"

    def page_res_json(self, page_no: int) -> Path:
        return self.pages_dir / f"{page_no:03d}.res.json"

    def page_render(self, page_no: int) -> Path:
        return self.pages_dir / f"{page_no:03d}.png"

    def table_html(self, page_no: int, table_no: int) -> Path:
        return self.tables_dir / f"{page_no:03d}_{table_no:02d}.html"

    def table_parquet(self, page_no: int, table_no: int) -> Path:
        return self.tables_dir / f"{page_no:03d}_{table_no:02d}.parquet"
