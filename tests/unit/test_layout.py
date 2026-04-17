from __future__ import annotations

from datetime import date
from pathlib import Path

from lapua_rag.storage.layout import DocumentLayout


def test_layout_creates_expected_paths(tmp_path: Path) -> None:
    layout = DocumentLayout.for_document(
        storage_root=tmp_path,
        tenant="lapua",
        doc_id="deadbeef00000000",
        bucket=date(2025, 4, 17),
    )
    assert layout.root == tmp_path / "lapua" / "2025" / "04" / "deadbeef00000000"
    assert layout.pages_dir.is_dir()
    assert layout.tables_dir.is_dir()
    assert layout.page_md(3).name == "003.md"
    assert layout.page_res_json(3).name == "003.res.json"
    assert layout.table_html(3, 1).name == "003_01.html"
