"""Clean up a raw PP-StructureV3 ``--save_path`` directory.

Keeps only the artefacts Lapua-RAG actually needs (per-page .md / .res.json,
full-page .png renders, .html tables, the source .pdf) and removes the
redundant .docx / .tex / cropped-image noise.

Usage:
    python scripts/cleanup_paddleocr_output.py <directory>
"""

from __future__ import annotations

import sys
from pathlib import Path

KEEP_SUFFIXES = {".md", ".json", ".html", ".pdf"}
KEEP_TABLE_SUFFIX = ".html"
DROP_EXTENSIONS = {".docx", ".tex", ".xlsx"}


def main(target: Path) -> None:
    if not target.is_dir():
        msg = f"not a directory: {target}"
        raise SystemExit(msg)

    removed = 0
    kept = 0
    for path in target.iterdir():
        if not path.is_file():
            continue
        ext = path.suffix.lower()
        if ext in DROP_EXTENSIONS:
            path.unlink()
            removed += 1
            continue
        if ext == ".png" and "_img_" in path.stem:
            path.unlink()
            removed += 1
            continue
        kept += 1
    print(f"kept={kept} removed={removed} dir={target}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        msg = "usage: cleanup_paddleocr_output.py <directory>"
        raise SystemExit(msg)
    main(Path(sys.argv[1]))
