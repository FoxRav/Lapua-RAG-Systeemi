"""Merge per-page artefacts into a single document.md.

We preferentially rebuild text from the per-page ``*.res.json`` files because
their ``rec_texts`` are reliable UTF-8 even when the ``.md`` writer occasionally
produces mojibake.
"""

from __future__ import annotations

import json
from pathlib import Path

from lapua_rag.postprocess.encoding import fix_finnish_mojibake


def _page_text_from_json(res_json_path: Path) -> str | None:
    try:
        data = json.loads(res_json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    res = data.get("res") if isinstance(data, dict) else None
    if not isinstance(res, dict):
        return None
    parsing = res.get("parsing_res_list")
    if isinstance(parsing, list) and parsing:
        blocks = [
            block.get("block_content", "")
            for block in parsing
            if isinstance(block, dict)
        ]
        joined = "\n\n".join(b for b in blocks if b).strip()
        if joined:
            return fix_finnish_mojibake(joined)
    ocr = res.get("overall_ocr_res", {})
    texts = ocr.get("rec_texts") if isinstance(ocr, dict) else None
    if isinstance(texts, list) and texts:
        return fix_finnish_mojibake("\n".join(str(t) for t in texts).strip())
    return None


def consolidate_markdown(
    *,
    pages_dir: Path,
    out_path: Path,
    page_count: int,
) -> Path:
    """Write a single consolidated ``document.md``.

    Returns the path. Idempotent (overwrites existing).
    """
    lines: list[str] = []
    for page_no in range(page_count):
        json_path = pages_dir / f"{page_no:03d}.res.json"
        md_path = pages_dir / f"{page_no:03d}.md"
        text = _page_text_from_json(json_path)
        if text is None and md_path.exists():
            text = fix_finnish_mojibake(md_path.read_text(encoding="utf-8", errors="replace"))
        if text is None:
            continue
        lines.append(f"\n\n<!-- page: {page_no} -->\n\n")
        lines.append(text)

    out_path.write_text("".join(lines).strip() + "\n", encoding="utf-8")
    return out_path
