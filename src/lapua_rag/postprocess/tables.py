"""Parse per-page HTML tables into pandas DataFrames and Parquet artefacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True, slots=True)
class ParsedTable:
    rows: int
    cols: int
    parquet_path: Path


def parse_table_html(*, html_path: Path, parquet_path: Path) -> ParsedTable | None:
    """Parse a single ``table.html`` file into Parquet.

    Returns None if the HTML contained no tables.
    """
    try:
        dfs = pd.read_html(html_path, encoding="utf-8")
    except ValueError:
        return None
    if not dfs:
        return None

    df = dfs[0]
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(parquet_path, engine="pyarrow", index=False)
    return ParsedTable(rows=int(df.shape[0]), cols=int(df.shape[1]), parquet_path=parquet_path)
