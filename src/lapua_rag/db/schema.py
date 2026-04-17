"""SQLModel tables for metadata + structured extraction."""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from sqlmodel import JSON, Column, Field, SQLModel


class DocumentRow(SQLModel, table=True):  # type: ignore[call-arg]
    __tablename__ = "documents"

    doc_id: str = Field(primary_key=True, max_length=32)
    tenant: str = Field(index=True, max_length=64)
    source_path: str
    sha256: str = Field(index=True, unique=True, max_length=64)
    doc_type: str = Field(index=True, max_length=32)
    paivamaara: date | None = Field(default=None, index=True)
    elin: str | None = Field(default=None, index=True, max_length=128)
    page_count: int = 0
    status: str = Field(index=True, max_length=16)
    created_at: datetime = Field(default_factory=datetime.utcnow, index=True)
    indexed_at: datetime | None = None
    extra: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))


class PageRow(SQLModel, table=True):  # type: ignore[call-arg]
    __tablename__ = "pages"

    id: int | None = Field(default=None, primary_key=True)
    doc_id: str = Field(foreign_key="documents.doc_id", index=True, max_length=32)
    page_no: int = Field(index=True)
    md_path: str
    json_path: str
    ocr_confidence_avg: float = 0.0


class ChunkRow(SQLModel, table=True):  # type: ignore[call-arg]
    __tablename__ = "chunks"

    chunk_id: str = Field(primary_key=True, max_length=48)
    doc_id: str = Field(foreign_key="documents.doc_id", index=True, max_length=32)
    tenant: str = Field(index=True, max_length=64)
    page_no: int = Field(index=True)
    section_id: str | None = Field(default=None, max_length=64)
    section_title: str | None = Field(default=None, max_length=256)
    doc_type: str = Field(index=True, max_length=32)
    text: str
    token_count: int = 0
    vector_id: str | None = Field(default=None, index=True, max_length=64)


class TableRow(SQLModel, table=True):  # type: ignore[call-arg]
    __tablename__ = "tables"

    table_id: str = Field(primary_key=True, max_length=48)
    doc_id: str = Field(foreign_key="documents.doc_id", index=True, max_length=32)
    page_no: int = Field(index=True)
    rows: int = 0
    cols: int = 0
    html_path: str
    parquet_path: str | None = None


class DecisionRow(SQLModel, table=True):  # type: ignore[call-arg]
    __tablename__ = "decisions"

    id: int | None = Field(default=None, primary_key=True)
    doc_id: str = Field(foreign_key="documents.doc_id", index=True, max_length=32)
    pykala: str = Field(index=True, max_length=32)
    otsikko: str
    paatos: str
    perustelut: str
    euro_summa: float | None = None
    paivamaara: date | None = Field(default=None, index=True)
    sivu: int = 0
    payload: dict[str, Any] = Field(default_factory=dict, sa_column=Column(JSON))
