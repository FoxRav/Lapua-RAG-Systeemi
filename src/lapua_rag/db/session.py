"""SQLModel engine + session helpers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache

from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

from lapua_rag.config import get_settings


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    is_sqlite = settings.database_url.startswith("sqlite")
    connect_args: dict[str, object] = {"check_same_thread": False} if is_sqlite else {}
    return create_engine(settings.database_url, echo=False, connect_args=connect_args)


def create_all() -> None:
    """Create all tables if they don't exist. Idempotent."""
    SQLModel.metadata.create_all(get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """Context-managed transactional scope."""
    session = Session(get_engine())
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
