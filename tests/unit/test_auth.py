"""Unit tests for the API-key auth dependency.

Auth resolution goes through :func:`lapua_rag.api.auth.require_api_key`
which reads :class:`Settings.auth_enabled` at call time and, when
enabled, looks up the key against the SQLite-backed ``api_keys`` table.

Tests use an in-memory SQLite DB with ``StaticPool`` (same pattern as
``test_audit_endpoint``) so no filesystem DB is required.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi import HTTPException
from sqlalchemy import Engine
from sqlalchemy.pool import StaticPool
from sqlmodel import Session, SQLModel, create_engine, select

from lapua_rag.api import auth as auth_module
from lapua_rag.api.auth import hash_api_key, require_api_key
from lapua_rag.config import Settings
from lapua_rag.db import session as db_session
from lapua_rag.db.schema import ApiKey


@pytest.fixture
def hermetic_db(monkeypatch: pytest.MonkeyPatch) -> Iterator[Engine]:
    engine: Engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    SQLModel.metadata.create_all(engine)
    monkeypatch.setattr(db_session, "get_engine", lambda: engine)
    yield engine


def _override_settings(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    """Swap :func:`get_settings` for a call that returns a fresh Settings.

    ``Settings(**overrides)`` bypasses the lru_cache so auth_enabled
    changes are always visible to the dependency under test.
    """
    base = {"tenant": "lapua", "auth_enabled": False}
    base.update(overrides)
    monkeypatch.setattr(auth_module, "get_settings", lambda: Settings(**base))  # type: ignore[arg-type]


def test_auth_disabled_returns_default_tenant(
    monkeypatch: pytest.MonkeyPatch,
    hermetic_db: Engine,
) -> None:
    """Dev/demo mode: missing header → configured tenant."""
    _override_settings(monkeypatch, auth_enabled=False, tenant="demo")
    assert require_api_key(x_api_key=None) == "demo"


def test_auth_enabled_missing_header_raises_401(
    monkeypatch: pytest.MonkeyPatch,
    hermetic_db: Engine,
) -> None:
    _override_settings(monkeypatch, auth_enabled=True)
    with pytest.raises(HTTPException) as exc:
        require_api_key(x_api_key=None)
    assert exc.value.status_code == 401
    assert "puuttuu" in exc.value.detail.lower()


def test_auth_enabled_valid_key_returns_tenant(
    monkeypatch: pytest.MonkeyPatch,
    hermetic_db: Engine,
) -> None:
    raw = "lrag_valid_test_key_xyz"
    with Session(hermetic_db) as s:
        s.add(ApiKey(key_hash=hash_api_key(raw), tenant="kunta-x", label="demo"))
        s.commit()

    _override_settings(monkeypatch, auth_enabled=True)
    assert require_api_key(x_api_key=raw) == "kunta-x"

    # last_used_at must be stamped after a successful auth.
    with Session(hermetic_db) as s:
        stored = s.exec(select(ApiKey)).first()
        assert stored is not None
        assert stored.last_used_at is not None


def test_auth_enabled_unknown_key_raises_401(
    monkeypatch: pytest.MonkeyPatch,
    hermetic_db: Engine,
) -> None:
    _override_settings(monkeypatch, auth_enabled=True)
    with pytest.raises(HTTPException) as exc:
        require_api_key(x_api_key="lrag_not_in_db")
    assert exc.value.status_code == 401


def test_auth_enabled_revoked_key_raises_401(
    monkeypatch: pytest.MonkeyPatch,
    hermetic_db: Engine,
) -> None:
    raw = "lrag_revoked"
    with Session(hermetic_db) as s:
        s.add(ApiKey(
            key_hash=hash_api_key(raw),
            tenant="kunta-y",
            is_active=False,
        ))
        s.commit()

    _override_settings(monkeypatch, auth_enabled=True)
    with pytest.raises(HTTPException) as exc:
        require_api_key(x_api_key=raw)
    assert exc.value.status_code == 401


def test_hash_api_key_is_deterministic() -> None:
    """Pure function: identical inputs produce identical hex digests."""
    assert hash_api_key("alpha") == hash_api_key("alpha")
    assert hash_api_key("alpha") != hash_api_key("beta")
    assert len(hash_api_key("alpha")) == 64  # SHA-256 hex
