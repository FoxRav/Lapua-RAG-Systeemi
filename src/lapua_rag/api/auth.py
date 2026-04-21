"""API-key authentication dependency for FastAPI routes.

The production contract is intentionally minimal:

* Clients send ``X-API-Key: <raw_key>``.
* The raw key is SHA-256 hashed and matched against :class:`ApiKey`.
* A match returns the bound ``tenant`` which the endpoint MUST use
  (the request-body ``tenant`` field is ignored whenever auth is on).
* A miss raises ``401``.

Auth is globally disabled via :pyattr:`Settings.auth_enabled` (default
``False``) so existing dev/demo deployments keep working. In that
bypass mode the dependency returns :pyattr:`Settings.tenant`, which
mirrors the pre-v0.9 behaviour where callers already defaulted to the
single-tenant ``lapua`` namespace.

Looking up ``Settings`` inside the dependency (not as a module-level
singleton) is deliberate: pytest can monkey-patch ``LAPUA_AUTH_ENABLED``
or override :func:`lapua_rag.config.get_settings` without racing a
process-wide cache.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Annotated

from fastapi import Header, HTTPException, status
from sqlmodel import col, select

from lapua_rag.config import Settings, get_settings
from lapua_rag.db.schema import ApiKey
from lapua_rag.db.session import session_scope
from lapua_rag.observability import get_logger

_log = get_logger(__name__)


def hash_api_key(raw: str) -> str:
    """Return the SHA-256 hex digest of a raw API key.

    Pure function — exposed so the ``lapua-rag keys`` CLI can store
    keys identically without depending on this module's FastAPI guts.
    """
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _resolve_api_key(raw: str, *, settings: Settings) -> str:
    """Look up an API key, update ``last_used_at``, return the tenant.

    Separated from :func:`require_api_key` so the session logic stays
    easy to test without the FastAPI Header dependency machinery.
    Raises ``HTTPException(401)`` if the key is unknown, revoked, or
    expired.
    """
    del settings  # reserved for future tenant-level overrides
    key_hash = hash_api_key(raw)
    with session_scope() as session:
        stmt = select(ApiKey).where(
            col(ApiKey.key_hash) == key_hash,
            col(ApiKey.is_active).is_(True),
        )
        db_key = session.exec(stmt).first()
        if db_key is None:
            _log.warning("auth.invalid_key", key_prefix=raw[:8])
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Virheellinen tai peruutettu API-avain",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        if db_key.expires_at is not None and db_key.expires_at < datetime.utcnow():
            _log.warning("auth.expired_key", tenant=db_key.tenant)
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="API-avain on vanhentunut",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        db_key.last_used_at = datetime.utcnow()
        session.add(db_key)
        _log.info("auth.ok", tenant=db_key.tenant, label=db_key.label)
        return db_key.tenant


def require_api_key(
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> str:
    """FastAPI dependency — returns the authenticated tenant.

    * Auth disabled → return ``settings.tenant`` (single-tenant fallback).
    * Auth enabled + header missing → ``401``.
    * Auth enabled + header invalid/expired → ``401``.
    """
    settings = get_settings()
    if not settings.auth_enabled:
        return settings.tenant
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="X-API-Key header puuttuu",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return _resolve_api_key(x_api_key, settings=settings)
