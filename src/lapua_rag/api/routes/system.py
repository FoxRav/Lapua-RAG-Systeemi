"""Systeemi endpoints: stats, versioning, coverage."""

from __future__ import annotations

from fastapi import APIRouter

from lapua_rag.config import get_settings
from lapua_rag.systeemi import (
    CoverageReport,
    SystemStats,
    SystemVersion,
    compute_coverage,
    compute_system_version,
    gather_stats,
)
from lapua_rag.systeemi.versioning import ModelFingerprint

router = APIRouter()


def _fingerprint() -> ModelFingerprint:
    settings = get_settings()
    return ModelFingerprint(
        embedder=settings.embedding_model,
        reranker=settings.reranker_model,
        llm_base=settings.llm_base,
        llm_lora=settings.llm_lora,
    )


@router.get("/system/stats", response_model=SystemStats)
def system_stats(tenant: str | None = None) -> SystemStats:
    return gather_stats(tenant=tenant)


@router.get("/system/version", response_model=SystemVersion)
def system_version(tenant: str | None = None) -> SystemVersion:
    return compute_system_version(tenant=tenant, models=_fingerprint())


@router.get("/system/coverage", response_model=CoverageReport)
def system_coverage(tenant: str | None = None) -> CoverageReport:
    return compute_coverage(tenant=tenant)
