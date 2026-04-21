"""Aggregation endpoint for COUNT / SUM questions.

The RAG path is great at "who/what/where" extractions but poor at
"kuinka monessa päätöksessä X on mukana" — that's a SQL problem, not a
retrieval problem. This router inspects the query for aggregate
keywords, runs the SQL against the already-populated ``decisions``
table (``db/schema.py::DecisionRow``), and returns a typed result.

The query classifier is a pure function so unit tests can pin its
behaviour without a DB; the endpoint itself is thin glue around
``session_scope()``.
"""

from __future__ import annotations

import re
import time
from typing import Annotated, Final, Literal

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import or_
from sqlmodel import col, func, select

from lapua_rag.api.auth import require_api_key
from lapua_rag.audit.service import log_entry
from lapua_rag.config import get_settings
from lapua_rag.db.schema import DecisionRow, DocumentRow
from lapua_rag.db.session import session_scope

router = APIRouter()

AggregateType = Literal["count", "sum", "not_supported"]

# Kysymyskuviot. Listat pidetään pieninä tarkoituksella — monimutkainen
# parsinta kuuluu myöhemmin lapua-llm-v3:n entity-extractioniin (ks.
# README §11.4). Tämä classifier on deterministinen "hyvä alku".
_COUNT_KEYWORDS: Final[tuple[str, ...]] = (
    "kuinka monta",
    "kuinka monessa",
    "montako",
    "montaako",
    "lukumäärä",
    "kuinka usein",
)
_SUM_KEYWORDS: Final[tuple[str, ...]] = (
    "kuinka paljon",
    "paljonko",
    "summa",
    "euroa",
    "yhteenlaskettu",
)

# Nimi-heuristiikka: kaksi peräkkäistä isolla alkavaa sanaa (Etunimi Sukunimi).
# Riittää "Sami Kuula" -tyyppisiin, ei yritä ratkaista yksinimistä.
_PERSON_NAME = re.compile(r"\b([A-ZÄÖÅ][a-zäöå]+\s+[A-ZÄÖÅ][a-zäöå]+)\b")

# Sanat jotka eivät kelpaa person-nimeksi vaikka ovat isolla alussa
# (lauseenalkuja, suomenkielisiä kohteliaisuuksia).
_NAME_BLOCKLIST: Final[frozenset[str]] = frozenset(
    {"Kuinka Monta", "Kuinka Monessa", "Kuinka Paljon", "Kuinka Usein"}
)


class AggregateRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str = Field(min_length=3, max_length=512)
    tenant: str | None = None


class AggregateResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    query: str
    result_type: AggregateType
    value: int | float | None
    unit: str | None = None
    entity: str | None = None
    explanation: str
    tenant: str


def classify_aggregate_query(query: str) -> tuple[AggregateType, str | None]:
    """Return ``(type, entity)`` for an aggregate question.

    Pure: no DB, no config. Order of checks matters: SUM keywords
    contain "paljon" which could clash with a future COUNT phrase, so
    the explicit substring check on COUNT keywords runs first.
    """
    lowered = query.lower()
    agg_type: AggregateType = "not_supported"
    if any(kw in lowered for kw in _COUNT_KEYWORDS):
        agg_type = "count"
    elif any(kw in lowered for kw in _SUM_KEYWORDS):
        agg_type = "sum"

    entity: str | None = None
    for match in _PERSON_NAME.finditer(query):
        candidate = match.group(1)
        if candidate in _NAME_BLOCKLIST:
            continue
        entity = candidate
        break
    return agg_type, entity


def _run_count(tenant: str, entity: str | None) -> int:
    """Count decisions for ``tenant``, optionally filtered by entity mention."""
    with session_scope() as db:
        stmt = (
            select(func.count(col(DecisionRow.id)))
            .join(DocumentRow, col(DocumentRow.doc_id) == col(DecisionRow.doc_id))
            .where(col(DocumentRow.tenant) == tenant)
        )
        if entity:
            like = f"%{entity}%"
            stmt = stmt.where(
                or_(
                    col(DecisionRow.otsikko).ilike(like),
                    col(DecisionRow.paatos).ilike(like),
                    col(DecisionRow.perustelut).ilike(like),
                )
            )
        raw = db.exec(stmt).one()
        return int(raw if not isinstance(raw, tuple) else raw[0])


def _run_sum(tenant: str) -> float:
    with session_scope() as db:
        stmt = (
            select(func.sum(col(DecisionRow.euro_summa)))
            .join(DocumentRow, col(DocumentRow.doc_id) == col(DecisionRow.doc_id))
            .where(col(DocumentRow.tenant) == tenant)
            .where(col(DecisionRow.euro_summa).is_not(None))
        )
        raw = db.exec(stmt).one()
        value = raw if not isinstance(raw, tuple) else raw[0]
        return float(value) if value is not None else 0.0


@router.post("/aggregate", response_model=AggregateResult)
def aggregate(
    request: AggregateRequest,
    background_tasks: BackgroundTasks,
    http_request: Request,
    auth_tenant: Annotated[str, Depends(require_api_key)],
) -> AggregateResult:
    settings = get_settings()
    tenant = auth_tenant if settings.auth_enabled else (request.tenant or auth_tenant)
    agg_type, entity = classify_aggregate_query(request.query)

    started = time.perf_counter()
    if agg_type == "count":
        count = _run_count(tenant, entity)
        explanation = (
            f"Systeemistä löytyi {count} päätöstä"
            + (f" henkilölle {entity}" if entity else "")
            + "."
        )
        result = AggregateResult(
            query=request.query,
            result_type="count",
            value=count,
            entity=entity,
            explanation=explanation,
            tenant=tenant,
        )
    elif agg_type == "sum":
        total = _run_sum(tenant)
        explanation = f"Rahasummien yhteenlaskettu summa: {total:,.2f} €.".replace(",", " ")
        result = AggregateResult(
            query=request.query,
            result_type="sum",
            value=round(total, 2),
            unit="EUR",
            entity=entity,
            explanation=explanation,
            tenant=tenant,
        )
    else:
        result = AggregateResult(
            query=request.query,
            result_type="not_supported",
            value=None,
            entity=entity,
            explanation=(
                "Tätä kysymystyyppiä ei tueta /v1/aggregate-endpointissa. "
                "Kokeile /v1/query-endpointia tai muotoile kysymys COUNT/SUM-muotoon."
            ),
            tenant=tenant,
        )

    latency_ms = int((time.perf_counter() - started) * 1000)
    client = http_request.client
    background_tasks.add_task(
        log_entry,
        tenant=tenant,
        endpoint="/v1/aggregate",
        query_text=request.query,
        mode=result.result_type,
        # not_supported is the aggregate equivalent of an abstain: we refused
        # to return a numeric answer because the classifier could not map the
        # question to SQL. Logging it this way lets the audit UI surface the
        # same "we didn't answer" state across both endpoints.
        abstained=(result.result_type == "not_supported"),
        abstain_reason="not_supported" if result.result_type == "not_supported" else None,
        max_source_score=None,
        source_doc_ids=[],
        latency_ms=latency_ms,
        client_ip=client.host if client is not None else None,
        user_agent=http_request.headers.get("user-agent"),
    )
    return result
