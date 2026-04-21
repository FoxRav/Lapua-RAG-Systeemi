"""Tests for the /v1/aggregate router.

Two layers:

* Pure-function tests of the query classifier (no DB).
* End-to-end FastAPI-TestClient tests against a file-backed SQLite DB
  populated with a few ``DecisionRow`` rows. We monkeypatch
  ``session_module.get_engine`` so the router's ``session_scope``
  context opens against our throwaway engine.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import Engine
from sqlmodel import Session, SQLModel, create_engine

from lapua_rag.api.routes import aggregate as aggregate_routes
from lapua_rag.api.routes.aggregate import classify_aggregate_query
from lapua_rag.db import session as session_module
from lapua_rag.db.schema import DecisionRow, DocumentRow


class TestClassifyQuery:
    def test_count_question(self) -> None:
        agg_type, entity = classify_aggregate_query("Kuinka monta päätöstä tehtiin viime kuussa?")
        assert agg_type == "count"
        assert entity is None

    def test_count_question_with_person(self) -> None:
        agg_type, entity = classify_aggregate_query(
            "Kuinka monessa päätöksessä Sami Kuula on ollut mukana?"
        )
        assert agg_type == "count"
        assert entity == "Sami Kuula"

    def test_sum_question(self) -> None:
        agg_type, _ = classify_aggregate_query("Paljonko investoinnit maksoivat?")
        assert agg_type == "sum"

    def test_unrecognised_question(self) -> None:
        agg_type, _ = classify_aggregate_query("Kuka on kaupunginjohtaja?")
        assert agg_type == "not_supported"

    def test_blocklist_prevents_false_entity(self) -> None:
        """Lauseenalun 'Kuinka Monessa' ei saa tulla entityksi."""
        _, entity = classify_aggregate_query("Kuinka Monessa kokouksessa päätettiin asiasta?")
        assert entity is None


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    db_path = tmp_path / "agg.db"
    engine = create_engine(
        f"sqlite:///{db_path}", connect_args={"check_same_thread": False}
    )
    SQLModel.metadata.create_all(engine)

    with Session(engine) as db:
        db.add(
            DocumentRow(
                doc_id="doc1",
                tenant="lapua",
                source_path="/tmp/doc1.pdf",
                sha256="a" * 64,
                doc_type="poytakirja",
                status="indexed",
            )
        )
        db.add(
            DocumentRow(
                doc_id="doc2",
                tenant="other",
                source_path="/tmp/doc2.pdf",
                sha256="b" * 64,
                doc_type="poytakirja",
                status="indexed",
            )
        )
        # lapua-tenantin päätökset
        db.add(
            DecisionRow(
                doc_id="doc1",
                pykala="§ 12",
                otsikko="Kaupunginhallituksen jäsenten valinta",
                paatos="Puheenjohtajaksi valittiin Sami Kuula",
                perustelut="Esityksen mukaisesti",
                euro_summa=None,
                paivamaara=date(2025, 1, 15),
                sivu=3,
            )
        )
        db.add(
            DecisionRow(
                doc_id="doc1",
                pykala="§ 13",
                otsikko="Investointipäätös",
                paatos="Hyväksyttiin uusi investointi",
                perustelut="Kaupunginhallituksen esitys",
                euro_summa=15000.50,
                paivamaara=date(2025, 2, 1),
                sivu=4,
            )
        )
        db.add(
            DecisionRow(
                doc_id="doc1",
                pykala="§ 14",
                otsikko="Sami Kuulan lausunto",
                paatos="Merkittiin tiedoksi",
                perustelut="Asia ei vaadi toimenpiteitä",
                euro_summa=2500.0,
                paivamaara=date(2025, 2, 1),
                sivu=5,
            )
        )
        # toisen tenantin päätös — ei pidä näkyä lapua-kyselyssä
        db.add(
            DecisionRow(
                doc_id="doc2",
                pykala="§ 1",
                otsikko="Eri kaupungin asia",
                paatos="Hyväksyttiin",
                perustelut="X",
                euro_summa=999.0,
                paivamaara=date(2025, 3, 1),
                sivu=1,
            )
        )
        db.commit()

    def _override_get_engine() -> Engine:
        return engine

    monkeypatch.setattr(session_module, "get_engine", _override_get_engine)

    app = FastAPI()
    app.include_router(aggregate_routes.router, prefix="/v1")
    yield TestClient(app)


def test_count_without_entity_returns_all_tenant_decisions(client: TestClient) -> None:
    resp = client.post(
        "/v1/aggregate", json={"query": "Kuinka monta päätöstä kaupunki teki?", "tenant": "lapua"}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_type"] == "count"
    assert body["value"] == 3
    assert body["entity"] is None
    assert body["tenant"] == "lapua"


def test_count_filters_by_person_entity(client: TestClient) -> None:
    resp = client.post(
        "/v1/aggregate",
        json={
            "query": "Kuinka monessa päätöksessä Sami Kuula on ollut mukana?",
            "tenant": "lapua",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_type"] == "count"
    assert body["value"] == 2
    assert body["entity"] == "Sami Kuula"
    assert "Sami Kuula" in body["explanation"]


def test_count_respects_tenant_filter(client: TestClient) -> None:
    resp = client.post(
        "/v1/aggregate",
        json={"query": "Kuinka monta päätöstä tehtiin?", "tenant": "other"},
    )
    assert resp.status_code == 200
    assert resp.json()["value"] == 1


def test_sum_returns_total_euros_for_tenant(client: TestClient) -> None:
    resp = client.post(
        "/v1/aggregate",
        json={"query": "Paljonko investoinnit maksoivat yhteensä?", "tenant": "lapua"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_type"] == "sum"
    assert body["value"] == pytest.approx(17500.50)
    assert body["unit"] == "EUR"


def test_not_supported_when_query_not_aggregate(client: TestClient) -> None:
    resp = client.post(
        "/v1/aggregate",
        json={"query": "Kuka on Lapuan kaupunginjohtaja?", "tenant": "lapua"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result_type"] == "not_supported"
    assert body["value"] is None
    assert "/v1/query" in body["explanation"]


def test_short_query_rejected_by_pydantic(client: TestClient) -> None:
    resp = client.post("/v1/aggregate", json={"query": "a"})
    assert resp.status_code == 422
