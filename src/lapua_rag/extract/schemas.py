"""JSON schemas for constrained decoding (lm-format-enforcer)."""

from __future__ import annotations

from lapua_rag.models.decisions import DecisionItem, DocumentStructure

DECISION_SCHEMA: dict[str, object] = DecisionItem.model_json_schema()
DOCUMENT_SCHEMA: dict[str, object] = DocumentStructure.model_json_schema()


SYSTEM_PROMPT = (
    "Olet Lapuan päätöstekstiasiantuntija. "
    "Vastaa muodolla: Johtopäätös → Perustelut → Lähteet (§, kokous, pvm)."
)

EXTRACTION_INSTRUCTION = (
    "Pura seuraavasta pykälätekstistä rakenteellinen tieto Pydantic-skeeman "
    "mukaisesti. Älä keksi kenttiä jotka puuttuvat. Jätä tuntemattomat kentät "
    "nulliksi."
)
