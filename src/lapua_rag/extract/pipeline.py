"""Map-reduce extraction over chunked document text."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import ValidationError

from lapua_rag.extract.llm import LlmClient
from lapua_rag.extract.schemas import (
    DECISION_SCHEMA,
    EXTRACTION_INSTRUCTION,
    SYSTEM_PROMPT,
)
from lapua_rag.models.decisions import DecisionItem, DocumentStructure
from lapua_rag.models.document import DocumentType
from lapua_rag.observability import get_logger
from lapua_rag.postprocess.chunking import RawChunk

_log = get_logger(__name__)


@dataclass(slots=True)
class ExtractionPipeline:
    client: LlmClient

    def extract_decision(self, *, chunk: RawChunk, doc_id: str) -> DecisionItem | None:
        prompt = (
            f"{EXTRACTION_INSTRUCTION}\n\n"
            f"Pykälä: {chunk.section_id or 'tuntematon'}\n"
            f"Sivu: {chunk.page_start}\n\n"
            f"Teksti:\n{chunk.text}"
        )
        try:
            raw = self.client.generate_json(
                system=SYSTEM_PROMPT,
                prompt=prompt,
                json_schema=DECISION_SCHEMA,
            )
            return DecisionItem.model_validate(raw)
        except (ValidationError, ValueError) as exc:
            _log.warning(
                "extract.decision_failed",
                doc_id=doc_id,
                section_id=chunk.section_id,
                error=str(exc),
            )
            return None

    def extract_document(
        self,
        *,
        doc_id: str,
        tenant: str,
        doc_type: DocumentType,
        chunks: list[RawChunk],
    ) -> DocumentStructure:
        decisions: list[DecisionItem] = []
        for chunk in chunks:
            if chunk.section_id is None:
                continue
            item = self.extract_decision(chunk=chunk, doc_id=doc_id)
            if item is not None:
                decisions.append(item)

        return DocumentStructure(
            doc_id=doc_id,
            tenant=tenant,
            doc_type=doc_type.value,  # type: ignore[arg-type]
            paatokset=decisions,
        )
