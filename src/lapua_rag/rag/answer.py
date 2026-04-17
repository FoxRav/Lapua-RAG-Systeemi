"""RAG answer service: *closed-book over Systeemi*.

The SLM (Qwen2.5-1.5B + LoRA ``lapua-llm-v2``) is only allowed to answer
from Systeemi (retrieved chunks). When Systeemi yields no sufficiently
relevant context the service *abstains*: no attempt is made to synthesise
an answer from the model's pretraining memory.

This is a product guarantee, not a hint to the model. The abstention
decision is made deterministically in Python **before** the LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from lapua_rag.extract.llm import LlmClient
from lapua_rag.observability import get_logger
from lapua_rag.retrieve.search import RetrievalResult, SearchService

_log = get_logger(__name__)

AbstainReason = Literal[
    "no_context",
    "below_threshold",
    "model_refused",
]


class RagSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    doc_id: str
    page_no: int
    section_id: str | None
    snippet: str


class RagAnswer(BaseModel):
    """Answer payload returned by the API.

    When ``abstained`` is True, ``johtopaatos`` contains a polite Finnish
    abstention message and ``perustelut`` names the reason; callers should
    never treat an abstained answer as factual.
    """

    model_config = ConfigDict(frozen=True)

    johtopaatos: str
    perustelut: str
    lahteet: list[RagSource] = Field(default_factory=list)
    abstained: bool = False
    abstain_reason: AbstainReason | None = None
    max_source_score: float | None = None


_ANSWER_SCHEMA = RagAnswer.model_json_schema()

_ANSWER_SYSTEM = (
    "Olet Lapuan päätöstekstiasiantuntija. "
    "Käytä VAIN annettua kontekstia Systeemistä. "
    "Et saa käyttää muistiasi tai päätellä kontekstin ulkopuolelta. "
    "Jos konteksti ei riitä vastaukseen, aseta kentät seuraavasti: "
    "abstained=true, abstain_reason='model_refused', "
    "johtopaatos='En löydä Systeemistä vastausta tähän kysymykseen.', "
    "perustelut='Kontekstissa ei ollut riittävää tietoa.' "
    "Muuten vastaa muodolla: Johtopäätös, Perustelut, Lähteet (§, kokous, pvm) "
    "ja aseta abstained=false."
)


_ABSTAIN_JOHTOPAATOS = "En löydä Systeemistä vastausta tähän kysymykseen."


def _build_context(
    results: list[RetrievalResult],
    *,
    max_chunks: int,
    max_chars_per_chunk: int,
) -> str:
    """Render retrieved chunks into the LLM prompt, with hard size bounds.

    The bounds protect us against OOM on CPU inference (quadratic attention
    memory) and keep the prompt within Qwen's effective context without
    requiring the model to stitch across dozens of excerpts.
    """
    blocks: list[str] = []
    for r in results[:max_chunks]:
        header = f"[{r.doc_id} | sivu {r.page_no}"
        if r.section_id:
            header += f" | {r.section_id}"
        header += "]"
        snippet = r.text if len(r.text) <= max_chars_per_chunk else r.text[:max_chars_per_chunk]
        blocks.append(f"{header}\n{snippet}")
    return "\n\n---\n\n".join(blocks)


def _abstain(
    *,
    reason: AbstainReason,
    results: list[RetrievalResult],
    explanation: str,
) -> RagAnswer:
    max_score = max((r.score for r in results), default=None)
    return RagAnswer(
        johtopaatos=_ABSTAIN_JOHTOPAATOS,
        perustelut=explanation,
        lahteet=[],
        abstained=True,
        abstain_reason=reason,
        max_source_score=max_score,
    )


@dataclass(slots=True)
class AnswerService:
    """Closed-book RAG answerer.

    Parameters
    ----------
    search
        Retrieval service over Systeemi.
    llm
        Qwen + LoRA client (local or vLLM).
    min_score
        Cross-encoder rerank score below which we consider the context
        insufficient and abstain without calling the LLM. BGE reranker-v2-m3
        typical ranges: ~0 relevant, negative irrelevant. 0.0 is a reasonable
        default; tune per tenant.
    """

    search: SearchService
    llm: LlmClient
    min_score: float = 0.0
    max_context_chunks: int = 3
    max_chars_per_chunk: int = 800

    def answer(self, *, query: str, tenant: str) -> RagAnswer:
        results = self.search.search(query=query, tenant=tenant)

        if not results:
            _log.info("rag.abstain", reason="no_context", tenant=tenant)
            return _abstain(
                reason="no_context",
                results=results,
                explanation=(
                    "Systeemistä ei löytynyt yhtään tämän kysymyksen kanssa "
                    "relevanttia katkelmaa."
                ),
            )

        top_score = max(r.score for r in results)
        if top_score < self.min_score:
            _log.info(
                "rag.abstain",
                reason="below_threshold",
                tenant=tenant,
                max_score=top_score,
                threshold=self.min_score,
            )
            return _abstain(
                reason="below_threshold",
                results=results,
                explanation=(
                    f"Parhaan vastaavuuden pisteet ({top_score:.3f}) jäivät luotettavuus"
                    f"kynnyksen ({self.min_score:.3f}) alle."
                ),
            )

        context = _build_context(
            results,
            max_chunks=self.max_context_chunks,
            max_chars_per_chunk=self.max_chars_per_chunk,
        )
        prompt = (
            "Kysymys:\n"
            f"{query}\n\n"
            "Konteksti Systeemistä (vain nämä katkelmat saa käyttää):\n"
            f"{context}\n"
        )
        raw = self.llm.generate_json(
            system=_ANSWER_SYSTEM,
            prompt=prompt,
            json_schema=_ANSWER_SCHEMA,
        )
        answer = RagAnswer.model_validate(raw)

        if answer.abstained:
            _log.info("rag.abstain", reason="model_refused", tenant=tenant)
            return answer.model_copy(update={"abstain_reason": "model_refused"})

        if not answer.lahteet:
            answer = answer.model_copy(
                update={
                    "lahteet": [
                        RagSource(
                            doc_id=r.doc_id,
                            page_no=r.page_no,
                            section_id=r.section_id,
                            snippet=r.text[:200],
                        )
                        for r in results
                    ],
                    "max_source_score": top_score,
                }
            )
        else:
            answer = answer.model_copy(update={"max_source_score": top_score})
        return answer
