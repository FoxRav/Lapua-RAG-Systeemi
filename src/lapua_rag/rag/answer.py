"""RAG answer service: *closed-book over Systeemi*.

Two operating modes (``Settings.answer_mode``):

* ``synth`` — Qwen2.5-1.5B + LoRA ``lapua-llm-v2`` synthesises a JSON
  answer from the top reranked chunks. Closed-book: the model is only
  allowed to use the retrieved context, never its pretraining memory.
* ``retrieve`` — skip the LLM entirely and return the top-N reranked
  chunks verbatim with citations. Use this when the synthesis model is
  unreliable (e.g. an overtrained-abstain LoRA) and the user is better
  served by reading the cited evidence directly. Same abstain
  guarantees apply: ``no_context`` and ``below_threshold`` still
  short-circuit before any results are returned.

The abstention decision in both modes is made deterministically in
Python **before** the LLM call (synth) or **before** the chunks are
returned (retrieve). This is a product guarantee, not a model hint.
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

AnswerMode = Literal["synth", "retrieve"]


class RagSource(BaseModel):
    model_config = ConfigDict(frozen=True)

    # chunk_id is optional for backwards compatibility with model outputs
    # that pre-date the field; new retrieval paths always populate it so
    # the UI can fetch the full chunk via GET /v1/chunks/{chunk_id}.
    chunk_id: str | None = None
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
    "Olet Lapuan kaupungin päätöstekstien asiantuntija. "
    "Tehtäväsi: vastaa kysymykseen kontekstissa annettujen Systeemi-katkelmien "
    "perusteella mahdollisimman täsmällisesti. "
    "Lue kaikki katkelmat huolellisesti — vastaus voi olla missä tahansa niistä, "
    "myös viimeisessä. "
    "Pakolliset säännöt: "
    "(1) Käytä vain annettua kontekstia, älä omaa muistiasi. "
    "(2) Jos kontekstissa on suora vastaus, anna se: täytä 'johtopaatos' "
    "yhdellä-kahdella lauseella, 'perustelut' lyhyellä viittauksella mihin "
    "katkelmaan vastaus perustuu, ja 'lahteet' niistä katkelmista joita käytit "
    "(doc_id, page_no, section_id otsikon [doc_id | sivu N | §...] mukaisesti). "
    "Aseta abstained=false. "
    "(3) Vain jos KONTEKSTISSA EI OLE vastauksen edellytyksiä lainkaan, aseta "
    "abstained=true, abstain_reason='model_refused', "
    "johtopaatos='En löydä Systeemistä vastausta tähän kysymykseen.', "
    "perustelut='Kontekstissa ei ollut riittävää tietoa.' "
    "Älä abstainaa varmuuden vuoksi — abstain on viimesijainen turvaverkko."
)

# Few-shot exemplar to counter lapua-llm-v2's over-strong abstain bias.
# The LoRA was trained heavily on refusal patterns and reflexively answers
# "En löydä Systeemistä vastausta" even when the literal answer is visible
# in the retrieved chunks. One concrete positive example primes it to
# extract when the context warrants it. Keep this short — every token
# competes for Qwen2.5-1.5B's small effective context.
_FEW_SHOT_EXAMPLE = (
    "Esimerkki — suora vastaus löytyy kontekstista:\n"
    "Kysymys: Kuka valittiin esimerkkikaupungin sivistystoimenjohtajaksi?\n\n"
    "Konteksti Systeemistä:\n"
    "[demo123 | sivu 5 | §42]\n"
    "## Päätös\n"
    "Kaupunginhallitus valitsi sivistystoimenjohtajaksi Maija Mallikkaan "
    "kaudelle 2025-2029.\n\n"
    "Odotettu vastaus:\n"
    "{\"johtopaatos\": \"Sivistystoimenjohtajaksi valittiin Maija Mallikas "
    "kaudelle 2025-2029.\", "
    "\"perustelut\": \"Päätös ilmenee suoraan katkelmasta [demo123 | sivu 5 | §42].\", "
    "\"lahteet\": [{\"doc_id\": \"demo123\", \"page_no\": 5, "
    "\"section_id\": \"§42\", "
    "\"snippet\": \"Kaupunginhallitus valitsi sivistystoimenjohtajaksi "
    "Maija Mallikkaan kaudelle 2025-2029.\"}], "
    "\"abstained\": false, \"abstain_reason\": null, "
    "\"max_source_score\": null}\n\n"
    "--- Varsinainen tehtävä alkaa nyt ---\n\n"
)


_ABSTAIN_JOHTOPAATOS = "En löydä Systeemistä vastausta tähän kysymykseen."

# Phrases that indicate the model wanted to abstain but forgot to flip the
# `abstained` flag. lapua-llm-v2 is trained to emit these strings; without
# this guard the API would expose a contradictory payload (abstained=false
# but johtopaatos = abstain template). Keep matchers narrow & lowercase.
_ABSTAIN_PATTERNS: tuple[str, ...] = (
    "en löydä systeemistä vastausta",
    "kontekstissa ei ollut riittävää tietoa",
    "kontekstissa ei ole riittävää tietoa",
)


def _looks_like_abstain(answer: RagAnswer) -> bool:
    text = f"{answer.johtopaatos}\n{answer.perustelut}".lower()
    return any(pat in text for pat in _ABSTAIN_PATTERNS)


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
    max_context_chunks: int = 5
    max_chars_per_chunk: int = 1500
    mode: AnswerMode = "synth"
    # Per-source snippet length when mode == "retrieve". Larger than the
    # synth-mode default (200) because the user reads the snippet directly.
    retrieve_snippet_chars: int = 600

    def answer(self, *, query: str, tenant: str) -> RagAnswer:
        results = self.search.search(query=query, tenant=tenant)

        gate = self._gate(results=results, tenant=tenant)
        if gate is not None:
            return gate

        top_score = max(r.score for r in results)
        if self.mode == "retrieve":
            return self._retrieve_answer(results=results, top_score=top_score, tenant=tenant)
        return self._synth_answer(
            query=query,
            results=results,
            top_score=top_score,
            tenant=tenant,
        )

    # ------------------------------------------------------------------
    # Pre-LLM gate: same abstain guarantees apply to both modes.
    def _gate(
        self,
        *,
        results: list[RetrievalResult],
        tenant: str,
    ) -> RagAnswer | None:
        if not results:
            _log.info("rag.abstain", reason="no_context", tenant=tenant, mode=self.mode)
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
                mode=self.mode,
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
        return None

    # ------------------------------------------------------------------
    # Retrieve mode: bypass the LLM, surface top-N cited chunks verbatim.
    # The user reads the evidence directly; no synthesis = no hallucination.
    def _retrieve_answer(
        self,
        *,
        results: list[RetrievalResult],
        top_score: float,
        tenant: str,
    ) -> RagAnswer:
        n = min(len(results), self.max_context_chunks)
        sources = [
            RagSource(
                chunk_id=r.chunk_id,
                doc_id=r.doc_id,
                page_no=r.page_no,
                section_id=r.section_id,
                snippet=r.text[: self.retrieve_snippet_chars],
            )
            for r in results[:n]
        ]
        _log.info(
            "rag.retrieve",
            tenant=tenant,
            n_sources=n,
            top_score=top_score,
        )
        return RagAnswer(
            johtopaatos=(
                f"Synteesi pois käytöstä (mode=retrieve). Löytyi {n} relevanttia "
                "katkelmaa Systeemistä; lue suoraan lähteistä alla."
            ),
            perustelut=(
                f"Reranker top-score {top_score:.3f}. "
                "Lähteet ovat järjestyksessä relevanttiudeltaan laskevasti."
            ),
            lahteet=sources,
            abstained=False,
            abstain_reason=None,
            max_source_score=top_score,
        )

    # ------------------------------------------------------------------
    # Synth mode: build prompt, call LLM, validate, recover abstain semantics.
    def _synth_answer(
        self,
        *,
        query: str,
        results: list[RetrievalResult],
        top_score: float,
        tenant: str,
    ) -> RagAnswer:
        context = _build_context(
            results,
            max_chunks=self.max_context_chunks,
            max_chars_per_chunk=self.max_chars_per_chunk,
        )
        prompt = (
            f"{_FEW_SHOT_EXAMPLE}"
            "Kysymys:\n"
            f"{query}\n\n"
            "Konteksti Systeemistä (vain nämä katkelmat saa käyttää):\n"
            f"{context}\n"
        )
        # Diagnostic event: how much context did the LLM actually see?
        # Useful to spot silent truncation / chunk-cap mismatches with retrieval.
        _log.info(
            "rag.llm_call",
            tenant=tenant,
            n_results=len(results),
            n_context_chunks=min(len(results), self.max_context_chunks),
            context_chars=len(context),
            top_score=top_score,
        )
        raw = self.llm.generate_json(
            system=_ANSWER_SYSTEM,
            prompt=prompt,
            json_schema=_ANSWER_SCHEMA,
        )
        answer = RagAnswer.model_validate(raw)

        # The LoRA sometimes emits abstain template text but forgets to set
        # abstained=true. Recover the intended semantics deterministically.
        if not answer.abstained and _looks_like_abstain(answer):
            _log.info(
                "rag.abstain_recovered",
                tenant=tenant,
                top_score=top_score,
            )
            answer = answer.model_copy(
                update={
                    "abstained": True,
                    "johtopaatos": _ABSTAIN_JOHTOPAATOS,
                    "perustelut": "Kontekstissa ei ollut riittävää tietoa.",
                    "lahteet": [],
                }
            )

        if answer.abstained:
            _log.info(
                "rag.abstain",
                reason="model_refused",
                tenant=tenant,
                top_score=top_score,
            )
            return answer.model_copy(
                update={
                    "abstain_reason": "model_refused",
                    "max_source_score": top_score,
                }
            )

        if not answer.lahteet:
            answer = answer.model_copy(
                update={
                    "lahteet": [
                        RagSource(
                            chunk_id=r.chunk_id,
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
