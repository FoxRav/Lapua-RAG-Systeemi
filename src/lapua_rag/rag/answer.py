"""RAG answer service: *closed-book over Systeemi*.

Three operating modes (``Settings.answer_mode``):

* ``synth`` — Qwen2.5-1.5B + LoRA ``lapua-llm-v2`` synthesises a JSON
  answer from the top reranked chunks. Closed-book: the model is only
  allowed to use the retrieved context, never its pretraining memory.
* ``retrieve`` — skip the LLM entirely and return the top-N reranked
  chunks verbatim with citations. Use this when the synthesis model is
  unreliable and the user is better served by reading the cited
  evidence directly.
* ``extract`` — bridge mode (v0.6) for use while ``lapua-llm-v3`` is
  in training. The LoRA performs only the narrow task of quoting the
  most relevant 1-3 sentences from the retrieved context (a task that
  empirically bypasses lapua-llm-v2's abstain bias because the prompt
  never asks the model to *answer* — only to *cite verbatim*). Python
  then renders one coherent answer from the quote. If the LLM still
  refuses or returns an empty quote, a deterministic Python fallback
  picks the highest-overlap sentence from the top chunk so the user
  always gets one cited answer rather than a 5-chunk dump.

In all modes ``no_context`` and ``below_threshold`` short-circuit
deterministically in Python **before** any LLM call. This is a product
guarantee, not a model hint.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from lapua_rag.extract.llm import LlmClient
from lapua_rag.observability import get_logger
from lapua_rag.retrieve.search import RetrievalResult, SearchService

_log = get_logger(__name__)

AbstainReason = Literal[
    "no_context",
    "below_threshold",
    "model_refused",
]

AnswerMode = Literal["synth", "retrieve", "extract"]


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


# ---------------------------------------------------------------------------
# Extract-mode helpers (v0.6 bridge while lapua-llm-v3 is in training).
# ---------------------------------------------------------------------------


class _ExtractResponse(BaseModel):
    """Narrow LLM contract for ``extract`` mode.

    The schema deliberately avoids the word "answer" or "vastaus" — the LoRA's
    abstain bias is keyed to those tokens. We ask only for a verbatim quote
    plus a 1-based index pointing back to the source chunk.
    """

    model_config = ConfigDict(frozen=True)

    quote: str = Field(default="", max_length=600)
    chunk_index: int = Field(default=1, ge=1)
    no_match: bool = False


_EXTRACT_SYSTEM = (
    "Olet Lapuan kaupungin päätöstekstien tarkkuus-extraktori. "
    "Saat numeroidut katkelmat (1, 2, 3, ...) ja kysymyksen. "
    "Tehtäväsi: poimi katkelmista tasan ne 1–3 virkettä jotka sisältävät "
    "kysymyksessä haetun tiedon. "
    "Säännöt: "
    "(1) Lainaa SANATARKASTI — älä muotoile uudelleen, älä lisää sanoja. "
    "(2) Aseta 'chunk_index' siihen numeroon (1..N) josta lainasit. "
    "(3) Jos katkelmissa EI OLE tietoa kysymykseen, aseta no_match=true ja "
    "jätä quote tyhjäksi. Älä keksi vastausta. "
    "Vastaa AINOASTAAN JSON-objektilla muodossa "
    '{"quote": "...", "chunk_index": N, "no_match": false} '
    "ilman selityksiä, koodilohkoja tai ympäröivää tekstiä."
)


# Match the first {...} block in the model's reply so we can recover even
# when the LoRA wraps the JSON in markdown fences or prepends a few words.
_JSON_OBJECT_RE = re.compile(r"\{.*?\}", re.DOTALL)


def _parse_extract_response(text: str) -> _ExtractResponse | None:
    """Lenient JSON parser for extract-mode plain-text responses.

    The LoRA was trained on JSON and usually emits it cleanly, but small
    quants sometimes wrap it in ``` fences or add a leading word. We accept
    the first JSON object we can find; if validation fails we surrender to
    the deterministic Python fallback rather than guess.
    """
    text = text.strip()
    if not text:
        return None
    candidates: list[str] = [text]
    match = _JSON_OBJECT_RE.search(text)
    if match is not None and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            raw = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        try:
            return _ExtractResponse.model_validate(raw)
        except ValidationError:
            continue
    return None


# Sentence-splitter regex tuned for OCR'd Finnish meeting minutes:
# split after .!? when followed by whitespace and a capital letter or digit
# (pages numbers, decision IDs). Keeps multi-sentence quoting intact.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-ZÅÄÖ0-9])")
_HTML_TAG = re.compile(r"<[^>]+>")
_MD_HEADER = re.compile(r"^\s*#{1,6}\s+", flags=re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"\w+", flags=re.UNICODE)
# Finnish stopwords kept tiny on purpose — we don't want to over-strip in
# short queries. Goal: filter only the most empty connectors so token-overlap
# scoring focuses on content nouns.
_QUERY_STOPWORDS: frozenset[str] = frozenset(
    {
        "on", "ei", "ja", "tai", "se", "että", "joka", "mikä",
        "kuka", "missä", "milloin", "miten", "mitä", "kuinka",
        "kenen", "ovat", "ole", "tämä", "tuo", "voi",
    }
)


def _split_sentences(text: str) -> list[str]:
    """Split OCR'd Finnish prose into sentences, dropping markup noise.

    Robust to HTML tables, markdown headers, and double newlines that the
    PP-StructureV3 pipeline emits. We deliberately keep the implementation
    naive — a heavier NLP splitter would be overkill for the
    quote-fallback path and add a startup cost we don't want.
    """
    cleaned = _HTML_TAG.sub(" ", text)
    cleaned = _MD_HEADER.sub("", cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if not cleaned:
        return []
    parts = _SENTENCE_SPLIT.split(cleaned)
    # Filter very short fragments — OCR list items (single names) score
    # poorly anyway and add noise to the ranking.
    return [p.strip() for p in parts if len(p.strip()) >= 20]


def _query_tokens(query: str) -> set[str]:
    return {
        t for t in (m.group(0).lower() for m in _WORD.finditer(query))
        if len(t) >= 4 and t not in _QUERY_STOPWORDS
    }


def _score_sentence(sentence: str, query_terms: set[str]) -> int:
    if not query_terms:
        return 0
    tokens = {m.group(0).lower() for m in _WORD.finditer(sentence)}
    return len(tokens & query_terms)


def _python_fallback_quote(query: str, chunk_text: str, max_chars: int = 600) -> str:
    """Pick the most query-relevant 1-2 sentences from ``chunk_text``.

    Used when the LoRA returns ``no_match=true`` or an empty quote despite
    the gate having allowed the chunk through. This guarantees the user
    always gets one cited sentence, not a silent abstain in extract mode.
    """
    sentences = _split_sentences(chunk_text)
    if not sentences:
        return chunk_text[:max_chars].strip()
    terms = _query_tokens(query)
    # Sort by (-score, length) so the most overlap wins; on ties prefer the
    # shorter, more focused sentence over a wall-of-text rebuttal.
    ranked = sorted(
        sentences,
        key=lambda s: (-_score_sentence(s, terms), len(s)),
    )
    chosen: list[str] = []
    total = 0
    for sent in ranked[:2]:
        budget = max_chars - total - (1 if chosen else 0)
        if budget <= 20:
            break
        chosen.append(sent[:budget])
        total += len(chosen[-1]) + 1
    return " ".join(chosen)


def _python_fallback_across_chunks(
    query: str,
    results: list[RetrievalResult],
    *,
    max_chars: int = 600,
) -> tuple[int, str]:
    """Find the highest-overlap sentence across *all* retrieved chunks.

    v0.6.2: when the LoRA correctly says ``no_match`` it usually means the
    reranker top-1 chunk does not actually answer the question (e.g. an
    attendance list selected for a "kuka on" query). Falling back to the
    top-1's text in that case surfaces noise. Searching across the entire
    retrieved set lets the deterministic layer recover the right chunk
    when it exists in the rerank pool.

    Returns ``(1-based chunk index in results, quote)``. Always returns
    something — if no sentence has any overlap we surface the top-1 chunk's
    truncated text so the user still gets a cited reply.
    """
    if not results:
        return 1, ""
    terms = _query_tokens(query)
    best_score = -1
    best_idx = 0
    best_sentence = ""
    for idx, r in enumerate(results):
        sentences = _split_sentences(r.text)
        for sent in sentences:
            score = _score_sentence(sent, terms)
            if score > best_score:
                best_score = score
                best_idx = idx
                best_sentence = sent
    if best_score <= 0:
        # No overlap anywhere — keep the existing top-1 behaviour for
        # backward compatibility (better than empty answer).
        return 1, _python_fallback_quote(query, results[0].text, max_chars=max_chars)
    quote = best_sentence[:max_chars].strip()
    return best_idx + 1, quote


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


def _build_numbered_context(
    results: list[RetrievalResult],
    *,
    max_chunks: int,
    max_chars_per_chunk: int,
) -> str:
    """Render context for extract mode with explicit 1-based chunk indices.

    The model is asked to return the index it quoted from, so each block
    must be prefixed with a stable, easily-tokenised number (the LoRA's
    tokeniser splits ``[1]`` cleanly).
    """
    blocks: list[str] = []
    for idx, r in enumerate(results[:max_chunks], start=1):
        header_meta = f"{r.doc_id} | sivu {r.page_no}"
        if r.section_id:
            header_meta += f" | {r.section_id}"
        snippet = r.text if len(r.text) <= max_chars_per_chunk else r.text[:max_chars_per_chunk]
        blocks.append(f"[{idx}] ({header_meta})\n{snippet}")
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
    # Tighter bounds for extract-mode prompts: the LoRA only needs to find a
    # single quote, and v0.6.0 smoke tests showed that 5 * 1500 chars +
    # outlines guided_json crashed vLLM's AsyncLLMEngine on 6 GB GPUs.
    extract_max_chunks: int = 3
    extract_max_chars_per_chunk: int = 800
    extract_max_new_tokens: int = 256

    def answer(self, *, query: str, tenant: str) -> RagAnswer:
        results = self.search.search(query=query, tenant=tenant)

        gate = self._gate(results=results, tenant=tenant)
        if gate is not None:
            return gate

        top_score = max(r.score for r in results)
        if self.mode == "retrieve":
            return self._retrieve_answer(results=results, top_score=top_score, tenant=tenant)
        if self.mode == "extract":
            return self._extract_answer(
                query=query,
                results=results,
                top_score=top_score,
                tenant=tenant,
            )
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
    # Extract mode: LoRA performs only a narrow quote-extraction task;
    # Python renders the final answer template. Falls back to a
    # deterministic sentence picker if the LoRA still refuses.
    def _extract_answer(
        self,
        *,
        query: str,
        results: list[RetrievalResult],
        top_score: float,
        tenant: str,
    ) -> RagAnswer:
        n = min(len(results), self.extract_max_chunks)
        context = _build_numbered_context(
            results,
            max_chunks=n,
            max_chars_per_chunk=self.extract_max_chars_per_chunk,
        )
        prompt = (
            f"Kysymys:\n{query}\n\n"
            f"Numeroidut katkelmat Systeemistä:\n{context}\n\n"
            "Palauta JSON jossa quote on sanatarkka lainaus (1–3 virkettä), "
            "chunk_index on sen katkelman numero (1..N) josta lainasit, ja "
            "no_match=true VAIN jos mikään katkelma ei sisällä vastausta."
        )
        _log.info(
            "rag.extract_call",
            tenant=tenant,
            n_results=len(results),
            n_context_chunks=n,
            context_chars=len(context),
            top_score=top_score,
        )

        used_index, quote, used_python_fallback = self._extract_quote(
            query=query,
            prompt=prompt,
            results=results,
            n_chunks=n,
            tenant=tenant,
        )

        used = results[used_index - 1]
        header = f"[{used.doc_id} | sivu {used.page_no}"
        if used.section_id:
            header += f" | {used.section_id}"
        header += "]"
        perustelut_source = (
            "Python-fallback: lainattu suoraan korkeimman pisteen lähteestä "
            if used_python_fallback
            else "Lainattu suoraan lähteestä "
        )
        perustelut = f"{perustelut_source}{header} (reranker-score {used.score:.3f})."

        # Lähteet: ensin lainauksen lähde, sitten muut top-N kontekstista
        # tukimateriaaliksi (lyhennetyillä snipeteillä). Käyttäjä näkee mistä
        # lainaus tulee mutta voi silti tarkistaa tukevat katkelmat.
        sources = [
            RagSource(
                chunk_id=used.chunk_id,
                doc_id=used.doc_id,
                page_no=used.page_no,
                section_id=used.section_id,
                snippet=used.text[: self.retrieve_snippet_chars],
            )
        ]
        for r in results[:n]:
            if r.chunk_id == used.chunk_id:
                continue
            sources.append(
                RagSource(
                    chunk_id=r.chunk_id,
                    doc_id=r.doc_id,
                    page_no=r.page_no,
                    section_id=r.section_id,
                    snippet=r.text[: self.retrieve_snippet_chars],
                )
            )

        return RagAnswer(
            johtopaatos=quote,
            perustelut=perustelut,
            lahteet=sources,
            abstained=False,
            abstain_reason=None,
            max_source_score=top_score,
        )

    def _extract_quote(
        self,
        *,
        query: str,
        prompt: str,
        results: list[RetrievalResult],
        n_chunks: int,
        tenant: str,
    ) -> tuple[int, str, bool]:
        """Run the LoRA extract call and apply the Python fallback.

        Returns ``(used_chunk_index, quote, used_python_fallback)``. The
        index is 1-based and clamped into ``[1, n_chunks]`` so downstream
        code can index ``results`` safely even when the LLM returns junk.

        v0.6.1: switched from ``generate_json`` (outlines guided_json) to
        ``generate_text`` + lenient JSON parser. The schema only has three
        fields, so guided decoding's overhead was killing the small GPU
        without buying us reliability we can't recover from in Python.
        """
        try:
            text = self.llm.generate_text(
                system=_EXTRACT_SYSTEM,
                prompt=prompt,
                max_new_tokens=256,
            )
        except Exception as exc:
            # Narrow except is hard here: httpx errors, retries exhausted,
            # connection refused. Logged structurally and a Python fallback
            # ensures the user still gets a coherent cited answer rather
            # than a raw error.
            _log.warning(
                "rag.extract_llm_failed",
                tenant=tenant,
                error_type=type(exc).__name__,
                error=str(exc)[:200],
            )
            chunk_idx, quote = _python_fallback_across_chunks(query, results)
            return chunk_idx, quote, True

        parsed = _parse_extract_response(text)
        if parsed is None:
            _log.info(
                "rag.extract_python_fallback",
                tenant=tenant,
                reason="parse_failed",
                raw_preview=text[:120],
            )
            chunk_idx, quote = _python_fallback_across_chunks(query, results)
            return chunk_idx, quote, True

        if parsed.no_match or not parsed.quote.strip():
            chunk_idx, quote = _python_fallback_across_chunks(query, results)
            _log.info(
                "rag.extract_python_fallback",
                tenant=tenant,
                reason="no_match" if parsed.no_match else "empty_quote",
                chosen_chunk_idx=chunk_idx,
            )
            return chunk_idx, quote, True

        # Clamp the model-supplied index to the range we actually rendered;
        # the LoRA occasionally emits 0 or an out-of-range integer despite
        # the schema's ge=1 constraint when temperature noise sneaks past.
        idx = max(1, min(parsed.chunk_index, n_chunks))
        return idx, parsed.quote.strip(), False

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
