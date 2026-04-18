"""Closed-book guardrail tests for AnswerService.

We do NOT load Qwen. Instead we mock the LLM client and retrieval service
to assert the service aborts deterministically before any LLM call whenever
Systeemi retrieval is empty or below the confidence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from lapua_rag.rag.answer import AnswerService, RagAnswer
from lapua_rag.retrieve.search import RetrievalResult


class _RecordingLlm:
    def __init__(self, payload: dict[str, object] | None = None) -> None:
        self.calls: list[tuple[str, str, dict[str, object]]] = []
        self._payload = payload or {
            "johtopaatos": "OK",
            "perustelut": "näin tein",
            "lahteet": [],
            "abstained": False,
        }

    def generate_json(
        self,
        *,
        system: str,
        prompt: str,
        json_schema: dict[str, object],
    ) -> dict[str, object]:
        self.calls.append((system, prompt, json_schema))
        return self._payload


@dataclass
class _FixedSearch:
    hits: list[RetrievalResult] = field(default_factory=list)

    def search(
        self,
        *,
        query: str,
        tenant: str,
        filters: dict[str, object] | None = None,
    ) -> list[RetrievalResult]:
        return self.hits


def test_abstains_when_no_context() -> None:
    llm = _RecordingLlm()
    svc = AnswerService(search=_FixedSearch(hits=[]), llm=llm, min_score=0.0)  # type: ignore[arg-type]

    ans = svc.answer(query="Mitä § 42?", tenant="lapua")

    assert ans.abstained is True
    assert ans.abstain_reason == "no_context"
    assert llm.calls == []  # LLM never invoked


def test_abstains_when_scores_below_threshold() -> None:
    llm = _RecordingLlm()
    hits = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            page_no=0,
            section_id="§ 1",
            score=-2.5,
            text="jotain",
        )
    ]
    svc = AnswerService(search=_FixedSearch(hits=hits), llm=llm, min_score=0.0)  # type: ignore[arg-type]

    ans = svc.answer(query="Mitä § 42?", tenant="lapua")

    assert ans.abstained is True
    assert ans.abstain_reason == "below_threshold"
    assert ans.max_source_score == -2.5
    assert llm.calls == []


def test_calls_llm_when_context_is_strong_enough() -> None:
    llm = _RecordingLlm(
        payload={
            "johtopaatos": "Hyväksyttiin talousarviomuutos.",
            "perustelut": "Kokouksessa äänestettiin...",
            "lahteet": [],
            "abstained": False,
        }
    )
    hits = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            page_no=3,
            section_id="§ 12",
            score=0.85,
            text="Kaupunginhallitus päätti hyväksyä talousarvion muutokset.",
        )
    ]
    svc = AnswerService(search=_FixedSearch(hits=hits), llm=llm, min_score=0.0)  # type: ignore[arg-type]

    ans = svc.answer(query="Mitä § 12 päätettiin?", tenant="lapua")

    assert ans.abstained is False
    assert ans.max_source_score == pytest.approx(0.85)
    assert len(ans.lahteet) == 1
    assert ans.lahteet[0].section_id == "§ 12"
    assert len(llm.calls) == 1


def test_backfills_sources_when_model_leaves_them_empty() -> None:
    llm = _RecordingLlm(
        payload={
            "johtopaatos": "Kyllä.",
            "perustelut": "...",
            "lahteet": [],
            "abstained": False,
        }
    )
    hits = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            page_no=1,
            section_id="§ 5",
            score=0.5,
            text="x" * 500,
        )
    ]
    svc = AnswerService(search=_FixedSearch(hits=hits), llm=llm, min_score=0.0)  # type: ignore[arg-type]

    ans = svc.answer(query="?", tenant="lapua")

    assert len(ans.lahteet) == 1
    assert ans.lahteet[0].snippet.startswith("x")
    assert len(ans.lahteet[0].snippet) == 200


def test_model_refusal_preserved() -> None:
    llm = _RecordingLlm(
        payload={
            "johtopaatos": "En löydä Systeemistä vastausta tähän kysymykseen.",
            "perustelut": "Kontekstissa ei ollut riittävää tietoa.",
            "lahteet": [],
            "abstained": True,
            "abstain_reason": "model_refused",
        }
    )
    hits = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            page_no=0,
            section_id=None,
            score=0.4,
            text="epärelevanttia",
        )
    ]
    svc = AnswerService(search=_FixedSearch(hits=hits), llm=llm, min_score=0.0)  # type: ignore[arg-type]

    ans = svc.answer(query="?", tenant="lapua")

    assert ans.abstained is True
    assert ans.abstain_reason == "model_refused"


def test_rag_answer_serializes() -> None:
    ans = RagAnswer(
        johtopaatos="Kyllä.",
        perustelut="x",
        lahteet=[],
        abstained=False,
    )
    data = ans.model_dump()
    assert data["abstained"] is False
    assert "abstain_reason" in data


def test_abstain_recovered_when_model_forgets_flag() -> None:
    """lapua-llm-v2 sometimes emits abstain template text but leaves
    abstained=False. The post-processor must flip the flag and clear
    sources to avoid a self-contradicting payload."""
    llm = _RecordingLlm(
        payload={
            "johtopaatos": "En löydä Systeemistä vastausta tähän kysymykseen.",
            "perustelut": "Kontekstissa ei ollut riittävää tietoa.",
            "lahteet": [],
            "abstained": False,
        }
    )
    hits = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            page_no=2,
            section_id="§ 9",
            score=0.92,
            text="Kaupunginhallitus valitsi puheenjohtajan.",
        )
    ]
    svc = AnswerService(search=_FixedSearch(hits=hits), llm=llm, min_score=0.0)  # type: ignore[arg-type]

    ans = svc.answer(query="?", tenant="lapua")

    assert ans.abstained is True
    assert ans.abstain_reason == "model_refused"
    assert ans.lahteet == []
    assert ans.max_source_score == pytest.approx(0.92)


def test_retrieve_mode_skips_llm_and_returns_cited_chunks() -> None:
    """Retrieve-mode bypasses the LLM and surfaces top-N reranked
    chunks verbatim with citations."""
    llm = _RecordingLlm()
    hits = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            page_no=5,
            section_id="§ 12",
            score=0.91,
            text="Kaupunginhallitus valitsi puheenjohtajaksi Maija Mallikkaan." * 20,
        ),
        RetrievalResult(
            chunk_id="c2",
            doc_id="d2",
            page_no=8,
            section_id=None,
            score=0.45,
            text="Toinen katkelma.",
        ),
    ]
    svc = AnswerService(
        search=_FixedSearch(hits=hits),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        max_context_chunks=5,
        mode="retrieve",
        retrieve_snippet_chars=300,
    )

    ans = svc.answer(query="Kuka valittiin?", tenant="lapua")

    assert llm.calls == []  # LLM never invoked in retrieve-mode
    assert ans.abstained is False
    assert ans.abstain_reason is None
    assert ans.max_source_score == pytest.approx(0.91)
    assert len(ans.lahteet) == 2
    assert ans.lahteet[0].doc_id == "d1"
    assert len(ans.lahteet[0].snippet) == 300  # truncated to retrieve_snippet_chars
    assert ans.lahteet[1].snippet == "Toinen katkelma."  # short text not padded


def test_retrieve_mode_respects_no_context_gate() -> None:
    """No retrieval results → abstain even in retrieve-mode."""
    llm = _RecordingLlm()
    svc = AnswerService(
        search=_FixedSearch(hits=[]),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="retrieve",
    )

    ans = svc.answer(query="?", tenant="lapua")

    assert ans.abstained is True
    assert ans.abstain_reason == "no_context"
    assert llm.calls == []


def test_retrieve_mode_respects_below_threshold_gate() -> None:
    """Top score below min_score → abstain even in retrieve-mode."""
    llm = _RecordingLlm()
    hits = [
        RetrievalResult(
            chunk_id="c1",
            doc_id="d1",
            page_no=0,
            section_id=None,
            score=-1.5,
            text="irrelevant",
        )
    ]
    svc = AnswerService(
        search=_FixedSearch(hits=hits),  # type: ignore[arg-type]
        llm=llm,
        min_score=0.0,
        mode="retrieve",
    )

    ans = svc.answer(query="?", tenant="lapua")

    assert ans.abstained is True
    assert ans.abstain_reason == "below_threshold"
    assert ans.max_source_score == pytest.approx(-1.5)
    assert llm.calls == []
