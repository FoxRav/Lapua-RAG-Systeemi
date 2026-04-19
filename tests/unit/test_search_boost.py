"""Tests for SearchService chunk-type boost (v0.6.2).

The boost layer post-processes BGE rerank scores so decision-bearing
chunks (those that actually answer "kuka on / valittiin"-style questions)
outrank attendance lists that merely enumerate names with role labels.

The patterns are anchored on chunk-start headers and selection verbs that
appear verbatim in Lapuan kaupunki minutes; tuning was guided by the
v0.6.1 smoke-test transcript (see README §11).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from lapua_rag.retrieve import search as search_mod
from lapua_rag.retrieve.search import (
    _ATTENDANCE_PENALTY,
    _DECISION_BOOST,
    SearchService,
    _chunk_type_boost,
)


def test_decision_chunk_gets_boost() -> None:
    text = (
        "## Päätös\n"
        "Kaupunginvaltuusto valitsi kaupunginhallitukseen jäseneksi "
        "Harri Seppälän puheenjohtajaksi kaudelle 2025-2029."
    )
    assert _chunk_type_boost(text) == _DECISION_BOOST


def test_attendance_list_gets_penalty() -> None:
    text = (
        "## Saapuvillaolleet jäsenet\n"
        "Anneli Jäätteenmäki, puheenjohtaja\n"
        "Kai Pöntinen, 1. varapuheenjohtaja\n"
    )
    assert _chunk_type_boost(text) == _ATTENDANCE_PENALTY


def test_decision_without_selection_verb_not_boosted() -> None:
    """A bare ## Päätös header with no selection verb (e.g. budget approval
    text) should not be boosted since it does not answer a "kuka on"-style
    question. Avoids false positives on every decision chunk."""
    text = (
        "## Päätös\n"
        "Kaupunginhallitus merkitsi tiedoksi talousarvion toteuman."
    )
    assert _chunk_type_boost(text) == 0.0


def test_neutral_chunk_no_adjustment() -> None:
    text = "Tämä on aivan tavallinen tekstikatkelma vailla erityispatterneja."
    assert _chunk_type_boost(text) == 0.0


def test_attendance_with_decision_text_nets_to_boost_minus_penalty() -> None:
    """Defensive: a chunk that contains both an attendance list and a
    selection verb later on should net to (boost + penalty), not crash.
    This documents the additive contract."""
    text = (
        "## Saapuvillaolleet jäsenet\n"
        "Anneli Jäätteenmäki\n"
        "## Päätös\n"
        "Kaupunginhallitus valitsi puheenjohtajaksi Maijan."
    )
    expected = _DECISION_BOOST + _ATTENDANCE_PENALTY
    assert abs(_chunk_type_boost(text) - expected) < 1e-9


# ---------------------------------------------------------------------------
# SearchService end-to-end with stubbed retriever + reranker.
# ---------------------------------------------------------------------------


@dataclass
class _StubRetriever:
    fused: list[tuple[str, float]] = field(default_factory=list)

    def retrieve(
        self,
        *,
        query: str,
        tenant: str,
        filters: dict[str, object] | None = None,
    ) -> list[tuple[str, float]]:
        return self.fused


@dataclass
class _StubReranker:
    """Returns the cross-encoder scores supplied at construction time."""

    scores: dict[str, float] = field(default_factory=dict)

    def rerank(
        self,
        *,
        query: str,
        candidates: list[tuple[str, str]],
        top_k: int,
    ) -> list[tuple[str, float]]:
        ranked = sorted(
            ((cid, self.scores.get(cid, 0.0)) for cid, _ in candidates),
            key=lambda p: p[1],
            reverse=True,
        )
        return ranked[:top_k]


def test_chunk_type_boost_can_flip_top1(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """Replicates the v0.6.1 Q1 failure: BGE ranks attendance list 0.99 vs
    decision chunk 0.85. With +0.20 / -0.15 the decision chunk wins.
    """
    chunks = {
        "c_attend": {
            "chunk_id": "c_attend",
            "doc_id": "d1",
            "page_no": 1,
            "section_id": None,
            "text": (
                "## Saapuvillaolleet jäsenet\n"
                "Anneli Jäätteenmäki, puheenjohtaja\n"
                "Kai Pöntinen, 1. varapuheenjohtaja"
            ),
        },
        "c_decision": {
            "chunk_id": "c_decision",
            "doc_id": "d2",
            "page_no": 12,
            "section_id": "§ 7",
            "text": (
                "## Päätös\n"
                "Kaupunginvaltuusto valitsi kaupunginhallituksen jäseneksi ja "
                "puheenjohtajaksi Harri Seppälän kaudelle 2025-2029."
            ),
        },
    }

    monkeypatch.setattr(
        search_mod,
        "_load_chunk_texts",
        lambda chunk_ids: {cid: chunks[cid]["text"] for cid in chunk_ids if cid in chunks},
    )
    monkeypatch.setattr(
        search_mod,
        "_load_chunk_facts",
        lambda chunk_ids: {cid: chunks[cid] for cid in chunk_ids if cid in chunks},
    )

    svc = SearchService(
        retriever=_StubRetriever(fused=[("c_attend", 0.0), ("c_decision", 0.0)]),  # type: ignore[arg-type]
        reranker=_StubReranker(scores={"c_attend": 0.99, "c_decision": 0.85}),  # type: ignore[arg-type]
        top_k_final=8,
    )

    results = svc.search(query="Kuka on kaupunginhallituksen puheenjohtaja?", tenant="lapua")

    assert [r.chunk_id for r in results] == ["c_decision", "c_attend"]
    assert results[0].score == 0.85 + _DECISION_BOOST
    assert results[1].score == 0.99 + _ATTENDANCE_PENALTY


def test_chunk_type_boost_can_be_disabled(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """``apply_chunk_type_boost=False`` preserves raw BGE rankings, used by
    callers that want to compare against the pre-v0.6.2 behaviour or by
    tests that pin the underlying reranker."""
    chunks = {
        "c_attend": {
            "chunk_id": "c_attend",
            "doc_id": "d1",
            "page_no": 1,
            "section_id": None,
            "text": "## Saapuvillaolleet jäsenet\nAnneli Jäätteenmäki",
        },
        "c_decision": {
            "chunk_id": "c_decision",
            "doc_id": "d2",
            "page_no": 12,
            "section_id": None,
            "text": "## Päätös\nKaupunginvaltuusto valitsi puheenjohtajaksi Maijan.",
        },
    }
    monkeypatch.setattr(
        search_mod,
        "_load_chunk_texts",
        lambda chunk_ids: {cid: chunks[cid]["text"] for cid in chunk_ids if cid in chunks},
    )
    monkeypatch.setattr(
        search_mod,
        "_load_chunk_facts",
        lambda chunk_ids: {cid: chunks[cid] for cid in chunk_ids if cid in chunks},
    )
    svc = SearchService(
        retriever=_StubRetriever(fused=[("c_attend", 0.0), ("c_decision", 0.0)]),  # type: ignore[arg-type]
        reranker=_StubReranker(scores={"c_attend": 0.99, "c_decision": 0.85}),  # type: ignore[arg-type]
        top_k_final=8,
        apply_chunk_type_boost=False,
    )

    results = svc.search(query="?", tenant="lapua")
    assert [r.chunk_id for r in results] == ["c_attend", "c_decision"]
    assert results[0].score == 0.99
    assert results[1].score == 0.85
